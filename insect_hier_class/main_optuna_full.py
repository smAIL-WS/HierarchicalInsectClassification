
"""
Modified main.py with Optuna integration.
Objective function rebuilds full training pipeline for each trial.
"""

# ---------------------------
# Allocator / runtime config
# ---------------------------
# Place these BEFORE any `import torch` so they apply to the whole process and all CUDA contexts.
import os
# # Set this to make stack traces accurate for debugging "illegal memory access"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Use the modern key and add a GC threshold to prevent fragmentation during many HPO trials
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")
# Keep cuDNN algorithm selection predictable to avoid large workspace jumps during HPO
os.environ.setdefault("CUDNN_BENCHMARK", "False")

import gc
from pathlib import Path
import random
import numpy as np
import time
import traceback
import timm
import sys
import torch
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    resnet18, resnet34, resnet50, resnext101_32x8d,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNeXt101_32X8D_Weights
)

import argparse
from torch.optim.lr_scheduler import OneCycleLR
from datetime import datetime
from multiprocessing import Manager
import signal
import sys
import optuna
from optuna.pruners import MedianPruner, PatientPruner
import config
from hierarchical_model import HIFD2, BackboneWrapper
from tree_loss import TreeLoss
from insect_dataset_loader import InsectDataset, DynamicTransform
from train_test import train, test
from transforms_utils import build_train_transform, build_test_transform

# Stabilize cuDNN memory footprint during HPO
torch.backends.cudnn.benchmark = False

# --- GLOBAL FLAG: set True when an interrupt is received ---
STOP_REQUESTED = False

def _signal_handler(sig, frame):
    global STOP_REQUESTED
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        print("\n[INTERRUPT] Received CTRL+C. Stopping study after current trials finish...")
    else:
        print("\n[INTERRUPT] Forced exit. Terminating immediately...")
        os._exit(1) # Hard kill on second CTRL+C

def _install_signal_handlers():
    """Register signal handlers for CTRL+C (SIGINT) and termination (SIGTERM)."""
    try:
        signal.signal(signal.SIGINT, _signal_handler)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        pass

def _stop_callback(study: optuna.Study, frozen_trial: optuna.trial.FrozenTrial):
    """
    Optuna callback called after each trial. If STOP_REQUESTED is set, stop the study.
    This ensures a clean stop with trials persisted to storage.
    """
    if STOP_REQUESTED:
        print("[INTERRUPT] Stop requested. Calling study.stop() ...")
        study.stop()

def _cleanup_on_exit(study: optuna.Study | None = None):
    """
    Run final cleanup before exiting: free GPU memory, print a brief summary,
    and close resources if any were opened outside Optuna.
    """
    try:
        # Free CUDA cache to avoid OOM issues in subsequent runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Brief summary of completed trials (optional)
        if study is not None:
            trials = study.get_trials(deepcopy=False)
            completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
            failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
            print("\n[SUMMARY] Trials:")
            print(f"  COMPLETE: {len(completed)}")
            print(f"  PRUNED  : {len(pruned)}")
            print(f"  FAIL    : {len(failed)}")
        print("[CLEANUP] Done. Exiting.")
    except Exception:
        print("[CLEANUP] Exception during cleanup:")
        traceback.print_exc()

# ---------------------------
# Utilities for per-trial teardown
# ---------------------------
def _flush_cuda(device: torch.device) -> None:
    """Empty CUDA cache safely with synchronization."""
    if torch.cuda.is_available() and device.type == "cuda":
        try:
            # Set the device context explicitly
            torch.cuda.set_device(device)
            # Ensure all kernels are finished before clearing memory
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            # Optional: second sync to ensure memory is released back to the driver
            torch.cuda.synchronize(device)
        except Exception as e:
            print(f"[CLEANUP] Warning: Could not flush CUDA on {device}: {e}")

def _del_and_collect(*objs) -> None:
    """Delete references and run GC."""
    for o in objs:
        try:
            # If it's a model, move it to CPU first to 'detach' from GPU memory
            if isinstance(o, torch.nn.Module):
                o.to("cpu")
            del o
        except Exception:
            pass
    gc.collect()

# Reuse helper functions from original script
def arg_parse():
    parser = argparse.ArgumentParser(description='Hierarchical Insect Classification Training')
    parser.add_argument('--worker', default=config.DEFAULT_NUM_WORKERS, type=int)
    parser.add_argument('--seed', type=int, default=config.DEFAULT_SEED)
    parser.add_argument('--epoch', type=int, default=config.DEFAULT_NUM_EPOCHS)
    parser.add_argument('--batch', type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument('--dataset', type=str, default=config.DATASET_NAME)
    parser.add_argument('--img_size', type=int, default=config.PROGRESSIVE_RESIZE_SCHEDULE['initial'])
    parser.add_argument('--lr_adjt', type=str, default='OneCycle', choices=['Cos', 'Step', 'Fixed', 'OneCycle'])
    parser.add_argument('--device', nargs='+', default=['0'])
    parser.add_argument('--backbone', type=str, default='resnet18', choices=list(config.BACKBONE_CONFIGS.keys()))
    parser.add_argument('--use_pretrained', action='store_true')
    return parser.parse_args()


def get_transform(image_size):
    return build_train_transform(image_size=image_size, aug=config.TRAIN_AUGMENTATION)


def get_test_transform(image_size):
    return build_test_transform(image_size=image_size, aug=config.TEST_AUGMENTATION)


def setup_backbone(backbone_name: str, use_pretrained: bool = False):
    backbone_name = backbone_name.lower()
    if backbone_name not in config.BACKBONE_CONFIGS:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    backbone_config = config.BACKBONE_CONFIGS[backbone_name]
    num_ftrs = backbone_config['num_ftrs']
    feature_sizes = backbone_config['feature_sizes']

    if backbone_name == 'resnet18':
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        raw_backbone = resnet18(weights=weights)

    elif backbone_name == 'resnet34':
        weights = ResNet34_Weights.DEFAULT if use_pretrained else None
        raw_backbone = resnet34(weights=weights)

    elif backbone_name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        raw_backbone = resnet50(weights=weights)

    elif backbone_name == 'resnext101':
        # torchvision’s name for 32x8d variant
        weights = ResNeXt101_32X8D_Weights.DEFAULT if use_pretrained else None
        raw_backbone = resnext101_32x8d(weights=weights)

    elif backbone_name == 'efficientnetv2_s':
        # timm uses 'pretrained' flag rather than weights classes
        raw_backbone = timm.create_model('efficientnetv2_s', pretrained=use_pretrained)

    elif backbone_name == 'mobilenetv3_small':
        weights = MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None
        raw_backbone = mobilenet_v3_small(weights=weights)

    else:
        raise ValueError(f"Backbone {backbone_name} not implemented")

    backbone = BackboneWrapper(raw_backbone)
    return backbone, num_ftrs, feature_sizes

# Optuna objective function

def objective(trial: optuna.trial.Trial, args):
    # Check if a stop was requested before even starting
    if STOP_REQUESTED:
        raise optuna.TrialPruned()

    # args = arg_parse()

    # -------- paths: keep run_folder; set per-trial output_folder --------
    base_run = Path(config.RUN_FOLDER)
    output_folder = base_run / "optuna_trials" / f"trial_{trial.number}"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Use a helper function for thread-safe logging instead of the Tee class
    log_path = output_folder / "trial.log"
    def log_print(msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[Trial {trial.number} | {timestamp}] {msg}"
        print(formatted_msg)  # Main console
        with open(log_path, "a") as f:
            f.write(formatted_msg + "\n")

    try:
        log_print(f"--- Starting Trial {trial.number} ---")

        # -------- per-trial seeds (deterministic per trial) --------
        trial_seed = args.seed + int(trial.number)
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        random.seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)

        # -------- one GPU per trial (round-robin) --------
        num_gpus = max(1, torch.cuda.device_count())
        gpu_id = trial.number % num_gpus

        # STAGGER START: Increase delay slightly and add trial-specific jitter
        # to prevent multiple workers from finishing preloading simultaneously.
        delay = (gpu_id * 45) + (random.random() * 15)
        time.sleep(delay)

        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        log_print(f"Assigned to {device}")


        # -------- disable heavy logs during HPO --------
        config.LOG_PER_CLASS = True
        config.LOG_PER_LEVEL = True
        config.LOG_CONFUSION = False


        # Suggest hyperparameters
        lr_classifier = trial.suggest_float("lr_classifier", 1e-5, 1e-2, log=True)
        lr_backbone = trial.suggest_float("lr_backbone", 5e-5, 1e-2, log=True) # Formerly 1e-6 to 1e-3
        batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096])
        hierarchy_weight = trial.suggest_float("hierarchy_weight", 1.0, 8.0)  # dataset-specific # Formerly 10.0 upper limit

        # Override config
        config.LR_CLASSIFIER = lr_classifier
        config.LR_BACKBONE = lr_backbone
        config.HIERARCHY_WEIGHT = hierarchy_weight

        # Build per-trial components inside try/finally to guarantee teardown
        trainloader = testloader = trainset = testset = None
        net = backbone = optimizer = scheduler = tree_loss = dynamic_transform = None

        try:
            # ---- transforms & datasets ----
            dynamic_transform = DynamicTransform(
                schedule=config.PROGRESSIVE_RESIZE_SCHEDULE,
                aug_config=config.TRAIN_AUGMENTATION,
                epoch_tracker={'current': 0}
            )

            trainset = InsectDataset(
                config.get_train_list_path(),
                input_transform=dynamic_transform
            )
            testset = InsectDataset(
                config.get_val_list_path(),
                input_transform=get_test_transform(args.img_size)
            )

            # Check if stop was requested during long preloading phase
            if STOP_REQUESTED:
                raise optuna.TrialPruned()

            # Use persistent_workers=True for throughput INSIDE the trial;
            # teardown in finally will ensure workers do NOT persist BETWEEN trials.
            trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=args.worker,
                pin_memory=True,
                persistent_workers=True
            )
            testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=max(0, args.worker // 2),
                pin_memory=True,
                persistent_workers=True
            )

            # ---- backbone & model ----
            backbone, num_ftrs, feature_sizes = setup_backbone(args.backbone, args.use_pretrained)
            net = HIFD2(model=backbone, backbone_name=args.backbone, dataset=args.dataset)
            net.to(device)

            # ---- loss ----
            tree_loss = TreeLoss(config.HIERARCHY, device, config.RUN_FOLDER, beta=config.TREE_LOSS_BETA)

            # ---- optimizer ----
            optimizer = optim.SGD([
                {'params': net.classifier_pf4.parameters(), 'lr': lr_classifier},
                {'params': net.classifier_pf3.parameters(), 'lr': lr_classifier},
                {'params': net.classifier_pf2.parameters(), 'lr': lr_classifier},
                {'params': net.classifier_pf1.parameters(), 'lr': lr_classifier},
                {'params': net.classifier_leaf.parameters(), 'lr': lr_classifier},
                {'params': net.fc_pf4.parameters(), 'lr': lr_classifier},
                {'params': net.fc_pf3.parameters(), 'lr': lr_classifier},
                {'params': net.fc_pf2.parameters(), 'lr': lr_classifier},
                {'params': net.fc_pf1.parameters(), 'lr': lr_classifier},
                {'params': net.fc_leaf_class.parameters(), 'lr': lr_classifier},
                {'params': net.conv_block_pf4.parameters(), 'lr': lr_classifier},
                {'params': net.conv_block_pf3.parameters(), 'lr': lr_classifier},
                {'params': net.conv_block_pf2.parameters(), 'lr': lr_classifier},
                {'params': net.conv_block_pf1.parameters(), 'lr': lr_classifier},
                {'params': net.conv_block_leaf_class.parameters(), 'lr': lr_classifier},
                {'params': net.backbone.parameters(), 'lr': lr_backbone}
            ], momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

            # ---- scheduler ----
            steps_per_epoch = len(trainloader)
            total_steps = args.epoch * steps_per_epoch
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[lr_classifier * 2.5] * 15 + [lr_backbone * 2.5],
                total_steps=total_steps,
                pct_start=0.4,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1e4
            )

            # Check one last time before starting training
            if STOP_REQUESTED:
                raise optuna.TrialPruned()

            # ---- train and return metrics ----
            return train(
                args.epoch, net, trainloader, testloader, optimizer, scheduler, args.lr_adjt,
                args.dataset, tree_loss, device, args.device, save_name=f"optuna_trial_{trial.number}",
                trainset=trainset, trees=config.HIERARCHY, get_transform=get_transform,
                get_test_transform=get_test_transform,
                run_folder=base_run,
                epoch_tracker={'current': 0},
                use_greedy_inference=False,
                trial=trial,
                output_folder=output_folder
            )

        except Exception as e:
            # Use log_print now to ensure the error is seen in the main console and log file
            log_print(f"[ERROR] Trial {trial.number} failed: {e}")
            traceback.print_exc()
            raise e  # Let Optuna handle the failure
        finally:
            # 1. Shutdown DataLoader workers by clearing the iterators
            for loader in [trainloader, testloader]:
                if loader is not None:
                    try:
                        if hasattr(loader, '_iterator'):
                            loader._iterator = None
                    except Exception:
                        pass

            # 2. Release dataset references (clears heavy RAM objects)
            try:
                if trainset is not None:
                    trainset.close()
                if testset is not None:
                    testset.close()
            except Exception:
                pass

            # 3. Drop all references and move model to CPU (frees VRAM safely)
            # _del_and_collect handles moving net/backbone to CPU
            _del_and_collect(trainloader, testloader, trainset, testset,
                             scheduler, optimizer, tree_loss, net, backbone, dynamic_transform)

            # 4. Final flush of the CUDA cache for this specific device
            _flush_cuda(device)

            # Optional: log peak GPU memory of the trial
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.set_device(device)
                try:
                    peak = torch.cuda.max_memory_allocated(device)
                    print(f"[MEM] Trial {trial.number} peak allocated on {device}: {peak / 1024 ** 3:.2f} GiB")
                except Exception:
                    pass


    finally:
        log_print(f"--- Finished Trial {trial.number} cleanup ---")

if __name__ == "__main__":
    args = arg_parse()
    DB_URL = "db/url/here"

    # Install signal handlers for graceful stopping
    _install_signal_handlers()

    # Pruner configuration matching your constraints:
    # - First 10 trials are never pruned (startup).
    # - Pruning checks start after 15 epochs (warmup; steps start at 0).
    # - Check every 5 epochs thereafter.
    # - Require at least 10 trials reporting at a step before comparing medians.
    # - Patience=2 checks before actually pruning a trial.
    base_pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=15,
        interval_steps=5,
        n_min_trials=10
    )
    pruner = PatientPruner(base_pruner, patience=2)

    # --- Constraints function: MCC floors (≤0 => feasible; >0 => violation) ---
    def constraints_func(frozen_trial: optuna.trial.FrozenTrial) -> list[float]:
        # Expect five MCCs in order [PF4, PF3, PF2, PF1, Leaf]
        vals = frozen_trial.values
        if vals is None or len(vals) != 5:
            # Treat as infeasible if values are missing/misaligned
            return [1.0] * 5
        # Enforce MCC >= 0.6 at every level
        return [0.6 - float(v) for v in vals]


    # --- Sampler: NSGA-III for many-objective optimization, with constraints ---
    # sampler = optuna.samplers.NSGAIISampler()  # If you prefer NSGA-II instead
    # Prefer NSGA-III for >=4 objectives:
    sampler = optuna.samplers.NSGAIIISampler(constraints_func=constraints_func)

    # --- Study creation: five objectives, all "maximize" ---
    study = optuna.create_study(
        storage=DB_URL,
        study_name="hierarchical_optuna_mcc",
        directions=["maximize"] * 5,  # one per level
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )

    # study = optuna.create_study(
    #     storage=DB_URL,
    #     study_name="hierarchical_optuna_1",
    #     direction="minimize",
    #     load_if_exists=True,
    #     pruner=pruner  # <-- enables pruning using per-epoch reports
    # )

    # --- Run optimization with graceful stop support ---
    try:
        # 'callbacks' lets us check STOP_REQUESTED after each trial and call study.stop()
        # 'catch' converts KeyboardInterrupt into a controlled return instead of bubbling up
        study.optimize(
            lambda t: objective(t, args),
            n_trials=30,
            n_jobs=4,
            callbacks=[_stop_callback],
            catch=(KeyboardInterrupt,)
        )
    except KeyboardInterrupt:
        # Rare case: if a worker raises KeyboardInterrupt despite catch,
        # we still run cleanup.
        print("\n[INTERRUPT] KeyboardInterrupt caught in main.")
    finally:
        _cleanup_on_exit(study)
