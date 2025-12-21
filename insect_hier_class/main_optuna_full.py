
"""
Modified main.py with Optuna integration.
Objective function rebuilds full training pipeline for each trial.
"""
from pathlib import Path
import random
import numpy as np
import timm
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

def objective(trial: optuna.trial.Trial):
    args = arg_parse()

    # -------- per-trial seeds (deterministic per trial) --------
    trial_seed = args.seed + int(trial.number)
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)
    random.seed(trial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)
    # For stricter determinism (optional; may slow training):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # See PyTorch reproducibility notes. [3](https://optuna.readthedocs.io/en/stable/reference/trial.html)

    # -------- one GPU per trial (round-robin) --------
    num_gpus = max(1, torch.cuda.device_count())
    gpu_id = trial.number % num_gpus
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    args.device = [str(gpu_id)]

    # -------- paths: keep run_folder; set per-trial output_folder --------
    base_run = Path(config.RUN_FOLDER)             # unchanged shared folder
    output_folder = base_run / "optuna_trials" / f"trial_{trial.number}"
    output_folder.mkdir(parents=True, exist_ok=True)

    # -------- disable heavy logs during HPO --------
    config.LOG_PER_CLASS = False
    config.LOG_PER_LEVEL = False
    config.LOG_CONFUSION = False


    # Suggest hyperparameters
    lr_classifier = trial.suggest_float("lr_classifier", 1e-5, 1e-2, log=True)
    lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096])
    hierarchy_weight = trial.suggest_float("hierarchy_weight", 1.0, 10.0)  # dataset-specific

    # Override config
    config.LR_CLASSIFIER = lr_classifier
    config.LR_BACKBONE = lr_backbone
    config.HIERARCHY_WEIGHT = hierarchy_weight

    # Data transforms and datasets
    dynamic_transform = DynamicTransform(
        schedule=config.PROGRESSIVE_RESIZE_SCHEDULE,
        aug_config=config.TRAIN_AUGMENTATION,
        epoch_tracker={'current': 0}
    )
    trainset = InsectDataset(config.get_train_list_path(), input_transform=dynamic_transform)
    testset = InsectDataset(config.get_val_list_path(), input_transform=get_test_transform(args.img_size))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=args.worker, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=args.worker // 2, pin_memory=True
    )

    # Backbone and model
    backbone, num_ftrs, feature_sizes = setup_backbone(args.backbone, args.use_pretrained)
    net = HIFD2(model=backbone, backbone_name=args.backbone, dataset=args.dataset)
    net.to(device)

    # TreeLoss
    tree_loss = TreeLoss(config.HIERARCHY, device, config.RUN_FOLDER, beta=config.TREE_LOSS_BETA)

    # Optimizer
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

    # Scheduler (OneCycleLR if selected)
    steps_per_epoch = len(trainloader)
    total_steps = args.epoch * steps_per_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[lr_classifier*2.5]*15 + [lr_backbone*2.5],
        total_steps=total_steps,
        pct_start=0.4,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )

    # Train and return final validation loss

    final_val_loss = train(
        args.epoch, net, trainloader, testloader, optimizer, scheduler, args.lr_adjt,
        args.dataset, tree_loss, device, args.device, save_name=f"optuna_trial_{trial.number}",
        trainset=trainset, trees=config.HIERARCHY, get_transform=get_transform,
        get_test_transform=get_test_transform,
        run_folder=base_run,  # keep shared base path
        epoch_tracker={'current': 0},
        use_greedy_inference=False,
        trial=trial,
        output_folder=output_folder  # write outputs only here during HPO
    )
    return final_val_loss

if __name__ == "__main__":
    DB_URL = "postgresql://myuser:HRNhrn@10.154.29.22:5432/mydb?sslmode=require"  # Replace with actual credentials

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

    study = optuna.create_study(
        storage=DB_URL,
        study_name="hierarchical_optuna_1",
        direction="minimize",
        load_if_exists=True,
        pruner=pruner  # <-- enables pruning using per-epoch reports
    )
    study.optimize(objective, n_trials=50, n_jobs=4)
