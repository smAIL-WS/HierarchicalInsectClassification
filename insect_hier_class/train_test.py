"""
Training and testing functions for hierarchical insect classification.
Includes per-epoch evaluation and metric logging.
"""
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch
import time
import os
import random
import numpy as np
from torch import GradScaler

import config
from hierarchical_target_generation_utils import (
    get_5_level_targets,
    get_valid_hierarchical_mask,
    cosine_anneal_schedule,
    compute_batch_entropy,
    load_level_weights_from_node_counts,
    compute_hierarchical_ce_loss
)
from logging_utils import (
    log_per_class_metrics,
    log_epoch_summary_to_csv,
    log_confusion_matrices_to_pdf,
    log_per_level_metrics
)

from hierarchical_classification_metrics import (
    compute_metrics,
    print_per_class_metrics,
    filter_invalid,
    load_level_name_maps,
    compute_level_accuracy_with_softmax
)

# Set random seeds for reproducibility
seed = config.DEFAULT_SEED
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.set_printoptions(precision=4, sci_mode=False)

# ================================================
# Debug: Print label ranges per level
print("Label Ranges per Level:")
for level, (start, end) in config.GLOBAL_INDEX_RANGES.items():
    print(f"  {level}: {start} to {end - 1} (total: {end - start})")
# ================================================


def train(epoches, net, trainloader, testloader, optimizer, scheduler, lr_adjt, 
          dataset, tree_loss, device, devices, save_name, trainset, trees, 
          get_transform, get_test_transform, run_folder, epoch_tracker):
    """
    Main training loop with progressive resizing and per-epoch evaluation.
    
    Args:
        epoches: Number of training epochs
        net: Model to train
        trainloader: Training data loader
        testloader: Test data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        lr_adjt: Learning rate adjustment strategy ('Cos', 'Step', or 'Fixed')
        dataset: Dataset name
        tree_loss: Tree loss function
        device: Main device
        devices: List of device IDs (for multi-GPU)
        save_name: Name for saving model checkpoints
        trainset: Training dataset (for transform updates)
        trees: Hierarchy tree structure
        get_transform: Function to get training transform
        get_test_transform: Function to get test transform
        run_folder: Path to run folder
    """

    # Load level-wise weights once at start
    level_weights = load_level_weights_from_node_counts(
        config.NODE_SAMPLE_COUNTS_FILE,
        run_folder,
        device
    )

    if level_weights:
        print("\nLevel-wise effective weights loaded:")
        for level, weights in level_weights.items():
            print(f"  {level}: mean={weights.mean():.4f}, std={weights.std():.4f}, "
                  f"min={weights.min():.4f}, max={weights.max():.4f}")
    else:
        print("\nWarning: No level-wise weights loaded. Using uniform weights.")

    # Scaler for mixed-precision training
    scaler = GradScaler()

    # Learning rates for different parameter groups
    lr_base = config.LR_CLASSIFIER
    lr_backbone = config.LR_BACKBONE
    lr = [lr_base] * (len(optimizer.param_groups) - 1) + [lr_backbone]
    
    max_val_acc = 0
    best_epoch = 0
    
    # Multi-GPU setup
    if len(devices) > 1:
        ids = list(map(int, devices))
        netp = torch.nn.DataParallel(net, device_ids=ids)
    else:
        netp = net
    
    # Initial evaluation before training
    print("\n" + "="*70)
    print("INITIAL EVALUATION (Before Training)")
    print("="*70)
    
    initial_image_size = config.PROGRESSIVE_RESIZE_SCHEDULE['initial']
    (initial_pf4_acc, initial_pf3_acc, initial_pf2_acc, initial_pf1_acc,
     initial_leaf_acc, initial_test_loss, initial_tree_loss, initial_ce_loss, initial_entropy_loss,
     _) = test(net, testloader, tree_loss, device, dataset, trainset, trees,
               get_test_transform, initial_image_size, run_folder)
    
    print(f"  PF4: {initial_pf4_acc:.2f}%")
    print(f"  PF3: {initial_pf3_acc:.2f}%")
    print(f"  PF2: {initial_pf2_acc:.2f}%")
    print(f"  PF1: {initial_pf1_acc:.2f}%")
    print(f"  Leaf: {initial_leaf_acc:.2f}%")
    print(f"  Loss: {initial_test_loss:.6f}")
    print(f"    - Tree Loss: {initial_tree_loss:.6f}")
    print(f"    - Cross-Entropy Loss: {initial_ce_loss:.6f}")
    print(f"    - Entropy Loss: {initial_entropy_loss:.6f}")
    
    # Training loop
    for epoch in range(epoches):
        # Update shared epoch tracker for worker_init_fn
        epoch_tracker["current"] = epoch

        # Update dataset transform
        if hasattr(trainset.transform, "set_epoch"):
            trainset.transform.set_epoch(epoch)
            print(f"[Train] Updated transform to epoch {epoch}")

        # Progressive resizing
        resize_schedule = config.PROGRESSIVE_RESIZE_SCHEDULE
        thresholds = resize_schedule['epoch_thresholds']
        
        if epoch < thresholds[0]:
            image_size = resize_schedule['initial']
        elif epoch < thresholds[1]:
            image_size = resize_schedule['mid']
        else:
            image_size = resize_schedule['final']

        print(f"[Epoch {epoch}] Starting epoch with image size {image_size}")
        
        epoch_start = time.time()
        print(f'\n{"="*70}')
        print(f'Epoch: {epoch} | Image Size: {image_size}')
        print("="*70)
        
        net.train()
        train_loss = 0
        train_tree_loss_sum = 0
        train_ce_loss_sum = 0
        train_entropy_loss_sum = 0
        
        # Initialize counters
        pf4_correct = pf3_correct = pf2_correct = pf1_correct = leaf_correct = 0
        pf4_total = pf3_total = pf2_total = pf1_total = leaf_total = 0
        
        pf4_preds, pf4_trues = [], []
        pf3_preds, pf3_trues = [], []
        pf2_preds, pf2_trues = [], []
        pf1_preds, pf1_trues = [], []
        leaf_preds, leaf_trues = [], []
        
        # Adjust learning rate
        if lr_adjt == 'Cos':
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, epoches, lr[nlr])
        elif lr_adjt == 'Step':
            # Step LR handled by scheduler.step() at end of epoch
            pass
        # else: Fixed LR (no adjustment)
        
        # Training iterations
        for batch_idx, (inputs, targets, indices, class_names) in enumerate(trainloader):
            # print(f"[Batch {batch_idx}] Loading batch...")

            inputs, targets = inputs.to(device), targets.to(device)
            # print(f"[Batch {batch_idx}] Starting GPU training...")

            # Debugging to check if the ground-truth labels are being converted correctly
            # print("Ground-truth labels in batch:", targets.tolist())

            optimizer.zero_grad()

            # Get hierarchical targets
            (pf4_targets, pf3_targets, pf2_targets, pf1_targets, 
             leaf_targets) = get_5_level_targets(
                targets, device, dataset, trees, run_folder=run_folder
            )

            # Forward pass
            with autocast(device_type="cuda"):
                (pf4_logits, pf3_logits, pf2_logits, pf1_logits, leaf_logits) = netp(inputs)

                # Apply sigmoid for tree loss
                pf4_sig = torch.sigmoid(pf4_logits)
                pf3_sig = torch.sigmoid(pf3_logits)
                pf2_sig = torch.sigmoid(pf2_logits)
                pf1_sig = torch.sigmoid(pf1_logits)
                leaf_class_sig = torch.sigmoid(leaf_logits)

                # Compute tree loss
                combined_output = torch.cat([pf4_sig, pf3_sig, pf2_sig, pf1_sig, leaf_class_sig], dim=1)
                tree_loss_val = tree_loss(combined_output, targets, device)

                # Compute cross-entropy loss on valid leaf targets
                valid_mask = get_valid_hierarchical_mask(
                    pf4_targets, pf3_targets, pf2_targets, pf1_targets, leaf_targets
                )

                # Compute hierarchical cross-entropy loss with raw logits
                logits_dict = {
                    'pf4': pf4_logits,
                    'pf3': pf3_logits,
                    'pf2': pf2_logits,
                    'pf1': pf1_logits,
                    'leaf': leaf_logits
                }

                targets_dict = {
                    'pf4': pf4_targets,
                    'pf3': pf3_targets,
                    'pf2': pf2_targets,
                    'pf1': pf1_targets,
                    'leaf': leaf_targets
                }

                hierarchical_ce_loss = compute_hierarchical_ce_loss(
                    logits_dict, targets_dict, valid_mask, level_weights, device
                )

                # Compute entropy for diversity regularization
                entropy_pf4 = compute_batch_entropy(pf4_sig)
                entropy_pf3 = compute_batch_entropy(pf3_sig)
                entropy_pf2 = compute_batch_entropy(pf2_sig)
                entropy_pf1 = compute_batch_entropy(pf1_sig)
                entropy_leaf = compute_batch_entropy(leaf_class_sig)

                # Aggregate entropy loss
                entropy_loss = entropy_pf4 + entropy_pf3 + entropy_pf2 + entropy_pf1 + entropy_leaf
                # Linear decay from START to TARGET over ENTROPY_WARMUP_EPOCHS
                if epoch < config.ENTROPY_WARMUP_EPOCHS:
                    lamda_entropy = config.LAMBDA_ENTROPY_START - (
                            (config.LAMBDA_ENTROPY_START - config.LAMBDA_ENTROPY_TARGET) * (
                                epoch / config.ENTROPY_WARMUP_EPOCHS)
                    )
                else:
                    lamda_entropy = config.LAMBDA_ENTROPY_TARGET

                # Combined loss
                loss = tree_loss_val + hierarchical_ce_loss + lamda_entropy * entropy_loss

                # # Print each component and the total loss
                # print(f"Tree Loss: {tree_loss_val.item():.4f}")
                # print(f"Hierarchical CE Loss: {hierarchical_ce_loss.item():.4f}")
                # print(f"Entropy Loss (scaled): {(lamda_entropy * entropy_loss).item():.4f}")
                # print(f"Total Combined Loss: {loss.item():.4f}")

            # Use scaler instead of direct backward
            scaler.scale(loss).backward()

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Clip gradients without logging
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)

            # # Clip gradients AND log original norm
            # total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            # print(f"Gradient norm before clipping: {total_norm:.4f}")

            scaler.step(optimizer)
            scaler.update()

            # # Backward pass
            # loss.backward()
            # optimizer.step()
            
            train_loss += loss.item()
            train_tree_loss_sum += tree_loss_val.item()
            train_ce_loss_sum += hierarchical_ce_loss.item()
            train_entropy_loss_sum += entropy_loss.item()

            # Check for NaN/Inf in losses
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n!!! NaN/Inf detected at epoch {epoch}, batch {batch_idx} !!!")
                print(f"  Combined loss: {loss.item()}")
                print(f"  Tree loss: {tree_loss_val.item()}")
                print(f"  CE loss: {hierarchical_ce_loss.item()}")
                print(f"  Entropy loss: {entropy_loss.item()}")
                print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
                raise ValueError("NaN/Inf loss detected - stopping training")

            # Compute accuracies
            with torch.no_grad():
                pf4_total, pf4_correct = compute_level_accuracy_with_softmax(
                    pf4_logits, pf4_targets, pf4_total, pf4_correct, pf4_preds, pf4_trues
                )

                pf3_total, pf3_correct = compute_level_accuracy_with_softmax(
                    pf3_logits, pf3_targets, pf3_total, pf3_correct, pf3_preds, pf3_trues
                )

                pf2_total, pf2_correct = compute_level_accuracy_with_softmax(
                    pf2_logits, pf2_targets, pf2_total, pf2_correct, pf2_preds, pf2_trues
                )

                pf1_total, pf1_correct = compute_level_accuracy_with_softmax(
                    pf1_logits, pf1_targets, pf1_total, pf1_correct, pf1_preds, pf1_trues
                )

                leaf_total, leaf_correct = compute_level_accuracy_with_softmax(
                    leaf_logits, leaf_targets, leaf_total, leaf_correct, leaf_preds, leaf_trues
                )

            # # Stop after first batch
            # if batch_idx == 0:
            #     print("Stopping after first batch for debugging.")
            #     break

        # Step scheduler if using Step LR
        if lr_adjt == 'Step':
            scheduler.step()
        
        # Compute epoch metrics
        label_maps = load_level_name_maps(config.LEVEL_NAME_MAPS_FILE, run_folder)
        
        # Filter invalid predictions
        pf4_trues_filt, pf4_preds_filt = filter_invalid(pf4_trues, pf4_preds)
        pf3_trues_filt, pf3_preds_filt = filter_invalid(pf3_trues, pf3_preds)
        pf2_trues_filt, pf2_preds_filt = filter_invalid(pf2_trues, pf2_preds)
        pf1_trues_filt, pf1_preds_filt = filter_invalid(pf1_trues, pf1_preds)
        leaf_trues_filt, leaf_preds_filt = filter_invalid(leaf_trues, leaf_preds)

        # # start debugging
        # # Filter invalid predictions
        # pf4_trues_filt, pf4_preds_filt = filter_invalid(pf4_trues, pf4_preds)
        # print(f"PF4 filtered trues (len={len(pf4_trues_filt)}): {pf4_trues_filt[:10]}")
        # print(f"PF4 filtered preds (len={len(pf4_preds_filt)}): {pf4_preds_filt[:10]}")
        #
        # pf3_trues_filt, pf3_preds_filt = filter_invalid(pf3_trues, pf3_preds)
        # print(f"PF3 filtered trues (len={len(pf3_trues_filt)}): {pf3_trues_filt[:10]}")
        # print(f"PF3 filtered preds (len={len(pf3_preds_filt)}): {pf3_preds_filt[:10]}")
        #
        # pf2_trues_filt, pf2_preds_filt = filter_invalid(pf2_trues, pf2_preds)
        # print(f"PF2 filtered trues (len={len(pf2_trues_filt)}): {pf2_trues_filt[:10]}")
        # print(f"PF2 filtered preds (len={len(pf2_preds_filt)}): {pf2_preds_filt[:10]}")
        #
        # pf1_trues_filt, pf1_preds_filt = filter_invalid(pf1_trues, pf1_preds)
        # print(f"PF1 filtered trues (len={len(pf1_trues_filt)}): {pf1_trues_filt[:10]}")
        # print(f"PF1 filtered preds (len={len(pf1_preds_filt)}): {pf1_preds_filt[:10]}")
        #
        # leaf_trues_filt_soft, leaf_preds_filt_soft = filter_invalid(leaf_trues, leaf_preds_soft)
        # print(f"Leaf soft filtered trues (len={len(leaf_trues_filt_soft)}): {leaf_trues_filt_soft[:10]}")
        # print(f"Leaf soft filtered preds (len={len(leaf_preds_filt_soft)}): {leaf_preds_filt_soft[:10]}")
        #
        # leaf_trues_filt_sig, leaf_preds_filt_sig = filter_invalid(leaf_trues, leaf_preds_sig)
        # print(f"Leaf sig filtered trues (len={len(leaf_trues_filt_sig)}): {leaf_trues_filt_sig[:10]}")
        # print(f"Leaf sig filtered preds (len={len(leaf_preds_filt_sig)}): {leaf_preds_filt_sig[:10]}")
        # # end debugging

        # Print per-class metrics
        print("\n--- Training Per-Class Metrics ---")
        print_per_class_metrics(pf4_trues_filt, pf4_preds_filt, "PF4", label_maps['parent_folder_4'])
        print_per_class_metrics(pf3_trues_filt, pf3_preds_filt, "PF3", label_maps['parent_folder_3'])
        print_per_class_metrics(pf2_trues_filt, pf2_preds_filt, "PF2", label_maps['parent_folder_2'])
        print_per_class_metrics(pf1_trues_filt, pf1_preds_filt, "PF1", label_maps['parent_folder_1'])
        print_per_class_metrics(leaf_trues_filt, leaf_preds_filt, "Leaf", label_maps['classification'])
        
        # Print aggregate metrics
        print("\n--- Training Aggregate Metrics ---")
        compute_metrics(pf4_trues_filt, pf4_preds_filt, "PF4")
        compute_metrics(pf3_trues_filt, pf3_preds_filt, "PF3")
        compute_metrics(pf2_trues_filt, pf2_preds_filt, "PF2")
        compute_metrics(pf1_trues_filt, pf1_preds_filt, "PF1")
        compute_metrics(leaf_trues_filt, leaf_preds_filt, "Leaf")
        
        # Compute accuracy percentages
        train_pf4_acc = 100. * pf4_correct / pf4_total if pf4_total > 0 else 0.0
        train_pf3_acc = 100. * pf3_correct / pf3_total if pf3_total > 0 else 0.0
        train_pf2_acc = 100. * pf2_correct / pf2_total if pf2_total > 0 else 0.0
        train_pf1_acc = 100. * pf1_correct / pf1_total if pf1_total > 0 else 0.0
        train_leaf_acc = 100. * leaf_correct / leaf_total if leaf_total > 0 else 0.0
        train_loss = train_loss / len(trainloader)
        train_tree_loss_avg = train_tree_loss_sum / len(trainloader)
        train_ce_loss_avg = train_ce_loss_sum / len(trainloader)
        train_entropy_loss_avg = train_entropy_loss_sum / len(trainloader)
        
        epoch_end = time.time()

        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train PF4: {train_pf4_acc:.2f}%')
        print(f'  Train PF3: {train_pf3_acc:.2f}%')
        print(f'  Train PF2: {train_pf2_acc:.2f}%')
        print(f'  Train PF1: {train_pf1_acc:.2f}%')
        print(f'  Train Leaf: {train_leaf_acc:.2f}%')
        print(f'  Train Loss: {train_loss:.6f}')
        print(f'    - Tree Loss: {train_tree_loss_avg:.6f}')
        print(f'    - CE Loss: {train_ce_loss_avg:.6f}')
        print(f'    - Entropy Loss: {train_entropy_loss_avg:.6f}')
        print(f'  Time: {epoch_end - epoch_start:.1f}s')
        
        # Log training metrics
        level_metrics_dict = {
            "PF4": (pf4_trues_filt, pf4_preds_filt),
            "PF3": (pf3_trues_filt, pf3_preds_filt),
            "PF2": (pf2_trues_filt, pf2_preds_filt),
            "PF1": (pf1_trues_filt, pf1_preds_filt),
            "Leaf": (leaf_trues_filt, leaf_preds_filt)
        }
        
        log_per_class_metrics(
            epoch=epoch,
            level_metrics_dict=level_metrics_dict,
            label_maps={
                "PF4": label_maps['parent_folder_4'],
                "PF3": label_maps['parent_folder_3'],
                "PF2": label_maps['parent_folder_2'],
                "PF1": label_maps['parent_folder_1'],
                "Leaf": label_maps['classification']
            },
            split="Train",
            run_folder=run_folder
        )

        log_per_level_metrics(
            epoch=epoch,
            level_metrics_dict=level_metrics_dict,
            split="Train",
            run_folder=run_folder
        )

        log_confusion_matrices_to_pdf(
            epoch=epoch,
            level_metrics_dict=level_metrics_dict,
            label_maps={
                "PF4": label_maps['parent_folder_4'],
                "PF3": label_maps['parent_folder_3'],
                "PF2": label_maps['parent_folder_2'],
                "PF1": label_maps['parent_folder_1'],
                "Leaf": label_maps['classification']
            },
            split="Train"
        )

        # Test evaluation
        (test_pf4_acc, test_pf3_acc, test_pf2_acc, test_pf1_acc,
         test_leaf_acc, test_loss, test_tree_loss_avg, test_ce_loss_avg, test_entropy_loss_avg,
         level_metrics_dict_test) = test(
            net, testloader, tree_loss, device, dataset, trainset, trees,
            get_test_transform, image_size, run_folder
        )
        
        # Log test metrics
        log_per_class_metrics(
            epoch=epoch,
            level_metrics_dict=level_metrics_dict_test,
            label_maps={
                "PF4": label_maps['parent_folder_4'],
                "PF3": label_maps['parent_folder_3'],
                "PF2": label_maps['parent_folder_2'],
                "PF1": label_maps['parent_folder_1'],
                "Leaf": label_maps['classification']
            },
            split="Test",
            run_folder=run_folder
        )

        log_per_level_metrics(
            epoch=epoch,
            level_metrics_dict=level_metrics_dict_test,
            split="Test",
            run_folder=run_folder
        )

        log_confusion_matrices_to_pdf(
            epoch=epoch,
            level_metrics_dict=level_metrics_dict_test,
            label_maps={
                "PF4": label_maps['parent_folder_4'],
                "PF3": label_maps['parent_folder_3'],
                "PF2": label_maps['parent_folder_2'],
                "PF1": label_maps['parent_folder_1'],
                "Leaf": label_maps['classification']
            },
            split="Test"
        )

        # Save epoch summary
        log_epoch_summary_to_csv(
            epoch=epoch,
            train_pf4_acc=train_pf4_acc,
            train_pf3_acc=train_pf3_acc,
            train_pf2_acc=train_pf2_acc,
            train_pf1_acc=train_pf1_acc,
            train_leaf_acc=train_leaf_acc,
            train_loss=train_loss,
            train_tree_loss=train_tree_loss_avg,
            train_ce_loss=train_ce_loss_avg,
            train_entropy_loss=train_entropy_loss_avg,
            test_pf4_acc=test_pf4_acc,
            test_pf3_acc=test_pf3_acc,
            test_pf2_acc=test_pf2_acc,
            test_pf1_acc=test_pf1_acc,
            test_leaf_acc=test_leaf_acc,
            test_loss=test_loss,
            test_tree_loss=test_tree_loss_avg,
            test_ce_loss=test_ce_loss_avg,
            test_entropy_loss=test_entropy_loss_avg,
            epoch_time=epoch_end - epoch_start,
            run_folder=run_folder
        )
        
        # Save best model
        if test_leaf_acc > max_val_acc:
            max_val_acc = test_leaf_acc
            best_epoch = epoch
            
            net.cpu()
            save_dir = config.MODELS_DIR
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f'model_{save_name}.pth'
            torch.save(net, save_path)
            print(f'\n✓ New best model saved: {save_path}')
            net.to(device)
    
    print(f'\n{"="*70}')
    print(f'TRAINING COMPLETE')
    print(f'{"="*70}')
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Validation Accuracy: {max_val_acc:.2f}%')
    print(f'{"="*70}\n')


def test(net, testloader, tree_loss, device, dataset, trainset, trees, 
         get_test_transform, image_size, run_folder):
    """
    Test/validation loop.
    
    Returns:
        Tuple of (pf4_acc, pf3_acc, pf2_acc, pf1_acc, 
                  leaf_acc, loss, tree_loss_avg, ce_loss_avg, entropy_loss_avg,
                  level_metrics_dict)
    """

    # Load level-wise weights once at start
    level_weights = load_level_weights_from_node_counts(
        config.NODE_SAMPLE_COUNTS_FILE,
        run_folder,
        device
    )

    epoch_start = time.time()
    net.eval()
    
    # Update test transform
    testloader.dataset.transform = get_test_transform(image_size)
    
    test_loss = 0
    test_tree_loss_sum = 0
    test_ce_loss_sum = 0
    test_entropy_loss_sum = 0
    pf4_correct = pf3_correct = pf2_correct = pf1_correct = leaf_correct = 0
    pf4_total = pf3_total = pf2_total = pf1_total = leaf_total = 0
    
    pf4_preds, pf4_trues = [], []
    pf3_preds, pf3_trues = [], []
    pf2_preds, pf2_trues = [], []
    pf1_preds, pf1_trues = [], []
    leaf_preds, leaf_trues = [], []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, indices, class_names) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            (pf4_targets, pf3_targets, pf2_targets, pf1_targets, 
             leaf_targets) = get_5_level_targets(
                targets, device, dataset, trees, run_folder=run_folder
            )

            # Forward pass
            pf4_logits, pf3_logits, pf2_logits, pf1_logits, leaf_logits = net(inputs)

            # Apply sigmoid for tree loss and accuracy
            pf4_sig = torch.sigmoid(pf4_logits)
            pf3_sig = torch.sigmoid(pf3_logits)
            pf2_sig = torch.sigmoid(pf2_logits)
            pf1_sig = torch.sigmoid(pf1_logits)
            leaf_class_sig = torch.sigmoid(leaf_logits)
            
            # Compute loss
            combined_output = torch.cat([pf4_sig, pf3_sig, pf2_sig, pf1_sig, leaf_class_sig], dim=1)
            tree_loss_val = tree_loss(combined_output, targets, device)
            
            valid_mask = get_valid_hierarchical_mask(
                pf4_targets, pf3_targets, pf2_targets, pf1_targets, leaf_targets
            )

            # Compute hierarchical CE loss
            logits_dict = {
                'pf4': pf4_logits,
                'pf3': pf3_logits,
                'pf2': pf2_logits,
                'pf1': pf1_logits,
                'leaf': leaf_logits
            }

            targets_dict = {
                'pf4': pf4_targets,
                'pf3': pf3_targets,
                'pf2': pf2_targets,
                'pf1': pf1_targets,
                'leaf': leaf_targets
            }

            hierarchical_ce_loss = compute_hierarchical_ce_loss(
                logits_dict, targets_dict, valid_mask, level_weights, device
            )

            # Compute entropy for test set too
            entropy_pf4 = compute_batch_entropy(pf4_sig)
            entropy_pf3 = compute_batch_entropy(pf3_sig)
            entropy_pf2 = compute_batch_entropy(pf2_sig)
            entropy_pf1 = compute_batch_entropy(pf1_sig)
            entropy_leaf = compute_batch_entropy(leaf_class_sig)
            entropy_loss = entropy_pf4 + entropy_pf3 + entropy_pf2 + entropy_pf1 + entropy_leaf
            lamda_entropy = config.LAMBDA_ENTROPY_TARGET

            loss = tree_loss_val + hierarchical_ce_loss + lamda_entropy * entropy_loss

            test_loss += loss.item()
            test_tree_loss_sum += tree_loss_val.item()
            test_ce_loss_sum += hierarchical_ce_loss.item()
            test_entropy_loss_sum += entropy_loss.item()

            # Compute accuracies using softmax
            from hierarchical_classification_metrics import compute_level_accuracy_with_softmax

            pf4_total, pf4_correct = compute_level_accuracy_with_softmax(
                pf4_logits, pf4_targets, pf4_total, pf4_correct, pf4_preds, pf4_trues
            )
            pf3_total, pf3_correct = compute_level_accuracy_with_softmax(
                pf3_logits, pf3_targets, pf3_total, pf3_correct, pf3_preds, pf3_trues
            )
            pf2_total, pf2_correct = compute_level_accuracy_with_softmax(
                pf2_logits, pf2_targets, pf2_total, pf2_correct, pf2_preds, pf2_trues
            )
            pf1_total, pf1_correct = compute_level_accuracy_with_softmax(
                pf1_logits, pf1_targets, pf1_total, pf1_correct, pf1_preds, pf1_trues
            )
            leaf_total, leaf_correct = compute_level_accuracy_with_softmax(
                leaf_logits, leaf_targets, leaf_total, leaf_correct, leaf_preds, leaf_trues
            )
    
    # Compute metrics
    test_pf4_acc = 100. * pf4_correct / pf4_total if pf4_total > 0 else 0.0
    test_pf3_acc = 100. * pf3_correct / pf3_total if pf3_total > 0 else 0.0
    test_pf2_acc = 100. * pf2_correct / pf2_total if pf2_total > 0 else 0.0
    test_pf1_acc = 100. * pf1_correct / pf1_total if pf1_total > 0 else 0.0
    test_leaf_acc = 100. * leaf_correct / leaf_total if leaf_total > 0 else 0.0
    test_loss = test_loss / len(testloader)
    test_tree_loss_avg = test_tree_loss_sum / len(testloader)
    test_ce_loss_avg = test_ce_loss_sum / len(testloader)
    test_entropy_loss_avg = test_entropy_loss_sum / len(testloader)
    
    epoch_end = time.time()

    print(f'\n--- Test Results ---')
    print(f'  PF4: {test_pf4_acc:.2f}%')
    print(f'  PF3: {test_pf3_acc:.2f}%')
    print(f'  PF2: {test_pf2_acc:.2f}%')
    print(f'  PF1: {test_pf1_acc:.2f}%')
    print(f'  Leaf (Soft): {test_leaf_acc:.2f}%')
    print(f'  Loss: {test_loss:.6f}')
    print(f'    - Tree Loss: {test_tree_loss_avg:.6f}')
    print(f'    - CE Loss: {test_ce_loss_avg:.6f}')
    print(f'    - Entropy Loss: {test_entropy_loss_avg:.6f}')
    print(f'  Time: {epoch_end - epoch_start:.1f}s')
    
    # Filter invalid predictions
    label_maps = load_level_name_maps(config.LEVEL_NAME_MAPS_FILE, run_folder)
    
    pf4_trues_filt, pf4_preds_filt = filter_invalid(pf4_trues, pf4_preds)
    pf3_trues_filt, pf3_preds_filt = filter_invalid(pf3_trues, pf3_preds)
    pf2_trues_filt, pf2_preds_filt = filter_invalid(pf2_trues, pf2_preds)
    pf1_trues_filt, pf1_preds_filt = filter_invalid(pf1_trues, pf1_preds)
    leaf_trues_filt, leaf_preds_filt = filter_invalid(leaf_trues, leaf_preds)
    
    # Print detailed metrics
    print("\n--- Test Per-Class Metrics ---")
    print_per_class_metrics(pf4_trues_filt, pf4_preds_filt, "PF4", label_maps['parent_folder_4'])
    print_per_class_metrics(pf3_trues_filt, pf3_preds_filt, "PF3", label_maps['parent_folder_3'])
    print_per_class_metrics(pf2_trues_filt, pf2_preds_filt, "PF2", label_maps['parent_folder_2'])
    print_per_class_metrics(pf1_trues_filt, pf1_preds_filt, "PF1", label_maps['parent_folder_1'])
    print_per_class_metrics(leaf_trues_filt, leaf_preds_filt, "Leaf", label_maps['classification'])

    
    print("\n--- Test Aggregate Metrics ---")
    compute_metrics(pf4_trues_filt, pf4_preds_filt, "PF4")
    compute_metrics(pf3_trues_filt, pf3_preds_filt, "PF3")
    compute_metrics(pf2_trues_filt, pf2_preds_filt, "PF2")
    compute_metrics(pf1_trues_filt, pf1_preds_filt, "PF1")
    compute_metrics(leaf_trues_filt, leaf_preds_filt, "Leaf")
    
    level_metrics_dict = {
        "PF4": (pf4_trues_filt, pf4_preds_filt),
        "PF3": (pf3_trues_filt, pf3_preds_filt),
        "PF2": (pf2_trues_filt, pf2_preds_filt),
        "PF1": (pf1_trues_filt, pf1_preds_filt),
        "Leaf": (leaf_trues_filt, leaf_preds_filt)
    }
    
    return (test_pf4_acc, test_pf3_acc, test_pf2_acc, test_pf1_acc,
            test_leaf_acc, test_loss, test_tree_loss_avg, test_ce_loss_avg, test_entropy_loss_avg,
            level_metrics_dict)
