# Updated for 5 levels of hierarchy.

import torch
from torch.nn.modules.activation import Softmax
from utils_insect_EffNet1 import *
import copy
import os
import time
import json
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import random
import numpy as np
from logging_EffNet import log_per_class_metrics, log_epoch_summary_to_csv

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.set_printoptions(precision=4, sci_mode=False)

def train(epoches, net, trainloader, testloader, optimizer, scheduler, lr_adjt, dataset, tree, device, devices, save_name, trainset, trees, get_transform, get_test_transform):
    lr = [0.002] * (len(optimizer.param_groups) - 1) + [0.0002]
    max_val_acc = 0
    best_epoch = 0
    if len(devices) > 1:
        ids = list(map(int, devices))
        netp = torch.nn.DataParallel(net, device_ids=ids)

    # Evaluate model accuracy before any learning has taken place:
    print("\nEvaluating initial accuracy before training...")
    initial_image_size = 96
    initial_pf4_acc, initial_pf3_acc, initial_pf2_acc, initial_pf1_acc, initial_leaf_acc_soft, initial_leaf_acc_sig, initial_test_loss, level_metrics_dict = \
        test(net, testloader, tree, device, dataset, trainset, trees, get_test_transform, initial_image_size)

    print("Initial Accuracy (Before Training):")
    print(f"  PF4: {initial_pf4_acc:.2f}%")
    print(f"  PF3: {initial_pf3_acc:.2f}%")
    print(f"  PF2: {initial_pf2_acc:.2f}%")
    print(f"  PF1: {initial_pf1_acc:.2f}%")
    print(f"  Leaf (Softmax): {initial_leaf_acc_soft:.2f}%")
    print(f"  Leaf (Sigmoid): {initial_leaf_acc_sig:.2f}%")
    print(f"  Initial Test Loss: {initial_test_loss:.6f}")


    # Training loop
    for epoch in range(epoches):
        # Progressive resizing logic
        if epoch < 10:
            image_size = 96
        elif epoch < 20:
            image_size = 112
        else:
            image_size = 128

        # Update transform in dataset
        trainset.transform = get_transform(image_size)
        epoch_start = time.time()
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0

        pf4_correct = 0 # coarsest parent folder, e.g 'Insecta'
        pf3_correct = 0
        pf2_correct = 0
        pf1_correct = 0
        leaf_class_correct_soft = 0 # softmax on leaf node class
        leaf_class_correct_sig = 0 # sigmoid on leaf node class

        pf4_total = 0
        pf3_total = 0
        pf2_total = 0
        pf1_total = 0
        leaf_class_total = 0

        pf4_preds, pf4_trues = [], []
        pf3_preds, pf3_trues = [], []
        pf2_preds, pf2_trues = [], []
        pf1_preds, pf1_trues = [], []
        leaf_preds_soft, leaf_trues = [], []
        leaf_preds_sig = []

        idx = 0
        if lr_adjt == 'Cos':
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, epoches, lr[nlr])
        else:
            print("Using fixed learning rate.")

        for batch_idx, (inputs, targets, indices, class_names) in enumerate(trainloader):
            idx = batch_idx
            # print(f"Batch {batch_idx}")
            # print(f"Inputs shape: {inputs.shape}")
            # print(f"Targets: {targets}")
            # print(f"Targets (leaf labels): {targets.tolist()}")
            # print(f"Sample indices: {indices.tolist()}")

            inputs, targets = inputs.to(device), targets.to(device)
            pf4_targets, pf3_targets, pf2_targets, pf1_targets, leaf_targets_sig, leaf_class_weights_tensor = get_5_level_targets(targets, device, dataset, trees, leaf_class_weights_file=None) # to be defined in utils_insect.py

            # print(f"leaf_targets_sig shape: {leaf_targets_sig.shape}")
            # print(f"pf4_targets shape: {pf4_targets.shape}")
            # print(f"pf3_targets shape: {pf3_targets.shape}")
            # print(f"pf2_targets shape: {pf2_targets.shape}")
            # print(f"pf1_targets shape: {pf1_targets.shape}")
            #
            # print(f"PF4 targets: {pf4_targets.tolist()}")
            # print(f"PF3 targets: {pf3_targets.tolist()}")
            # print(f"PF2 targets: {pf2_targets.tolist()}")
            # print(f"PF1 targets: {pf1_targets.tolist()}")

            optimizer.zero_grad()

            if len(devices) > 1:
                pf4_sig, pf3_sig, pf2_sig, pf1_sig, leaf_class_sig, leaf_class_soft = netp(inputs) #sigmoid and softmax outputs from RFM_insect.py
            else:
                pf4_sig, pf3_sig, pf2_sig, pf1_sig, leaf_class_sig, leaf_class_soft = net(inputs)

            # Inspect each output
            # print(f"pf4_sig shape: {pf4_sig.shape}")
            # print(f"pf4_sig values:\n{pf4_sig}")
            #
            # print(f"pf3_sig shape: {pf3_sig.shape}")
            # print(f"pf3_sig values:\n{pf3_sig}")
            #
            # print(f"pf2_sig shape: {pf2_sig.shape}")
            # print(f"pf2_sig values:\n{pf2_sig}")
            #
            # print(f"pf1_sig shape: {pf1_sig.shape}")
            # print(f"pf1_sig values:\n{pf1_sig}")
            #
            # print(f"leaf_class_sig shape: {leaf_class_sig.shape}")
            # print(f"leaf_class_sig values:\n{leaf_class_sig}")
            #
            # print(f"leaf_class_soft shape: {leaf_class_soft.shape}")
            # print(f"leaf_class_soft values:\n{leaf_class_soft}")

            # Compute tree loss
            tree_loss = tree(torch.cat([pf4_sig, pf3_sig, pf2_sig, pf1_sig, leaf_class_sig], 1), targets,
                             device)

            valid_mask = get_valid_hierarchical_mask(pf4_targets, pf3_targets, pf2_targets, pf1_targets,
                                                     leaf_targets_sig)

            if valid_mask.sum() > 0:
                selected_logits = leaf_class_soft[valid_mask]
                selected_targets = leaf_targets_sig[valid_mask]

                if leaf_class_weights_tensor is not None:
                    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=leaf_class_weights_tensor.to(torch.float64))
                else:
                    ce_loss_fn = torch.nn.CrossEntropyLoss()

                ce_loss_leaf_node = ce_loss_fn(selected_logits.to(torch.float64), selected_targets)
                loss = ce_loss_leaf_node + tree_loss
                # Debug: print individual loss components
                # print(f"ce_loss_leaf_node: {ce_loss_leaf_node.item():.6f}")
                # print(f"tree_loss: {tree_loss.item():.6f}")
            else:
                loss = tree_loss

            #
            # # Detect leaf class labels from the tree
            # leaf_class_labels = find_leaf_class_labels(trees)
            # leaf_class_labels_tensor = torch.tensor(leaf_class_labels, device=device)
            #
            # # Create a mask for leaf samples in the batch
            # leaf_mask = torch.isin(targets, leaf_class_labels_tensor)
            # leaf_labels = torch.nonzero(leaf_mask, as_tuple=False)
            #
            # # Apply cross-entropy loss only to leaf samples
            # if leaf_mask.any():
            #     select_leaf_labels = targets[leaf_mask]
            #     select_fc_soft = leaf_class_soft[leaf_mask]
            #
            #     # Compute CE loss
            #     ce_loss_leaf_node = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)

                # # Identify leaf node samples (Insect dataset only)
            # leaf_labels = torch.nonzero(targets > 31, as_tuple=False)  # 0-indexed labels
            #
            # # Debug: print number of leaf labels
            # # print(f"Leaf labels shape: {leaf_labels.shape}")
            #
            # # Compute total loss
            # if leaf_labels.shape[0] > 0:
            #     select_leaf_labels = torch.index_select(targets, 0,
            #                                             leaf_labels.squeeze()) - 32  # adjust for 1-indexed labels
            #     select_fc_soft = torch.index_select(leaf_class_soft, 0, leaf_labels.squeeze())
            #
            #     # print(f"select_leaf_labels: {select_leaf_labels}")
            #     # print(f"select_fc_soft shape: {select_fc_soft.shape}")
            #
            #     # Compute CE loss
            #     ce_loss_leaf_node = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)

                # Debug: print individual loss components
                # print(f"ce_loss_leaf_node: {ce_loss_leaf_node.item():.6f}")
                # print(f"tree_loss: {tree_loss.item():.6f}")
            #
            #     loss = ce_loss_leaf_node + tree_loss
            # else:
            #     # print(f"tree_loss (only): {tree_loss.item():.6f}")
            #     loss = tree_loss

            # Final loss print
            # print(f"Total loss: {loss.item():.6f}")

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Stop after 1 or 2 batches - only for debugging
            # if batch_idx == 0:
            #      break

            with torch.no_grad():
                # PF4
                _, pf4_predicted = torch.max(pf4_sig.data, 1)
                valid_pf4_mask = pf4_targets != -1
                if valid_pf4_mask.sum() > 0:
                    pf4_predicted_selected = pf4_predicted[valid_pf4_mask]
                    pf4_targets_selected = pf4_targets[valid_pf4_mask]
                    pf4_total += pf4_targets_selected.size(0)
                    pf4_correct += pf4_predicted_selected.eq(pf4_targets_selected).cpu().sum().item()
                    pf4_preds.extend(pf4_predicted_selected.cpu().numpy())
                    pf4_trues.extend(pf4_targets_selected.cpu().numpy())

                # PF3
                _, pf3_predicted = torch.max(pf3_sig.data, 1)
                valid_pf3_mask = pf3_targets != -1
                if valid_pf3_mask.sum() > 0:
                    pf3_predicted_selected = pf3_predicted[valid_pf3_mask]
                    pf3_targets_selected = pf3_targets[valid_pf3_mask]
                    pf3_total += pf3_targets_selected.size(0)
                    pf3_correct += pf3_predicted_selected.eq(pf3_targets_selected).cpu().sum().item()
                    pf3_preds.extend(pf3_predicted_selected.cpu().numpy())
                    pf3_trues.extend(pf3_targets_selected.cpu().numpy())

                # PF2
                _, pf2_predicted = torch.max(pf2_sig.data, 1)
                valid_pf2_mask = pf2_targets != -1
                if valid_pf2_mask.sum() > 0:
                    pf2_predicted_selected = pf2_predicted[valid_pf2_mask]
                    pf2_targets_selected = pf2_targets[valid_pf2_mask]
                    pf2_total += pf2_targets_selected.size(0)
                    pf2_correct += pf2_predicted_selected.eq(pf2_targets_selected).cpu().sum().item()
                    pf2_preds.extend(pf2_predicted_selected.cpu().numpy())
                    pf2_trues.extend(pf2_targets_selected.cpu().numpy())

                # PF1
                _, pf1_predicted = torch.max(pf1_sig.data, 1)
                valid_pf1_mask = pf1_targets != -1
                if valid_pf1_mask.sum() > 0:
                    pf1_predicted_selected = pf1_predicted[valid_pf1_mask]
                    pf1_targets_selected = pf1_targets[valid_pf1_mask]
                    pf1_total += pf1_targets_selected.size(0)
                    pf1_correct += pf1_predicted_selected.eq(pf1_targets_selected).cpu().sum().item()
                    pf1_preds.extend(pf1_predicted_selected.cpu().numpy())
                    pf1_trues.extend(pf1_targets_selected.cpu().numpy())

                # print("PF4 predictions with indices:")
                # for idx, pred in zip(indices, pf4_predicted.tolist()):
                #     print(f"Index: {idx}, Predicted PF4: {pred}")
                #
                # print("PF4 targets with indices:")
                # for idx, target in zip(indices, pf4_targets.tolist()):
                #     print(f"Index: {idx}, Target PF4: {target}")
                #
                # print("PF3 predictions with indices:")
                # for idx, pred in zip(indices, pf3_predicted.tolist()):
                #     print(f"Index: {idx}, Predicted PF3: {pred}")
                #
                # print("PF3 targets with indices:")
                # for idx, target in zip(indices, pf3_targets.tolist()):
                #     print(f"Index: {idx}, Target PF3: {target}")
                #
                # print("PF2 predictions with indices:")
                # for idx, pred in zip(indices, pf2_predicted.tolist()):
                #     print(f"Index: {idx}, Predicted PF2: {pred}")
                #
                # print("PF2 targets with indices:")
                # for idx, target in zip(indices, pf2_targets.tolist()):
                #     print(f"Index: {idx}, Target PF2: {target}")
                #
                # print("PF1 predictions with indices:")
                # for idx, pred in zip(indices, pf1_predicted.tolist()):
                #     print(f"Index: {idx}, Predicted PF1: {pred}")
                #
                # print("PF1 targets with indices:")
                # for idx, target in zip(indices, pf1_targets.tolist()):
                #     print(f"Index: {idx}, Target PF1: {target}")

                # Leaf class
                if valid_mask.sum() > 0:
                    select_leaf_class_soft = leaf_class_soft[valid_mask]
                    select_leaf_class_sig = leaf_class_sig[valid_mask]
                    select_leaf_labels = leaf_targets_sig[valid_mask]

                    num_leaf_classes = num_classes_per_level['leaf']  # assuming this is defined in your model

                    if select_leaf_labels.min() < 0 or select_leaf_labels.max() >= num_leaf_classes:
                        print("Invalid leaf target detected:", select_leaf_labels)
                    else:
                        _, class_predicted_soft = torch.max(select_leaf_class_soft.data, 1)
                        _, class_predicted_sig = torch.max(select_leaf_class_sig.data, 1)

                        leaf_class_total += select_leaf_labels.size(0)
                        leaf_class_correct_soft += class_predicted_soft.eq(select_leaf_labels).cpu().sum().item()
                        leaf_class_correct_sig += class_predicted_sig.eq(select_leaf_labels).cpu().sum().item()

                        leaf_preds_soft.extend(class_predicted_soft.cpu().numpy())
                        leaf_preds_sig.extend(class_predicted_sig.cpu().numpy())
                        leaf_trues.extend(select_leaf_labels.cpu().numpy())

                        # print("leaf class predictions with indices:")
                        # for idx, pred in zip(indices, class_predicted_sig.tolist()):
                        #     print(f"Index: {idx}, Predicted leaf_class_sig: {pred}")
                        #
                        # print("Leaf class targets with indices:")
                        # for idx, target in zip(indices, select_leaf_labels.tolist()):
                        #     print(f"Index: {idx}, Target leaf_class_sig: {target}")
                else:
                    pass # Skip print statement al model is functioning
                    # print("Skipping leaf accuracy: no valid leaf targets in batch.")


        if lr_adjt == 'Step':
            scheduler.step()

        # Sklearn metrics
        def compute_metrics(y_true, y_pred, level_name):
            print(f"\nMetrics for {level_name}:")
            print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
            print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
            print(f"Recall: {recall_score(y_true, y_pred, average='macro'):.4f}")
            print(f"F1 Score: {f1_score(y_true, y_pred, average='macro'):.4f}")

        def print_per_class_metrics(y_true, y_pred, level_name, label_map=None):
            precision = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
            unique_classes = sorted(set(y_true))

            print(f"\nPer-class metrics for {level_name}:")
            for i, cls in enumerate(unique_classes):
                name = label_map.get(str(cls), f"Class {cls}") if label_map else f"Class {cls}"
                print(f"Class {cls} ({name}): Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")

        def filter_invalid(y_true, y_pred, sentinel=-1):
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            valid_mask = y_true != sentinel
            return y_true[valid_mask], y_pred[valid_mask]

        def load_level_name_maps(filename):
            script_dir = os.path.dirname(__file__)
            path = os.path.join(script_dir, filename)
            with open(path, 'r') as f:
                return json.load(f)

        label_maps = load_level_name_maps('level_name_maps_13Oct25.json')  # contains mappings for all levels

        # Filter invalid entries
        pf4_trues_filt, pf4_preds_filt = filter_invalid(pf4_trues, pf4_preds)
        pf3_trues_filt, pf3_preds_filt = filter_invalid(pf3_trues, pf3_preds)
        pf2_trues_filt, pf2_preds_filt = filter_invalid(pf2_trues, pf2_preds)
        pf1_trues_filt, pf1_preds_filt = filter_invalid(pf1_trues, pf1_preds)
        leaf_trues_filt_soft, leaf_preds_filt_soft = filter_invalid(leaf_trues, leaf_preds_soft)
        leaf_trues_filt_sig, leaf_preds_filt_sig = filter_invalid(leaf_trues, leaf_preds_sig)

        print_per_class_metrics(pf4_trues_filt, pf4_preds_filt, "PF4", label_maps['parent_folder_4'])
        print_per_class_metrics(pf3_trues_filt, pf3_preds_filt, "PF3", label_maps['parent_folder_3'])
        print_per_class_metrics(pf2_trues_filt, pf2_preds_filt, "PF2", label_maps['parent_folder_2'])
        print_per_class_metrics(pf1_trues_filt, pf1_preds_filt, "PF1", label_maps['parent_folder_1'])
        print_per_class_metrics(leaf_trues_filt_soft, leaf_preds_filt_soft, "Leaf Class (Softmax)", label_maps['classification'])
        print_per_class_metrics(leaf_trues_filt_sig, leaf_preds_filt_sig, "Leaf Class (Sigmoid)", label_maps['classification'])

        compute_metrics(pf4_trues_filt, pf4_preds_filt, "PF4")
        compute_metrics(pf3_trues_filt, pf3_preds_filt, "PF3")
        compute_metrics(pf2_trues_filt, pf2_preds_filt, "PF2")
        compute_metrics(pf1_trues_filt, pf1_preds_filt, "PF1")
        compute_metrics(leaf_trues_filt_soft, leaf_preds_filt_soft, "Leaf Class (Softmax)")
        compute_metrics(leaf_trues_filt_sig, leaf_preds_filt_sig, "Leaf Class (Sigmoid)")

        train_pf4_acc = 100. * pf4_correct/pf4_total if pf4_total > 0 else 0.0
        train_pf3_acc = 100. * pf3_correct/pf3_total if pf3_total > 0 else 0.0
        train_pf2_acc = 100. * pf2_correct/pf2_total if pf2_total > 0 else 0.0
        train_pf1_acc = 100. * pf1_correct/pf1_total if pf1_total > 0 else 0.0
        train_class_acc_soft = 100.*leaf_class_correct_soft/leaf_class_total if leaf_class_total > 0 else 0.0
        train_class_acc_sig = 100.*leaf_class_correct_sig/leaf_class_total if leaf_class_total > 0 else 0.0
        train_loss = train_loss/(idx+1)
        epoch_end = time.time()
        print('Iteration %d, train_pf4_acc = %.5f,train_pf3_acc = %.5f,train_pf2_acc = %.5f,train_pf1_acc = %.5f,train_class_acc_soft = %.5f,train_class_acc_sig = %.5f, train_loss = %.6f, Time = %.1fs' % \
            (epoch, train_pf4_acc, train_pf3_acc, train_pf2_acc, train_pf1_acc, train_class_acc_soft, train_class_acc_sig, train_loss, (epoch_end - epoch_start)))

        # Level metrics for per-class logging
        level_metrics_dict = {
            "PF4": (pf4_trues_filt, pf4_preds_filt),
            "PF3": (pf3_trues_filt, pf3_preds_filt),
            "PF2": (pf2_trues_filt, pf2_preds_filt),
            "PF1": (pf1_trues_filt, pf1_preds_filt),
            "Leaf Class (Softmax)": (leaf_trues_filt_soft, leaf_preds_filt_soft),
            "Leaf Class (Sigmoid)": (leaf_trues_filt_sig, leaf_preds_filt_sig)
        }

        # Log per-class metrics to csv
        log_per_class_metrics(
            epoch=epoch,
            level_metrics_dict=level_metrics_dict,
            label_maps={
                "PF4": label_maps['parent_folder_4'],
                "PF3": label_maps['parent_folder_3'],
                "PF2": label_maps['parent_folder_2'],
                "PF1": label_maps['parent_folder_1'],
                "Leaf Class (Softmax)": label_maps['classification'],
                "Leaf Class (Sigmoid)": label_maps['classification']
            },
            split="Train"
        )

        (
            test_pf4_acc, test_pf3_acc, test_pf2_acc, test_pf1_acc,
            test_class_acc_soft, test_class_acc_sig, test_loss,
            level_metrics_dict_test
        ) = test(net, testloader, tree, device, dataset, trainset, trees, get_test_transform, image_size)

        log_per_class_metrics(
            epoch=epoch,
            level_metrics_dict=level_metrics_dict_test,
            label_maps={
                "PF4": label_maps['parent_folder_4'],
                "PF3": label_maps['parent_folder_3'],
                "PF2": label_maps['parent_folder_2'],
                "PF1": label_maps['parent_folder_1'],
                "Leaf Class (Softmax)": label_maps['classification'],
                "Leaf Class (Sigmoid)": label_maps['classification']
            },
            split="Test"
        )

        # Save per-epoch summary to csv
        log_epoch_summary_to_csv(
            epoch=epoch,
            train_pf4_acc=train_pf4_acc,
            train_pf3_acc=train_pf3_acc,
            train_pf2_acc=train_pf2_acc,
            train_pf1_acc=train_pf1_acc,
            train_leaf_acc_soft=train_class_acc_soft,
            train_leaf_acc_sig=train_class_acc_sig,
            train_loss=train_loss,
            test_pf4_acc=test_pf4_acc,
            test_pf3_acc=test_pf3_acc,
            test_pf2_acc=test_pf2_acc,
            test_pf1_acc=test_pf1_acc,
            test_leaf_acc_soft=test_class_acc_soft,
            test_leaf_acc_sig=test_class_acc_sig,
            test_loss=test_loss,
            epoch_time=epoch_end - epoch_start
        )

        if test_class_acc_soft > max_val_acc:
            max_val_acc = test_class_acc_soft
            best_epoch = epoch
            net.cpu()

            # Ensure the save directory exists
            save_dir = f'./models_{dataset}'
            os.makedirs(save_dir, exist_ok=True)

            # Save the model
            torch.save(net, './models_'+dataset+'/model_'+save_name+'.pth')
            net.to(device)

    print('\n\nBest Epoch: %d, Best Results: %.5f' % (best_epoch, max_val_acc))


def test(net, testloader, tree, device, dataset, trainset, trees, get_test_transform, image_size):
    epoch_start = time.time()
    net.eval()

    # Update test transform based on current image size
    testloader.dataset.transform = get_test_transform(image_size)

    test_loss = 0
    pf4_correct = pf3_correct = pf2_correct = pf1_correct = 0
    leaf_class_correct_soft = leaf_class_correct_sig = 0
    pf4_total = pf3_total = pf2_total = pf1_total = leaf_class_total = 0

    pf4_preds, pf4_trues = [], []
    pf3_preds, pf3_trues = [], []
    pf2_preds, pf2_trues = [], []
    pf1_preds, pf1_trues = [], []
    leaf_preds_soft, leaf_preds_sig, leaf_trues = [], [], []

    # leaf_class_labels = find_leaf_class_labels(trees)
    # leaf_class_labels_tensor = torch.tensor(leaf_class_labels, device=device)

    for batch_idx, (inputs, targets, indices, class_names) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        pf4_targets, pf3_targets, pf2_targets, pf1_targets, leaf_targets_sig, leaf_class_weights_tensor = get_5_level_targets(targets, device, dataset, trees, leaf_class_weights_file=None) # to be defined in utils_insect.py

        # # After constructing leaf_targets_sig
        # print("DEBUG: leaf_targets_sig shape:", leaf_targets_sig.shape)
        # print("DEBUG: leaf_targets_sig contents:", leaf_targets_sig)

        # Optional: check if it's empty
        # if leaf_targets_sig.numel() == 0:
        #     print("WARNING: No valid leaf targets found for these samples.")

        # print("leaf_targets_sig shape:", leaf_targets_sig.shape)
        # print("leaf_indices shape:", len(leaf_indices))

        pf4_sig, pf3_sig, pf2_sig, pf1_sig, leaf_class_soft, leaf_class_sig = net(inputs)

        # Compute tree loss
        tree_loss = tree(torch.cat([pf4_sig, pf3_sig, pf2_sig, pf1_sig, leaf_class_sig], 1), targets, device)

        # Only compute CE loss if there are valid leaf targets
        valid_mask = get_valid_hierarchical_mask(pf4_targets, pf3_targets, pf2_targets, pf1_targets,
                                                 leaf_targets_sig)

        if valid_mask.sum() > 0:
            selected_logits = leaf_class_soft[valid_mask]
            selected_targets = leaf_targets_sig[valid_mask]

            if leaf_class_weights_tensor is not None:
                ce_loss_fn = torch.nn.CrossEntropyLoss(weight=leaf_class_weights_tensor.to(torch.float64))
            else:
                ce_loss_fn = torch.nn.CrossEntropyLoss()

            ce_loss_leaf_node = ce_loss_fn(selected_logits.to(torch.float64), selected_targets)
            loss = ce_loss_leaf_node + tree_loss
        else:
            loss = tree_loss

        # leaf_mask = torch.isin(targets, leaf_class_labels_tensor)
        # leaf_labels = torch.nonzero(leaf_mask, as_tuple=False)
        #
        # if leaf_mask.any():
        #     select_leaf_labels = targets[leaf_mask]
        #     select_fc_soft = leaf_class_soft[leaf_mask]
        #     ce_loss_leaf_node = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)
        #     loss = ce_loss_leaf_node + tree_loss
        # else:
        #     loss = tree_loss

        test_loss += loss.item()

        # Predictions
        with torch.no_grad():
            # PF4 Accuracy
            _, pf4_predicted = torch.max(pf4_sig.data, 1)
            valid_pf4_mask = pf4_targets != -1
            if valid_pf4_mask.sum() > 0:
                pf4_predicted_selected = pf4_predicted[valid_pf4_mask]
                pf4_targets_selected = pf4_targets[valid_pf4_mask]
                pf4_total += pf4_targets_selected.size(0)
                pf4_correct += pf4_predicted_selected.eq(pf4_targets_selected).cpu().sum().item()
                pf4_preds.extend(pf4_predicted_selected.cpu().numpy())
                pf4_trues.extend(pf4_targets_selected.cpu().numpy())

            # PF3 Accuracy
            _, pf3_predicted = torch.max(pf3_sig.data, 1)
            valid_pf3_mask = pf3_targets != -1
            if valid_pf3_mask.sum() > 0:
                pf3_predicted_selected = pf3_predicted[valid_pf3_mask]
                pf3_targets_selected = pf3_targets[valid_pf3_mask]
                pf3_total += pf3_targets_selected.size(0)
                pf3_correct += pf3_predicted_selected.eq(pf3_targets_selected).cpu().sum().item()
                pf3_preds.extend(pf3_predicted_selected.cpu().numpy())
                pf3_trues.extend(pf3_targets_selected.cpu().numpy())

            # PF2 Accuracy
            _, pf2_predicted = torch.max(pf2_sig.data, 1)
            valid_pf2_mask = pf2_targets != -1
            if valid_pf2_mask.sum() > 0:
                pf2_predicted_selected = pf2_predicted[valid_pf2_mask]
                pf2_targets_selected = pf2_targets[valid_pf2_mask]
                pf2_total += pf2_targets_selected.size(0)
                pf2_correct += pf2_predicted_selected.eq(pf2_targets_selected).cpu().sum().item()
                pf2_preds.extend(pf2_predicted_selected.cpu().numpy())
                pf2_trues.extend(pf2_targets_selected.cpu().numpy())

            # PF1 Accuracy
            _, pf1_predicted = torch.max(pf1_sig.data, 1)
            valid_pf1_mask = pf1_targets != -1
            if valid_pf1_mask.sum() > 0:
                pf1_predicted_selected = pf1_predicted[valid_pf1_mask]
                pf1_targets_selected = pf1_targets[valid_pf1_mask]
                pf1_total += pf1_targets_selected.size(0)
                pf1_correct += pf1_predicted_selected.eq(pf1_targets_selected).cpu().sum().item()
                pf1_preds.extend(pf1_predicted_selected.cpu().numpy())
                pf1_trues.extend(pf1_targets_selected.cpu().numpy())

            # Leaf
            # Leaf Accuracy (Softmax and Sigmoid)
            valid_leaf_mask = leaf_targets_sig != -1

            if valid_mask.sum() > 0:
                select_leaf_class_soft = leaf_class_soft[valid_mask]
                select_leaf_class_sig = leaf_class_sig[valid_mask]
                select_leaf_targets = leaf_targets_sig[valid_mask]

                num_leaf_classes = num_classes_per_level['leaf']

                if select_leaf_targets.min() < 0 or select_leaf_targets.max() >= num_leaf_classes:
                    print("Invalid leaf target detected:", select_leaf_targets)
                else:
                    _, class_predicted_soft = torch.max(select_leaf_class_soft.data, 1)
                    _, class_predicted_sig = torch.max(select_leaf_class_sig.data, 1)

                    leaf_class_total += select_leaf_targets.size(0)
                    leaf_class_correct_soft += class_predicted_soft.eq(select_leaf_targets).cpu().sum().item()
                    leaf_class_correct_sig += class_predicted_sig.eq(select_leaf_targets).cpu().sum().item()

                    leaf_preds_soft.extend(class_predicted_soft.cpu().numpy())
                    leaf_preds_sig.extend(class_predicted_sig.cpu().numpy())
                    leaf_trues.extend(select_leaf_targets.cpu().numpy())
            else:
                pass
                # print("Skipping leaf accuracy: no valid leaf targets in batch.")

            # else:
            #     print("Skipping leaf accuracy: no valid leaf targets in batch.")

    # Accuracy
    test_pf4_acc = 100. * pf4_correct / pf4_total if pf4_total > 0 else 0.0
    test_pf3_acc = 100. * pf3_correct / pf3_total if pf3_total > 0 else 0.0
    test_pf2_acc = 100. * pf2_correct / pf2_total if pf2_total > 0 else 0.0
    test_pf1_acc = 100. * pf1_correct / pf1_total if pf1_total > 0 else 0.0
    test_class_acc_soft = 100. * leaf_class_correct_soft / leaf_class_total if leaf_class_total > 0 else 0.0
    test_class_acc_sig = 100. * leaf_class_correct_sig / leaf_class_total if leaf_class_total > 0 else 0.0
    test_loss = test_loss / (batch_idx + 1)
    epoch_end = time.time()

    print(f'\nTest Results:')
    print(f'PF4 Acc: {test_pf4_acc:.4f}, PF3 Acc: {test_pf3_acc:.4f}, PF2 Acc: {test_pf2_acc:.4f}, PF1 Acc: {test_pf1_acc:.4f}')
    print(f'Leaf Acc (Softmax): {test_class_acc_soft:.4f}, Leaf Acc (Sigmoid): {test_class_acc_sig:.4f}, Loss: {test_loss:.6f}, Time: {epoch_end - epoch_start:.2f}s')

    # Sklearn metrics
    def compute_metrics(y_true, y_pred, level_name):
        print(f"\nMetrics for {level_name}:")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred, average='macro'):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred, average='macro'):.4f}")

    def print_per_class_metrics(y_true, y_pred, level_name, label_map=None):
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        unique_classes = sorted(set(y_true))

        print(f"\nPer-class metrics for {level_name}:")
        for i, cls in enumerate(unique_classes):
            name = label_map.get(str(cls), f"Class {cls}") if label_map else f"Class {cls}"
            print(f"Class {cls} ({name}): Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")

    def filter_invalid(y_true, y_pred, sentinel=-1):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        valid_mask = y_true != sentinel
        return y_true[valid_mask], y_pred[valid_mask]

    def load_level_name_maps(filename):
        script_dir = os.path.dirname(__file__)
        path = os.path.join(script_dir, filename)
        with open(path, 'r') as f:
            return json.load(f)

    label_maps = load_level_name_maps('level_name_maps_13Oct25.json')  # contains mappings for all levels

    # Filter invalid entries
    pf4_trues_filt, pf4_preds_filt = filter_invalid(pf4_trues, pf4_preds)
    pf3_trues_filt, pf3_preds_filt = filter_invalid(pf3_trues, pf3_preds)
    pf2_trues_filt, pf2_preds_filt = filter_invalid(pf2_trues, pf2_preds)
    pf1_trues_filt, pf1_preds_filt = filter_invalid(pf1_trues, pf1_preds)
    leaf_trues_filt_soft, leaf_preds_filt_soft = filter_invalid(leaf_trues, leaf_preds_soft)
    leaf_trues_filt_sig, leaf_preds_filt_sig = filter_invalid(leaf_trues, leaf_preds_sig)

    print_per_class_metrics(pf4_trues_filt, pf4_preds_filt, "PF4", label_maps['parent_folder_4'])
    print_per_class_metrics(pf3_trues_filt, pf3_preds_filt, "PF3", label_maps['parent_folder_3'])
    print_per_class_metrics(pf2_trues_filt, pf2_preds_filt, "PF2", label_maps['parent_folder_2'])
    print_per_class_metrics(pf1_trues_filt, pf1_preds_filt, "PF1", label_maps['parent_folder_1'])
    print_per_class_metrics(leaf_trues_filt_soft, leaf_preds_filt_soft, "Leaf Class (Softmax)", label_maps['classification'])
    print_per_class_metrics(leaf_trues_filt_sig, leaf_preds_filt_sig, "Leaf Class (Sigmoid)", label_maps['classification'])

    compute_metrics(pf4_trues_filt, pf4_preds_filt, "PF4")
    compute_metrics(pf3_trues_filt, pf3_preds_filt, "PF3")
    compute_metrics(pf2_trues_filt, pf2_preds_filt, "PF2")
    compute_metrics(pf1_trues_filt, pf1_preds_filt, "PF1")
    compute_metrics(leaf_trues_filt_soft, leaf_preds_filt_soft, "Leaf Class (Softmax)")
    compute_metrics(leaf_trues_filt_sig, leaf_preds_filt_sig, "Leaf Class (Sigmoid)")

    # Level metrics for per-class logging
    level_metrics_dict = {
        "PF4": (pf4_trues_filt, pf4_preds_filt),
        "PF3": (pf3_trues_filt, pf3_preds_filt),
        "PF2": (pf2_trues_filt, pf2_preds_filt),
        "PF1": (pf1_trues_filt, pf1_preds_filt),
        "Leaf Class (Softmax)": (leaf_trues_filt_soft, leaf_preds_filt_soft),
        "Leaf Class (Sigmoid)": (leaf_trues_filt_sig, leaf_preds_filt_sig)
    }

    return (
        test_pf4_acc, test_pf3_acc, test_pf2_acc, test_pf1_acc,
        test_class_acc_soft, test_class_acc_sig, test_loss,
        {
            "PF4": (pf4_trues_filt, pf4_preds_filt),
            "PF3": (pf3_trues_filt, pf3_preds_filt),
            "PF2": (pf2_trues_filt, pf2_preds_filt),
            "PF1": (pf1_trues_filt, pf1_preds_filt),
            "Leaf Class (Softmax)": (leaf_trues_filt_soft, leaf_preds_filt_soft),
            "Leaf Class (Sigmoid)": (leaf_trues_filt_sig, leaf_preds_filt_sig)
        }
    )
    

def test_AP(model, dataset, test_set, test_data_loader, device):
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        model.eval()
        for j, (images, labels) in enumerate(test_data_loader):
            images = images.to(device)
            labels = labels.to(device)
            select_labels = labels[:, test_set.to_eval]
            if dataset in ['Insect']:
                y_pf4_sig, y_pf3_sig, y_pf2_sig, y_pf1_sig, y_leaf_class_sig, y_leaf_class_soft = model(images)
                batch_pMargin = torch.cat([
                    y_pf4_sig,
                    y_pf3_sig,
                    y_pf2_sig,
                    y_pf1_sig,
                    torch.softmax(y_leaf_class_soft, dim=1)
                ], dim=1).data
            else:
                y_pf4_sig: object
                y_pf4_sig, y_leaf_class_soft, y_leaf_class_sig = model(images)
                batch_pMargin = torch.cat([y_pf4_sig, torch.softmax(y_leaf_class_soft, dim=1)], dim=1).data
            
            predicted = batch_pMargin > 0.5
            total += select_labels.size(0) * select_labels.size(1)
            correct += (predicted.to(torch.float64) == select_labels).sum()
            cpu_batch_pMargin = batch_pMargin.to('cpu')
            y = select_labels.to('cpu')
            if j == 0:
                test = cpu_batch_pMargin
                test_y = y
            else:
                test = torch.cat((test, cpu_batch_pMargin), dim=0)
                test_y = torch.cat((test_y, y), dim=0)
        score = average_precision_score(test_y, test, average='micro')
        print('Accuracy:' + str(float(correct) / float(total)))
        print('Precision score:' + str(score))