"""
Metrics computation and reporting for hierarchical classification.
Includes per-class and aggregate metrics.
"""

import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

import config


def build_hierarchy_structure():
    """
    Build a dictionary mapping parent node IDs to their child node IDs
    from config.HIERARCHY.

    Returns:
        Dict mapping parent_id -> [child_ids]
    """
    hierarchy_structure = {}

    for path in config.HIERARCHY:
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i + 1]

            if parent not in hierarchy_structure:
                hierarchy_structure[parent] = []
            if child not in hierarchy_structure[parent]:
                hierarchy_structure[parent].append(child)

    return hierarchy_structure


def greedy_top_down_prediction(pMargin_np, hierarchy_structure):
    """
    Perform greedy top-down prediction to enforce hierarchical consistency.

    Args:
        pMargin_np: Numpy array of marginal probabilities [batch_size, num_nodes]
        hierarchy_structure: Dict mapping parent node IDs to their child node IDs
                             Example: {pf4_id: [pf3_ids], pf3_id: [pf2_ids], ...}

    Returns:
        List of predicted paths per sample.
        Each path is a list of local indices for each level.
        Stops when the hierarchy ends.
    """
    batch_size = pMargin_np.shape[0]
    predicted_paths = []

    for i in range(batch_size):
        path = []

        # Start at pf4 (root level)
        pf4_nodes = range(*config.GLOBAL_INDEX_RANGES['pf4'])
        pf4_pred_global = max(pf4_nodes, key=lambda n: pMargin_np[i, n])
        pf4_pred_local = pf4_pred_global - config.GLOBAL_INDEX_RANGES['pf4'][0]
        path.append(pf4_pred_local)

        # Move down the hierarchy greedily until no children exist
        current_parent = pf4_pred_global
        for level_name in ["pf3", "pf2", "pf1", "leaf"]:
            child_nodes = hierarchy_structure.get(current_parent, [])
            if not child_nodes:
                break  # Stop here because this is a leaf in the hierarchy

            # Choose child with highest marginal probability
            pred_global = max(child_nodes, key=lambda n: pMargin_np[i, n])
            pred_local = pred_global - config.GLOBAL_INDEX_RANGES[level_name][0]
            path.append(pred_local)
            current_parent = pred_global

        predicted_paths.append(path)

    return predicted_paths


def compute_metrics(y_true, y_pred, level_name):
    """
    Compute and print aggregate metrics for a hierarchy level.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        level_name: Name of the hierarchy level
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"{level_name}: No valid samples to compute metrics")
        return

    print(f"\n{level_name} Metrics:")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  F1 Score:  {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  MCC:       {matthews_corrcoef(y_true, y_pred):.4f}")


def print_per_class_metrics(y_true, y_pred, level_name, label_map=None):
    """
    Print per-class precision, recall, and F1 score.
    Handles classes with zero predictions by including all classes present in ground truth.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        level_name: Name of the hierarchy level
        label_map: Dict mapping label indices to names
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"\n{level_name} Per-Class Metrics: No valid samples")
        return

    # Get all unique classes from ground truth
    unique_classes = sorted(set(y_true))

    # Compute metrics with explicit labels to ensure all classes are included
    precision = precision_score(y_true, y_pred, labels=unique_classes, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=unique_classes, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=unique_classes, average=None, zero_division=0)

    # Count predictions per class
    pred_counts = {cls: 0 for cls in unique_classes}
    for pred in y_pred:
        if pred in pred_counts:
            pred_counts[pred] += 1

    print(f"\n{level_name} Per-Class Metrics:")
    for i, cls in enumerate(unique_classes):
        name = label_map.get(str(cls), f"Class {cls}") if label_map else f"Class {cls}"
        pred_count = pred_counts[cls]
        zero_flag = " [ZERO PRED]" if pred_count == 0 else ""
        print(f"  {cls:3d} ({name:30s}): P={precision[i]:.4f}, R={recall[i]:.4f}, "
              f"F1={f1[i]:.4f}, Preds={pred_count:4d}{zero_flag}")


def filter_invalid(y_true, y_pred, sentinel=-1):
    """
    Filter out invalid (sentinel) values from predictions and ground truth.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        sentinel: Value indicating invalid labels
        
    Returns:
        Tuple of (filtered_y_true, filtered_y_pred)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    valid_mask = y_true != sentinel
    return y_true[valid_mask], y_pred[valid_mask]


def load_level_name_maps(filename, run_folder):
    """
    Load level name mappings from JSON file.
    
    Args:
        filename: Name of JSON file
        run_folder: Path to run folder
        
    Returns:
        Dict with mappings for each hierarchy level
    """
    path = run_folder / filename
    if not path.exists():
        print(f"Warning: Level name maps file not found at {path}")
        return {}
    
    with open(path, 'r') as f:
        return json.load(f)


def compute_hierarchical_accuracy_with_inference(
        combined_output,
        pf4_targets, pf3_targets, pf2_targets, pf1_targets, leaf_targets,
        tree_loss, device, config_module, counters, sentinel=-1, use_greedy=False
):
    """
    Compute hierarchical accuracy using tree loss inference method.
    Optimized:
    - Static caching for hierarchy and level nodes
    - Vectorized accuracy computation for non-greedy mode
    """

    # --- Static cache for hierarchy and level nodes ---
    if not hasattr(compute_hierarchical_accuracy_with_inference, "_cache"):
        compute_hierarchical_accuracy_with_inference._cache = {
            "hierarchy_structure": build_hierarchy_structure() if use_greedy else None,
            "level_nodes": {
                "pf4": range(*config.GLOBAL_INDEX_RANGES['pf4']),
                "pf3": range(*config.GLOBAL_INDEX_RANGES['pf3']),
                "pf2": range(*config.GLOBAL_INDEX_RANGES['pf2']),
                "pf1": range(*config.GLOBAL_INDEX_RANGES['pf1']),
                "leaf": range(*config.GLOBAL_INDEX_RANGES['leaf'])
            },
            "level_names": ["pf4", "pf3", "pf2", "pf1", "leaf"]
        }

    cache = compute_hierarchical_accuracy_with_inference._cache
    hierarchy_structure = cache["hierarchy_structure"]
    level_nodes = cache["level_nodes"]
    level_names_const = cache["level_names"]

    # Get marginal probabilities
    pMargin = tree_loss.inference(combined_output, device)
    pMargin_np = pMargin.cpu().numpy()

    # Convert targets to NumPy arrays
    targets_np = [t.cpu().numpy() for t in [pf4_targets, pf3_targets, pf2_targets, pf1_targets, leaf_targets]]

    if use_greedy:
        # Greedy mode: still sequential per sample
        for i in range(pMargin_np.shape[0]):
            # Build true path dynamically
            true_path = []
            level_names = []
            for level_name, targets in zip(level_names_const, targets_np):
                if targets[i] != sentinel:
                    true_path.append(targets[i])
                    level_names.append(level_name)

            if not true_path:
                continue

            # Predict path using greedy
            pred_paths = greedy_top_down_prediction(pMargin_np[i:i + 1], hierarchy_structure)
            pred_path = pred_paths[0]

            # Align predictions if greedy stops early
            if len(pred_path) < len(level_names):
                level_names = level_names[:len(pred_path)]
                true_path = true_path[:len(pred_path)]

            # Update counters
            for idx, level_name in enumerate(level_names):
                counters[level_name]["total"] += 1
                counters[level_name]["trues"].append(true_path[idx])
                counters[level_name]["preds"].append(pred_path[idx])
                if pred_path[idx] == true_path[idx]:
                    counters[level_name]["correct"] += 1

    else:
        # Non-greedy mode: vectorized
        for level_name, node_ids, targets in zip(level_names_const, level_nodes.values(), targets_np):
            valid_mask = targets != sentinel
            if not np.any(valid_mask):
                continue

            # Predict for all samples at this level
            probs = pMargin_np[:, node_ids]  # [batch_size, num_nodes_in_level]
            preds_global = np.argmax(probs, axis=1)
            preds_global = np.array(node_ids)[preds_global]

            # Convert to local IDs
            offset = config.GLOBAL_INDEX_RANGES[level_name][0]
            preds_local = preds_global - offset

            # Update counters in bulk
            counters[level_name]["total"] += valid_mask.sum()
            counters[level_name]["trues"].extend(targets[valid_mask].tolist())
            counters[level_name]["preds"].extend(preds_local[valid_mask].tolist())
            counters[level_name]["correct"] += np.sum(preds_local[valid_mask] == targets[valid_mask])

    return counters



# def compute_level_accuracy_with_softmax(logits, targets, total, correct, preds, trues, sentinel=-1):
#     """
#     Compute accuracy for a single hierarchy level using softmax predictions.
#
#     Args:
#         logits: Raw logits output for this level [batch, num_classes]
#         targets: Ground truth targets for this level
#         total: Running total of samples
#         correct: Running count of correct predictions
#         preds: List to append predictions to
#         trues: List to append ground truth to
#         sentinel: Value indicating invalid targets
#
#     Returns:
#         Tuple of (updated_total, updated_correct)
#     """
#     # Apply softmax to get probabilities
#     probs = torch.softmax(logits, dim=1)
#     predicted = torch.argmax(probs, dim=1)
#
#     valid_mask = targets != sentinel
#
#     if valid_mask.sum() > 0:
#         selected_pred = predicted[valid_mask]
#         selected_targets = targets[valid_mask]
#
#         total += selected_targets.size(0)
#         correct += selected_pred.eq(selected_targets).cpu().sum().item()
#
#         preds.extend(selected_pred.cpu().numpy())
#         trues.extend(selected_targets.cpu().numpy())
#
#     return total, correct

# def compute_level_accuracy(sig_output, targets, total, correct, preds, trues, sentinel=-1):
#     """
#     Compute accuracy for a single hierarchy level.
#
#     Args:
#         sig_output: Model sigmoid/softmax output for this level
#         targets: Ground truth targets for this level
#         total: Running total of samples
#         correct: Running count of correct predictions
#         preds: List to append predictions to
#         trues: List to append ground truth to
#         sentinel: Value indicating invalid targets
#
#     Returns:
#         Tuple of (updated_total, updated_correct)
#     """
#     predicted = torch.argmax(sig_output.data, dim=1)
#     valid_mask = targets != sentinel
#
#     if valid_mask.sum() > 0:
#         selected_pred = predicted[valid_mask]
#         selected_targets = targets[valid_mask]
#
#         total += selected_targets.size(0)
#         correct += selected_pred.eq(selected_targets).cpu().sum().item()
#
#         preds.extend(selected_pred.cpu().numpy())
#         trues.extend(selected_targets.cpu().numpy())
#
#     return total, correct
#
#
# def compute_leaf_accuracy(leaf_soft, leaf_sig, leaf_targets, valid_mask, num_leaf_classes,
#                          total, correct_soft, correct_sig, preds_soft, preds_sig, trues,
#                          sentinel=-1, verbose=False):
#     """
#     Compute accuracy for leaf level with both softmax and sigmoid outputs.
#
#     Args:
#         leaf_soft: Softmax output for leaf level
#         leaf_sig: Sigmoid output for leaf level
#         leaf_targets: Ground truth leaf targets
#         valid_mask: Mask indicating valid samples
#         num_leaf_classes: Number of leaf classes
#         total: Running total of samples
#         correct_soft: Running count of correct predictions (softmax)
#         correct_sig: Running count of correct predictions (sigmoid)
#         preds_soft: List to append softmax predictions to
#         preds_sig: List to append sigmoid predictions to
#         trues: List to append ground truth to
#         sentinel: Value indicating invalid targets
#         verbose: Whether to print debug info
#
#     Returns:
#         Tuple of (updated_total, updated_correct_soft, updated_correct_sig)
#     """
#     if valid_mask.sum() > 0:
#         selected_soft = leaf_soft[valid_mask]
#         selected_sig = leaf_sig[valid_mask]
#         selected_targets = leaf_targets[valid_mask]
#
#         # Validate target range
#         if selected_targets.min() < 0 or selected_targets.max() >= num_leaf_classes:
#             if verbose:
#                 print(f"Warning: Invalid leaf target detected. "
#                       f"Min: {selected_targets.min()}, Max: {selected_targets.max()}, "
#                       f"Expected range: [0, {num_leaf_classes-1}]")
#             return total, correct_soft, correct_sig
#
#         # Compute predictions
#         pred_soft = torch.argmax(selected_soft.data, dim=1)
#         pred_sig = torch.argmax(selected_sig.data, dim=1)
#
#         # Update counters
#         total += selected_targets.size(0)
#         correct_soft += pred_soft.eq(selected_targets).cpu().sum().item()
#         correct_sig += pred_sig.eq(selected_targets).cpu().sum().item()
#
#         # Store predictions
#         preds_soft.extend(pred_soft.cpu().numpy())
#         preds_sig.extend(pred_sig.cpu().numpy())
#         trues.extend(selected_targets.cpu().numpy())
#     elif verbose:
#         print("Info: No valid leaf targets in this batch.")
#
#     return total, correct_soft, correct_sig


def compute_confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Compute confusion matrix, ensuring all classes are represented.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes (auto-detected if None)

    Returns:
        Confusion matrix as numpy array
    """
    from sklearn.metrics import confusion_matrix

    if num_classes is None:
        # Include all classes from ground truth
        all_classes = sorted(set(y_true))
    else:
        all_classes = list(range(num_classes))

    return confusion_matrix(y_true, y_pred, labels=all_classes)


# def print_confusion_matrix(y_true, y_pred, level_name, label_map=None):
#     """
#     Print confusion matrix in readable format.
#
#     Args:
#         y_true: Ground truth labels
#         y_pred: Predicted labels
#         level_name: Name of hierarchy level
#         label_map: Optional mapping from indices to names
#     """
#     cm = compute_confusion_matrix(y_true, y_pred)
#     unique_classes = sorted(set(y_true))
#
#     print(f"\nConfusion Matrix for {level_name}:")
#     print("True\\Pred", end="")
#     for cls in unique_classes:
#         name = label_map.get(str(cls), str(cls))[:10] if label_map else str(cls)
#         print(f"\t{name}", end="")
#     print()
#
#     for i, cls_true in enumerate(unique_classes):
#         name = label_map.get(str(cls_true), str(cls_true))[:10] if label_map else str(cls_true)
#         print(f"{name}", end="")
#         for j, cls_pred in enumerate(unique_classes):
#             print(f"\t{cm[i, j]}", end="")
#         print()
