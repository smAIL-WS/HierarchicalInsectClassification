"""
Metrics computation and reporting for hierarchical classification.
Includes per-class and aggregate metrics.
"""

import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

import config


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
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        level_name: Name of the hierarchy level
        label_map: Dict mapping label indices to names
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"\n{level_name} Per-Class Metrics: No valid samples")
        return
    
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    unique_classes = sorted(set(y_true))
    
    print(f"\n{level_name} Per-Class Metrics:")
    for i, cls in enumerate(unique_classes):
        if i < len(precision):
            name = label_map.get(str(cls), f"Class {cls}") if label_map else f"Class {cls}"
            print(f"  {cls:3d} ({name:30s}): P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}")


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


def compute_level_accuracy_with_softmax(logits, targets, total, correct, preds, trues, sentinel=-1):
    """
    Compute accuracy for a single hierarchy level using softmax predictions.

    Args:
        logits: Raw logits output for this level [batch, num_classes]
        targets: Ground truth targets for this level
        total: Running total of samples
        correct: Running count of correct predictions
        preds: List to append predictions to
        trues: List to append ground truth to
        sentinel: Value indicating invalid targets

    Returns:
        Tuple of (updated_total, updated_correct)
    """
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=1)
    predicted = torch.argmax(probs, dim=1)

    valid_mask = targets != sentinel

    if valid_mask.sum() > 0:
        selected_pred = predicted[valid_mask]
        selected_targets = targets[valid_mask]

        total += selected_targets.size(0)
        correct += selected_pred.eq(selected_targets).cpu().sum().item()

        preds.extend(selected_pred.cpu().numpy())
        trues.extend(selected_targets.cpu().numpy())

    return total, correct

def compute_level_accuracy(sig_output, targets, total, correct, preds, trues, sentinel=-1):
    """
    Compute accuracy for a single hierarchy level.
    
    Args:
        sig_output: Model sigmoid/softmax output for this level
        targets: Ground truth targets for this level
        total: Running total of samples
        correct: Running count of correct predictions
        preds: List to append predictions to
        trues: List to append ground truth to
        sentinel: Value indicating invalid targets
        
    Returns:
        Tuple of (updated_total, updated_correct)
    """
    predicted = torch.argmax(sig_output.data, dim=1)
    valid_mask = targets != sentinel
    
    if valid_mask.sum() > 0:
        selected_pred = predicted[valid_mask]
        selected_targets = targets[valid_mask]
        
        total += selected_targets.size(0)
        correct += selected_pred.eq(selected_targets).cpu().sum().item()
        
        preds.extend(selected_pred.cpu().numpy())
        trues.extend(selected_targets.cpu().numpy())
    
    return total, correct


def compute_leaf_accuracy(leaf_soft, leaf_sig, leaf_targets, valid_mask, num_leaf_classes,
                         total, correct_soft, correct_sig, preds_soft, preds_sig, trues, 
                         sentinel=-1, verbose=False):
    """
    Compute accuracy for leaf level with both softmax and sigmoid outputs.
    
    Args:
        leaf_soft: Softmax output for leaf level
        leaf_sig: Sigmoid output for leaf level
        leaf_targets: Ground truth leaf targets
        valid_mask: Mask indicating valid samples
        num_leaf_classes: Number of leaf classes
        total: Running total of samples
        correct_soft: Running count of correct predictions (softmax)
        correct_sig: Running count of correct predictions (sigmoid)
        preds_soft: List to append softmax predictions to
        preds_sig: List to append sigmoid predictions to
        trues: List to append ground truth to
        sentinel: Value indicating invalid targets
        verbose: Whether to print debug info
        
    Returns:
        Tuple of (updated_total, updated_correct_soft, updated_correct_sig)
    """
    if valid_mask.sum() > 0:
        selected_soft = leaf_soft[valid_mask]
        selected_sig = leaf_sig[valid_mask]
        selected_targets = leaf_targets[valid_mask]
        
        # Validate target range
        if selected_targets.min() < 0 or selected_targets.max() >= num_leaf_classes:
            if verbose:
                print(f"Warning: Invalid leaf target detected. "
                      f"Min: {selected_targets.min()}, Max: {selected_targets.max()}, "
                      f"Expected range: [0, {num_leaf_classes-1}]")
            return total, correct_soft, correct_sig
        
        # Compute predictions
        pred_soft = torch.argmax(selected_soft.data, dim=1)
        pred_sig = torch.argmax(selected_sig.data, dim=1)
        
        # Update counters
        total += selected_targets.size(0)
        correct_soft += pred_soft.eq(selected_targets).cpu().sum().item()
        correct_sig += pred_sig.eq(selected_targets).cpu().sum().item()
        
        # Store predictions
        preds_soft.extend(pred_soft.cpu().numpy())
        preds_sig.extend(pred_sig.cpu().numpy())
        trues.extend(selected_targets.cpu().numpy())
    elif verbose:
        print("Info: No valid leaf targets in this batch.")
    
    return total, correct_soft, correct_sig


def compute_confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes (auto-detected if None)
        
    Returns:
        Confusion matrix as numpy array
    """
    from sklearn.metrics import confusion_matrix
    
    if num_classes is None:
        num_classes = max(max(y_true), max(y_pred)) + 1
    
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def print_confusion_matrix(y_true, y_pred, level_name, label_map=None):
    """
    Print confusion matrix in readable format.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        level_name: Name of hierarchy level
        label_map: Optional mapping from indices to names
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    unique_classes = sorted(set(y_true))
    
    print(f"\nConfusion Matrix for {level_name}:")
    print("True\\Pred", end="")
    for cls in unique_classes:
        name = label_map.get(str(cls), str(cls))[:10] if label_map else str(cls)
        print(f"\t{name}", end="")
    print()
    
    for i, cls_true in enumerate(unique_classes):
        name = label_map.get(str(cls_true), str(cls_true))[:10] if label_map else str(cls_true)
        print(f"{name}", end="")
        for j, cls_pred in enumerate(unique_classes):
            print(f"\t{cm[i, j]}", end="")
        print()
