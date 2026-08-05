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


def compute_metrics(y_true, y_pred, level_name, related_info=None):
    """
    Compute and print aggregate metrics for a hierarchy level.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        level_name: Name of the hierarchy level
        related_info: List of (true_id, was_related) tuples
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

    if related_info and len(related_info) > 0:
        rel_acc = np.mean([item[1] for item in related_info])
        print(f"  Related Accuracy: {rel_acc:.4f}")


def print_per_class_metrics(y_true, y_pred, level_name, label_map=None, related_info=None):
    """
    Print per-class precision, recall, and F1 score.
    Handles classes with zero predictions by including all classes present in ground truth.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        level_name: Name of the hierarchy level
        label_map: Dict mapping label indices to names
        related_info: List of (true_id, was_related) tuples
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

    # Map related errors to class IDs
    related_map = {}
    if related_info:
        for tid, was_rel in related_info:
            if tid not in related_map:
                related_map[tid] = []
            related_map[tid].append(was_rel)

    print(f"\n{level_name} Per-Class Metrics:")
    for i, cls in enumerate(unique_classes):
        name = label_map.get(str(cls), f"Class {cls}") if label_map else f"Class {cls}"
        pred_count = pred_counts[cls]
        zero_flag = " [ZERO PRED]" if pred_count == 0 else ""

        rel_str = ""
        if cls in related_map:
            rel_pct = np.mean(related_map[cls])
            rel_str = f", RelErr={rel_pct:.4f}"

        print(f"  {cls:3d} ({name:30s}): P={precision[i]:.4f}, R={recall[i]:.4f}, "
              f"F1={f1[i]:.4f}{rel_str}, Preds={pred_count:4d}{zero_flag}")


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


@torch.no_grad()
def per_level_marginal_argmax(marginals, level_order, level_nodes):
    """Fallback: choose argmax of marginals per level."""
    device = marginals.device
    per_level = {}
    for lvl in level_order:
        ids = torch.as_tensor(level_nodes[lvl], device=device, dtype=torch.long)
        per_level[lvl] = ids[marginals[:, ids].argmax(dim=1)]
    return per_level


@torch.no_grad()
def map_decode_paths(fs_sigmoid, stateSpace_unweighted, level_order, level_nodes):
    """Vectorized MAP decoder over legal states."""
    device = fs_sigmoid.device
    S = stateSpace_unweighted.to(device)
    scores = torch.matmul(S, fs_sigmoid.T)
    best_state_idx = scores.argmax(dim=0)
    best_states = S[best_state_idx]

    path_nodes_global = {}
    for lvl in level_order:
        ids = torch.as_tensor(level_nodes[lvl], device=device, dtype=torch.long)
        active = best_states[:, ids] > 0
        lvl_pred = torch.full((fs_sigmoid.shape[0],), -1, device=device, dtype=torch.long)
        has_any = active.any(dim=1)
        idx_in_level = active.float().argmax(dim=1)
        lvl_pred[has_any] = ids[idx_in_level[has_any]]
        path_nodes_global[lvl] = lvl_pred
    return {"path_nodes_global": path_nodes_global, "best_state_idx": best_state_idx}


@torch.no_grad()
def map_truncate_with_threshold(marginals, map_paths, targets_per_level, level_order, global_index_ranges, threshold,
                                sentinel=-1):
    """Robust back-off truncation logic that handles imbalanced paths (-1 indices)."""
    device = marginals.device
    B = marginals.shape[0]
    avail = torch.stack([(targets_per_level[lvl] != sentinel) for lvl in level_order], dim=1).to(device)
    deepest_idx = (len(level_order) - 1) - torch.flip(avail.int(), dims=[1]).argmax(dim=1)

    final_node = torch.full((B,), -1, device=device, dtype=torch.long)
    final_level = deepest_idx.clone()
    final_conf = torch.zeros(B, device=device)

    for d in reversed(range(len(level_order))):
        lvl = level_order[d]
        mask = (final_level >= d) & (final_node == -1)
        if not mask.any(): continue

        candidates = map_paths[lvl][mask]

        # FIX: Handle cases where the MAP path is shorter than the current level (-1)
        valid_cand_mask = (candidates >= 0)
        confs = torch.zeros_like(candidates, dtype=torch.float32, device=device)

        if valid_cand_mask.any():
            # Only gather confidence for batch indices that actually have a node at this level
            valid_indices = candidates[valid_cand_mask]
            gathered = torch.gather(marginals[mask][valid_cand_mask], 1, valid_indices.unsqueeze(1)).squeeze(1)
            confs[valid_cand_mask] = gathered

        # If confidence >= threshold OR if backed all the way to root, accept
        accept = (confs >= threshold) | (d == 0)

        indices = torch.where(mask)[0][accept]
        final_node[indices] = candidates[accept]
        final_level[indices] = d
        final_conf[indices] = confs[accept]

    return {"final_selected_node_global": final_node, "final_selected_level_idx": final_level,
            "final_selected_confidence": final_conf}


@torch.no_grad()
def hierarchical_predict_map_truncate(fs_sigmoid, tree_loss_module, targets_per_level, level_order, global_index_ranges,
                                      device, threshold, sentinel=-1):
    """Main prediction pipeline."""
    S = tree_loss_module.stateSpace_unweighted
    marginals = tree_loss_module.compute_true_marginals(fs_sigmoid, device)
    level_nodes_list = {lvl: list(range(*global_index_ranges[lvl])) for lvl in level_order}

    map_out = map_decode_paths(fs_sigmoid, S, level_order, level_nodes_list)
    map_paths = map_out["path_nodes_global"]

    zero_mask = (S[map_out["best_state_idx"]].sum(dim=1) == 0)
    if zero_mask.any():
        fallback = per_level_marginal_argmax(marginals, level_order, level_nodes_list)
        for lvl in level_order:
            map_paths[lvl][zero_mask] = fallback[lvl][zero_mask]

    map_trunc = map_truncate_with_threshold(marginals, map_paths, targets_per_level, level_order, global_index_ranges,
                                            threshold, sentinel)
    return {"marginals": marginals, "map_path_nodes_global": map_paths, **map_trunc}


def compute_hierarchical_accuracy_with_inference(
        combined_output,
        pf4_targets, pf3_targets, pf2_targets, pf1_targets, leaf_targets,
        tree_loss, device, config_module, counters, sentinel=-1, **kwargs
):
    """Refactored metrics entry point."""
    level_order = ["pf4", "pf3", "pf2", "pf1", "leaf"]
    targets_dict = {'pf4': pf4_targets, 'pf3': pf3_targets, 'pf2': pf2_targets, 'pf1': pf1_targets,
                    'leaf': leaf_targets}

    results = hierarchical_predict_map_truncate(
        combined_output, tree_loss, targets_dict, level_order, config.GLOBAL_INDEX_RANGES,
        device, config.HIER_CONF_THRESHOLD, sentinel
    )

    map_paths = results["map_path_nodes_global"]
    for i, lvl in enumerate(level_order):
        t = targets_dict[lvl].to(device)
        valid = (t != sentinel)
        if not valid.any(): continue

        preds_global = map_paths[lvl]

        # Robustly handle samples where the MAP path is shorter than the ground truth level
        has_pred = (preds_global >= 0) & valid
        if not has_pred.any(): continue

        preds_local = preds_global[has_pred] - config.GLOBAL_INDEX_RANGES[lvl][0]
        v_targets = t[has_pred].cpu().numpy()
        v_preds = preds_local.cpu().numpy()
        v_confs = combined_output[has_pred, preds_global[has_pred]].cpu().numpy()

        counters[lvl]["total"] += len(v_targets)
        counters[lvl]["trues"].extend(v_targets.tolist())
        counters[lvl]["preds"].extend(v_preds.tolist())
        counters[lvl]["confs"].extend(v_confs.tolist())
        counters[lvl]["correct"] += np.sum(v_preds == v_targets)

        # Related error tracking: predicted shared parent with true?
        if i > 0:
            parent_lvl = level_order[i - 1]
            t_parent = targets_dict[parent_lvl].to(device)
            # Sample must have valid target at both levels
            mask = has_pred & (t_parent != sentinel) & (map_paths[parent_lvl] >= 0)
            if mask.any():
                incorrect = (map_paths[lvl][mask] != (t[mask] + config.GLOBAL_INDEX_RANGES[lvl][0]))
                if incorrect.any():
                    p_pred = map_paths[parent_lvl][mask][incorrect]
                    p_true = t_parent[mask][incorrect] + config.GLOBAL_INDEX_RANGES[parent_lvl][0]
                    related = (p_pred == p_true).cpu().numpy()

                    if "related_hits" not in counters[lvl]:
                        counters[lvl]["related_hits"] = []

                    hits = list(zip(t[mask][incorrect].cpu().numpy().tolist(), related.tolist()))
                    counters[lvl]["related_hits"].extend(hits)

    return counters

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

