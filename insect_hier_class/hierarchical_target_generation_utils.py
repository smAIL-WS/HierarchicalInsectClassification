"""
Utility functions for hierarchical insect classification.
Includes target generation, validation masks, and data augmentation helpers.
"""

import os
import json
import math
import numpy as np
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random

import config


# Cache for loaded JSON files to avoid repeated file I/O
_json_cache = {}


def load_json_cached(filepath):
    """
    Load JSON file with caching to avoid repeated disk reads.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded JSON data (dict)
    """
    filepath_str = str(filepath)
    if filepath_str not in _json_cache:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                _json_cache[filepath_str] = json.load(f)
        else:
            _json_cache[filepath_str] = None
    return _json_cache[filepath_str]


def get_5_level_targets(targets, device, dataset, trees, run_folder,
                        sentinel=-1):
    """
    Generate targets for all 5 hierarchy levels from leaf-level targets.

    Args:
        targets: Tensor of leaf-level target indices
        device: Device to place tensors on
        dataset: Dataset name (for compatibility)
        trees: Hierarchy tree structure
        run_folder: Path to run folder (not used anymore, kept for compatibility)
        sentinel: Value for invalid/missing targets

    Returns:
        Tuple of (pf4_targets, pf3_targets, pf2_targets, pf1_targets, leaf_targets_sig)
        Note: No longer returns weights tensor
    """
    batch_size = targets.size(0)

    # Initialize with sentinel values
    pf4_targets = torch.full((batch_size,), sentinel, dtype=torch.long, device=device)
    pf3_targets = torch.full((batch_size,), sentinel, dtype=torch.long, device=device)
    pf2_targets = torch.full((batch_size,), sentinel, dtype=torch.long, device=device)
    pf1_targets = torch.full((batch_size,), sentinel, dtype=torch.long, device=device)
    leaf_targets_sig = torch.full((batch_size,), sentinel, dtype=torch.long, device=device)

    # Map targets to hierarchy levels
    for i, target in enumerate(targets):
        for path in trees:
            if len(path) > 0 and path[-1] == target.item():
                if len(path) > 0:
                    pf4_targets[i] = path[0] - config.GLOBAL_INDEX_RANGES['pf4'][0]
                if len(path) > 1:
                    pf3_targets[i] = path[1] - config.GLOBAL_INDEX_RANGES['pf3'][0]
                if len(path) > 2:
                    pf2_targets[i] = path[2] - config.GLOBAL_INDEX_RANGES['pf2'][0]
                if len(path) > 3:
                    pf1_targets[i] = path[3] - config.GLOBAL_INDEX_RANGES['pf1'][0]
                if len(path) > 4:
                    leaf_targets_sig[i] = path[4] - config.GLOBAL_INDEX_RANGES['leaf'][0]
                break

    return pf4_targets, pf3_targets, pf2_targets, pf1_targets, leaf_targets_sig



def load_ce_class_weights_from_node_counts(node_counts_file, run_folder, device):
    """
    Load effective (class-balanced) weights for all hierarchy levels from node sample counts.
    Uses the same effective-number formula as tree loss:

        w_c = (1 - beta) / (1 - beta^n_c)

    Weights are normalized to mean 1 per level to keep CE loss magnitudes stable.

    Args:
        node_counts_file: Filename of node counts JSON (e.g., 'node_sample_counts.json')
        run_folder: Path to run folder containing the JSON file
        device: Device to place tensors on

    Returns:
        Dict[level_name -> torch.FloatTensor[num_classes]] or None if file not found
    """
    node_counts_path = run_folder / node_counts_file

    # Load node counts (with caching)
    node_counts_dict = load_json_cached(node_counts_path)
    if node_counts_dict is None:
        print(f"[CE Weights] Warning: Node counts file not found at {node_counts_path}")
        return None

    ce_class_weights = {}
    beta = config.TREE_LOSS_BETA  # use beta from config (already imported in this module)

    # Process each hierarchy level
    for level_name in ['pf4', 'pf3', 'pf2', 'pf1', 'leaf']:
        if level_name not in config.GLOBAL_INDEX_RANGES:
            print(f"[CE Weights] Warning: GLOBAL_INDEX_RANGES missing for level '{level_name}'. Skipping.")
            continue

        start_idx, end_idx = config.GLOBAL_INDEX_RANGES[level_name]
        num_classes = end_idx - start_idx

        # Initialize weights tensor
        weights = torch.ones(num_classes, dtype=torch.float32)

        # Step 1: Compute raw effective weights for this level
        # w_c = (1 - beta) / (1 - beta^n_c); for n_c == 0, use 1.0 (will be normalized later)
        raw_values = []
        for global_idx in range(start_idx, end_idx):
            count = int(node_counts_dict.get(str(global_idx), 0))
            if count > 0:
                denom = 1.0 - (beta ** count)
                if denom <= 0.0:  # numeric safety if beta ~1 and count large
                    denom = 1e-12
                w = (1.0 - beta) / denom
            else:
                w = 1.0
            raw_values.append(w)

        # Step 2: Normalize to mean 1 (keeps loss scale stable across levels)
        mean_w = (sum(raw_values) / len(raw_values)) if raw_values else 1.0
        if mean_w <= 0.0:
            mean_w = 1.0

        # Step 3: Populate tensor with normalized weights
        for local_idx, w in enumerate(raw_values):
            weights[local_idx] = w / mean_w

        ce_class_weights[level_name] = weights.to(device)

    # Optional summary
    if ce_class_weights:
        print("\n[CE Weights] Level-wise effective weights (mean≈1):")
        for lvl, w in ce_class_weights.items():
            print(f"  {lvl}: mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}")

    return ce_class_weights


def init_classifier_biases_from_counts(
    net,
    node_counts_file: str,
    run_folder: Path,
    device: torch.device,
    levels=('pf4', 'pf3', 'pf2', 'pf1', 'leaf'),
    epsilon: float = 1e-8,
    tiny_prior_scale: float = 1e-3,
):
    """
    Initialize classifier biases for multi-class CE heads using log class priors.

    For each level:
      - p_c = count_c / sum_counts
      - bias_c = log(p_c + epsilon)
    Fallbacks:
      - If sum_counts == 0 (no data for level): uniform p_c = 1/C
      - If some counts are zero but not all: assign zero-count classes a tiny prior
        tiny_prior = (1/C) * tiny_prior_scale before normalization.

    Args:
        net: Model instance containing classifier heads:
             net.classifier_pf4, classifier_pf3, classifier_pf2, classifier_pf1, classifier_leaf
        node_counts_file: Filename of node counts JSON
        run_folder: Path to folder containing the JSON file
        device: Torch device
        levels: Iterable of hierarchy levels to process
        epsilon: Small constant to avoid log(0)
        tiny_prior_scale: Scale for tiny prior used for zero-count classes when sum_counts > 0

    Returns:
        None (in-place bias initialization)
    """
    node_counts_path = run_folder / node_counts_file

    counts_dict = load_json_cached(node_counts_path)
    if counts_dict is None:
        print(f"[Bias Init] Warning: Node counts file not found at {node_counts_path}. Skipping bias init.")
        return

    # Map level -> classifier attribute name on `net`
    level_to_attr = {
        'pf4': 'classifier_pf4',
        'pf3': 'classifier_pf3',
        'pf2': 'classifier_pf2',
        'pf1': 'classifier_pf1',
        'leaf': 'classifier_leaf',
    }

    for level_name in levels:
        # Resolve classifier module
        attr = level_to_attr.get(level_name, None)
        if attr is None or not hasattr(net, attr):
            print(f"[Bias Init] Warning: Model has no classifier for level '{level_name}'. Skipping.")
            continue
        classifier = getattr(net, attr)

        # Get global index range for this level
        if level_name not in config.GLOBAL_INDEX_RANGES:
            print(f"[Bias Init] Warning: GLOBAL_INDEX_RANGES missing for level '{level_name}'. Skipping.")
            continue

        start_idx, end_idx = config.GLOBAL_INDEX_RANGES[level_name]
        num_classes = end_idx - start_idx

        # Collect raw counts in per-level local order
        counts = torch.zeros(num_classes, dtype=torch.float32)
        for global_idx in range(start_idx, end_idx):
            c = float(int(counts_dict.get(str(global_idx), 0)))
            local_idx = global_idx - start_idx
            if 0 <= local_idx < num_classes:
                counts[local_idx] = c

        sum_counts = float(counts.sum().item())
        priors = torch.empty_like(counts)

        if sum_counts <= 0.0:
            # No data for this level -> uniform priors
            priors.fill_(1.0 / num_classes)
        else:
            # Assign tiny prior to zero-count classes, then renormalize
            num_classes_f = float(num_classes)
            tiny_prior = (1.0 / num_classes_f) * tiny_prior_scale

            # Start with raw priors
            priors = counts / sum_counts

            # For any zeros, bump to tiny_prior first
            zero_mask = (priors == 0)
            if zero_mask.any():
                priors[zero_mask] = tiny_prior
                # Renormalize to sum 1
                priors = priors / float(priors.sum().item())

        # Compute bias = log(prior + epsilon)
        biases = torch.log(priors + epsilon).to(device)

        # Copy into classifier bias (safely, no grad)
        if getattr(classifier, 'bias', None) is None:
            # Some Linear layers can be created without bias; if so, warn.
            print(f"[Bias Init] Warning: Classifier '{attr}' has no bias parameter. Skipping.")
            continue

        if classifier.bias.data.shape[0] != num_classes:
            print(f"[Bias Init] Warning: Bias length mismatch at '{attr}': "
                  f"{classifier.bias.data.shape[0]} vs {num_classes}. Skipping.")
            continue

        with torch.no_grad():
            classifier.bias.data.copy_(biases)

        print(f"[Bias Init] Level '{level_name}': set biases from priors (sum_counts={sum_counts:.0f}).")



def build_valid_masks_from_targets(targets_dict, sentinel=-1, device='cuda'):
    """
    Build per-level validity masks from targets_dict for a batch.

    Args:
        targets_dict: Dict[str, Tensor], keys among {'pf4','pf3','pf2','pf1','leaf'}
                      Each tensor is shape [B] with integer targets, sentinel for invalid.
        sentinel:     Integer marking invalid targets.
        device:       Torch device.

    Returns:
        masks_dict: Dict[str, BoolTensor], shape [B] per level.
    """
    levels = ['pf4', 'pf3', 'pf2', 'pf1', 'leaf']

    # Get batch size from any available level
    # If none exist, return empty dict
    if not targets_dict:
        return {}

    # Find the first present level to determine B
    first_level = next((lvl for lvl in levels if lvl in targets_dict), None)
    if first_level is None:
        return {}

    B = targets_dict[first_level].size(0)
    masks_dict = {}

    for level in levels:
        if level not in targets_dict:
            masks_dict[level] = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            t = targets_dict[level].to(device)
            masks_dict[level] = (t != sentinel)
    return masks_dict


def compute_hierarchical_ce_loss(
    logits_dict,
    targets_dict,
    ce_class_weights=None,
    level_weight_alpha=None,
    invert=None,
    device='cuda',
    use_focal=False,
    focal_alpha=0.25,
    focal_gamma=2.0,
    sentinel=-1,
    eps=1e-8,
    label_smoothing=None,
):

    """
    Compute combined cross-entropy loss (or optionally Focal loss) across all hierarchy levels with:
      - Per-level validity masking (apply CE only to samples valid at that level)
      - Per-sample normalization of depth weights over the valid levels for that sample
      - Optional class weights per level (ce_class_weights)

    Args:
        logits_dict: Dict[str, Tensor], mapping level name -> raw logits [B, C_level]
        targets_dict: Dict[str, Tensor], mapping level name -> target indices [B], sentinel for invalid samples
        ce_class_weights: Optional Dict[str, Tensor], class weights per level (shape [C_level])
        level_weight_alpha: Float; base alpha for depth weighting (defaults to config.ALPHA_TARGET if available)
        invert: Bool; if True, deeper levels get higher weights; if False, coarser levels get higher weights
        device: Torch device
        use_focal: Bool; if True, apply focal modulation on CE
        focal_alpha: Float; focal alpha
        focal_gamma: Float; focal gamma
        sentinel: Int; invalid target marker in targets_dict[level]
        eps: Float; small constant to avoid div-by-zero

    Returns:
        Scalar tensor: combined loss over batch (averaged across samples)
    """
    # Configure levels and depths (pf4=0 ... leaf=4)
    levels = ['pf4', 'pf3', 'pf2', 'pf1', 'leaf']
    depth_map = {level: i for i, level in enumerate(levels)}

    # Pull defaults from config if present (safe fallback if not)
    try:
        import config  # noqa: F401
        if level_weight_alpha is None:
            level_weight_alpha = getattr(config, 'ALPHA_TARGET', 0.9)
        if invert is None:
            invert = getattr(config, 'ALPHA_LOSS_INVERT', False)
    except Exception:
        if level_weight_alpha is None:
            level_weight_alpha = 0.9
        if invert is None:
            invert = False

    # Batch size consistency check
    # Assumes all levels exist in dicts and share the same batch size
    any_level = next(iter(logits_dict))
    B = logits_dict[any_level].size(0)

    # ----- 1) Build per-level validity masks -----
    valid_masks = {}
    for level in levels:
        if level not in targets_dict:
            # If a level is absent from targets, treat all as invalid for that level
            valid_masks[level] = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            t = targets_dict[level].to(device)
            valid_masks[level] = (t != sentinel)

    # ----- 2) Compute base depth weights (global), then per-sample normalized weights -----
    # Base depth weight per level (scalar per level)
    base_depth_weights = []
    for level in levels:
        d = depth_map[level]
        w = math.exp(level_weight_alpha * d) if invert else math.exp(-level_weight_alpha * d)
        base_depth_weights.append(w)
    base_depth_weights = torch.tensor(base_depth_weights, dtype=torch.float32, device=device)  # [L]

    # Build [B, L] validity matrix and expand base weights across batch
    validity_mat = torch.stack([valid_masks[level] for level in levels], dim=1).float()  # [B, L]
    W = validity_mat * base_depth_weights.unsqueeze(0)                                   # [B, L], zeroed where invalid

    # Row-normalize per sample over valid levels
    row_sums = W.sum(dim=1, keepdim=True)  # [B, 1]
    W_norm = W / (row_sums + eps)          # if a row has zero valid levels, weights stay ~0

    # ----- 3) Compute per-level CE (or focal CE) per sample, then weight by W_norm and aggregate -----
    total_weighted_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

    for li, level in enumerate(levels):
        # Skip level if logits/targets missing
        if level not in logits_dict or level not in targets_dict:
            continue

        logits = logits_dict[level].to(device)  # [B, C_level]
        targets = targets_dict[level].to(device)  # [B]
        valid_mask = valid_masks[level]  # [B]

        # If no valid samples for this level in the batch, skip
        if valid_mask.sum().item() == 0:
            continue

        # Select valid samples only, compute per-sample CE (reduction='none')
        class_w = None
        if ce_class_weights and (level in ce_class_weights) and (ce_class_weights[level] is not None):
            class_w = ce_class_weights[level].to(device)

        ce_per_valid = F.cross_entropy(logits[valid_mask], targets[valid_mask], weight=class_w, reduction='none', label_smoothing=label_smoothing)  # [N_valid]

        if use_focal:
            # Standard focal wrapping: pt = exp(-ce), loss = alpha * (1 - pt)^gamma * ce
            pt = torch.exp(-ce_per_valid)
            ce_per_valid = focal_alpha * (1.0 - pt) ** focal_gamma * ce_per_valid  # [N_valid]

        # Scatter back into a full [B] vector (zeros for invalid)
        ce_per_sample = torch.zeros(B, dtype=torch.float32, device=device)
        ce_per_sample[valid_mask] = ce_per_valid  # [B]

        # Weight this level's per-sample CE by W_norm[:, li] and sum
        level_weighted = ce_per_sample * W_norm[:, li]  # [B]
        total_weighted_loss = total_weighted_loss + level_weighted.sum()

    # ----- 4) Normalize by number of samples that had at least one valid level -----
    # Count samples with ANY valid level in this batch
    samples_with_any_valid = (validity_mat.sum(dim=1) > 0).sum().clamp(min=1)  # prevent div-by-zero
    total_loss = total_weighted_loss / samples_with_any_valid

    return total_loss


def cosine_anneal_schedule(t, nb_epoch, lr):
    """
    Cosine annealing learning rate schedule.
    
    Args:
        t: Current epoch
        nb_epoch: Total number of epochs
        lr: Base learning rate
        
    Returns:
        Adjusted learning rate
    """
    cos_inner = np.pi * (t % nb_epoch)
    cos_inner /= nb_epoch
    cos_out = np.cos(cos_inner) + 1
    return float(lr / 2 * cos_out)


def compute_batch_entropy(logits: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Computes average entropy of predictions across a batch from raw logits.
    Returns NEGATIVE entropy so minimizing the loss MAXIMIZES entropy (encourages diversity).

    Args:
        logits (torch.Tensor): Tensor of shape [batch_size, num_classes] with raw logits.
        epsilon (float): Small constant to avoid log(0) and division by zero.

    Returns:
        torch.Tensor: Negative scalar entropy (minimize this to encourage diverse predictions).
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1)  # [batch_size, num_classes]

    # Clamp to avoid log(0)
    probs_clamped = torch.clamp(probs, epsilon, 1.0)

    # Compute log probabilities
    log_probs = torch.log(probs_clamped)

    # Entropy for each sample: -sum(p * log(p))
    sample_entropies = -torch.sum(probs_clamped * log_probs, dim=1)  # [batch_size]

    # Average across batch and return negative
    return -torch.mean(sample_entropies)
