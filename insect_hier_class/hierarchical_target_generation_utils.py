"""
Utility functions for hierarchical insect classification.
Includes target generation, validation masks, and data augmentation helpers.
"""

import os
import json
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
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


def load_level_weights_from_node_counts(node_counts_file, run_folder, device):
    """
    Load effective weights for all hierarchy levels from node sample counts.
    Uses the same class-balanced weighting formula as tree loss.

    Args:
        node_counts_file: Filename of node counts JSON (e.g., 'node_sample_counts.json')
        run_folder: Path to run folder containing the JSON file
        device: Device to place tensors on

    Returns:
        Dict mapping level name to weight tensor, or None if file not found
    """
    node_counts_path = run_folder / node_counts_file

    # Load node counts (with caching)
    node_counts_dict = load_json_cached(node_counts_path)

    if node_counts_dict is None:
        print(f"Warning: Node counts file not found at {node_counts_path}")
        return None

    level_weights = {}
    beta = config.TREE_LOSS_BETA

    # Process each hierarchy level
    for level_name in ['pf4', 'pf3', 'pf2', 'pf1', 'leaf']:
        start_idx, end_idx = config.GLOBAL_INDEX_RANGES[level_name]
        num_classes = end_idx - start_idx

        # Initialize weights tensor
        weights = torch.ones(num_classes, dtype=torch.float32)

        # Step 1: Compute effective weights for this level
        effective_weights = {}
        for global_idx in range(start_idx, end_idx):
            count = int(node_counts_dict.get(str(global_idx), 0))
            if count > 0:
                # Effective number of samples: (1 - beta) / (1 - beta^n)
                weight = (1 - beta) / (1 - beta ** count)
            else:
                weight = 1.0
            effective_weights[global_idx] = weight

        # Step 2: Normalize to mean 1
        if effective_weights:
            mean_weight = sum(effective_weights.values()) / len(effective_weights)
        else:
            mean_weight = 1.0

        # Step 3: Populate tensor with normalized weights
        for global_idx, weight in effective_weights.items():
            local_idx = global_idx - start_idx
            if 0 <= local_idx < num_classes:
                weights[local_idx] = weight / mean_weight

        level_weights[level_name] = weights.to(device)

    return level_weights


def get_valid_hierarchical_mask(pf4_targets, pf3_targets, pf2_targets, pf1_targets, 
                                leaf_targets_sig, sentinel=-1):
    """
    Returns a mask indicating which samples have valid targets across all hierarchy levels.
    
    Args:
        pf4_targets: Targets for PF4 level
        pf3_targets: Targets for PF3 level
        pf2_targets: Targets for PF2 level
        pf1_targets: Targets for PF1 level
        leaf_targets_sig: Targets for leaf level
        sentinel: Sentinel value indicating invalid targets
        
    Returns:
        Boolean tensor mask of shape (batch_size,) where True means all levels are valid
    """
    return (
        (pf4_targets != sentinel) &
        (pf3_targets != sentinel) &
        (pf2_targets != sentinel) &
        (pf1_targets != sentinel) &
        (leaf_targets_sig != sentinel)
    )


def compute_hierarchical_ce_loss(logits_dict, targets_dict, valid_mask,
                                 level_weights=None, device='cuda'):
    """
    Compute combined cross-entropy loss across all hierarchy levels.

    Args:
        logits_dict: Dict mapping level name to raw logits tensor
                    {'pf4': tensor [batch, num_classes], 'pf3': tensor, ...}
        targets_dict: Dict mapping level name to target indices
                     {'pf4': tensor [batch], 'pf3': tensor, ...}
        valid_mask: Boolean mask [batch] indicating valid samples
        level_weights: Dict mapping level name to weight tensor (optional)
        device: Device for computation

    Returns:
        Combined cross-entropy loss (scalar tensor)
    """
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    num_valid_levels = 0

    for level_name in ['pf4', 'pf3', 'pf2', 'pf1', 'leaf']:
        if level_name not in logits_dict or level_name not in targets_dict:
            continue

        logits = logits_dict[level_name]
        targets = targets_dict[level_name]

        # Apply valid mask
        if valid_mask.sum() == 0:
            continue

        selected_logits = logits[valid_mask]
        selected_targets = targets[valid_mask]

        # Create loss function with optional weights
        if level_weights and level_name in level_weights:
            ce_loss_fn = torch.nn.CrossEntropyLoss(
                weight=level_weights[level_name].to(torch.float32)
            )
        else:
            ce_loss_fn = torch.nn.CrossEntropyLoss()

        # Compute loss for this level
        level_loss = ce_loss_fn(selected_logits, selected_targets)
        total_loss += level_loss
        num_valid_levels += 1

    # Average across levels (more stable than sum)
    if num_valid_levels > 0:
        total_loss = total_loss / num_valid_levels

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
