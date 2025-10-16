# Adjust block size (towards end of script) depending on resolution of the images being used. (jigsaw_generator not currently used)/

import os
import json
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import random
import networkx as nx

import torch

# Define number of classes per level
num_classes_per_level = {
    'pf4': 2,
    'pf3': 11,
    'pf2': 13,
    'pf1': 15,
    'leaf': 22
}

# Automatically compute global index ranges
global_index_ranges = {}
start = 0
for level, count in num_classes_per_level.items():
    global_index_ranges[level] = (start, start + count)
    start += count

def load_json_from_script_dir(filename):
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, filename)
    with open(path, 'r') as f:
        return json.load(f)

def get_5_level_targets(targets, device, dataset, trees, leaf_class_weights_file=None, sentinel=-1):
    batch_size = targets.size(0)

    # Initialize with sentinel values
    pf4_targets = torch.full((batch_size,), sentinel, dtype=torch.long, device=device)
    pf3_targets = torch.full((batch_size,), sentinel, dtype=torch.long, device=device)
    pf2_targets = torch.full((batch_size,), sentinel, dtype=torch.long, device=device)
    pf1_targets = torch.full((batch_size,), sentinel, dtype=torch.long, device=device)
    leaf_targets_sig = torch.full((batch_size,), sentinel, dtype=torch.long, device=device)

    for i, target in enumerate(targets):
        for path in trees:
            if len(path) > 0 and path[-1] == target.item():
                if len(path) > 0:
                    pf4_targets[i] = path[0] - global_index_ranges['pf4'][0]
                if len(path) > 1:
                    pf3_targets[i] = path[1] - global_index_ranges['pf3'][0]
                if len(path) > 2:
                    pf2_targets[i] = path[2] - global_index_ranges['pf2'][0]
                if len(path) > 3:
                    pf1_targets[i] = path[3] - global_index_ranges['pf1'][0]
                if len(path) > 4:
                    leaf_targets_sig[i] = path[4] - global_index_ranges['leaf'][0]
                break

    # Load and prepare leaf class weights
    leaf_class_weights_tensor = None
    if leaf_class_weights_file:
        weights_dict = load_json_from_script_dir(leaf_class_weights_file)
        num_leaf_classes = global_index_ranges['leaf'][1] - global_index_ranges['leaf'][0]
        leaf_class_weights_tensor = torch.ones(num_leaf_classes, dtype=torch.float32, device=device)
        for idx_str, weight in weights_dict.items():
            global_idx = int(idx_str)
            local_idx = global_idx - global_index_ranges['leaf'][0]
            if 0 <= local_idx < num_leaf_classes:
                leaf_class_weights_tensor[local_idx] = weight

    return pf4_targets, pf3_targets, pf2_targets, pf1_targets, leaf_targets_sig, leaf_class_weights_tensor

def get_valid_hierarchical_mask(pf4_targets, pf3_targets, pf2_targets, pf1_targets, leaf_targets_sig, sentinel=-1):
    """
    Returns a mask indicating which samples have valid targets across all hierarchy levels.

    Args:
        pf4_targets (Tensor): Targets for PF4 level.
        pf3_targets (Tensor): Targets for PF3 level.
        pf2_targets (Tensor): Targets for PF2 level.
        pf1_targets (Tensor): Targets for PF1 level.
        leaf_targets_sig (Tensor): Targets for leaf level.
        sentinel (int): Sentinel value indicating invalid targets.

    Returns:
        Tensor: Boolean mask of shape (batch_size,) where True means all levels are valid.
    """
    return (
        (pf4_targets != sentinel) &
        (pf3_targets != sentinel) &
        (pf2_targets != sentinel) &
        (pf1_targets != sentinel) &
        (leaf_targets_sig != sentinel)
    )

# cosine anneal schedule for the learning rate
def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

# jigsaw_generator is not currently used in any script, although it is defined here. Therefore the block_size is only
# relevant if it is called by another script.
def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws
