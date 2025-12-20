
"""
Tree loss implementation for hierarchical classification.
Uses conditional probability framework with class-balanced weighting.
"""

import os
import json
import torch
import torch.nn as nn
import math

import config


class TreeLoss(nn.Module):
    """
    Tree-based loss for hierarchical classification using conditional probabilities.
    Supports effective class-balanced weighting.
    """
    
    def __init__(
        self,
        hierarchy,
        device,
        run_folder,
        alpha=None,
        invert=None,
        sample_count_file=None,
        beta=None
    ):
        """
        Args:
            hierarchy: List of hierarchy paths (e.g., [[1], [1, 8], [1, 5, 15], ...])
            device: Device to place tensors on
            run_folder: Path to folder containing sample count files
            alpha: Depth weighting parameter (default from config)
            invert: Whether to invert depth weighting (default from config)
            sample_count_file: Sample count JSON filename (default from config)
            beta: Effective sample weighting parameter (default from config)
        """
        super(TreeLoss, self).__init__()
        
        # Use config defaults if not specified
        alpha = alpha if alpha is not None else config.TREE_LOSS_ALPHA # NO LONGER USED
        invert = invert if invert is not None else config.ALPHA_LOSS_INVERT
        sample_count_file = sample_count_file if sample_count_file is not None else config.NODE_SAMPLE_COUNTS_FILE
        beta = beta if beta is not None else config.TREE_LOSS_BETA

        self.hierarchy = hierarchy
        self.total_nodes = max(max(path) for path in hierarchy) + 1
        self.device = device
        self.alpha = alpha
        self.invert = invert
        self.beta = beta
        self.run_folder = run_folder

        # --- Cache precomputed structures ---
        # hierarchy remains a list; these are new attributes
        self.level_weights: dict[int, float] = self.compute_level_weights(self.hierarchy)

        # Precompute state spaces for training and inference
        self.stateSpace_weighted: torch.Tensor = self.generateStateSpace(self.hierarchy, weighted=True).to(device)
        self.stateSpace_unweighted: torch.Tensor = self.generateStateSpace(self.hierarchy, weighted=False).float().to(device)

        self.mask = (self.stateSpace_unweighted > 0).float()  # [num_states, num_nodes]
        self.state_counts = torch.sum(self.mask, dim=0)  # [num_nodes]
        self.num_nodes = self.stateSpace_unweighted.shape[1]
        self.num_states = self.stateSpace_unweighted.shape[0]

        # Generate state space
        self.stateSpace = self.generateStateSpace(self.hierarchy, weighted=True).to(device)
            # hierarchy,
            # alpha=self.alpha,
            # invert=self.invert
        # ).to(device)
        self.has_printed = False
        
        # Load sample counts and compute effective weights
        sample_count_path = run_folder / sample_count_file
        self.sample_counts = self._load_sample_counts(sample_count_path)
        
        if self.sample_counts is None:
            print("Warning: Sample counts not loaded. Effective weighting will not be applied.")
            self.effective_weights = None
        else:
            print(f"Loaded sample counts for {len(self.sample_counts)} nodes.")
            self.effective_weights = self._compute_effective_weights(self.sample_counts, self.beta)
            self._print_effective_weights()
    
    def _load_sample_counts(self, filepath):
        """Load sample counts from JSON file."""
        if not os.path.exists(filepath):
            print(f"Warning: Sample count file not found at {filepath}")
            return None
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _compute_effective_weights(self, sample_counts, beta):
        """
        Compute effective class-balanced weights using the formula:
        weight = (1 - beta) / (1 - beta^n)
        where n is the number of samples for that node.
        
        Reference: Cui et al. "Class-Balanced Loss Based on Effective Number of Samples"
        """
        if sample_counts is None:
            return None
        
        weights = {}
        for idx_str, count in sample_counts.items():
            count = int(count)
            if count > 0:
                weight = (1 - beta) / (1 - beta ** count)
            else:
                weight = 1.0
            weights[int(idx_str)] = weight
        
        # Normalize to mean 1
        mean_weight = sum(weights.values()) / len(weights) if weights else 1.0
        normalized_weights = {
            idx: weight / mean_weight
            for idx, weight in weights.items()
        }
        
        return normalized_weights
    
    def _print_effective_weights(self):
        """Print effective weights for debugging."""
        if self.effective_weights is not None:
            print("\n--- Effective Class Weights ---")
            for idx, weight in sorted(self.effective_weights.items()):
                print(f"  Node {idx:3d}: Weight = {weight:.6f}")
            print()
    
    def forward(self, fs, labels, device):
        """
        Compute tree loss.
        
        Args:
            fs: Model outputs (logits) concatenated across all levels
            labels: Ground truth leaf labels
            device: Device to place tensors on
            
        Returns:
            Scalar loss value
        """
        # Ensure tensors are on correct device and contiguous
        stateSpace = self.stateSpace_weighted.contiguous()
        fs = fs.to(device).contiguous()
        labels = labels.to(device)
        
        # Check for anomalies (NaN/Inf)
        if torch.isnan(fs).any() or torch.isinf(fs).any():
            print("Warning: Detected NaN or Inf in model outputs!")
            print(f"  NaNs: {torch.isnan(fs).sum().item()}")
            print(f"  Infs: {torch.isinf(fs).sum().item()}")
        
        # Compute state-space activations
        index = torch.mm(stateSpace, fs.T)  # [num_states, batch_size]

        # Print state space once per training for debugging
        # Print index once per epoch for visualization
        if not self.has_printed:
            torch.set_printoptions(threshold=float('inf'), linewidth=200, precision=1)
            print("StateSpace:\n", stateSpace)
            # print("Joint probabilities:\n", torch.exp(index))
            self.has_printed = True
        
        # Log-Sum-Exp trick for numerical stability
        max_vals = torch.max(index, dim=0, keepdim=True)[0]  # [1, batch_size]
        stable_index = index - max_vals
        joint_unweighted = torch.exp(stable_index)  # [num_states, batch_size]
        
        # Apply effective weights if available
        if self.effective_weights is not None:
            weight_tensor = torch.ones(joint_unweighted.shape[0], device=device)
            for idx, weight in self.effective_weights.items():
                # Note: stateSpace has an extra row at index 0, so we shift by 1
                if idx + 1 < weight_tensor.shape[0]:
                    weight_tensor[idx + 1] = weight
            joint = joint_unweighted * weight_tensor.unsqueeze(1)
        else:
            joint = joint_unweighted
        
        # Partition function (normalization)
        z = torch.sum(joint, dim=0)  # [batch_size]
        log_z = torch.log(z + 1e-10) + max_vals.squeeze(0)  # Add epsilon for stability
        
        # Compute marginal probabilities and loss for each sample
        loss = torch.zeros(fs.shape[0], dtype=torch.float32, device=device)
        for i in range(len(labels)):
            label = labels[i].item()
            if label < 0 or label >= stateSpace.shape[1]:
                continue  # Skip invalid labels
            
            # Find states where this label is present
            label_mask = stateSpace[:, label] > 0
            marginal = torch.sum(joint[:, i][label_mask])
            log_marginal = torch.log(marginal + 1e-10) + max_vals[0, i]
            loss[i] = -(log_marginal - log_z[i])
        
        return torch.mean(loss)

    def inference(self, fs, device):
        """
        Compute marginal probabilities for all nodes (vectorized, optimized).

        Args:
            fs: Model outputs [batch_size, num_nodes]
            device: Device

        Returns:
            Marginal probability matrix [batch_size, num_nodes]
        """
        with torch.no_grad():
            stateSpace = self.stateSpace_unweighted.to(device)  # [num_states, num_nodes]
            fs = fs.to(device)  # [batch_size, num_nodes]

            # Compute joint probabilities
            index = torch.mm(stateSpace, fs.T)  # [num_states, batch_size]
            joint = torch.exp(index)  # [num_states, batch_size]
            z = torch.sum(joint, dim=0)  # [batch_size]

            # Mask for active states per node
            mask = (stateSpace > 0).float()  # [num_states, num_nodes]
            state_counts = torch.sum(mask, dim=0)  # [num_nodes]

            # Compute raw sums for all nodes and batches using einsum
            # 'sn,sb->nb' means: sum over states (s), multiply mask(sn) with joint(sb)
            raw_sum = torch.einsum('sn,sb->nb', mask, joint)  # [num_nodes, batch_size]

            # Normalize by state_counts and z (broadcasted)
            raw_sum /= (state_counts.unsqueeze(1) + 1e-10)  # [num_nodes, batch_size]
            raw_sum /= z.unsqueeze(0)  # [num_nodes, batch_size]

            # Transpose to [batch_size, num_nodes]
            pMargin = raw_sum.T.contiguous()

            return pMargin

    # def inference(self, fs, device):
    #     """
    #     Compute marginal probabilities for all nodes (for inference).
    #
    #     Args:
    #         fs: Model outputs
    #         device: Device
    #
    #     Returns:
    #         Marginal probability matrix [batch_size, num_nodes]
    #     """
    #     with torch.no_grad():
    #         stateSpace = self.stateSpace_unweighted
    #         fs = fs.to(device)
    #
    #         index = torch.mm(stateSpace, fs.T)
    #         joint = torch.exp(index)
    #         z = torch.sum(joint, dim=0)
    #
    #         # pMargin = torch.zeros((fs.shape[0], fs.shape[1]), dtype=torch.float32, device=device)
    #         # for i in range(fs.shape[0]):
    #         #     for j in range(fs.shape[1]):
    #         #         mask_indices = torch.where(stateSpace[:, j] > 0)[0]
    #         #         if len(mask_indices) > 0:
    #         #             pMargin[i, j] = torch.sum(joint[mask_indices, i]) / z[i]
    #
    #         pMargin = torch.zeros((fs.shape[0], fs.shape[1]), dtype=torch.float32, device=device)
    #         state_counts = torch.sum(stateSpace > 0, dim=0)  # [num_nodes], count of states per node
    #
    #         for i in range(fs.shape[0]):
    #             for j in range(fs.shape[1]):
    #                 mask_indices = torch.where(stateSpace[:, j] > 0)[0]
    #                 if len(mask_indices) > 0:
    #                     raw_sum = torch.sum(joint[mask_indices, i])
    #                     normalized_sum = raw_sum / (state_counts[j] + 1e-10)  # avoid div by zero
    #                     pMargin[i, j] = normalized_sum / z[i]
    #
    #         return pMargin

    def compute_level_weights(self, hierarchy):
        """
        Compute linear normalized weights for nodes in a hierarchical structure.
        - Default: Root(s) = 1.0, deepest leaves = config.HIERARCHY_WEIGHT,
          intermediate nodes linearly interpolated along paths.
        - Singleton-branch override: If a branch has only a single node (root with no children),
          that node gets config.HIERARCHY_WEIGHT instead of 1.0.

        Args:
            hierarchy (list[list[int]]): List of paths representing hierarchy,
                e.g. [[rootA], [rootA, c1], [rootA, c1, c2], [rootB], ...]

        Returns:
            dict[int, float]: Mapping node_id -> weight
        """
        HIERARCHY_WEIGHT = config.HIERARCHY_WEIGHT

        if not hierarchy:
            return {}

        # --- Precompute parent (has-children) information ---
        # A node is a parent if it appears at any position i < len(path)-1 on any path.
        parent_nodes = set()
        for path in hierarchy:
            for i in range(len(path) - 1):
                parent_nodes.add(path[i])

        # Roots are the first element of each path (can be multiple roots in a forest)
        roots = {path[0] for path in hierarchy if path}

        # Singleton roots are roots that have NO children anywhere (i.e., never appear as parent)
        # and whose path appears as length==1 in the input.
        path_len_by_root = {path[0]: len(path) for path in hierarchy if path}
        singleton_roots = {r for r in roots if (r not in parent_nodes) and (path_len_by_root.get(r, 0) == 1)}

        # --- Find global maximum depth (length of the longest path) ---
        max_depth = max(len(path) for path in hierarchy)

        # Special case: global depth == 1 (the whole hierarchy is just single-node roots)
        # All such singletons should get HIERARCHY_WEIGHT.
        if max_depth == 1:
            # There could be multiple single roots; assign all to HIERARCHY_WEIGHT
            weights = {}
            for path in hierarchy:
                node = path[0]
                weights[node] = HIERARCHY_WEIGHT
            return weights

        # Otherwise, compute the global step for linear interpolation along the longest branch.
        global_step = (HIERARCHY_WEIGHT - 1.0) / (max_depth - 1)
        print(f"Longest path length: {max_depth}")
        print(f"Global step: {global_step}")

        # Sort paths by length (longest first) so we seed using the deepest branch
        sorted_paths = sorted(hierarchy, key=len, reverse=True)

        weights = {}

        # --- Assign default weights for roots ---
        # Default: 1.0, EXCEPT singleton roots -> HIERARCHY_WEIGHT
        for root in roots:
            if root in singleton_roots:
                weights[root] = HIERARCHY_WEIGHT
            else:
                weights[root] = 1.0

        # --- Assign weights along the longest branch first (respect existing assignments) ---
        longest_branch = sorted_paths[0]
        current_weight = 1.0
        for i, node in enumerate(longest_branch):
            if i == 0:
                # Root of the longest branch: ensure it already has either 1.0 or HIERARCHY_WEIGHT (singleton case)
                # Do not overwrite an existing singleton override.
                if node not in weights:
                    weights[node] = 1.0
            elif i == len(longest_branch) - 1:
                weights[node] = HIERARCHY_WEIGHT
            else:
                current_weight += global_step
                # Only set if not already assigned (avoid overwriting)
                if node not in weights:
                    weights[node] = current_weight

        # --- Process the remaining branches ---
        for path in sorted_paths[1:]:
            # Find deepest ancestor on this path that already has an assigned weight
            ancestor_index = -1
            for i, node in enumerate(path):
                if node in weights:
                    ancestor_index = i
                else:
                    break

            if ancestor_index == -1:
                # No ancestor with weight found; start from root
                ancestor_index = 0
                ancestor_node = path[ancestor_index]
                ancestor_weight = weights.get(ancestor_node,
                                              HIERARCHY_WEIGHT if ancestor_node in singleton_roots else 1.0)
                # ensure the root has a weight set (respect singleton override)
                weights[ancestor_node] = ancestor_weight
            else:
                ancestor_weight = weights[path[ancestor_index]]

            remaining_nodes = len(path) - ancestor_index - 1
            if remaining_nodes <= 0:
                # Entire path is already weighted
                continue

            # Interpolate linearly from ancestor_weight to HIERARCHY_WEIGHT across remaining_nodes
            local_step = (HIERARCHY_WEIGHT - ancestor_weight) / remaining_nodes
            current_weight = ancestor_weight

            for j, node in enumerate(path[ancestor_index + 1:], start=1):
                if j == remaining_nodes:
                    # Last node on this path is a leaf -> force to HIERARCHY_WEIGHT
                    weights[node] = HIERARCHY_WEIGHT
                else:
                    current_weight += local_step
                    # Only set if not already assigned (avoid overwriting existing decisions)
                    if node not in weights:
                        weights[node] = current_weight

        return weights

    def generateStateSpace(self, hierarchy, weighted=True):
        """
        Generate state space matrix with depth-based linear normalized weighting.
        Uses compute_level_weights() to assign weights.

        Args:
            hierarchy (list): List of hierarchy paths.
            weighted (bool): If True, apply hierarchical weights (for training).
                            If False, use unweighted state space (for inference).

        Returns:
            torch.Tensor: State space matrix [num_states + 1, num_nodes]
        """
        import torch

        # Compute weights for all nodes
        if weighted:
            node_weights = self.compute_level_weights(hierarchy)
            # After node_weights is computed in generateStateSpace
            all_nodes = {node for path in hierarchy for node in path}
            missing_nodes = all_nodes - set(node_weights.keys())

            if missing_nodes:
                print("\n[Warning] Missing nodes in hierarchical weights:")
                print(f"Total missing: {len(missing_nodes)}")
                print(f"Nodes: {sorted(missing_nodes)}")
            else:
                print("\n[Info] All hierarchy nodes have assigned weights.")
        else:
            # Unweighted: all active nodes get weight = 1.0
            node_weights = {node: 1.0 for path in hierarchy for node in path}

        total_nodes = max(max(path) for path in hierarchy) + 1
        stateSpace = torch.zeros(total_nodes + 1, total_nodes, dtype=torch.float32)
        recorded = torch.zeros(total_nodes, dtype=torch.bool)

        i = 1  # State counter (row 0 is kept as all zeros)

        for path in hierarchy:
            if len(path) == 0:
                continue

            classification = path[-1]
            parents = path[:-1]

            # Handle single-level path (no parents)
            if len(parents) == 0:
                if not recorded[classification]:
                    stateSpace[i, classification] = node_weights[classification]
                    recorded[classification] = True
                    i += 1
                continue

            # Record intermediate states for each parent
            for d in range(len(parents)):
                node = parents[d]
                if not recorded[node]:
                    for j in range(d + 1):
                        parent_node = parents[j]
                        stateSpace[i, parent_node] = node_weights[parent_node]
                    recorded[node] = True
                    i += 1

            # Final state: all parents + classification
            if not recorded[classification]:
                for parent_node in parents:
                    stateSpace[i, parent_node] = node_weights[parent_node]
                stateSpace[i, classification] = node_weights[classification]
                recorded[classification] = True
                i += 1

        # Validate state space
        expected_states = total_nodes + 1
        if i != expected_states:
            print(f"Warning: State space generation mismatch. Expected {expected_states} states, got {i}")

        return stateSpace

    # def generateStateSpace(self, hierarchy, alpha=0.3, invert=True):
    #     """
    #     Generate state space matrix with depth-based weighting.
    #
    #     Args:
    #         hierarchy: List of hierarchy paths
    #         alpha: Depth weighting parameter
    #         invert: If True, deeper nodes get higher weights
    #
    #     Returns:
    #         State space matrix [num_states + 1, num_nodes]
    #     """
    #     total_nodes = max(max(path) for path in hierarchy) + 1
    #
    #     # Initialize state space and tracking arrays
    #     stateSpace = torch.zeros(total_nodes + 1, total_nodes, dtype=torch.float32)
    #     recorded = torch.zeros(total_nodes, dtype=torch.bool)
    #     node_depths = torch.full((total_nodes,), -1, dtype=torch.int)
    #
    #     # Record depth for each node
    #     for path in hierarchy:
    #         for depth, node in enumerate(path):
    #             if node_depths[node] == -1:
    #                 node_depths[node] = depth
    #
    #     i = 1  # State counter (row 0 is kept as all zeros)
    #
    #     for path in hierarchy:
    #         if len(path) == 0:
    #             continue
    #
    #         classification = path[-1]
    #         parents = path[:-1]
    #
    #         # Handle single-level path (no parents)
    #         if len(parents) == 0:
    #             if not recorded[classification]:
    #                 depth_val = node_depths[classification].item()
    #                 weight = math.exp(alpha * depth_val) if invert else math.exp(-alpha * depth_val)
    #                 stateSpace[i, classification] = weight
    #                 recorded[classification] = True
    #                 i += 1
    #             continue
    #
    #         # Record intermediate states for each parent
    #         for d in range(len(parents)):
    #             node = parents[d]
    #             if not recorded[node]:
    #                 for j in range(d + 1):
    #                     parent_node = parents[j]
    #                     depth_val = node_depths[parent_node].item()
    #                     weight = math.exp(alpha * depth_val) if invert else math.exp(-alpha * depth_val)
    #                     stateSpace[i, parent_node] = weight
    #                 recorded[node] = True
    #                 i += 1
    #
    #         # Final state: all parents + classification
    #         if not recorded[classification]:
    #             for parent_node in parents:
    #                 depth_val = node_depths[parent_node].item()
    #                 weight = math.exp(alpha * depth_val) if invert else math.exp(-alpha * depth_val)
    #                 stateSpace[i, parent_node] = weight
    #
    #             depth_val = node_depths[classification].item()
    #             weight = math.exp(alpha * depth_val) if invert else math.exp(-alpha * depth_val)
    #             stateSpace[i, classification] = weight
    #             recorded[classification] = True
    #             i += 1
    #
    #     # Validate state space
    #     expected_states = total_nodes + 1
    #     if i != expected_states:
    #         print(f"Warning: State space generation mismatch. Expected {expected_states} states, got {i}")
    #
    #     return stateSpace
