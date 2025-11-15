
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
        alpha = alpha if alpha is not None else config.TREE_LOSS_ALPHA
        invert = invert if invert is not None else config.TREE_LOSS_INVERT
        sample_count_file = sample_count_file if sample_count_file is not None else config.NODE_SAMPLE_COUNTS_FILE
        beta = beta if beta is not None else config.TREE_LOSS_BETA
        
        self.total_nodes = max(max(path) for path in hierarchy) + 1
        self.device = device
        self.alpha = alpha
        self.invert = invert
        self.beta = beta
        self.run_folder = run_folder
        
        # Generate state space
        self.stateSpace = self.generateStateSpace(
            hierarchy, 
            alpha=self.alpha, 
            invert=self.invert
        ).to(device)
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
        stateSpace = self.stateSpace.to(device).contiguous()
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
            torch.set_printoptions(threshold=float('inf'), linewidth=200, precision=3)
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
        Compute marginal probabilities for all nodes (for inference).
        
        Args:
            fs: Model outputs
            device: Device
            
        Returns:
            Marginal probability matrix [batch_size, num_nodes]
        """
        with torch.no_grad():
            stateSpace = self.stateSpace.to(device)
            fs = fs.to(device)
            
            index = torch.mm(stateSpace, fs.T)
            joint = torch.exp(index)
            z = torch.sum(joint, dim=0)
            
            pMargin = torch.zeros((fs.shape[0], fs.shape[1]), dtype=torch.float32, device=device)
            for i in range(fs.shape[0]):
                for j in range(fs.shape[1]):
                    mask_indices = torch.where(stateSpace[:, j] > 0)[0]
                    if len(mask_indices) > 0:
                        pMargin[i, j] = torch.sum(joint[mask_indices, i]) / z[i]
            
            return pMargin
    
    def generateStateSpace(self, hierarchy, alpha=0.3, invert=True):
        """
        Generate state space matrix with depth-based weighting.
        
        Args:
            hierarchy: List of hierarchy paths
            alpha: Depth weighting parameter
            invert: If True, deeper nodes get higher weights
            
        Returns:
            State space matrix [num_states + 1, num_nodes]
        """
        total_nodes = max(max(path) for path in hierarchy) + 1
        
        # Initialize state space and tracking arrays
        stateSpace = torch.zeros(total_nodes + 1, total_nodes, dtype=torch.float32)
        recorded = torch.zeros(total_nodes, dtype=torch.bool)
        node_depths = torch.full((total_nodes,), -1, dtype=torch.int)
        
        # Record depth for each node
        for path in hierarchy:
            for depth, node in enumerate(path):
                if node_depths[node] == -1:
                    node_depths[node] = depth
        
        i = 1  # State counter (row 0 is kept as all zeros)
        
        for path in hierarchy:
            if len(path) == 0:
                continue
            
            classification = path[-1]
            parents = path[:-1]
            
            # Handle single-level path (no parents)
            if len(parents) == 0:
                if not recorded[classification]:
                    depth_val = node_depths[classification].item()
                    weight = math.exp(alpha * depth_val) if invert else math.exp(-alpha * depth_val)
                    stateSpace[i, classification] = weight
                    recorded[classification] = True
                    i += 1
                continue
            
            # Record intermediate states for each parent
            for d in range(len(parents)):
                node = parents[d]
                if not recorded[node]:
                    for j in range(d + 1):
                        parent_node = parents[j]
                        depth_val = node_depths[parent_node].item()
                        weight = math.exp(alpha * depth_val) if invert else math.exp(-alpha * depth_val)
                        stateSpace[i, parent_node] = weight
                    recorded[node] = True
                    i += 1
            
            # Final state: all parents + classification
            if not recorded[classification]:
                for parent_node in parents:
                    depth_val = node_depths[parent_node].item()
                    weight = math.exp(alpha * depth_val) if invert else math.exp(-alpha * depth_val)
                    stateSpace[i, parent_node] = weight
                
                depth_val = node_depths[classification].item()
                weight = math.exp(alpha * depth_val) if invert else math.exp(-alpha * depth_val)
                stateSpace[i, classification] = weight
                recorded[classification] = True
                i += 1
        
        # Validate state space
        expected_states = total_nodes + 1
        if i != expected_states:
            print(f"Warning: State space generation mismatch. Expected {expected_states} states, got {i}")
        
        return stateSpace
