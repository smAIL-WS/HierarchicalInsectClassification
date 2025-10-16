# Updated for 5 levels of state space.

import os
import json
import torch.nn as nn
import torch
import math

def load_node_weights(filename='node_weights_06Oct25.json'):
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, filename)
    with open(path, 'r') as f:
        return json.load(f)

def load_node_sample_counts(filename='node_sample_counts_13Oct25.json'):
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, filename)
    if not os.path.exists(path):
        print(f"Warning: Sample count file '{filename}' not found. Proceeding without weighting.")
        return None
    with open(path, 'r') as f:
        return json.load(f)

class TreeLoss(nn.Module):
    def __init__(
        self,
        hierarchy,
        device,
        alpha=0.3,
        invert=True,
        weight_file='node_weights_06Oct25.json',
        sample_count_file='node_sample_counts_13Oct25.json',
        beta=0.9999
    ):
        super(TreeLoss, self).__init__()
        self.total_nodes = max(max(path) for path in hierarchy) + 1
        self.device = device
        self.alpha = alpha
        self.invert = invert
        self.beta = beta

        self.stateSpace = self.generateStateSpace(hierarchy, alpha=self.alpha, invert=self.invert).to(device)
        self.has_printed = False

        # Load node weights (legacy or fallback)
        self.node_weights = load_node_weights(weight_file)

        # Load sample counts and compute effective weights
        self.sample_counts = self.load_node_sample_counts(sample_count_file)
        if self.sample_counts is None:
            print("Sample counts not loaded — effective weighting will not be applied.")
        else:
            print(f"Loaded sample counts for {len(self.sample_counts)} nodes.")
        self.effective_weights = self.compute_effective_weights(self.sample_counts, self.beta)

        # Debug: print effective weights
        if self.effective_weights is not None:
            print("\n--- Effective Weights ---")
            for idx, weight in sorted(self.effective_weights.items()):
                print(f"Index {idx}: Weight = {weight:.6f}")

    def load_node_sample_counts(self, filename):
        script_dir = os.path.dirname(__file__)
        path = os.path.join(script_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: Sample count file '{filename}' not found. Proceeding without effective weighting.")
            return None
        with open(path, 'r') as f:
            return json.load(f)

    def compute_effective_weights(self, sample_counts, beta):
        if sample_counts is None:
            return None

        weights = {}
        for idx_str, count in sample_counts.items():
            count = int(count)
            weight = (1 - beta) / (1 - beta ** count)
            weights[int(idx_str)] = weight

        # Normalize to mean 1
        mean_weight = sum(weights.values()) / len(weights)
        normalized_weights = {
            idx: weight / mean_weight
            for idx, weight in weights.items()
        }

        return normalized_weights

    def forward(self, fs, labels, device):
        # Ensure tensors are on GPU and contiguous
        stateSpace = self.stateSpace.to(device).contiguous()
        fs_T = fs.T.to(device).contiguous()
        labels = labels.to(device)

        # Disable tensor truncation
        torch.set_printoptions(threshold=float('inf'), linewidth=200, precision=3)

        # Print tensors for debugging
        # Print stateSpace only once
        if not self.has_printed:
            print("State Space:\n", stateSpace)
            self.has_printed = True

        # print("fs.T (Transposed Model Output):\n", fs_T)
        # print("Labels:\n", labels)

        # Check for anomalies
        has_nan = torch.isnan(fs_T).any().item()
        has_inf = torch.isinf(fs_T).any().item()

        if has_nan or has_inf:
            print("Debug Info: Anomaly detected in input tensors.")
            print("stateSpace device:", stateSpace.device)
            print("fs device:", fs_T.device)
            print("stateSpace shape:", stateSpace.shape)
            print("fs.T shape:", fs_T.shape)
            print("Any NaNs in fs:", has_nan)
            print("Any Infs in fs:", has_inf)
            print("stateSpace dtype:", stateSpace.dtype)
            print("fs dtype:", fs_T.dtype)

        index = torch.mm(self.stateSpace, fs.T)

        # Log-Sum-Exp trick for numerical stability
        max_vals = torch.max(index, dim=0, keepdim=True)[0]  # shape: [1, batch_size]
        stable_index = index - max_vals
        joint_unweighted = torch.exp(stable_index)  # shape: [num_states, batch_size]

        # Apply effective weights if available
        if self.effective_weights is not None:
            weight_tensor = torch.ones(joint_unweighted.shape[0], device=device)
            for idx, weight in self.effective_weights.items():
                weight_tensor[idx + 1] = weight  # shift weights by 1 to skip the zero row
            joint_weighted = joint_unweighted * weight_tensor.unsqueeze(1)  # broadcast across batch
            joint = joint_weighted
        else:
            joint = joint_unweighted

        # Partition function (z) in log-space
        z = torch.sum(joint, dim=0)  # shape: [batch_size]
        log_z = torch.log(z) + max_vals.squeeze(0)  # shape: [batch_size]

        # Optional: Debug prints
        # print("\n--- Joint Probabilities (Weighted or Unweighted) ---")
        # print(joint)
        # print("\n--- Partition Function (z) ---")
        # print(z)
        # print("\n--- Log Partition Function (log_z) ---")
        # print(log_z)

        # Compute marginal probabilities for each sample
        loss = torch.zeros(fs.shape[0], dtype=torch.float64).to(device)
        for i in range(len(labels)):
            # Find indices in stateSpace where label is present
            label_mask = self.stateSpace[:, labels[i]] > 0
            marginal = torch.sum(joint[:, i][label_mask])
            log_marginal = torch.log(marginal) + max_vals[0, i]
            loss[i] = -(log_marginal - log_z[i])

        return torch.mean(loss)

    def inference(self, fs, device):
        with torch.no_grad():
            index = torch.mm(self.stateSpace, fs.T)
            joint = torch.exp(index)
            z = torch.sum(joint, dim=0)
            pMargin = torch.zeros((fs.shape[0], fs.shape[1]), dtype=torch.float64).to(device)
            for i in range(fs.shape[0]):
                for j in range(fs.shape[1]):
                    pMargin[i, j] = torch.sum(torch.index_select(joint[:, i], 0, torch.where(self.stateSpace[:, j] > 0)[0]))
            return pMargin

    # Modified generateStateSpace function with flexible depth-based weighting
    def generateStateSpace(self, hierarchy, alpha=0.3, invert=True): # invert = True so weighting favours leaf layers
        # Dynamically determine total number of nodes
        total_nodes = max(max(path) for path in hierarchy) + 1

        # Initialize stateSpace and recorded tensors
        stateSpace = torch.zeros(total_nodes + 1, total_nodes, dtype=torch.float32)
        recorded = torch.zeros(total_nodes)
        node_depths = torch.full((total_nodes,), -1)  # -1 means uninitialized
        i = 1

        for path in hierarchy:
            for depth, node in enumerate(path):
                if node_depths[node] == -1:
                    node_depths[node] = depth

            root = path[0]
            classification = path[-1]
            parents = path[:-1]

            # Handle single-level path (no parents)
            if len(parents) == 0:
                if recorded[classification] == 0:
                    recorded[classification] = 1
                depth_val = node_depths[classification].item()
                weight = math.exp(alpha * depth_val) if invert else math.exp(-alpha * depth_val)
                stateSpace[i, classification] = weight
                i += 1
                continue

            # Record intermediate states for each parent node
            for d in range(len(parents)):
                node = parents[d]
                if recorded[node] == 0:
                    for j in range(d + 1):
                        depth_val = node_depths[parents[j]].item()
                        weight = math.exp(alpha * depth_val) if invert else math.exp(-alpha * depth_val)
                        stateSpace[i, parents[j]] = weight
                    recorded[node] = 1
                    i += 1

            # Final row: all parents + classification
            if recorded[classification] == 0:
                for node in parents:
                    depth_val = node_depths[node].item()
                    weight = math.exp(alpha * depth_val) if invert else math.exp(-alpha * depth_val)
                    stateSpace[i, node] = weight
                depth_val = node_depths[classification].item()
                weight = math.exp(alpha * depth_val) if invert else math.exp(-alpha * depth_val)
                stateSpace[i, classification] = weight
                recorded[classification] = 1
                i += 1

        if i == total_nodes + 1:
            return stateSpace
        else:
            print(f'Invalid StateSpace! i={i}, expected={total_nodes + 1}')
            return None


    # # For paths in the form [root, , , leaf]
    # def generateStateSpace(self, hierarchy):
    #     # Dynamically determine total number of nodes
    #     total_nodes = max(max(path) for path in hierarchy) + 1
    #
    #     # Initialize stateSpace and recorded tensors
    #     stateSpace = torch.zeros(total_nodes + 1, total_nodes)
    #     recorded = torch.zeros(total_nodes)
    #     i = 1
    #
    #     for path in hierarchy:
    #         root = path[0]
    #         classification = path[-1]
    #         parents = path[:-1]
    #
    #         # Handle single-level path (no parents)
    #         if len(parents) == 0:
    #             if recorded[classification] == 0:
    #                 recorded[classification] = 1
    #             stateSpace[i, classification] = 1
    #             i += 1
    #             continue
    #
    #         # Record intermediate states for each parent node
    #         for d in range(len(parents)):
    #             node = parents[d]
    #             if recorded[node] == 0:
    #                 for j in range(d + 1):
    #                     stateSpace[i, parents[j]] = 1
    #                 recorded[node] = 1
    #                 i += 1
    #
    #         # Final row: all parents + classification
    #         if recorded[classification] == 0:
    #             for node in parents:
    #                 stateSpace[i, node] = 1
    #             stateSpace[i, classification] = 1
    #             recorded[classification] = 1
    #             i += 1
    #
    #     if i == total_nodes + 1:
    #         return stateSpace
    #     else:
    #         print(f'Invalid StateSpace! i={i}, expected={total_nodes + 1}')
    #         return None

    # For paths in the form [leaf, root, , , ]
    # def generateStateSpace(self, hierarchy):
    #     # Dynamically determine total number of nodes
    #     total_nodes = max(max(path) for path in hierarchy) + 1
    #
    #     # Initialize stateSpace and recorded tensors
    #     stateSpace = torch.zeros(total_nodes + 1, total_nodes)
    #     recorded = torch.zeros(total_nodes)
    #     i = 1
    #
    #     for path in hierarchy:
    #         classification = path[0]
    #         parents = path[1:]
    #
    #         # Handle single-level path (no parents)
    #         if len(parents) == 0:
    #             if recorded[classification] == 0:
    #                 recorded[classification] = 1
    #             stateSpace[i, classification] = 1
    #             i += 1
    #             continue
    #
    #         # Record intermediate states for each parent node
    #         for d in range(len(parents)):
    #             node = parents[d]
    #             if recorded[node] == 0:
    #                 for j in range(d + 1):
    #                     stateSpace[i, parents[j]] = 1
    #                 recorded[node] = 1
    #                 i += 1
    #
    #         # Final row: all parents + classification
    #         if recorded[classification] == 0:
    #             for node in parents:
    #                 stateSpace[i, node] = 1
    #             stateSpace[i, classification] = 1
    #             recorded[classification] = 1
    #             i += 1
    #
    #     if i == total_nodes + 1:
    #         return stateSpace
    #     else:
    #         print(f'Invalid StateSpace! i={i}, expected={total_nodes + 1}')
    #         return None
