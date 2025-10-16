import json

beta = 0.9999

with open("node_sample_counts_13Oct25.json", "r") as f:
    sample_counts = json.load(f)

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

for idx in sorted(normalized_weights.keys()):
    print(f"Node {idx}: Count = {sample_counts[str(idx)]}, Weight = {normalized_weights[idx]:.6f}")