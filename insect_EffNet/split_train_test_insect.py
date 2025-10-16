import duckdb
import pandas as pd
import re
import random
import json
from collections import defaultdict, Counter
from pathlib import Path

# Configuration
random.seed(42)
db_path = '/Projects/FAIR_Device_data/Zaki/preprocess/insect_images.duckdb'
table_name = 'cropped_images_cleaned'
base_dir = Path(__file__).resolve().parent
train_txt_path = base_dir / 'insect_train_list_13Oct25.txt'
test_txt_path = base_dir / 'insect_test_list_13Oct25.txt'
tree_file_path = base_dir / 'hierarchy_tree_13Oct25.txt'
name_map_path = base_dir / 'level_name_maps_13Oct25.json'

# Sampling configuration
min_train_images = 800
min_test_images = 200
max_train_images = 16000
max_test_images = 4000
require_multiple_dates = True

# Connect to DuckDB and load data
con = duckdb.connect(database=db_path, read_only=True)
df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
print(f"Total rows loaded: {len(df)}")

# Exclude unwanted classification
df = df[df['classification'] != 'Ichneumonidae'] # has 6 levels of hierarchy, so doesn't work with 5-level setup
print(f"Rows after excluding 'Ichneumonidae': {len(df)}")

# Extract base_date from folder_path
folder_pattern = re.compile(r'(?:FAIR-D\d*_)?(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})')
df['base_date'] = df['folder_path'].apply(lambda x: folder_pattern.search(x).group(1) if folder_pattern.search(x) else None)
df = df.dropna(subset=['base_date'])
print(f"Rows after dropping missing base_date: {len(df)}")

# Validity check that cell contains a taxonomic name and excludes empty or with 'nan' labelled cells
def is_valid(entry):
    return pd.notna(entry) and str(entry).strip().lower() != 'nan' and str(entry).strip() != ''

# Define hierarchy column order (used throughout)
hierarchy_levels = ['parent_folder_4', 'parent_folder_3', 'parent_folder_2', 'parent_folder_1', 'classification']

def realign_hierarchy(row):
    # Define the original hierarchy columns
    raw_levels = ['parent_folder_4', 'parent_folder_3', 'parent_folder_2', 'parent_folder_1', 'classification']

    # Collect all valid labels in order
    valid_labels = [row[col] for col in raw_levels if is_valid(row[col])]

    # Left-align the labels so the coarsest is in parent_folder_4
    aligned = [''] * 5
    for i, label in enumerate(valid_labels):
        aligned[i] = label

    return pd.Series(aligned, index=raw_levels)

# Store original classification
df['original_classification'] = df['classification']

# Filter classes with >1 unique date
if require_multiple_dates:
    folder_counts = df.groupby('classification')['base_date'].nunique()
    valid_classes = folder_counts[folder_counts > 1].index
    df = df[df['classification'].isin(valid_classes)]
    print(f"Rows after filtering to classes with >1 date: {len(df)}")

# Group by classification and base_date
class_date_groups = defaultdict(lambda: defaultdict(list))
for _, row in df.iterrows():
    class_date_groups[row['classification']][row['base_date']].append(row)

# Select classes with ≥min and ≤max train/test images
train_rows, test_rows = [], []
class_image_summary = {}

for cls, date_groups in class_date_groups.items():
    if len(date_groups) < 2:
        continue
    sorted_dates = sorted(date_groups.items(), key=lambda x: len(x[1]), reverse=True)
    train_pool, test_pool = [], []
    for i, (date, rows) in enumerate(sorted_dates):
        if i % 2 == 0:
            train_pool.extend(rows)
        else:
            test_pool.extend(rows)
    if len(train_pool) >= min_train_images and len(test_pool) >= min_test_images:
        train_sample_size = min(len(train_pool), max_train_images)
        test_sample_size = min(len(test_pool), max_test_images)
        sampled_train = random.sample(train_pool, train_sample_size)
        sampled_test = random.sample(test_pool, test_sample_size)
        train_rows.extend(sampled_train)
        test_rows.extend(sampled_test)
        class_image_summary[cls] = {
            'train': len(sampled_train),
            'test': len(sampled_test)
        }

selected_rows = train_rows + test_rows
selected_df = pd.DataFrame(selected_rows)

# Realign hierarchy only for selected data
selected_df[hierarchy_levels] = selected_df.apply(realign_hierarchy, axis=1)

# Filter for valid hierarchy
selected_df = selected_df[selected_df[hierarchy_levels].apply(lambda row: any(is_valid(val) for val in row), axis=1)]

# Print image counts per class
print("\nImage counts per class:")
for cls, counts in class_image_summary.items():
    print(f"  {cls}: {counts['train']} train, {counts['test']} test")

# Print image counts per hierarchy level (after realignment)
print("\nImage counts per hierarchy level (realigned):")
for level in hierarchy_levels:
    level_counter = Counter()
    for val in selected_df[level]:
        if is_valid(val):
            level_counter[val] += 1
    print(f"\n{level}:")
    for node, count in level_counter.items():
        print(f"  {node}: {count}")

# Create level-wise node mapping (filtered with is_valid)
global_node_map = {}
node_counter = 0
for level in hierarchy_levels:
    unique_nodes = sorted(set(val for val in selected_df[level] if is_valid(val)))
    for node in unique_nodes:
        if node not in global_node_map:
            global_node_map[node] = node_counter
            node_counter += 1

# Create per-level name maps: {level: {local_index: name}}
level_name_maps = {}
for level in hierarchy_levels:
    level_name_maps[level] = {}
    unique_nodes = sorted(set(val for val in selected_df[level] if is_valid(val)))
    for node in unique_nodes:
        global_idx = global_node_map[node]
        local_idx = global_idx - min(global_node_map[n] for n in unique_nodes)
        level_name_maps[level][local_idx] = node

# # === Compute weights for cross-entropy loss (leaf classes only) ===
#
# # Get valid leaf classes from selected_df['classification']
# valid_leaf_classes = set(
#     selected_df['classification'].dropna().unique()
# )
#
# # Count training samples per valid leaf class
# leaf_class_counts = Counter(
#     row['classification']
#     for row in train_rows
#     if is_valid(row['classification']) and row['classification'] in valid_leaf_classes
# )
#
# # Total number of leaf samples
# total_leaf_samples = sum(leaf_class_counts.values())
#
# # Inverse frequency weights
# leaf_class_weights = {
#     cls: total_leaf_samples / count
#     for cls, count in leaf_class_counts.items()
# }
#
# # Normalize to mean 1
# mean_leaf_weight = sum(leaf_class_weights.values()) / len(leaf_class_weights)
# leaf_class_weights = {
#     cls: weight / mean_leaf_weight
#     for cls, weight in leaf_class_weights.items()
# }
#
# # Map to global indices
# leaf_class_weights_by_index = {
#     global_node_map[cls]: leaf_class_weights[cls]
#     for cls in leaf_class_weights
#     if cls in global_node_map
# }
#
# # Save to JSON
# leaf_weights_path = base_dir / 'leaf_class_weights_06Oct25.json'
# with open(leaf_weights_path, 'w') as f:
#     json.dump(leaf_class_weights_by_index, f, indent=4)
# print(f"Saved leaf class weights to: {leaf_weights_path}")
#
# # Print leaf class weights
# print("\nLeaf class weights (cross-entropy loss):")
# for cls in sorted(leaf_class_weights):
#     global_idx = global_node_map[cls]
#     weight = leaf_class_weights[cls]
#     print(f"  {global_idx:4d} | {cls:30s} | weight: {weight:.4f}")

# === Compute weights for tree loss (all nodes in hierarchy) ===

# Count node occurrences across all hierarchy levels in training data
node_counts = Counter()
for row in train_rows:
    for level in hierarchy_levels:
        node = row[level]
        if is_valid(node):
            node_counts[node] += 1

# Map to global indices
node_counts_by_index = {
    global_node_map[node]: count
    for node, count in node_counts.items()
    if node in global_node_map
}

# Save to JSON
node_counts_path = base_dir / 'node_sample_counts_13Oct25.json'
with open(node_counts_path, 'w') as f:
    json.dump(node_counts_by_index, f, indent=4)
print(f"Saved node sample counts to: {node_counts_path}")

# Print node counts (ordered by global index)
print("\nNode sample counts:")
sorted_nodes_by_index = sorted(
    node_counts.items(),
    key=lambda item: global_node_map[item[0]]
)

for node, count in sorted_nodes_by_index:
    global_idx = global_node_map[node]
    print(f"  {global_idx:4d} | {node:30s} | count: {count}")

# # Inverse frequency weights
# node_weights = {
#     node: 1.0 / count
#     for node, count in node_counts.items()
# }
#
# # Normalize to mean 1
# mean_node_weight = sum(node_weights.values()) / len(node_weights)
# node_weights = {
#     node: weight / mean_node_weight
#     for node, weight in node_weights.items()
# }
#
# # Map to global indices
# node_weights_by_index = {
#     global_node_map[node]: weight
#     for node, weight in node_weights.items()
#     if node in global_node_map
# }
#
# # Save to JSON
# node_weights_path = base_dir / 'node_weights_06Oct25.json'
# with open(node_weights_path, 'w') as f:
#     json.dump(node_weights_by_index, f, indent=4)
# print(f"Saved node weights to: {node_weights_path}")
#
# # Print node weights (tree loss), ordered by global index
# print("\nNode weights (tree loss):")
# sorted_nodes_by_index = sorted(
#     node_weights.items(),
#     key=lambda item: global_node_map[item[0]]
# )
#
# for node, weight in sorted_nodes_by_index:
#     global_idx = global_node_map[node]
#     print(f"  {global_idx:4d} | {node:30s} | weight: {weight:.4f}")

# Print hierarchy label mappings
print("\nHierarchy label mappings for selected train/test data:")
for level in hierarchy_levels:
    print(f"\n{level} mapping:")
    unique_nodes = sorted(set(val for val in selected_df[level] if is_valid(val)))
    for node in unique_nodes:
        idx = global_node_map[node]
        print(f"{idx}: {node}")

# Print number of unique classes at each hierarchy level
print("\nUnique classes per hierarchy level in selected train/test data:")
for level in hierarchy_levels:
    valid_values = selected_df[level][selected_df[level].apply(is_valid)]
    unique_count = valid_values.nunique()
    print(f"{level}: {unique_count} unique classes")
    print(f"  Sample values: {valid_values.unique()[:5]}")

# Generate output lines for train and test sets
train_lines = []
for row in train_rows:
    target_index = global_node_map[row['original_classification']]
    class_name = row['original_classification']
    train_lines.append(f"{row['image_path']} {target_index} {class_name}")

test_lines = []
for row in test_rows:
    target_index = global_node_map[row['original_classification']]
    class_name = row['original_classification']
    test_lines.append(f"{row['image_path']} {target_index} {class_name}")

# Save train and test list files
with open(train_txt_path, 'w') as f:
    for line in train_lines:
        f.write(line + '\n')
print(f"\nSaved training list to: {train_txt_path}")

with open(test_txt_path, 'w') as f:
    for line in test_lines:
        f.write(line + '\n')
print(f"Saved test list to: {test_txt_path}")

# Build hierarchy tree
tree_paths = selected_df.apply(
    lambda row: [global_node_map[row[level]] for level in hierarchy_levels if is_valid(row[level])],
    axis=1
).drop_duplicates().tolist()

# Sort tree paths by length
tree_paths_sorted = sorted(tree_paths, key=lambda x: len(x))

# Save tree file
with open(tree_file_path, 'w') as f:
    f.write("trees = [\n")
    for path in tree_paths_sorted:
        f.write(f"    {path},\n")
    f.write("]\n")
print(f"Saved hierarchy tree to: {tree_file_path}")

# Save per-level name maps to JSON
with open(name_map_path, 'w') as f:
    json.dump(level_name_maps, f, indent=4)
print(f"Saved level name mappings to: {name_map_path}")

