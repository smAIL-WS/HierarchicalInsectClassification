import datetime
import duckdb
import pandas as pd
import re
import random
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

# Configuration
random.seed(42)
db_path = '/Projects/FAIR_Device_data/Zaki/preprocess/insect_images.duckdb'
table_name = 'cropped_images_cleaned'
base_dir = Path(__file__).resolve().parent
runs_dir = base_dir / 'runs'
runs_dir.mkdir(exist_ok=True)
date_str = datetime.date.today().isoformat()
run_folder = runs_dir / date_str
run_folder.mkdir(exist_ok=True)
log_path = run_folder / 'training_info.txt'
class DualLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
sys.stdout = DualLogger(log_path)
train_txt_path = run_folder / 'insect_train_list.txt'
test_txt_path = run_folder / 'insect_test_list.txt'
tree_file_path = run_folder / 'hierarchy_tree.txt'
name_map_path = run_folder / 'level_name_maps.json'
leaf_counts_path = run_folder / 'leaf_sample_counts.json'
node_counts_path = run_folder / 'node_sample_counts.json'

# Sampling configuration
min_train_images = 450
min_test_images = 150
min_train_ratio = 0.75
max_train_images = 13500
max_test_images = 4500
require_multiple_dates = True

# Connect to DuckDB and load data
con = duckdb.connect(database=db_path, read_only=True)
df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
print(f"Total rows loaded: {len(df)}")

# # Exclude unwanted classification
# df = df[df['classification'] != 'Ichneumonidae'] # has 6 levels of hierarchy, so doesn't work with 5-level setup
# print(f"Rows after excluding 'Ichneumonidae': {len(df)}")

# --- Extract and normalize base_date ---
folder_pattern = re.compile(r'(?:FAIR-D\d*_)?(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})')

# Extract full timestamp, then strip time to keep only YYYY-MM-DD
df['base_date'] = df['folder_path'].apply(
    lambda x: folder_pattern.search(x).group(1).split('_')[0] if folder_pattern.search(x) else None
)

# Drop rows without a valid date
df = df.dropna(subset=['base_date'])
print(f"Rows after extracting and normalizing base_date: {len(df)}")

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

# Select classes with ≥min train/test images and flexible train ratio
train_rows, test_rows = [], []
class_image_summary = {}

for cls, date_groups in class_date_groups.items():
    # Each date_groups[date] is already a list of rows for (class, date)
    if len(date_groups) < 2:
        print(f"Skipping class '{cls}' due to insufficient date groups.")
        continue

    # Sort date groups by size (largest first)
    sorted_dates = sorted(date_groups.items(), key=lambda x: len(x[1]), reverse=True)
    total_images = sum(len(rows) for _, rows in sorted_dates)
    target_train_images = int(total_images * min_train_ratio)

    train_pool, test_pool = [], []
    accumulated = 0

    # Primary allocation: fill train until target ratio using whole date groups
    for date, rows in sorted_dates:
        if accumulated < target_train_images:
            train_pool.extend(rows)
            accumulated += len(rows)
        else:
            test_pool.extend(rows)

    # Fallback: alternate entire date groups if minimums not met
    if len(train_pool) < min_train_images or len(test_pool) < min_test_images:
        train_pool, test_pool = [], []
        for i, (date, rows) in enumerate(sorted_dates):
            if i % 2 == 0:
                train_pool.extend(rows)
            else:
                test_pool.extend(rows)

    # Final check: skip class if still insufficient
    if len(train_pool) < min_train_images or len(test_pool) < min_test_images:
        print(f"Skipping class '{cls}' due to insufficient images after fallback.")
        continue

    # --- Apply maximum caps with random sampling ---
    if max_train_images and len(train_pool) > max_train_images:
        train_pool = random.sample(train_pool, max_train_images)

    if max_test_images and len(test_pool) > max_test_images:
        test_pool = random.sample(test_pool, max_test_images)

    # Add to global pools
    train_rows.extend(train_pool)
    test_rows.extend(test_pool)

    # Update summary
    class_image_summary[cls] = {
        'train': len(train_pool),
        'test': len(test_pool)
    }

print(f"Total train images: {len(train_rows)}, Total test images: {len(test_rows)}")

# --- Safety check: ensure no (class, date) overlap ---
train_pairs = {(r['classification'], r['base_date']) for r in train_rows}
test_pairs = {(r['classification'], r['base_date']) for r in test_rows}
overlap = train_pairs.intersection(test_pairs)
assert not overlap, f"Overlap detected for class/date pairs: {overlap}"

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

# Add split column to selected_df
selected_df['split'] = ['train'] * len(train_rows) + ['test'] * len(test_rows)

# Print image counts per hierarchy level (realigned, with train/test split)
print("\nImage counts per hierarchy level (realigned, with train/test split):")
for level in hierarchy_levels:
    train_counter = Counter()
    test_counter = Counter()

    for _, row in selected_df.iterrows():
        val = row[level]
        if is_valid(val):
            if row['split'] == 'train':
                train_counter[val] += 1
            else:
                test_counter[val] += 1

    print(f"\n{level}:")
    all_nodes = set(train_counter.keys()) | set(test_counter.keys())
    for node in all_nodes:
        train_count = train_counter[node]
        test_count = test_counter[node]
        total_count = train_count + test_count
        print(f"  {node}: train={train_count}, test={test_count}, total={total_count}")

# # Print image counts per hierarchy level (after realignment)
# print("\nImage counts per hierarchy level (realigned):")
# for level in hierarchy_levels:
#     level_counter = Counter()
#     for val in selected_df[level]:
#         if is_valid(val):
#             level_counter[val] += 1
#     print(f"\n{level}:")
#     for node, count in level_counter.items():
#         print(f"  {node}: {count}")

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

# # === Count samples for valid leaf classes ===
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
# # Map to global indices
# leaf_class_counts_by_index = {
#     global_node_map[cls]: leaf_class_counts[cls]
#      for cls in leaf_class_counts
#      if cls in global_node_map
#  }
#
# # Save to JSON
# with open(leaf_counts_path, 'w') as f:
#     json.dump(leaf_class_counts_by_index, f, indent=4)
# print(f"Saved leaf sample counts to: {leaf_counts_path}")

# === Count samples for use in effective weights for tree loss (all nodes in hierarchy) ===

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

