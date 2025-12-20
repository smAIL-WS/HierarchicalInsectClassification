
import datetime
import duckdb
import pandas as pd
import re
import random
import json
import sys
import math
from collections import defaultdict, Counter
from pathlib import Path

# =========================
# Configuration
# =========================
random.seed(42)

db_path = '/Projects/FAIR_Device_data/Zaki/preprocess/insect_images_final.duckdb'
table_name = 'cropped_images_optim'

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
val_txt_path = run_folder / 'insect_val_list.txt'  # NEW (only written when enabled)

tree_file_path = run_folder / 'hierarchy_tree.txt'
name_map_path = run_folder / 'level_name_maps.json'
leaf_counts_path = run_folder / 'leaf_sample_counts.json'
node_counts_path = run_folder / 'node_sample_counts.json'

# =========================
# Sampling configuration
# =========================
min_train_images = 450
min_test_images = 150
min_train_ratio = 0.75
max_train_images = 18000
max_test_images = 6000
require_multiple_dates = True

# --- Validation controls (NEW) ---
train_val_test_split = False               # Set False to keep original train/test-only behavior
val_ratio_within_train = 0.20             # 20% of train per class (target, before guardrails)
allow_train_val_leak = True               # Allowed only when a class has exactly one train device-day group

# Guardrails to prevent starving training
min_train_retention_ratio = 0.65          # After val split, retain at least 65% of pre-val train images
max_val_ratio_within_train = 0.35         # After split, val ≤ 35% of (train+val) for the class
val_select_small_groups_first = True      # Prefer smaller groups for val to reduce overshoot

# =========================
# Load data
# =========================
con = duckdb.connect(database=db_path, read_only=True)
df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
print(f"Total rows loaded: {len(df)}")

# Exclude unwanted classification
df = df[df['classification'] != 'lacewing']
print(f"Rows after excluding 'lacewing': {len(df)}")

# --- Extract and normalize base_date and device_id from folder_path ---
folder_pattern = re.compile(r'(?:FAIR-(D\d+)_)?(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})')

df['device_id'] = df['folder_path'].apply(
    lambda x: folder_pattern.search(x).group(1) if folder_pattern.search(x) and folder_pattern.search(x).group(1) else 'unknown'
)
df['base_date'] = df['folder_path'].apply(
    lambda x: folder_pattern.search(x).group(2).split('_')[0] if folder_pattern.search(x) else None
)

# Drop rows without a valid date
df = df.dropna(subset=['base_date'])
print(f"Rows after extracting and normalizing base_date: {len(df)}")

# --- Helpers for hierarchy validity and alignment ---
def is_valid(entry):
    return pd.notna(entry) and str(entry).strip().lower() != 'nan' and str(entry).strip() != ''

hierarchy_levels = ['parent_folder_4', 'parent_folder_3', 'parent_folder_2', 'parent_folder_1', 'classification']

def realign_hierarchy(row):
    raw_levels = ['parent_folder_4', 'parent_folder_3', 'parent_folder_2', 'parent_folder_1', 'classification']
    valid_labels = [row[col] for col in raw_levels if is_valid(row[col])]
    aligned = [''] * 5
    for i, label in enumerate(valid_labels):
        aligned[i] = label
    return pd.Series(aligned, index=raw_levels)

# Store original classification (for output lists)
df['original_classification'] = df['classification']

# --- Filter classes with >1 unique date (optional, as in original script) ---
if require_multiple_dates:
    folder_counts = df.groupby('classification')['base_date'].nunique()
    valid_classes = folder_counts[folder_counts > 1].index
    df = df[df['classification'].isin(valid_classes)]
    print(f"Rows after filtering to classes with >1 date: {len(df)}")

# --- Group by classification, then by (base_date, device_id) ---
class_date_device_groups = defaultdict(lambda: defaultdict(list))
for _, row in df.iterrows():
    key = (row['base_date'], row['device_id'])
    class_date_device_groups[row['classification']][key].append(row)

# =========================
# Train/Test selection (UNCHANGED)
# =========================
train_rows, test_rows = [], []
initial_class_image_summary = {}  # train/test counts (pre-val), for diagnostics

for cls, date_device_groups in class_date_device_groups.items():
    # Skip classes with <2 groups
    if len(date_device_groups) < 2:
        print(f"Skipping class '{cls}' due to insufficient (date, device) groups.")
        continue

    # Sort groups by size (largest first)
    sorted_groups = sorted(date_device_groups.items(), key=lambda x: len(x[1]), reverse=True)
    total_images = sum(len(rows) for _, rows in sorted_groups)
    target_train_images = int(total_images * min_train_ratio)

    train_pool, test_pool = [], []
    accumulated = 0

    # Primary allocation: fill train until target using whole (date, device) groups
    for (date, device), rows in sorted_groups:
        if accumulated < target_train_images:
            train_pool.extend(rows)
            accumulated += len(rows)
        else:
            test_pool.extend(rows)

    # Fallback: alternate whole groups if minimums not met
    if len(train_pool) < min_train_images or len(test_pool) < min_test_images:
        train_pool, test_pool = [], []
        for i, ((date, device), rows) in enumerate(sorted_groups):
            if i % 2 == 0:
                train_pool.extend(rows)
            else:
                test_pool.extend(rows)

    # Final check: skip class if still insufficient
    if len(train_pool) < min_train_images or len(test_pool) < min_test_images:
        print(f"Skipping class '{cls}' due to insufficient images after fallback.")
        continue

    # Apply maximum caps with random sampling
    if max_train_images and len(train_pool) > max_train_images:
        train_pool = random.sample(train_pool, max_train_images)
    if max_test_images and len(test_pool) > max_test_images:
        test_pool = random.sample(test_pool, max_test_images)

    # Add to global pools
    train_rows.extend(train_pool)
    test_rows.extend(test_pool)

    # Record class-level counts (pre-val)
    initial_class_image_summary[cls] = {
        'train': len(train_pool),
        'test': len(test_pool)
    }

print(f"Total train images (pre-val): {len(train_rows)}, Total test images: {len(test_rows)}")

# --- Safety check: ensure no (class, date, device) overlap between TRAIN and TEST (unchanged) ---
train_pairs = {(r['classification'], r['base_date'], r['device_id']) for r in train_rows}
test_pairs = {(r['classification'], r['base_date'], r['device_id']) for r in test_rows}
overlap_train_test = train_pairs.intersection(test_pairs)
assert not overlap_train_test, f"Overlap detected for class/date/device pairs (train vs test): {overlap_train_test}"

# =========================
# Optional: Train -> (Train + Val) split with guardrails
# =========================
val_rows = []
train_rows_final = train_rows  # default: unchanged
val_split_notes = {}           # per-class: {'split': 'non-leaky'|'leaky'|'none', 'val_groups': int, 'sampled': int, 'adjustments': [...]}

if train_val_test_split:
    # Build per-class indexing of train/test rows
    train_rows_by_class = defaultdict(list)
    test_rows_by_class = defaultdict(list)
    for r in train_rows:
        train_rows_by_class[r['classification']].append(r)
    for r in test_rows:
        test_rows_by_class[r['classification']].append(r)

    # Collect new train rows (post-val) here
    train_rows_final = []

    # Track classes allowed to have train<->val overlap (leaky single-group case)
    leaky_allowed_classes = set()

    # Process each class present in train/test selections
    classes_seen = set(train_rows_by_class.keys()) | set(test_rows_by_class.keys())
    for cls in sorted(classes_seen):
        cls_train = train_rows_by_class.get(cls, [])
        cls_test = test_rows_by_class.get(cls, [])

        # Group train rows by (base_date, device_id)
        train_groups = defaultdict(list)
        for r in cls_train:
            key = (r['base_date'], r['device_id'])
            train_groups[key].append(r)

        num_train_groups = len(train_groups)
        num_test_groups = len({(r['base_date'], r['device_id']) for r in cls_test})

        target_val_images = round(val_ratio_within_train * len(cls_train))

        cls_val_rows = []
        cls_train_rows_final = []
        adjustments = []

        # Helper: compute retention and val caps
        total_pre_val_train = len(cls_train)
        retention_min_train = math.ceil(min_train_retention_ratio * total_pre_val_train)
        max_val_cap = math.floor(max_val_ratio_within_train * total_pre_val_train)  # val share of (train+val) = total_pre_val_train

        if num_train_groups >= 2 and target_val_images > 0:
            # Non-leaky: select whole groups
            # Order groups by size according to preference
            group_items = list(train_groups.items())
            group_items.sort(key=lambda x: len(x[1]), reverse=not val_select_small_groups_first)

            val_selected_keys = []
            acc_val = 0
            remaining_groups = num_train_groups

            for key, rows in group_items:
                # ensure at least one group remains in train
                if remaining_groups <= 1:
                    break

                proposed_val = acc_val + len(rows)
                proposed_train = total_pre_val_train - proposed_val

                # Guardrail: training retention
                if proposed_train < retention_min_train:
                    # selecting this group would starve training — skip it
                    adjustments.append('retention')
                    continue

                # Guardrail: max val ratio
                if proposed_val > max_val_cap:
                    adjustments.append('max_val')
                    break

                # Accept group for val
                val_selected_keys.append(key)
                acc_val = proposed_val
                remaining_groups -= 1

                # Stop once we reached/just exceeded target
                if acc_val >= target_val_images:
                    break

            # Edge case: keep at least one group in train (already enforced by remaining_groups check)

            # Assign rows
            for key, rows in train_groups.items():
                if key in val_selected_keys:
                    cls_val_rows.extend(rows)
                else:
                    cls_train_rows_final.extend(rows)

            if len(val_selected_keys) > 0:
                split_type = 'non-leaky'
                val_split_notes[cls] = {
                    'split': split_type,
                    'val_groups': len(val_selected_keys),
                    'adjustments': sorted(set(adjustments))
                }
            else:
                # No val allocated due to guardrails
                split_type = 'none'
                val_split_notes[cls] = {
                    'split': split_type,
                    'val_groups': 0,
                    'adjustments': sorted(set(adjustments))
                }

        elif num_train_groups == 1 and allow_train_val_leak and target_val_images > 0:
            # Leaky: sample within the single group (respect guardrails)
            only_key = next(iter(train_groups.keys()))
            only_rows = train_groups[only_key]

            # Max allowed by ratio cap
            cap_by_ratio = max_val_cap
            # Max allowed by retention (leave at least retention_min_train in train)
            cap_by_retention = max(0, total_pre_val_train - retention_min_train)
            # Always leave at least one image for train
            cap_leave_one = max(0, total_pre_val_train - 1)

            sample_n = min(target_val_images, cap_by_ratio, cap_by_retention, cap_leave_one)

            if sample_n > 0:
                sampled_indices = set(random.sample(range(len(only_rows)), sample_n))
                for i, r in enumerate(only_rows):
                    if i in sampled_indices:
                        cls_val_rows.append(r)
                    else:
                        cls_train_rows_final.append(r)
                adj = []
                if sample_n < target_val_images:
                    # We were capped by some guardrail(s)
                    if sample_n == cap_by_ratio:
                        adj.append('max_val')
                    if sample_n == cap_by_retention:
                        adj.append('retention')
                val_split_notes[cls] = {
                    'split': 'leaky',
                    'sampled': sample_n,
                    'adjustments': sorted(set(adj))
                }
                leaky_allowed_classes.add(cls)
            else:
                # target or guardrails prevent val; all remain in train
                cls_train_rows_final.extend(only_rows)
                val_split_notes[cls] = {
                    'split': 'none',
                    'val_groups': 0,
                    'adjustments': ['retention'] if retention_min_train > 0 else []
                }

        else:
            # No val for this class (either target is 0 or no train groups)
            for rows in train_groups.values():
                cls_train_rows_final.extend(rows)
            val_split_notes[cls] = {
                'split': 'none',
                'val_groups': 0,
                'adjustments': []
            }

        # Accumulate
        train_rows_final.extend(cls_train_rows_final)
        val_rows.extend(cls_val_rows)

    # --- Safety checks involving VAL ---
    val_pairs = {(r['classification'], r['base_date'], r['device_id']) for r in val_rows}
    # 1) No overlap between VAL and TEST
    overlap_val_test = val_pairs.intersection(test_pairs)
    assert not overlap_val_test, f"Overlap detected between val and test pairs: {overlap_val_test}"

    # 2) Overlap between TRAIN and VAL only allowed for classes with a single train group (leaky case)
    train_final_pairs = {(r['classification'], r['base_date'], r['device_id']) for r in train_rows_final}
    overlap_train_val = train_final_pairs.intersection(val_pairs)
    if overlap_train_val:
        offending = {(cls, date, dev) for (cls, date, dev) in overlap_train_val if cls not in leaky_allowed_classes}
        assert not offending, f"Illegal train<->val overlap detected for classes with >=2 train groups: {offending}"

    print(f"Total images after val split — train: {len(train_rows_final)}, val: {len(val_rows)}, test: {len(test_rows)}")

# =========================
# Build selected_df for downstream artifacts
# =========================
if train_val_test_split:
    selected_rows = train_rows_final + val_rows + test_rows
else:
    selected_rows = train_rows + test_rows

selected_df = pd.DataFrame(selected_rows)

# Realign hierarchy only for selected data
selected_df[hierarchy_levels] = selected_df.apply(realign_hierarchy, axis=1)

# Filter for valid hierarchy (at least one valid label in row)
selected_df = selected_df[selected_df[hierarchy_levels].apply(lambda row: any(is_valid(val) for val in row), axis=1)]

# =========================
# Concise per-class logging (post-val if enabled; otherwise pre-val)
# =========================
print("\nPer-class image counts:")
def count_by_class(rows):
    counts = defaultdict(int)
    for r in rows:
        counts[r['classification']] += 1
    return counts

train_counts_by_class = count_by_class(train_rows_final if train_val_test_split else train_rows)
val_counts_by_class = count_by_class(val_rows) if train_val_test_split else defaultdict(int)
test_counts_by_class = count_by_class(test_rows)

def group_count(rows):
    d = defaultdict(set)
    for r in rows:
        d[r['classification']].add((r['base_date'], r['device_id']))
    return {cls: len(keys) for cls, keys in d.items()}

# Group counts based on original train/test grouping (source)
train_group_counts = group_count(train_rows)
test_group_counts = group_count(test_rows)

all_classes = sorted(set(list(train_counts_by_class.keys()) + list(test_counts_by_class.keys())))
for cls in all_classes:
    tr = train_counts_by_class.get(cls, 0)
    va = val_counts_by_class.get(cls, 0)
    te = test_counts_by_class.get(cls, 0)
    tg = train_group_counts.get(cls, 0)
    sg = test_group_counts.get(cls, 0)
    note = val_split_notes.get(cls, {'split': 'none', 'adjustments': []})
    if train_val_test_split:
        split_type = note.get('split', 'none')
        if split_type == 'non-leaky':
            print(f"Class: {cls} | train={tr} val={va} test={te} | train_groups={tg} test_groups={sg} | val_split=non-leaky | val_groups={note.get('val_groups', 0)} | adjustments={note.get('adjustments', [])}")
        elif split_type == 'leaky':
            print(f"Class: {cls} | train={tr} val={va} test={te} | train_groups={tg} test_groups={sg} | val_split=leaky | sampled={note.get('sampled', 0)} from single group | adjustments={note.get('adjustments', [])}")
        else:
            print(f"Class: {cls} | train={tr} val={va} test={te} | train_groups={tg} test_groups={sg} | val_split=none | adjustments={note.get('adjustments', [])}")
    else:
        print(f"Class: {cls} | train={tr} test={te} | train_groups={tg} test_groups={sg}")

# =========================
# Add split column to selected_df
# =========================
if train_val_test_split:
    selected_df['split'] = (['train'] * len(train_rows_final)) + (['val'] * len(val_rows)) + (['test'] * len(test_rows))
else:
    selected_df['split'] = (['train'] * len(train_rows)) + (['test'] * len(test_rows))

# =========================
# Image counts per hierarchy level (with split)
# =========================
print("\nImage counts per hierarchy level (realigned, with splits):")
for level in hierarchy_levels:
    split_counters = {'train': Counter(), 'test': Counter()}
    if train_val_test_split:
        split_counters['val'] = Counter()

    for _, row in selected_df.iterrows():
        val = row[level]
        if is_valid(val):
            split_counters[row['split']][val] += 1

    print(f"\n{level}:")
    # Collect all nodes observed across splits
    all_nodes = set()
    for c in split_counters.values():
        all_nodes |= set(c.keys())
    for node in all_nodes:
        parts = []
        for split_name in ['train', 'val', 'test'] if train_val_test_split else ['train', 'test']:
            cnt = split_counters.get(split_name, Counter()).get(node, 0)
            parts.append(f"{split_name}={cnt}")
        total_count = sum(split_counters[s].get(node, 0) for s in split_counters.keys())
        print(f"  {node}: " + ", ".join(parts) + f", total={total_count}")

# =========================
# Create global node map from selected_df (train+val+test when enabled; train+test otherwise)
# =========================
global_node_map = {}
node_counter = 0
for level in hierarchy_levels:
    unique_nodes = sorted(set(val for val in selected_df[level] if is_valid(val)))
    for node in unique_nodes:
        if node not in global_node_map:
            global_node_map[node] = node_counter
            node_counter += 1

# Per-level name maps: {level: {local_index: name}}
level_name_maps = {}
for level in hierarchy_levels:
    level_name_maps[level] = {}
    unique_nodes = sorted(set(val for val in selected_df[level] if is_valid(val)))
    if unique_nodes:
        min_idx_in_level = min(global_node_map[n] for n in unique_nodes)
    else:
        min_idx_in_level = 0
    for node in unique_nodes:
        global_idx = global_node_map[node]
        local_idx = global_idx - min_idx_in_level
        level_name_maps[level][local_idx] = node

# =========================
# Count node occurrences in TRAIN ONLY
# =========================
node_counts = Counter()
rows_for_training_counts = train_rows_final if train_val_test_split else train_rows
for row in rows_for_training_counts:
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

# Save train-only node counts
with open(node_counts_path, 'w') as f:
    json.dump(node_counts_by_index, f, indent=4)
print(f"Saved node sample counts to: {node_counts_path}")

# Print node counts (ordered by global index)
print("\nNode sample counts (train only):")
sorted_nodes_by_index = sorted(
    node_counts.items(),
    key=lambda item: global_node_map[item[0]]
)
for node, count in sorted_nodes_by_index:
    global_idx = global_node_map[node]
    print(f"  {global_idx:4d}  {node:30s}  count: {count}")

# Print hierarchy label mappings
print("\nHierarchy label mappings for selected data:")
for level in hierarchy_levels:
    print(f"\n{level} mapping:")
    unique_nodes = sorted(set(val for val in selected_df[level] if is_valid(val)))
    for node in unique_nodes:
        idx = global_node_map[node]
        print(f"{idx}: {node}")

# Print number of unique classes at each hierarchy level
print("\nUnique classes per hierarchy level in selected data:")
for level in hierarchy_levels:
    valid_values = selected_df[level][selected_df[level].apply(is_valid)]
    unique_count = valid_values.nunique()
    print(f"{level}: {unique_count} unique classes")
    print(f"  Sample values: {valid_values.unique()[:5]}")

# =========================
# Generate output lines for train / val / test sets
# =========================
def lines_for_rows(rows):
    out = []
    for row in rows:
        target_index = global_node_map[row['original_classification']]
        class_name = row['original_classification']
        out.append(f"{row['image_path']} {target_index} {class_name}")
    return out

# TRAIN
train_lines = lines_for_rows(train_rows_final if train_val_test_split else train_rows)
with open(train_txt_path, 'w') as f:
    for line in train_lines:
        f.write(line + '\n')
print(f"\nSaved training list to: {train_txt_path}")

# VAL (only when enabled)
if train_val_test_split:
    val_lines = lines_for_rows(val_rows)
    with open(val_txt_path, 'w') as f:
        for line in val_lines:
            f.write(line + '\n')
    print(f"Saved validation list to: {val_txt_path}")

# TEST
test_lines = lines_for_rows(test_rows)
with open(test_txt_path, 'w') as f:
    for line in test_lines:
        f.write(line + '\n')
print(f"Saved test list to: {test_txt_path}")

# =========================
# Build hierarchy tree from selected_df
# =========================
tree_paths = selected_df.apply(
    lambda row: [global_node_map[row[level]] for level in hierarchy_levels if is_valid(row[level])],
    axis=1
).drop_duplicates().tolist()

tree_paths_sorted = sorted(tree_paths, key=lambda x: len(x))
with open(tree_file_path, 'w') as f:
    f.write("trees = [\n")
    for path in tree_paths_sorted:
        f.write(f"  {path},\n")
    f.write("]\n")
print(f"Saved hierarchy tree to: {tree_file_path}")

# =========================
# Save per-level name maps to JSON
# =========================
with open(name_map_path, 'w') as f:
    json.dump(level_name_maps, f, indent=4)
print(f"Saved level name mappings to: {name_map_path}")
