import duckdb
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
import random
from collections import defaultdict
# Connect to DuckDB
db_path = '/Projects/FAIR_Device_data/Zaki/preprocess/insect_images.duckdb'
con = duckdb.connect(db_path)

# Load full table
df = con.execute("SELECT * FROM cropped_images_cleaned").fetchdf()
print(f"Total rows loaded: {len(df)}")

# Define hierarchy columns
hierarchy_cols = ['classification', 'parent_folder_1', 'parent_folder_2', 'parent_folder_3', 'parent_folder_4']

# Validation function
def is_valid(entry):
    return pd.notna(entry) and str(entry).strip().lower() != 'nan' and str(entry).strip() != ''

# Filter for exact 5-level hierarchy
df = df[
    df[hierarchy_cols].apply(lambda row: all(is_valid(val) for val in row), axis=1) &
    ~df['parent_folder_5'].apply(is_valid)
]

# Extract base_date from folder_path
folder_pattern = re.compile(r'(?:FAIR-D\\d*_)?(\\d{4}-\\d{2}-\\d{2}_\\d{2}-\\d{2}-\\d{2})')
df['base_date'] = df['folder_path'].apply(lambda x: folder_pattern.search(x).group(1) if folder_pattern.search(x) else None)
df = df.dropna(subset=['base_date'])
print(f"Rows after dropping missing base_date: {len(df)}")

# Filter classes with >1 unique date
folder_counts = df.groupby('classification')['base_date'].nunique()
valid_classes = folder_counts[folder_counts > 1].index
df = df[df['classification'].isin(valid_classes)]
print(f"Rows after filtering to classes with >1 date: {len(df)}")
print(f"Valid classes with data from more than one date: {list(valid_classes)}")

# Group by classification and base_date
class_date_groups = defaultdict(lambda: defaultdict(list))
for _, row in df.iterrows():
    class_date_groups[row['classification']][row['base_date']].append(row)

# Select classes with ≥800 train and ≥200 test images
train_rows, test_rows = [], []
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
    if len(train_pool) >= 800 and len(test_pool) >= 200:
        train_rows.extend(random.sample(train_pool, 800))
        test_rows.extend(random.sample(test_pool, 200))

# Combine selected rows
filtered_df = pd.DataFrame(train_rows + test_rows)

# Create directed graph
G = nx.DiGraph()
for _, row in filtered_df.iterrows():
    hierarchy = [row[f'parent_folder_{i}'] for i in range(5, 0, -1)] + [row['classification']]
    hierarchy = [node for node in hierarchy if is_valid(node)]
    for parent, child in zip(hierarchy, hierarchy[1:]):
        G.add_edge(parent, child)

# Force 'Insecta' and 'Arachnida' to be root nodes
for root in ['Insecta', 'Arachnida']:
    if root in G:
        for pred in list(G.predecessors(root)):
            G.remove_edge(pred, root)

# Layout and coloring
pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
node_colors = ['gold' if G.out_degree(node) == 0 else 'lightblue' for node in G.nodes()]

# Draw graph
fig, ax = plt.subplots(figsize=(24, 18))
nx.draw(G, pos, with_labels=True, node_size=750, node_color=node_colors,
        font_size=20, arrows=True, edge_color='grey', ax=ax)

fig.suptitle("Filtered Hierarchy Tree of Classifications", fontsize=28, y=0.96)
plt.tight_layout(rect=[0, -0.05, 1, 0.99])
plt.savefig("/Projects/FAIR_Device_data/Zaki/preprocess/train_insect_hier_23class", dpi=300)
plt.show()
