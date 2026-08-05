# -*- coding: utf-8 -*-
import duckdb
import pandas as pd
from pathlib import Path
import config

# ------------------------------------------------------------
# User-configurable paths
# ------------------------------------------------------------
db_path = config.DUCKDB_PATH
table_name = 'cropped_images_optim'

# Replace these with your actual text file paths:
train_txt_path = config.get_train_list_path()
val_txt_path = config.get_val_list_path()

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def read_list_file(txt_path):
    """
    Reads a list file where each line is:
        <image_path> <idx> <class_name>
    - image_path may contain spaces
    - idx is an integer
    - class_name is a single token (no spaces)
    Returns a pandas DataFrame with columns: image_path, idx, class_name
    """
    rows = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                # Skip malformed lines
                continue
            # Last two tokens are idx and class; the rest form the path
            class_name = parts[-1]
            idx_str = parts[-2]
            image_path = ' '.join(parts[:-2])
            try:
                idx = int(idx_str)
            except ValueError:
                # If idx isn't an integer, skip the line
                # (or you could set idx = None if you prefer)
                continue
            rows.append((image_path, idx, class_name))
    return pd.DataFrame(rows, columns=['image_path', 'idx', 'class_name'])

# ------------------------------------------------------------
# Load train and val lists and combine
# ------------------------------------------------------------
train_df = read_list_file(train_txt_path)
val_df = read_list_file(val_txt_path)

combined_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

# Optional sanity checks
print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Combined: {len(combined_df)}")
print("Sample rows from combined list:")
print(combined_df.head(5))

# ------------------------------------------------------------
# Query DuckDB to compute average area per class_name (from text files)
# ------------------------------------------------------------
con = duckdb.connect(db_path, read_only=True)

# Register the combined DataFrame as a view for joining
con.register('combined_list', combined_df)

# Build and run the query:
# - Join combined_list.image_path to table.image_path
# - Compute AVG(width * height) per class_name
# - Leave rows unsorted (no ORDER BY)
query = f"""
WITH joined AS (
    SELECT
        cl.class_name,
        ci.width,
        ci.height
    FROM combined_list AS cl
    JOIN {table_name} AS ci
      ON cl.image_path = ci.image_path
    -- If you need to guard against missing width/height, you can add:
    -- WHERE ci.width IS NOT NULL AND ci.height IS NOT NULL
)
SELECT
    class_name,
    ROUND(AVG(CAST(width AS DOUBLE) * CAST(height AS DOUBLE))) AS avg_area,
    COUNT(*) AS n_images
FROM joined
GROUP BY class_name
"""

result_df = con.execute(query).df()

# Show results
print("\nAverage area per class (from train + val lists):")
print(result_df)

# Optionally save to CSV (uncomment if desired)
# out_csv = Path.cwd() / "avg_area_per_class.csv"
# result_df.to_csv(out_csv, index=False)
# print(f"\nSaved results to: {out_csv}")

# Clean up
con.close()