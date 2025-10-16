import duckdb

db_path = '/Projects/FAIR_Device_data/Zaki/preprocess/insect_images.duckdb'
con = duckdb.connect(database=db_path, read_only=True)

# Load all relevant columns
query = """
SELECT parent_folder_1, parent_folder_2, parent_folder_3, parent_folder_4, parent_folder_5
FROM cropped_images_cleaned;
"""
df = con.execute(query).fetchdf()

# Inspect each column
for col in df.columns:
    print(f"\n--- {col} ---")
    print(f"Data type: {df[col].dtype}")
    print(f"Null count: {df[col].isnull().sum()}")
    print("Value counts (top 10):")
    print(df[col].value_counts(dropna=False).head(10))
