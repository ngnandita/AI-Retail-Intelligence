import pandas as pd

# ==============================
# STEP 1: Load Dataset
# ==============================
df = pd.read_csv("dataset/sample_-_superstore.csv")
print(" Dataset Loaded Successfully!\n")

# ==============================
# STEP 2: Remove Extra Spaces
# ==============================
df.columns = df.columns.str.strip()

# ==============================
# STEP 3: First 5 Rows
# ==============================
print(" First 5 Rows:")
print(df.head())

# ==============================
# STEP 4: Shape
# ==============================
print("\n Shape:")
print(df.shape)

# ==============================
# STEP 5: Column Names
# ==============================
print("\n Columns:")
print(df.columns.tolist())

# ==============================
# STEP 6: Data Types
# ==============================
print("\n Data Types:")
print(df.dtypes)

# ==============================
# STEP 7: Missing Values
# ==============================
print("\n Missing Values:")
print(df.isnull().sum())

# ==============================
# STEP 8: Duplicate Rows
# ==============================
duplicates = df.duplicated().sum()

print("\n Duplicate Rows:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()
    print(" Duplicate Rows Removed Successfully!")

# ==============================
# STEP 9: Statistical Summary
# ==============================
print("\n Statistical Summary:")
print(df.describe())

# ==============================
# STEP 10: Save Clean Dataset
# ==============================
df.to_csv("dataset/clean_superstore.csv", index=False)

print("\n Clean Dataset Saved Successfully!")