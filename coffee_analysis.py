import pandas as pd

# Load the data
df = pd.read_csv(r"D:\project\Afficionado_Coffee_Roasters_xlsx_-_Transactions.csv")

# Basic info
print("=" * 50)
print("DATASET SHAPE (rows, columns):")
print(df.shape)

print("\n" + "=" * 50)
print("FIRST 5 ROWS:")
print(df.head())

print("\n" + "=" * 50)
print("COLUMN NAMES & DATA TYPES:")
print(df.dtypes)

print("\n" + "=" * 50)
print("MISSING VALUES PER COLUMN:")
print(df.isnull().sum())

print("\n" + "=" * 50)
print("BASIC STATISTICS:")
print(df[['transaction_qty', 'unit_price']].describe())

print("\n" + "=" * 50)
print("UNIQUE STORES:")
print(df['store_location'].unique())

print("\n" + "=" * 50)
print("UNIQUE PRODUCT CATEGORIES:")
print(df['product_category'].unique())