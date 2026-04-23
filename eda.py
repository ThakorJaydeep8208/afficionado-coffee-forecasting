import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv(r"D:\project\Afficionado Coffee Roasters.xlsx - Transactions.csv")

# --- Prep ---
df['transaction_time'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S')
df['hour'] = df['transaction_time'].dt.hour
df['revenue'] = df['transaction_qty'] * df['unit_price']

output_folder = r"D:\project\charts"
os.makedirs(output_folder, exist_ok=True)
sns.set_theme(style="whitegrid")

# --- Chart 1: Revenue by Store ---
plt.figure(figsize=(8, 5))
store_rev = df.groupby('store_location')['revenue'].sum().sort_values(ascending=False)
sns.barplot(x=store_rev.index, y=store_rev.values, hue=store_rev.index, palette='Set2', legend=False)
plt.title('Total Revenue by Store')
plt.ylabel('Revenue ($)')
plt.xlabel('Store')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'chart1_revenue_by_store.png'))
plt.close()
print("Chart 1 saved.")

# --- Chart 2: Transactions by Hour ---
plt.figure(figsize=(10, 5))
hourly = df.groupby('hour')['transaction_id'].count()
sns.lineplot(x=hourly.index, y=hourly.values, marker='o', color='steelblue')
plt.title('Number of Transactions by Hour of Day')
plt.ylabel('Transaction Count')
plt.xlabel('Hour')
plt.xticks(range(6, 21))
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'chart2_transactions_by_hour.png'))
plt.close()
print("Chart 2 saved.")

# --- Chart 3: Hourly Revenue by Store ---
hourly_rev = df.groupby(['hour', 'store_location'])['revenue'].sum().reset_index()
plt.figure(figsize=(10, 5))
for store in df['store_location'].unique():
    subset = hourly_rev[hourly_rev['store_location'] == store]
    plt.plot(subset['hour'], subset['revenue'], marker='o', label=store)
plt.title('Hourly Revenue by Store')
plt.ylabel('Revenue ($)')
plt.xlabel('Hour of Day')
plt.legend()
plt.xticks(range(6, 21))
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'chart3_hourly_revenue_by_store.png'))
plt.close()
print("Chart 3 saved.")

# --- Chart 4: Top 10 Products by Revenue ---
plt.figure(figsize=(10, 6))
top_products = df.groupby('product_type')['revenue'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_products.values, y=top_products.index, hue=top_products.index, palette='Blues_r', legend=False)
plt.title('Top 10 Product Types by Revenue')
plt.xlabel('Revenue ($)')
plt.ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'chart4_top_products.png'))
plt.close()
print("Chart 4 saved.")

# --- Chart 5: Revenue by Product Category ---
plt.figure(figsize=(8, 5))
cat_rev = df.groupby('product_category')['revenue'].sum().sort_values(ascending=False)
sns.barplot(x=cat_rev.values, y=cat_rev.index, hue=cat_rev.index, palette='Oranges_r', legend=False)
plt.title('Revenue by Product Category')
plt.xlabel('Revenue ($)')
plt.ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'chart5_revenue_by_category.png'))
plt.close()
print("Chart 5 saved.")

# --- Chart 6: Transaction Qty Distribution ---
plt.figure(figsize=(7, 4))
sns.countplot(x='transaction_qty', data=df, hue='transaction_qty', palette='Set2', legend=False)
plt.title('Distribution of Transaction Quantity')
plt.xlabel('Quantity per Transaction')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'chart6_qty_distribution.png'))
plt.close()
print("Chart 6 saved.")

print("\nAll 6 charts saved to:", output_folder)
