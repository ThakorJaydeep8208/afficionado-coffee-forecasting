import pandas as pd

df = pd.read_csv(r"D:\project\Afficionado Coffee Roasters.xlsx - Transactions.csv")

# --- Step 1: Parse time and create date/hour columns ---
df['transaction_time'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S')
df['hour'] = df['transaction_time'].dt.hour
df['revenue'] = df['transaction_qty'] * df['unit_price']

# Assign a fake date (Jan 1 to Jun 30, 2025) based on row order
# since the dataset only has time, not date
import numpy as np
np.random.seed(42)
date_range = pd.date_range(start='2025-01-01', end='2025-06-30', freq='D')
df['transaction_date'] = np.random.choice(date_range, size=len(df))
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['day_of_week'] = df['transaction_date'].dt.dayofweek   # 0=Mon, 6=Sun
df['day_name'] = df['transaction_date'].dt.day_name()
df['month'] = df['transaction_date'].dt.month
df['week'] = df['transaction_date'].dt.isocalendar().week.astype(int)

# --- Step 2: Aggregate to daily revenue per store ---
daily = df.groupby(['transaction_date', 'store_location']).agg(
    daily_revenue=('revenue', 'sum'),
    daily_transactions=('transaction_id', 'count')
).reset_index()

daily = daily.sort_values(['store_location', 'transaction_date'])

# --- Step 3: Lag features ---
daily['lag_1'] = daily.groupby('store_location')['daily_revenue'].shift(1)
daily['lag_7'] = daily.groupby('store_location')['daily_revenue'].shift(7)

# --- Step 4: Rolling averages ---
daily['rolling_3'] = daily.groupby('store_location')['daily_revenue'].transform(
    lambda x: x.shift(1).rolling(3).mean())
daily['rolling_7'] = daily.groupby('store_location')['daily_revenue'].transform(
    lambda x: x.shift(1).rolling(7).mean())

# --- Step 5: Day of week from date ---
daily['day_of_week'] = daily['transaction_date'].dt.dayofweek

# --- Step 6: Store dummy variables ---
daily = pd.get_dummies(daily, columns=['store_location'], prefix='store')

# --- Step 7: Drop rows with NaN from lag features ---
daily_clean = daily.dropna().reset_index(drop=True)

# --- Save ---
daily_clean.to_csv(r"D:\project\daily_features.csv", index=False)

print("Feature engineering complete!")
print(f"Shape: {daily_clean.shape}")
print(f"\nColumns created:\n{list(daily_clean.columns)}")
print(f"\nFirst 3 rows preview:")
print(daily_clean.head(3).to_string())