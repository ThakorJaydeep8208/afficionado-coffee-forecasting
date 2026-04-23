import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings, os
warnings.filterwarnings('ignore')

df = pd.read_csv(r"D:\project\daily_features.csv")
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
output_folder = r"D:\project\charts"
os.makedirs(output_folder, exist_ok=True)

results = []

for store in ['store_Astoria', "store_Hell's Kitchen", 'store_Lower Manhattan']:
    store_name = store.replace('store_', '')
    store_df = df[df[store] == True].sort_values('transaction_date').reset_index(drop=True)
    
    revenue = store_df['daily_revenue'].values
    n = len(revenue)
    split = int(n * 0.8)
    
    train, test = revenue[:split], revenue[split:]
    test_dates = store_df['transaction_date'].values[split:]
    
    # --- Model 1: Naive (last value) ---
    naive_pred = np.full(len(test), train[-1])
    naive_mae = mean_absolute_error(test, naive_pred)
    naive_rmse = np.sqrt(mean_squared_error(test, naive_pred))
    
    # --- Model 2: Moving Average ---
    ma_pred = np.full(len(test), np.mean(train[-7:]))
    ma_mae = mean_absolute_error(test, ma_pred)
    ma_rmse = np.sqrt(mean_squared_error(test, ma_pred))
    
    # --- Model 3: Exponential Smoothing ---
    es_model = ExponentialSmoothing(train, trend='add', seasonal=None)
    es_fit = es_model.fit()
    es_pred = es_fit.forecast(len(test))
    es_mae = mean_absolute_error(test, es_pred)
    es_rmse = np.sqrt(mean_squared_error(test, es_pred))
    
    # --- Model 4: Gradient Boosting ---
    features = ['lag_1', 'lag_7', 'rolling_3', 'rolling_7', 'day_of_week']
    X = store_df[features].values
    y = store_df['daily_revenue'].values
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    
    results.append({
        'Store': store_name,
        'Naive MAE': round(naive_mae, 2),
        'MovingAvg MAE': round(ma_mae, 2),
        'ExpSmoothing MAE': round(es_mae, 2),
        'GradientBoosting MAE': round(gb_mae, 2),
        'Best Model': min([
            ('Naive', naive_mae),
            ('Moving Average', ma_mae),
            ('Exp Smoothing', es_mae),
            ('Gradient Boosting', gb_mae)
        ], key=lambda x: x[1])[0]
    })
    
    # --- Plot for this store ---
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, test, label='Actual', color='black', linewidth=2)
    plt.plot(test_dates, naive_pred, label='Naive', linestyle='--', alpha=0.7)
    plt.plot(test_dates, ma_pred, label='Moving Average', linestyle='--', alpha=0.7)
    plt.plot(test_dates, es_pred, label='Exp Smoothing', linestyle='--', alpha=0.7)
    plt.plot(test_dates, gb_pred, label='Gradient Boosting', linestyle='-', color='green', linewidth=2)
    plt.title(f'Forecast vs Actual — {store_name}')
    plt.ylabel('Daily Revenue ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.tight_layout()
    safe_name = store_name.replace("'", "").replace(" ", "_")
    plt.savefig(os.path.join(output_folder, f'forecast_{safe_name}.png'))
    plt.close()
    print(f"Chart saved for {store_name}")

# --- Print results table ---
results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("MODEL COMPARISON (MAE = lower is better)")
print("=" * 70)
print(results_df.to_string(index=False))
results_df.to_csv(r"D:\project\model_results.csv", index=False)
print("\nResults saved to model_results.csv")