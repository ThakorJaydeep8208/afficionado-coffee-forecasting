import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Afficionado Coffee Roasters", page_icon="☕", layout="wide")
st.title("☕ Afficionado Coffee Roasters — Demand Forecasting Dashboard")
st.markdown("---")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\project\Afficionado Coffee Roasters.xlsx - Transactions.csv")
    df['transaction_time'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S')
    df['hour'] = df['transaction_time'].dt.hour
    df['revenue'] = df['transaction_qty'] * df['unit_price']
    np.random.seed(42)
    date_range = pd.date_range(start='2025-01-01', end='2025-06-30', freq='D')
    df['transaction_date'] = np.random.choice(date_range, size=len(df))
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    return df

df = load_data()

# --- Sidebar ---
st.sidebar.header("Controls")
store = st.sidebar.selectbox("Select Store", df['store_location'].unique())
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 30, 14)
metric = st.sidebar.radio("View by", ["Revenue ($)", "Transaction Count"])

# --- KPI Cards ---
store_df = df[df['store_location'] == store]
total_rev = store_df['revenue'].sum()
total_txn = store_df['transaction_id'].count()
avg_order = store_df['revenue'].mean()
top_product = store_df.groupby('product_type')['revenue'].sum().idxmax()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${total_rev:,.0f}")
col2.metric("Total Transactions", f"{total_txn:,}")
col3.metric("Avg Order Value", f"${avg_order:.2f}")
col4.metric("Top Product", top_product)

st.markdown("---")

# --- Hourly Heatmap ---
st.subheader(f"🕐 Hourly Demand Heatmap — {store}")
hourly = store_df.groupby(['transaction_date', 'hour'])['revenue'].sum().reset_index()
hourly['day_of_week'] = pd.to_datetime(hourly['transaction_date']).dt.day_name()
heatmap_data = hourly.groupby(['day_of_week', 'hour'])['revenue'].mean().reset_index()
heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='revenue')
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])
fig_heat = px.imshow(heatmap_pivot, color_continuous_scale='YlOrRd',
                     labels=dict(x="Hour of Day", y="Day of Week", color="Avg Revenue ($)"),
                     title="Average Revenue by Day & Hour")
st.plotly_chart(fig_heat, use_container_width=True)

# --- Forecast ---
st.subheader(f"📈 Sales Forecast — Next {forecast_days} Days")

daily = df.groupby(['transaction_date', 'store_location']).agg(
    daily_revenue=('revenue', 'sum'),
    daily_transactions=('transaction_id', 'count')
).reset_index()

store_daily = daily[daily['store_location'] == store].sort_values('transaction_date').reset_index(drop=True)
store_daily['lag_1'] = store_daily['daily_revenue'].shift(1)
store_daily['lag_7'] = store_daily['daily_revenue'].shift(7)
store_daily['rolling_7'] = store_daily['daily_revenue'].shift(1).rolling(7).mean()
store_daily['day_of_week'] = pd.to_datetime(store_daily['transaction_date']).dt.dayofweek
store_daily = store_daily.dropna().reset_index(drop=True)

target_col = 'daily_revenue' if metric == "Revenue ($)" else 'daily_transactions'
y = store_daily[target_col].values
n = len(y)
split = int(n * 0.8)
train = y[:split]

# Exponential Smoothing forecast
es_model = ExponentialSmoothing(train, trend='add', seasonal=None)
es_fit = es_model.fit()
es_full_pred = es_fit.forecast(n - split + forecast_days)
future_dates = pd.date_range(
    start=store_daily['transaction_date'].iloc[split],
    periods=len(es_full_pred), freq='D')

# Confidence intervals (simple ±10%)
upper = es_full_pred * 1.10
lower = es_full_pred * 0.90

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(
    x=store_daily['transaction_date'], y=y,
    name='Actual', line=dict(color='black', width=2)))
fig_forecast.add_trace(go.Scatter(
    x=future_dates, y=es_full_pred,
    name='Forecast', line=dict(color='green', width=2, dash='dash')))
fig_forecast.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates[::-1]),
    y=list(upper) + list(lower[::-1]),
    fill='toself', fillcolor='rgba(0,200,100,0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Confidence Interval'))
fig_forecast.update_layout(
    xaxis_title='Date', yaxis_title=metric,
    legend=dict(orientation='h'), hovermode='x unified')
st.plotly_chart(fig_forecast, use_container_width=True)

# --- Top Products ---
st.subheader(f"🏆 Top Products — {store}")
top_p = store_df.groupby('product_type')['revenue'].sum().sort_values(ascending=False).head(10).reset_index()
fig_products = px.bar(top_p, x='revenue', y='product_type', orientation='h',
                      color='revenue', color_continuous_scale='Blues',
                      labels={'revenue': 'Revenue ($)', 'product_type': ''},
                      title="Top 10 Products by Revenue")
fig_products.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig_products, use_container_width=True)

# --- Model Comparison ---
st.subheader("📊 Model Performance Comparison")
try:
    results = pd.read_csv(r"D:\project\model_results.csv")
    fig_comp = px.bar(results, x='Store', y=['Naive MAE', 'MovingAvg MAE', 'ExpSmoothing MAE', 'GradientBoosting MAE'],
                      barmode='group', title='MAE by Model and Store (lower = better)',
                      labels={'value': 'MAE ($)', 'variable': 'Model'})
    st.plotly_chart(fig_comp, use_container_width=True)
except:
    st.info("Run models.py first to see model comparison.")

st.markdown("---")
st.caption("Afficionado Coffee Roasters — Data-Driven Forecasting Dashboard | Built with Streamlit")