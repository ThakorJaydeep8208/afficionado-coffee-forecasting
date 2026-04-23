# ☕ Afficionado Coffee Roasters — Demand Forecasting

Data-Driven Forecasting & Peak Demand Prediction using Python, Machine Learning & Streamlit.

## 📁 Project Structure

| File | Description |
|------|-------------|
| `coffee_analysis.py` | Data loading & validation |
| `eda.py` | Exploratory Data Analysis — generates 6 charts |
| `feature_engineering.py` | Feature engineering — lag, rolling avg, dummies |
| `models.py` | 4 forecasting models + performance comparison |
| `app.py` | Interactive Streamlit dashboard |
| `report.py` | Generates the PDF research report |
| `Afficionado_Coffee_Report.pdf` | Final research report |

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install pandas matplotlib seaborn prophet scikit-learn statsmodels streamlit plotly reportlab
```

### 2. Run each script in order
```bash
python coffee_analysis.py
python eda.py
python feature_engineering.py
python models.py
python report.py
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```

## 📊 Models Used
- Naive Forecast
- Moving Average
- Exponential Smoothing
- Gradient Boosting Regression

## 🏆 Results

| Store | Best Model | MAE |
|-------|-----------|-----|
| Astoria | Exponential Smoothing | $72.74 |
| Hell's Kitchen | Gradient Boosting | $97.38 |
| Lower Manhattan | Exponential Smoothing | $65.87 |

## 🛠️ Tech Stack
Python · Pandas · Scikit-learn · Statsmodels · Streamlit · Plotly · Matplotlib · ReportLab

## 👤 Author
ThakorJaydeep8208
