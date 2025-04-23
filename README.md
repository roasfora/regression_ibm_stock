# 📈 IBM Stock Price Regression

A simple regression pipeline to predict IBM's stock closing price using historical data. Includes preprocessing, model training, evaluation, and experiment tracking via MLflow.

---

## 🧠 Objective

Predict IBM’s **daily closing stock price** based on historical data and temporal features.

---

## ⚙️ How It Works

### `src/data_loader.py`
- Loads IBM historical stock data
- Renames columns and parses dates
- Adds temporal features (year, month, day, weekday)
- Returns `X` (features) and `y` (target: closing price)

---

### `train_and_log.py`
- Splits data into training and test sets
- Trains 4 models:
  - Linear Regression
  - Ridge Regression
  - Random Forest
  - XGBoost
- Evaluates and logs metrics to MLflow (MAE, RMSE, R²)
- Selects and registers the best model (lowest MAE)

```bash
python train_and_log.py
