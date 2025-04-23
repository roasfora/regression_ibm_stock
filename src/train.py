import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from src.data_loader import load_data
import numpy as np

mlflow.set_experiment("Regressao_IBM_Local")

def eval_and_log(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Métricas
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        # Log no MLflow
        mlflow.log_param("modelo", model_name)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(model, "modelo")

        print(f"[{model_name}] MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

        return {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelos = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Ridge": Ridge(alpha=1.0),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }

    resultados = []

    for nome, modelo in modelos.items():
        resultado = eval_and_log(modelo, nome, X_train, X_test, y_train, y_test)
        resultados.append(resultado)

    # Encontrar o melhor modelo (com menor MAE)
    melhor = sorted(resultados, key=lambda x: x["mae"])[0]
    print(f"\n🔝 Melhor modelo: {melhor['model_name']} (MAE: {melhor['mae']:.2f})")

    # Registrar o melhor modelo
    mlflow.register_model(
        model_uri=f"runs:/{melhor['run_id']}/modelo",
        name="RegressaoIBMModel"
    )
    print(" Modelo registrado como: RegressaoIBMModel")

if __name__ == "__main__":
    train()
