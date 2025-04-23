# src/data_loader.py
import pandas as pd

def load_data(path="data/ibm_daily_raw.csv"):
    df = pd.read_csv(path)

    # Ajusta nomes e tipos
    df.rename(columns={
        "1. open": "preco_abertura",
        "2. high": "preco_maximo",
        "3. low": "preco_minimo",
        "4. close": "preco_fechamento",
        "5. volume": "volume"
    }, inplace=True)

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values("date", inplace=True)

    # Features temporais
    df['ano'] = df['date'].dt.year
    df['mes'] = df['date'].dt.month
    df['dia'] = df['date'].dt.day
    df['dia_semana'] = df['date'].dt.weekday

    # Define X e y
    features = [
        "preco_abertura", "preco_maximo", "preco_minimo", "volume",
        "ano", "mes", "dia", "dia_semana"
    ]
    df = df.dropna()

    X = df[features]
    y = df["preco_fechamento"]

    return X, y
