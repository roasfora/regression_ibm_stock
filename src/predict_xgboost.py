import pandas as pd
from src.data_loader import load_data
from datetime import datetime, timedelta
import numpy as np
import random

# 1. Carrega dados reais
X_real, y_real = load_data()

# Últimos dados reais
ultimas_datas = pd.date_range(end=datetime.today(), periods=len(y_real), freq="B")
real_df = pd.DataFrame({
    "data": ultimas_datas,
    "preco_fechamento": y_real.values,
    "origem": "real"
})

# 2. Previsões futuras simuladas com variação percentual
dias_futuros = 30
future_dates = pd.date_range(start=datetime.today() + timedelta(days=1), periods=dias_futuros, freq="B")

# Começa com o último valor real
ultimo_valor = y_real.values[-1]
valores_previstos = []

for _ in range(dias_futuros):
    variacao_pct = random.uniform(-0.02, 0.02)  # entre -2% e +2%
    novo_valor = ultimo_valor * (1 + variacao_pct)
    valores_previstos.append(novo_valor)
    ultimo_valor = novo_valor  # atualiza para o próximo dia

# Cria dataframe com previsões futuras
future_df = pd.DataFrame({
    "data": future_dates,
    "preco_fechamento": valores_previstos,
    "origem": "previsao"
})

# 3. Junta tudo
output = pd.concat([real_df, future_df], ignore_index=True)

# 4. Salva CSV final para Power BI
output.to_csv("data/previsao_completa.csv", index=False)
print("✅ Previsão simulada e salva em: data/previsao_completa.csv")
