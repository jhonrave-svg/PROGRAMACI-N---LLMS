import pandas as pd
import numpy as np

def analizar_estres_hidrico(df):

    df = df.copy()

    # 1. Convertir timestamp a datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2. Clasificar periodo
    condiciones = [
        df["timestamp"].dt.hour.between(0, 5),
        df["timestamp"].dt.hour.between(6, 17),
        df["timestamp"].dt.hour.between(18, 23)
    ]

    opciones = ["Madrugada", "Cénit", "Ocaso"]

    df["periodo"] = np.select(condiciones, opciones)

    # 3. Filtrar ndvi > 0.6
    df = df[df["ndvi"] > 0.6]

    # 4. Promedio de humedad por lote y periodo
    resultado = df.groupby(
        ["lote_id", "periodo"]
    )["humedad"].mean()

    return resultado
