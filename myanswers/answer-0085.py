import pandas as pd

def analizar_estres_hidrico(df):

    df = df.copy()

    # Convertir timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Crear periodo
    horas = df["timestamp"].dt.hour

    df["periodo"] = "Ocaso"
    df.loc[horas < 6, "periodo"] = "Madrugada"
    df.loc[(horas >= 6) & (horas < 18), "periodo"] = "Cénit"

    # Filtrar ndvi
    df = df[df["ndvi"] > 0.6]

    # Devolver DataFrame en lugar de Serie
    resultado = (
        df.groupby(["lote_id", "periodo"])[["humedad"]]
        .mean()
    )

    return resultado
