import pandas as pd

def analizar_estres_hidrico(df):

    df = df.copy()

    # Convertir timestamp a datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Clasificar periodo
    horas = df["timestamp"].dt.hour

    df["periodo"] = "Ocaso"
    df.loc[horas < 6, "periodo"] = "Madrugada"
    df.loc[(horas >= 6) & (horas < 18), "periodo"] = "Cénit"

    # Filtrar ndvi > 0.6
    df = df[df["ndvi"] > 0.6]

    # Promedio de humedad por lote y periodo
    resultado = (
        df.groupby(["lote_id", "periodo"])[["humedad"]]
        .mean()
    )

    return resultado
