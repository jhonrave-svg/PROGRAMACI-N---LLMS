import pandas as pd

def analizar_estres_hidrico(df):

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    horas = df["timestamp"].dt.hour

    df["periodo"] = "Ocaso"
    df.loc[horas < 6, "periodo"] = "Madrugada"
    df.loc[(horas >= 6) & (horas < 18), "periodo"] = "Cénit"

    df = df[df["ndvi"] > 0.6]

    return df.groupby(
        ["lote_id", "periodo"]
    )["humedad"].mean()
