import pandas as pd

def analizar_estres_hidrico(df):

    df = df.copy()

    # 1. Convertir timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2. Clasificar periodo
    horas = df["timestamp"].dt.hour

    df["periodo"] = "Ocaso"
    df.loc[horas < 6, "periodo"] = "Madrugada"
    df.loc[(horas >= 6) & (horas < 18), "periodo"] = "Cénit"

    # 3. Filtrar NDVI
    df = df[df["ndvi"] > 0.6]

    # 4. Agrupar y devolver DataFrame
    resultado = (
        df.groupby(["lote_id", "periodo"])[["humedad"]]
        .mean()
    )

    return resultado
