import pandas as pd

def analizar_estres_hidrico(df):

    # 1. Convertir timestamp a datetime
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2. Crear columna periodo
    def clasificar_periodo(hora):
        if 0 <= hora <= 5:
            return "Madrugada"
        elif 6 <= hora <= 17:
            return "Cénit"
        else:
            return "Ocaso"

    df["periodo"] = df["timestamp"].dt.hour.apply(clasificar_periodo)

    # 3. Filtrar registros con ndvi > 0.6
    df_filtrado = df[df["ndvi"] > 0.6]

    # 4. Promedio de humedad agrupado por lote_id y periodo
    resultado = df_filtrado.groupby(
        ["lote_id", "periodo"]
    )["humedad"].mean()

    return resultado
