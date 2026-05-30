import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def segmentar_productos_tienda(df, n_clusters):

    # 1. Eliminar filas con precio NaN o <= 0
    df = df[df["precio"].notna() & (df["precio"] > 0)].copy()

    # 2. Rellenar NaN de calificacion con la mediana
    df["calificacion"] = df["calificacion"].fillna(df["calificacion"].median())

    # 3. Rellenar NaN de n_ventas con 0
    df["n_ventas"] = df["n_ventas"].fillna(0)

    # 4. Crear ingreso_estimado
    df["ingreso_estimado"] = df["precio"] * df["n_ventas"]

    # 5. Seleccionar features
    features = df[
        ["precio", "n_ventas", "calificacion", "ingreso_estimado"]
    ]

    # 6. Escalar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # 7. Aplicar KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["segmento"] = kmeans.fit_predict(X_scaled)

    # 8. Reiniciar índice
    df = df.reset_index(drop=True)

    # 9. Perfil de segmentos
    perfil_segmentos = (
        df.groupby("segmento")[
            ["precio", "n_ventas", "calificacion", "ingreso_estimado"]
        ]
        .mean()
        .round(2)
    )

    return df, perfil_segmentos
