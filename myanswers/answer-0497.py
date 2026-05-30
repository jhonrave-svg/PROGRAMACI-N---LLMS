import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def ajustar_regresion_polinomica(df, target_col, grado):

    # Separar variables predictoras y objetivo
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # Generar características polinómicas
    poly = PolynomialFeatures(
        degree=grado,
        include_bias=False
    )
    X_poly = poly.fit_transform(X)

    # Entrenar modelo
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predicciones
    predicciones = model.predict(X_poly)

    # Coeficientes
    coeficientes = model.coef_

    return predicciones, coeficientes
