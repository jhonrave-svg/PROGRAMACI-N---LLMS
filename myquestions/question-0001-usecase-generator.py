import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import random

def transformar_datos(df):
    X = df.copy()

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    transformer = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    return transformer.fit_transform(X)


def generar_caso_de_uso_transformar_datos():
    n_rows = random.randint(5, 10)

    df = pd.DataFrame({
        "edad": np.random.randint(18, 70, n_rows),
        "ingresos": np.random.randint(1000, 5000, n_rows),
        "genero": np.random.choice(["M", "F"], n_rows)
    })

    num_cols = ["edad", "ingresos"]
    cat_cols = ["genero"]

    transformer = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X_expected = transformer.fit_transform(df)

    input_data = {"df": df.copy()}
    output_data = X_expected

    return input_data, output_data


# EJEMPLO DE USO
entrada, salida_esperada = generar_caso_de_uso_transformar_datos()

print("Input:")
print(entrada["df"])

resultado = transformar_datos(entrada["df"])

print("\nOutput esperado (shape):", salida_esperada.shape)
print("Output obtenido (shape):", resultado.shape)
