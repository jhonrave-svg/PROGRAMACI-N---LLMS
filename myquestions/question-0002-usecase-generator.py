import pandas as pd
import numpy as np
import random

def detectar_outliers_iqr(df, col):
    df = df.copy()

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df["es_outlier"] = (df[col] < lower) | (df[col] > upper)

    return df


def generar_caso_de_uso_detectar_outliers_iqr():
    n = random.randint(10, 20)

    data = np.random.normal(50, 10, n)

    data[random.randint(0, n-1)] = 200
    data[random.randint(0, n-1)] = -50

    df = pd.DataFrame({"valores": data})

    Q1 = df["valores"].quantile(0.25)
    Q3 = df["valores"].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_expected = df.copy()
    df_expected["es_outlier"] = (df["valores"] < lower) | (df["valores"] > upper)

    input_data = {"df": df.copy(), "col": "valores"}
    output_data = df_expected

    return input_data, output_data


# EJEMPLO DE USO
entrada, salida_esperada = generar_caso_de_uso_detectar_outliers_iqr()

print("\nInput:")
print(entrada["df"])

resultado = detectar_outliers_iqr(entrada["df"], entrada["col"])

print("\nResultado:")
print(resultado.head())
