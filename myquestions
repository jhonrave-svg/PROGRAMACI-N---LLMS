import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import random

def generar_caso_de_uso_preparar_datos():
    """
    Genera un caso de uso aleatorio para una función que separa X de y,
    imputa valores nulos con la media y escala los datos.
    """
    # 1. Configuración aleatoria
    n_rows = random.randint(5, 10)
    n_cols = random.randint(2, 4)
    target_name = random.choice(['target', 'label', 'clase'])
    
    # 2. Crear datos sintéticos con nulos
    data = np.random.rand(n_rows, n_cols) * 100
    df = pd.DataFrame(data, columns=[f'feat_{i}' for i in range(n_cols)])
    
    # Insertar algunos NaNs aleatorios
    for _ in range(2):
        df.iloc[random.randint(0, n_rows-1), random.randint(0, n_cols-1)] = np.nan
        
    # Crear la columna objetivo (y)
    df[target_name] = np.random.randint(0, 2, size=n_rows)
    
    # 3. Calcular el OUTPUT esperado (Lógica de la función preparar_datos)
    X_raw = df.drop(columns=[target_name])
    y_expected = df[target_name].values
    
    # Imputación
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_raw)
    
    # Escalado
    scaler = StandardScaler()
    X_expected = scaler.fit_transform(X_imputed)
    
    # 4. Formatear Retorno
    input_data = {
        "df": df.copy(),
        "target_col": target_name
    }
    
    output_data = {
        "X": X_expected,
        "y": y_expected
    }
    
    return input_data, output_data

# Ejemplo de ejecución:
# entrada, salida = generar_caso_de_uso_preparar_datos()
# print("Input DF:\n", entrada['df'])
# print("\nOutput X (Escalado):\n", salida['X'])
