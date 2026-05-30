import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
 
 
def analizar_estres_hidrico(df):
    """
    Analiza el estrés hídrico de cultivos a partir de datos de drones.
 
    Parámetros:
        df (pd.DataFrame): DataFrame con columnas lote_id, humedad, ndvi, timestamp.
 
    Retorna:
        pd.Series: Promedio de humedad agrupado por lote_id y periodo (índice multinivel).
    """
 
    # 1. Convertir la columna timestamp a objetos datetime
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
 
    # 2. Clasificar el momento de medición en 'periodo' según la hora
    def clasificar_periodo(hora):
        if 0 <= hora <= 5:
            return 'Madrugada'
        elif 6 <= hora <= 17:
            return 'Cénit'
        else:
            return 'Ocaso'
 
    df['periodo'] = df['timestamp'].dt.hour.apply(clasificar_periodo)
 
    # 3. Filtrar solo registros con ndvi > 0.6 (plantas sanas)
    df_sano = df[df['ndvi'] > 0.6]
 
    # 4. Retornar la Serie con promedio de humedad agrupado por lote_id y periodo
    resultado = df_sano.groupby(['lote_id', 'periodo'])['humedad'].mean()
 
    return resultado
 
 
# ─────────────────────────────────────────────
# Generador del caso de uso (código original)
# ─────────────────────────────────────────────
 
def generar_caso_de_uso_analizar_estres_hidrico(n_registros=1000):
    lotes = [f'LOTE_{str(i).zfill(2)}' for i in range(1, 11)]
    data = {
        'lote_id': [random.choice(lotes) for _ in range(n_registros)],
        'humedad': [round(random.uniform(5.0, 60.0), 2) for _ in range(n_registros)],
        'ndvi': [round(random.uniform(0.1, 0.95), 2) for _ in range(n_registros)],
        'timestamp': [
            (datetime(2024, 5, 1) + timedelta(
                days=random.randint(0, 15),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )).strftime('%Y-%m-%d %H:%M:%S')
            for _ in range(n_registros)
        ]
    }
    df = pd.DataFrame(data)
    return {"df": df}, df
 
 
# ─────────────────────────────────────────────
# Ejecución y validación
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    inputs, expected_output = generar_caso_de_uso_analizar_estres_hidrico(1000)
    df_input = inputs['df']
 
    print("=" * 55)
    print("Dataset generado exitosamente.")
    print(f"Filas totales: {len(df_input)}")
    print("\nPrimeras filas del DataFrame:")
    print(df_input.head(5))
 
    print("\n" + "=" * 55)
    print("Ejecutando analizar_estres_hidrico()...")
    resultado = analizar_estres_hidrico(df_input)
 
    print("\nTipo del resultado:", type(resultado))
    print("Nombre del índice:", resultado.index.names)
    print(f"\nRegistros con ndvi > 0.6: {len(df_input[df_input['ndvi'] > 0.6])}")
    print(f"Combinaciones lote/periodo en resultado: {len(resultado)}")
 
    print("\nPrimeras filas del resultado (promedio humedad):")
    print(resultado.head(15).round(2))
 
    print("\n" + "=" * 55)
    print("Verificaciones:")
    assert isinstance(resultado, pd.Series), "❌ El resultado debe ser una pd.Series"
    assert resultado.index.names == ['lote_id', 'periodo'], "❌ El índice debe ser ['lote_id', 'periodo']"
    periodos_validos = {'Madrugada', 'Cénit', 'Ocaso'}
    periodos_encontrados = set(resultado.index.get_level_values('periodo'))
    assert periodos_encontrados.issubset(periodos_validos), f"❌ Periodos inesperados: {periodos_encontrados}"
    print("✅ Tipo correcto: pd.Series")
    print("✅ Índice multinivel: ['lote_id', 'periodo']")
    print(f"✅ Periodos encontrados: {periodos_encontrados}")
    print("✅ Todas las validaciones pasaron")
