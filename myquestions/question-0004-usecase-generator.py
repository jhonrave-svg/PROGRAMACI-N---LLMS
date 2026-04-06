from sklearn.decomposition import PCA
import numpy as np
import random

def reducir_dimensionalidad(X, n_componentes):
    pca = PCA(n_components=n_componentes)
    return pca.fit_transform(X)


def generar_caso_de_uso_reducir_dimensionalidad():
    n = random.randint(20, 50)
    n_features = random.randint(4, 6)
    n_components = random.randint(2, n_features-1)

    X = np.random.rand(n, n_features)

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    input_data = {
        "X": X.copy(),
        "n_componentes": n_components
    }

    output_data = X_reduced

    return input_data, output_data


# EJEMPLO DE USO
entrada, salida_esperada = generar_caso_de_uso_reducir_dimensionalidad()

resultado = reducir_dimensionalidad(
    entrada["X"],
    entrada["n_componentes"]
)

print("\nShape esperado:", salida_esperada.shape)
print("Shape obtenido:", resultado.shape)
