from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import random

def evaluar_f1(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return f1_score(y_test, y_pred)


def generar_caso_de_uso_evaluar_f1():
    n = random.randint(50, 100)

    X = np.random.rand(n, 3)
    y = np.random.choice([0, 1], n)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    input_data = {"X": X.copy(), "y": y.copy()}
    output_data = f1

    return input_data, output_data


# EJEMPLO DE USO
entrada, salida_esperada = generar_caso_de_uso_evaluar_f1()

resultado = evaluar_f1(entrada["X"], entrada["y"])

print("\nF1 esperado:", salida_esperada)
print("F1 obtenido:", resultado)
