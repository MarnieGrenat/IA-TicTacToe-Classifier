import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time


def test(model, X_test, y_test):
    test_start = time.time()
    pred = model.predict(X_test.to_numpy())
    test_end = time.time()

    # 2) Calcular e exibir métricas de classificação
    print(10 * '=' + ' MÉTRICAS ' + 10 * '=')
    print(f"Tempo de Execução de Testes = {test_end - test_start:.4f} segundos")
    print(classification_report(y_test, pred, digits=4))

    print(10 * '=' + ' ANÁLISE DE RESULTADOS ' + 10 * '=')

    cm_display = ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=['X_WIN', 'DRAW', 'ONGOING', 'O_WIN'])
    cm_display.plot()
    plt.title("Matriz de Confusão")
    plt.show()