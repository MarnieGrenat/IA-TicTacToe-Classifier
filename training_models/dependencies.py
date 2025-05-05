import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve, auc
)

from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve


def test_model(model, X_test, y_test, class_names=None):
    """
    Avalia um modelo MLP multiclass (4 categorias) para o projeto TicTacToe.

    Parâmetros
    ----------
    model : estimator já treinado com método predict() e predict_proba()
    X_test : array-like ou DataFrame — features de teste
    y_test : array-like — labels de teste (inteiros ou strings)
    class_names : list of str, opcional — nomes das classes na ordem dos rótulos
    """
    # Prepare dados
    X = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
    y = np.asarray(y_test)
    if class_names is None:
        class_names = list(model.classes_)

    # 1) Previsões e tempo de inferência
    t0 = time.time()
    y_pred = model.predict(X)
    t1 = time.time()
    print(f"[INFO] Tempo de inferência: {t1 - t0:.4f} s\n")

    # 2) Relatório de classificação
    print("="*10, "CLASSIFICATION REPORT", "="*10)
    print(classification_report(y, y_pred, target_names=class_names, digits=4))

    # 3) Matriz de Confusão
    print("\n", "="*10, "CONFUSION MATRIX", "="*10)
    disp = ConfusionMatrixDisplay.from_predictions(
        y, y_pred,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=None
    )
    disp.ax_.set_title("Matriz de Confusão")
    plt.show()

    # 4) Curvas ROC One-vs-Rest
    print("\n", "="*10, "ROC CURVES (OVR)", "="*10)
    y_bin = label_binarize(y, classes=model.classes_)
    proba = model.predict_proba(X)
    n_classes = y_bin.shape[1]

    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr,
                 label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0,1],[0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves One-vs-Rest")
    plt.legend(loc="lower right")
    plt.show()

    # 5) Curvas de Calibração por classe
    print("\n", "="*10, "CALIBRATION CURVES", "="*10)
    plt.figure()
    for i in range(n_classes):
        prob_true, prob_pred = calibration_curve(y_bin[:, i], proba[:, i], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=class_names[i])

    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel("Probabilidade média prevista")
    plt.ylabel("Fração real de positivos")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.show()
