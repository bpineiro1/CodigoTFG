import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    ndcg_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
)
from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score

def recall_at_k(y_true, y_prob, k: int = 10):
    sort = y_prob.argsort()[::-1][:k]
    pos_sum = np.sum(y_true)
    if pos_sum == 0:
        return 0.0
    return float(np.sum(y_true[sort]) / pos_sum)

def precision_at_k(y_true, y_prob, k: int = 10):
    sort = y_prob.argsort()[::-1][:k]
    return float(np.sum(y_true[sort]) / k)

def best_threshold(y_true: np.ndarray, y_prob: np.ndarray, criterion: str = "gmean") -> float:
    """Umbral de decisión que maximiza `criterion` ("gmean" o "f1") sobre unas predicciones
    de validación. Se usa en el bucle interno, nunca sobre el test."""
    grid = np.linspace(0.01, 0.99, 99)
    scores = []
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        if criterion == "f1":
            scores.append(f1_score(y_true, y_pred, zero_division=0))
        else:
            scores.append(geometric_mean_score(y_true, y_pred))
    return float(grid[int(np.argmax(scores))])

def predict_top_fraction(y_prob: np.ndarray, fraction: float) -> np.ndarray:
    """Marca como positivos el `fraction` de ejemplos con mayor puntuación.
    Los empates se resuelven de forma determinista (orden estable)."""
    k = int(round(fraction * len(y_prob)))
    k = min(max(k, 1), len(y_prob))
    order = np.argsort(-y_prob, kind="stable")
    y_pred = np.zeros(len(y_prob), dtype=int)
    y_pred[order[:k]] = 1
    return y_pred

def calculate_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, y_pred=None) -> dict:
    """Las métricas de umbral (sensibilidad, especificidad, precisión, F1, G-Mean) usan
    `threshold`, o directamente `y_pred` si se proporciona; AUC-ROC, AUC-PR y las
    métricas @10 no dependen del umbral."""
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)

    test_sens = sensitivity_score(y_true, y_pred)
    test_spec = specificity_score(y_true, y_pred)
    test_gmean = geometric_mean_score(y_true, y_pred)
    
    try:
        test_auc_roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        test_auc_roc = 0.5

    test_precision = precision_score(y_true, y_pred, zero_division=0)
    test_f1 = f1_score(y_true, y_pred, zero_division=0)

    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    test_auc_pr = auc(recalls, precisions)

    test_ndcg_at_k = ndcg_score(y_true.reshape(1, -1), y_prob.reshape(1, -1), k=10)
    test_recall_at_k = recall_at_k(y_true, y_prob, k=10)
    test_precision_at_k = precision_at_k(y_true, y_prob, k=10)

    return {
        "sensitivity": test_sens,
        "specificity": test_spec,
        "precision": test_precision,
        "f1": test_f1,
        "gmean": test_gmean,
        "auc_roc": test_auc_roc,
        "auc_pr": test_auc_pr,
        "recall_at_10": test_recall_at_k,
        "ndcg_at_10": test_ndcg_at_k,
        "precision_at_10": test_precision_at_k,
    }
