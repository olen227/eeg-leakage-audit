"""Метрики оператора (ROC-AUC, EER, калибровка порога) и детектора (F1)."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

def pair_distances(emb, pair_idx):
    a = emb[pair_idx[:, 0]]; b = emb[pair_idx[:, 1]]
    return np.linalg.norm(a - b, axis=1)

def roc_auc_from_scores(same_labels, dist):
    """same_labels=1 если пара одного субъекта. Оператор: чем меньше dist, тем ближе.
    Скор разделимости = -dist (больше => вероятнее один субъект)."""
    return roc_auc_score(same_labels, -dist)

def eer_from_scores(same_labels, dist):
    """EER по ROC-кривой скора -dist."""
    fpr, tpr, thr = roc_curve(same_labels, -dist)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[i] + fnr[i]) / 2
    return float(eer), float(-thr[i])  # порог по dist

def calibrate_threshold(intra_dist, inter_dist, alpha=1.0, beta=1.0):
    """Формула (2.10): tau* = argmin[alpha*FAR + beta*FRR].
    FAR — доля межсубъектных пар принятых за одного (dist<tau).
    FRR — доля внутрисубъектных пар отклонённых (dist>=tau)."""
    cand = np.unique(np.concatenate([intra_dist, inter_dist]))
    best_tau, best_cost = None, np.inf
    for tau in cand:
        far = np.mean(inter_dist < tau) if len(inter_dist) else 0.0
        frr = np.mean(intra_dist >= tau) if len(intra_dist) else 0.0
        cost = alpha * far + beta * frr
        if cost < best_cost:
            best_cost, best_tau = cost, tau
    return float(best_tau), float(best_cost)

def f1_binary(y_true, y_pred):
    return float(f1_score(y_true, y_pred, zero_division=0))
