"""Metrika, calibration və threshold köməkçi funksiyaları."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def tpr_at_fpr(y_true, y_score, fpr_target=0.05):
    """FPR <= fpr_target olan ən yüksək recall (BAF məqaləsinin əməliyyat metrikası)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    return float(tpr[max(idx, 0)])


def score_report(y_true, y_score):
    """3 əsas metrika: PR-AUC (əsas), ROC-AUC, TPR@5%FPR."""
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "tpr_at_5pct_fpr": tpr_at_fpr(y_true, y_score, 0.05),
    }


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE: kvantil bin-lərdə |real fraud faizi - orta proqnoz| çəkili ortası."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] / g["n"].sum() * (g["y"] - g["p"]).abs()).sum())


def candidate_thresholds(y_prob, n=300):
    """Skorların kvantillərindən threshold namizədləri (yuxarı 50%-lik hissə)."""
    return np.unique(np.quantile(y_prob, np.linspace(0.5, 0.9999, n)))


def operating_point(y_true, y_prob, threshold, c_fn, c_fp):
    """Verilən threshold-da confusion matrix, metrikalar və biznes xərci."""
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_prob) >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "fpr": fp / (fp + tn),
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "flag_rate": (tp + fp) / len(y),
        "cost_per_1000": (c_fn * fn + c_fp * fp) / len(y) * 1000,
    }


def threshold_table(y_true, y_prob, c_fn, c_fp, thresholds=None):
    """Bütün namizəd threshold-lar üçün operating_point cədvəli."""
    thresholds = candidate_thresholds(y_prob) if thresholds is None else thresholds
    return pd.DataFrame([operating_point(y_true, y_prob, t, c_fn, c_fp) for t in thresholds])


def threshold_at_fpr(y_true, y_prob, fpr_target=0.05):
    """FPR <= fpr_target şərtini ödəyən ən aşağı threshold."""
    fpr, _, thr = roc_curve(y_true, y_prob)
    ok = np.where(fpr <= fpr_target)[0]
    return float(thr[ok[-1]])
