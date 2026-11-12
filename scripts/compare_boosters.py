"""LightGBM / XGBoost / CatBoost vs HistGradientBoosting müqayisəsi (eyni split, eyni pipeline).

Quraşdırma: pip install -r requirements-extra.txt
İşə salma (repo kökündən): python scripts/compare_boosters.py
Nəticə: results/booster_comparison.csv

Qeyd: boosterlər sabit, ağlabatan parametrlərlə işləyir (ayrıca tuning yoxdur). HGB isə notebook-da
tuning olunmuş parametrlərlə (HGB tuned) və tuning-siz (HGB baseline) iki sətirlə göstərilir ki, müqayisə ədalətli olsun.
Quraşdırılmamış kitabxana atlanır, xəta verən model qeyd olunub davam edilir.
"""
import sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from src.config import *
from src.evaluation import score_report
from src.features import build_pipeline

df = pd.read_csv(DATA_PATH)
DROP = [c for c in df.columns if df[c].nunique(dropna=False) <= 1] + [TIME_COL]


def part(months):
    """Verilən aylara aid (X, y) qaytarır."""
    d = df[df[TIME_COL].isin(months)]
    return d.drop(columns=[TARGET]), d[TARGET]


X_tr, y_tr = part(TRAIN_MONTHS)
X_va, y_va = part(VALID_MONTHS)
X_te, y_te = part(TEST_MONTHS)
tr_eval = X_tr.sample(min(200_000, len(X_tr)), random_state=RANDOM_STATE)
pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()  # XGBoost üçün imbalance çəkisi


def hgb(**kw):
    """HistGradientBoosting (class_weight='balanced')."""
    return HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", early_stopping=False,
                                          random_state=RANDOM_STATE, **kw)


def candidates():
    """Namizəd modellər: (ad, model_yaradan_funksiya). Quraşdırılmayanlar atlanır."""
    out = [
        ("HGB baseline", lambda: hgb(learning_rate=0.1, max_depth=6)),
        # Notebook-da tuning ilə seçilmiş parametrlər
        ("HGB tuned", lambda: hgb(learning_rate=0.1, max_depth=3, min_samples_leaf=20, l2_regularization=0.0)),
    ]
    try:
        from lightgbm import LGBMClassifier
        out.append(("LightGBM", lambda: LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)))
    except ImportError:
        print("LightGBM quraşdırılmayıb, atlanır")
    try:
        from xgboost import XGBClassifier
        out.append(("XGBoost", lambda: XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, scale_pos_weight=pos_weight, tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)))
    except ImportError:
        print("XGBoost quraşdırılmayıb, atlanır")
    try:
        from catboost import CatBoostClassifier
        out.append(("CatBoost", lambda: CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=4, auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=0)))
    except ImportError:
        print("CatBoost quraşdırılmayıb, atlanır")
    return out


def paired_bootstrap_delta(y, p_a, p_b, n_boot=200, seed=RANDOM_STATE):
    """Δ PR-AUC (b - a) üçün orta və 95% inam intervalı (paired bootstrap)."""
    rng, y, d = np.random.default_rng(seed), np.asarray(y), []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        d.append(average_precision_score(y[i], p_b[i]) - average_precision_score(y[i], p_a[i]))
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5])


rows, valid_scores = {}, {}
for name, make in candidates():
    t0 = time.time()
    try:
        pipe = build_pipeline(make(), DROP).fit(X_tr, y_tr)
        r = {}
        for split, (X, y) in {"train": (tr_eval, y_tr.loc[tr_eval.index]), "valid": (X_va, y_va), "test": (X_te, y_te)}.items():
            p = pipe.predict_proba(X)[:, 1]
            s = score_report(y, p)
            r.update({f"{split}_pr_auc": s["pr_auc"], f"{split}_roc_auc": s["roc_auc"], f"{split}_tpr@5%fpr": s["tpr_at_5pct_fpr"]})
            if split == "valid":
                valid_scores[name] = p
        r["seconds"] = time.time() - t0
        rows[name] = r
        print(f"{name}: valid PR-AUC {r['valid_pr_auc']:.4f} | test PR-AUC {r['test_pr_auc']:.4f} | {r['seconds']:.0f}s", flush=True)
    except Exception:
        print(f"{name}: XƏTA, atlanır"); traceback.print_exc()

res = pd.DataFrame(rows).T
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(res.round(4).to_string())

# HGB tuned ilə paired bootstrap (valid): booster HGB-dən statistik olaraq yaxşıdırmı?
if "HGB tuned" in valid_scores:
    print("\nΔ valid PR-AUC (model - HGB tuned), 95% CI:")
    for name, p in valid_scores.items():
        if name == "HGB tuned":
            continue
        m, (lo, hi) = paired_bootstrap_delta(y_va, valid_scores["HGB tuned"], p)
        verdict = "HGB-dən yaxşı" if lo > 0 else ("HGB-dən pis" if hi < 0 else "fərq əhəmiyyətli deyil")
        print(f"  {name}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {verdict}")

Path("results").mkdir(exist_ok=True)
res.to_csv("results/booster_comparison.csv")
