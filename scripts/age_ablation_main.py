"""Yaş ablation: customer_age (və ondan törəyən feature) çıxarılsa performans və yaş qruplarında FPR necə dəyişir?

İki variant (eyni tuned HGB): "yaşlı" (final model) və "yaşsız". Hər variant üçün threshold valid-də ümumi FPR<=5% şərti ilə
seçilir, test-də 50+ və <50 qrupları üzrə FPR / recall / alarm nisbəti hesablanır. Beləliklə iki variant eyni əməliyyat nöqtəsində müqayisə olunur.
İşə salma (repo kökündən): python scripts/age_ablation.py
Nəticə: results/age_ablation.csv
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import *
from src.evaluation import score_report, threshold_at_fpr
from src.features import build_pipeline

df = pd.read_csv(DATA_PATH)
DROP = [c for c in df.columns if df[c].nunique(dropna=False) <= 1] + [TIME_COL]
AGE_DERIVED = ["customer_age", "current_addr_share_of_life"]  # ikincisi yaşdan hesablanır (FeatureEngineer)


def part(months):
    """Verilən aylara aid (X, y) qaytarır."""
    d = df[df[TIME_COL].isin(months)]
    return d.drop(columns=[TARGET]), d[TARGET]


X_tr, y_tr = part(TRAIN_MONTHS)
X_va, y_va = part(VALID_MONTHS)
X_te, y_te = part(TEST_MONTHS)


def make_model():
    """Notebook-da tuning ilə seçilmiş HGB parametrləri."""
    return HistGradientBoostingClassifier(learning_rate=0.1, max_depth=3, min_samples_leaf=20, l2_regularization=0.0,
                                          max_iter=200, class_weight="balanced", early_stopping=False, random_state=RANDOM_STATE)


def group_stats(y, p, thr):
    """Verilən threshold-da bir qrup üzrə FPR, recall və alarm nisbəti."""
    y, pred = np.asarray(y).astype(bool), np.asarray(p) >= thr
    return {"fpr": (pred & ~y).sum() / max((~y).sum(), 1), "recall": (pred & y).sum() / max(y.sum(), 1), "flag_rate": pred.mean(), "n": len(y)}


full = build_pipeline(make_model(), DROP).fit(X_tr, y_tr)
all_cols = full.named_steps["select"].cols_
variants = {"yaşlı (final model)": full}
t0 = time.time()
keep = [c for c in all_cols if c not in AGE_DERIVED]
variants["yaşsız"] = build_pipeline(make_model(), DROP, keep_features=keep).fit(X_tr, y_tr)
print(f"yaşsız model {time.time() - t0:.0f}s, feature sayı: {len(all_cols)} -> {len(keep)}", flush=True)

older = (X_te["customer_age"] >= 50).values
rows = {}
for name, pipe in variants.items():
    p_va, p_te = pipe.predict_proba(X_va)[:, 1], pipe.predict_proba(X_te)[:, 1]
    thr = threshold_at_fpr(y_va, p_va, 0.05)
    r = {f"test_{k}": v for k, v in score_report(y_te, p_te).items()}
    r["threshold"] = thr
    for label, m in [("50+", older), ("<50", ~older)]:
        for k, v in group_stats(y_te[m], p_te[m], thr).items():
            r[f"{label}_{k}"] = v
    r["fpr_gap_50plus_minus_under50"] = r["50+_fpr"] - r["<50_fpr"]
    rows[name] = r
out = pd.DataFrame(rows).T.round(4)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(out.T.to_string())
Path("results").mkdir(exist_ok=True)
out.to_csv("results/age_ablation.csv")
