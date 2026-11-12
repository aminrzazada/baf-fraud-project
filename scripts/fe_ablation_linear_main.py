"""Feature engineering ablation Logistic Regression ilə (xətti model interaksiyanı özü öyrənə bilmir).

Sual: FE ağac modelində qazanc vermədi, çünki boosting kombinasiyaları özü tapır? Xətti modeldə FE faydalıdırsa, bu izahı dəstəkləyir.
İşə salma (repo kökündən): python scripts/fe_ablation_linear.py
Nəticə: results/fe_ablation_linear.csv
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import *
from src.evaluation import score_report
from src.features import BAFCleaner, FeatureEngineer, FeatureSelector, build_preprocessor

df = pd.read_csv(DATA_PATH)
DROP = [c for c in df.columns if df[c].nunique(dropna=False) <= 1] + [TIME_COL]


def part(months):
    """Verilən aylara aid sətirləri qaytarır."""
    return df[df[TIME_COL].isin(months)]


tr, va, te = part(TRAIN_MONTHS), part(VALID_MONTHS), part(TEST_MONTHS)


def make_model():
    """Modeldəki baseline parametrlər: güclü L2 (C=0.1), imbalance üçün class_weight."""
    return LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)


variants = {
    "clean only (raw + missing indikatorlar)": [("clean", BAFCleaner(DROP))],
    "clean + feature engineering (tam)": [("clean", BAFCleaner(DROP)), ("fe", FeatureEngineer())],
}
rows = {}
for name, steps in variants.items():
    t0 = time.time()
    pipe = Pipeline(steps + [("prep", build_preprocessor()), ("select", FeatureSelector()), ("model", make_model())])
    pipe.fit(tr.drop(columns=[TARGET]), tr[TARGET])
    r = {"n_features": len(pipe.named_steps["select"].cols_)}
    for sname, d in [("valid", va), ("test", te)]:
        s = score_report(d[TARGET], pipe.predict_proba(d.drop(columns=[TARGET]))[:, 1])
        r.update({f"{sname}_pr_auc": s["pr_auc"], f"{sname}_roc_auc": s["roc_auc"], f"{sname}_tpr@5%fpr": s["tpr_at_5pct_fpr"]})
    rows[name] = r
    print(name, f"{time.time() - t0:.0f}s", flush=True)

out = pd.DataFrame(rows).T.round(4)
print(out.to_string())
Path("results").mkdir(exist_ok=True)
out.to_csv("results/fe_ablation_linear.csv")
