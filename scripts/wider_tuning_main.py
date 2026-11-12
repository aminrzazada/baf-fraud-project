"""Genişləndirilmiş HGB tuning: notebook-dakı axtarışın nəticəsi şəbəkənin kənarında idi, burada aralıq genişləndirilib.

Valid ayında (zamana görə, təsadüfi CV yox) təsadüfi axtarış, seçim valid PR-AUC ilə. Test yalnız ən yaxşı və mövcud parametr üçün bir dəfə hesablanır.
İşə salma (repo kökündən): python scripts/wider_tuning.py [N_ITER]   (default 20; hər sınaq tam data ilə bir neçə dəqiqə çəkə bilər)
Nəticə: results/wider_tuning.csv
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

from src.config import *
from src.evaluation import score_report
from src.features import build_pipeline

N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 20
df = pd.read_csv(DATA_PATH)
DROP = [c for c in df.columns if df[c].nunique(dropna=False) <= 1] + [TIME_COL]


def part(months):
    """Verilən aylara aid (X, y) qaytarır."""
    d = df[df[TIME_COL].isin(months)]
    return d.drop(columns=[TARGET]), d[TARGET]


X_tr, y_tr = part(TRAIN_MONTHS)
X_va, y_va = part(VALID_MONTHS)
X_te, y_te = part(TEST_MONTHS)

# Notebook-dakı şəbəkə: lr {0.03,0.05,0.1}, depth {3,5,8}, leaf {20,100,300}, l2 {0,1,10}. Burada hər ox genişləndirilib
GRID = {
    "learning_rate": [0.02, 0.03, 0.05, 0.1, 0.15],
    "max_depth": [2, 3, 4, 5, 6],
    "min_samples_leaf": [5, 10, 20, 50, 100, 300],
    "l2_regularization": [0.0, 0.1, 1.0, 10.0, 100.0],
    "max_iter": [100, 200, 400],
}
CURRENT = {"learning_rate": 0.1, "max_depth": 3, "min_samples_leaf": 20, "l2_regularization": 0.0, "max_iter": 200}


def fit_eval(params):
    """Verilən parametrlərlə pipeline-ı öyrədib valid PR-AUC və öyrənmə müddətini qaytarır."""
    t0 = time.time()
    model = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=RANDOM_STATE, **params)
    pipe = build_pipeline(model, DROP).fit(X_tr, y_tr)
    return pipe, average_precision_score(y_va, pipe.predict_proba(X_va)[:, 1]), time.time() - t0


trials = [CURRENT] + list(ParameterSampler(GRID, n_iter=N_ITER, random_state=RANDOM_STATE))
rows, pipes = [], []
for i, params in enumerate(trials):
    pipe, score, sec = fit_eval(params)
    rows.append({**params, "valid_pr_auc": score, "seconds": sec, "is_current": i == 0})
    pipes.append(pipe)
    print(f"{i:>2} valid PR-AUC {score:.4f} | {params} | {sec:.0f}s", flush=True)

res = pd.DataFrame(rows).sort_values("valid_pr_auc", ascending=False).reset_index(drop=True)
best = res.iloc[0]
cur = res[res["is_current"]].iloc[0]
print(f"\nMövcud parametrlər: valid PR-AUC {cur['valid_pr_auc']:.4f}")
print(f"Ən yaxşı: valid PR-AUC {best['valid_pr_auc']:.4f} (fərq {best['valid_pr_auc'] - cur['valid_pr_auc']:+.4f})")
for p in GRID:  # ən yaxşı dəyər şəbəkənin kənarındadırmı?
    vals = sorted(GRID[p])
    if best[p] in (vals[0], vals[-1]):
        print(f"  Diqqət: {p}={best[p]} şəbəkənin kənarındadır")

best_idx = rows.index(next(r for r in rows if all(r[k] == best[k] for k in GRID) and r["valid_pr_auc"] == best["valid_pr_auc"]))
for name, pipe in [("mövcud", pipes[0]), ("ən yaxşı", pipes[best_idx])]:
    s = score_report(y_te, pipe.predict_proba(X_te)[:, 1])
    print(f"Test ({name}): PR-AUC {s['pr_auc']:.4f} | ROC-AUC {s['roc_auc']:.4f} | TPR@5%FPR {s['tpr_at_5pct_fpr']:.4f}")
Path("results").mkdir(exist_ok=True)
res.to_csv("results/wider_tuning.csv", index=False)
