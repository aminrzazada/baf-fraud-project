"""Streamlit app üçün model yükləmə və skorinq məntiqi (UI-dan ayrıdır ki, test oluna bilsin)."""
import json
from pathlib import Path

import joblib
import pandas as pd


def load_artifacts(root: Path):
    """models/ qovluğundan kalibrə olunmuş modeli və meta məlumatı yükləyir."""
    model = joblib.load(root / "models" / "model.joblib")
    meta = json.loads((root / "models" / "meta.json").read_text(encoding="utf-8"))
    return model, meta


def values_to_frame(values: dict, meta: dict) -> pd.DataFrame:
    """Tək müraciətin dəyərlərini modelin gözlədiyi sütun sırası və tiplə DataFrame edir."""
    schema = meta["input_schema"]
    row = {}
    for col, spec in schema.items():
        v = values[col]
        row[col] = int(v) if spec["type"] == "int" else (float(v) if spec["type"] == "float" else v)
    return pd.DataFrame([row], columns=list(schema))


def score_frame(model, meta: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Cədvəli skorlayır: fraud_probability və flagged (threshold-a görə) sütunları əlavə edir."""
    cols = list(meta["input_schema"])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Çatışmayan sütunlar: {missing}")
    # Yalnız sxemdəki sütunlar: artıq sütun (məs. fraud_bool) pipeline-a düşməsin
    p = model.predict_proba(df[cols])[:, 1]
    out = df.copy()
    out["fraud_probability"] = p
    out["flagged"] = p >= meta["threshold"]
    return out


def flag_by_capacity(prob: pd.Series, capacity_pct: float) -> pd.Series:
    """Yoxlama komandasının tutumuna görə: ən yüksək riskli capacity_pct% müraciəti işarələyir."""
    cut = prob.quantile(1 - capacity_pct / 100)
    return prob >= cut
