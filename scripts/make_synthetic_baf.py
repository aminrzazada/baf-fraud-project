"""BAF sxemini təqlid edən SİNTETİK data (yalnız pipeline-ın işlədiyini yoxlamaq üçün).

Bu datadan alınan metrikalar real nəticə DEYİL. Real nəticələr üçün Kaggle-dan
Base.csv istifadə edin.
Istifadə: python scripts/make_synthetic_baf.py --n 150000 --out data/Base_synthetic.csv
"""
import argparse

import numpy as np
import pandas as pd


def make(n: int, seed: int = 0) -> pd.DataFrame:
    """n sətirlik BAF-a bənzər cədvəl yaradır (~1.2% fraud, zamanla artan)."""
    r = np.random.default_rng(seed)
    d = pd.DataFrame(
        {
            "income": r.choice(np.arange(0.1, 1.0, 0.1).round(1), n),
            "name_email_similarity": r.beta(2, 2, n),
            "prev_address_months_count": np.where(r.random(n) < 0.71, -1, r.integers(5, 380, n)),
            "current_address_months_count": np.where(r.random(n) < 0.004, -1, r.gamma(2, 40, n).astype(int)),
            "customer_age": r.choice([10, 20, 30, 40, 50, 60, 70, 80, 90], n, p=[.01, .1, .3, .22, .17, .12, .06, .015, .005]),
            "days_since_request": r.exponential(0.5, n),
            "intended_balcon_amount": np.where(r.random(n) < 0.74, -r.uniform(0, 15, n), r.exponential(10, n)),
            "payment_type": r.choice(list("ABCDE"), n, p=[.25, .37, .26, .1, .02]),
            "zip_count_4w": r.normal(1500, 1000, n).clip(1).astype(int),
            "velocity_6h": r.normal(5600, 3000, n),
            "velocity_24h": r.normal(4800, 1500, n),
            "velocity_4w": r.normal(4800, 900, n),
            "bank_branch_count_8w": r.exponential(180, n).astype(int),
            "date_of_birth_distinct_emails_4w": r.poisson(9, n),
            "employment_status": r.choice(["CA", "CB", "CC", "CD", "CE", "CF", "CG"], n, p=[.73, .13, .07, .02, .03, .01, .01]),
            "credit_risk_score": r.normal(130, 70, n).astype(int),
            "email_is_free": (r.random(n) < 0.53).astype(int),
            "housing_status": r.choice(["BA", "BB", "BC", "BD", "BE", "BF", "BG"], n, p=[.26, .03, .17, .29, .01, .23, .01]),
            "phone_home_valid": (r.random(n) < 0.42).astype(int),
            "phone_mobile_valid": (r.random(n) < 0.9).astype(int),
            "bank_months_count": np.where(r.random(n) < 0.25, -1, r.integers(1, 32, n)),
            "has_other_cards": (r.random(n) < 0.22).astype(int),
            "proposed_credit_limit": r.choice([200, 500, 1000, 1500, 1900, 2000], n, p=[.55, .15, .1, .08, .02, .1]),
            "foreign_request": (r.random(n) < 0.03).astype(int),
            "source": r.choice(["INTERNET", "TELEAPP"], n, p=[.99, .01]),
            "session_length_in_minutes": np.where(r.random(n) < 0.02, -1, r.exponential(7, n)),
            "device_os": r.choice(["linux", "windows", "macintosh", "x11", "other"], n, p=[.33, .32, .02, .1, .23]),
            "keep_alive_session": (r.random(n) < 0.5).astype(int),
            "device_distinct_emails_8w": r.choice([-1, 0, 1, 2], n, p=[.001, .01, .96, .029]),
            "device_fraud_count": 0,
            "month": r.integers(0, 8, n),
        }
    )
    # Fraud ehtimalı: bir neçə məntiqli siqnal + səs
    z = (
        2.0 * (1 - d["name_email_similarity"])
        + 0.8 * d["email_is_free"]
        + 0.012 * d["credit_risk_score"]
        + 0.0002 * d["velocity_6h"]
        + 0.5 * (d["device_distinct_emails_8w"] > 1)
        + 1.2 * (d["source"] == "TELEAPP")
        + 0.3 * (d["customer_age"] >= 50)
        - 0.5 * (d["session_length_in_minutes"] > 10)
        + 0.1 * d["month"]
        + r.normal(0, 1.0, n)
    )
    lo, hi = -30.0, 30.0
    for _ in range(60):  # intercept-i ~1.2% fraud üçün bisect ilə tapırıq
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(z + mid)))).mean() < 0.012 else (lo, mid)
    p = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
    d.insert(0, "fraud_bool", (r.random(n) < p).astype(int))
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150_000)
    ap.add_argument("--out", default="data/Base_synthetic.csv")
    a = ap.parse_args()
    df = make(a.n)
    df.to_csv(a.out, index=False)
    print(df.shape, f"fraud rate={df.fraud_bool.mean():.3%}")
