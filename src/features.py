"""Preprocessing və feature engineering addımları (sklearn uyğun transformerlər).

Hamısı Pipeline daxilindədir ki, Streamlit app-da xam sətir verib eyni
emal zəncirindən keçsin (train/serving uyğunsuzluğu olmasın).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, MISSING_MINUS1, NEGATIVE_MISSING

EPS = 1e-6


class BAFCleaner(BaseEstimator, TransformerMixin):
    """Gizli missing kodlarını (-1 / mənfi) NaN edir, indikator əlavə edir, drop_cols silir."""

    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        """Öyrənilən parametr yoxdur (stateless)."""
        return self

    def transform(self, X):
        """-1/mənfi kodları NaN edir, *_missing indikatorlarını əlavə edir, drop_cols-u silir."""
        X = X.copy()
        for col in MISSING_MINUS1:
            if col in X:
                mask = X[col] == -1
                X[f"{col}_missing"] = mask.astype("int8")  # missing-in özü siqnal ola bilər
                X[col] = X[col].mask(mask)
        for col in NEGATIVE_MISSING:
            if col in X:
                mask = X[col] < 0
                X[f"{col}_missing"] = mask.astype("int8")
                X[col] = X[col].mask(mask)
        drop = [c for c in (self.drop_cols or []) if c in X]
        return X.drop(columns=drop)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Domen biliyinə əsaslanan yeni feature-lər (velocity, nisbətlər, interaksiyalar)."""

    def fit(self, X, y=None):
        """Öyrənilən parametr yoxdur (stateless)."""
        return self

    @staticmethod
    def _nansum(df, cols):
        """Sütunların cəmi; hamısı NaN-dırsa NaN qaytarır (min_count=1)."""
        return df[cols].sum(axis=1, min_count=1)

    def transform(self, X):
        """Domen əsaslı yeni feature-ləri əlavə edir (velocity, nisbətlər, interaksiyalar)."""
        X = X.copy()

        # 1) Velocity: qısa müddətli aktivlik uzunmüddətliyə nisbətən sıçrayıbmı
        X["velocity_ratio_6h_24h"] = X["velocity_6h"] / (X["velocity_24h"] + EPS)
        X["velocity_ratio_24h_4w"] = X["velocity_24h"] / (X["velocity_4w"] + EPS)
        X["velocity_accel"] = X["velocity_6h"] - X["velocity_4w"]

        # 2) Maliyyə nisbətləri: tələb olunan limit gəlirə / risk skoruna görə
        X["credit_limit_to_income"] = X["proposed_credit_limit"] / (X["income"] + 0.01)
        X["credit_risk_per_limit"] = X["credit_risk_score"] / (X["proposed_credit_limit"] + EPS)
        X["balcon_to_limit"] = X["intended_balcon_amount"] / (X["proposed_credit_limit"] + EPS)

        # 3) Yaşayış stabilliyi (customer_age onluqlarla verilib, ay = age*12)
        X["addr_months_total"] = self._nansum(
            X, ["current_address_months_count", "prev_address_months_count"]
        )
        X["current_addr_share_of_life"] = X["current_address_months_count"] / (
            X["customer_age"] * 12 + EPS
        )

        # 4) Kimlik/email anomaliyaları
        X["distinct_emails_total"] = self._nansum(
            X, ["date_of_birth_distinct_emails_4w", "device_distinct_emails_8w"]
        )
        X["phone_valid_count"] = X["phone_home_valid"] + X["phone_mobile_valid"]
        X["email_risk"] = X["email_is_free"] * (1 - X["name_email_similarity"])

        # 5) Sessiya davranışı
        X["session_log"] = np.log1p(X["session_length_in_minutes"])
        X["short_session"] = (X["session_length_in_minutes"] <= 1).astype("int8")

        # 6) Ağır quyruqlu say sütunları üçün log (xətti model üçün faydalıdır)
        X["zip_count_log"] = np.log1p(X["zip_count_4w"])
        X["branch_count_log"] = np.log1p(X["bank_branch_count_8w"])

        # 7) İnteraksiya: telefon kanalı + xaricdən sorğu
        X["teleapp_foreign"] = ((X["source"] == "TELEAPP") & (X["foreign_request"] == 1)).astype("int8")

        return X.replace([np.inf, -np.inf], np.nan)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Verilmiş feature adlarını saxlayır (keep=None olsa hamısını buraxır)."""

    def __init__(self, keep=None):
        self.keep = keep

    def fit(self, X, y=None):
        """Saxlanacaq sütunların siyahısını (cols_) müəyyən edir."""
        self.cols_ = list(X.columns) if self.keep is None else [c for c in self.keep if c in X.columns]
        return self

    def transform(self, X):
        """Yalnız seçilmiş sütunları qaytarır."""
        return X[self.cols_]


def _numeric_cols(df):
    """Kateqorik olmayan bütün sütunlar rəqəmsal hesab olunur."""
    return [c for c in df.columns if c not in CATEGORICAL]


def build_preprocessor():
    """Rəqəmsal: median imputation + StandardScaler. Kateqorik: OneHot."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    prep = ColumnTransformer(
        [("num", numeric, _numeric_cols), ("cat", categorical, CATEGORICAL)],
        verbose_feature_names_out=False,
    )
    return prep.set_output(transform="pandas")


def build_pipeline(model, drop_cols=None, keep_features=None):
    """Tam pipeline: clean -> feature engineering -> preprocess -> select -> model."""
    return Pipeline(
        [
            ("clean", BAFCleaner(drop_cols)),
            ("fe", FeatureEngineer()),
            ("prep", build_preprocessor()),
            ("select", FeatureSelector(keep_features)),
            ("model", model),
        ]
    )
