"""Layihənin sabitləri (bütün notebook və app üçün vahid mənbə)."""
import os

RANDOM_STATE = 42
DATA_PATH = os.environ.get("BAF_DATA_PATH", "data/Base.csv")
TARGET = "fraud_bool"
TIME_COL = "month"

# BAF-da çatışmayan dəyərlər -1 ilə kodlanıb
MISSING_MINUS1 = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]
# Bu sütunda mənfi dəyərlər "yoxdur" mənasındadır
NEGATIVE_MISSING = ["intended_balcon_amount"]

CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

# Zamana görə split (month: 0..7)
TRAIN_MONTHS = [0, 1, 2, 3, 4]
VALID_MONTHS = [5]
TEST_MONTHS = [6, 7]

# Biznes xərci fərziyyələri (şərti vahid). Həssaslıq analizi notebook-dadır.
COST_FN = 500.0  # buraxılmış fraud: təxmini itki
COST_FP = 10.0   # yanlış alarm: manual yoxlama + müştəri sürtünməsi
