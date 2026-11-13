"""
Configuration module for Credit Risk Prediction System.
Defines column schemas, valid options, file paths, model parameters, and risk thresholds.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR
RAW_DATA_PATH = BASE_DIR / "extracted_data" / "german.data"
CSV_DATA_PATH = BASE_DIR / "credit_data.csv"
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
CONFUSION_MATRICES_PATH = MODELS_DIR / "confusion_matrices.joblib"
TRAIN_DATA_PATH = MODELS_DIR / "train_data.joblib"

# Target Column Definition
TARGET_COL = "credit_risk"
TARGET_LABELS = {
    0: "Low Risk / Repaid (Good)",
    1: "High Risk / Default (Bad)"
}

# Feature Columns
NUMERICAL_COLS = [
    "duration",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents"
]

CATEGORICAL_COLS = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status_sex",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]

ALL_FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# Feature metadata and display names for user forms
NUMERICAL_SPECS = {
    "duration": {"label": "Loan Duration (Months)", "min": 4, "max": 72, "default": 24, "step": 1},
    "credit_amount": {"label": "Credit Amount (DM)", "min": 250, "max": 20000, "default": 3200, "step": 100},
    "installment_rate": {"label": "Installment Rate (% of Income)", "min": 1, "max": 4, "default": 3, "step": 1},
    "residence_since": {"label": "Years at Current Residence", "min": 1, "max": 4, "default": 2, "step": 1},
    "age": {"label": "Applicant Age (Years)", "min": 18, "max": 75, "default": 35, "step": 1},
    "existing_credits": {"label": "Existing Credits at Bank", "min": 1, "max": 4, "default": 1, "step": 1},
    "num_dependents": {"label": "Number of Dependents", "min": 1, "max": 2, "default": 1, "step": 1}
}

CATEGORICAL_OPTIONS = {
    "checking_status": [
        "< 0 DM",
        "0 to < 200 DM",
        ">= 200 DM",
        "no checking account"
    ],
    "credit_history": [
        "critical account / other credits existing",
        "existing credits paid duly till now",
        "delay in paying off in the past",
        "all credits at this bank paid duly",
        "no credits taken / all paid duly"
    ],
    "purpose": [
        "radio/television",
        "car (new)",
        "furniture/equipment",
        "car (used)",
        "business",
        "education",
        "repairs",
        "domestic appliances",
        "retraining",
        "vacation",
        "others"
    ],
    "savings_status": [
        "< 100 DM",
        "100 to < 500 DM",
        "500 to < 1000 DM",
        ">= 1000 DM",
        "unknown / no savings account"
    ],
    "employment": [
        "< 1 year",
        "1 to < 4 years",
        "4 to < 7 years",
        ">= 7 years",
        "unemployed"
    ],
    "personal_status_sex": [
        "male : single",
        "female : divorced/separated/married",
        "male : married/widowed",
        "male : divorced/separated",
        "female : single"
    ],
    "other_parties": [
        "none",
        "guarantor",
        "co-applicant"
    ],
    "property_magnitude": [
        "real estate",
        "building society savings / life insurance",
        "car or other",
        "unknown / no property"
    ],
    "other_payment_plans": [
        "none",
        "bank",
        "stores"
    ],
    "housing": [
        "own",
        "rent",
        "for free"
    ],
    "job": [
        "skilled employee/official",
        "unskilled - resident",
        "management/self-employed/highly qualified",
        "unemployed/unskilled - non-resident"
    ],
    "telephone": [
        "yes",
        "none"
    ],
    "foreign_worker": [
        "yes",
        "no"
    ]
}

# Risk Tier Thresholds
# Documented justification:
# Baseline population default rate is 30%.
# Cost of a false negative (granting loan to defaulter) is 5x the cost of a false positive.
# Low Risk: P(Default) < 0.28 (Strong creditworthiness, low default risk below base rate)
# Medium Risk: 0.28 <= P(Default) < 0.58 (Moderate risk around base rate, requires underwriter manual review)
# High Risk: P(Default) >= 0.58 (Elevated delinquency probability, high loss risk)
RISK_THRESHOLD_LOW = 0.28
RISK_THRESHOLD_HIGH = 0.58

def classify_risk_tier(default_prob: float) -> tuple[str, str, str]:
    """
    Categorizes default probability into Low, Medium, or High risk with styling and decision guidance.
    Returns: (Risk_Category, Badge_Color, Decision_Recommendation)
    """
    if default_prob < RISK_THRESHOLD_LOW:
        return (
            "Low Risk",
            "#22c55e",  # emerald green
            "Favorable credit profile. Recommend standard approval or fast-track processing."
        )
    elif default_prob < RISK_THRESHOLD_HIGH:
        return (
            "Medium Risk",
            "#f59e0b",  # amber
            "Borderline credit profile. Recommend manual underwriter review, proof of income, or secondary guarantor."
        )
    else:
        return (
            "High Risk",
            "#ef4444",  # red
            "High likelihood of default. Recommend declining application or requiring substantial secured collateral."
        )
