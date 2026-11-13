"""
Data processing and preprocessing pipeline for Credit Risk Prediction.
Ensures zero data leakage, consistent encoding, and robust applicant transformation.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.config import (
    CSV_DATA_PATH,
    RAW_DATA_PATH,
    TARGET_COL,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    ALL_FEATURE_COLS
)

def ensure_dataset_exists() -> pd.DataFrame:
    """Loads credit_data.csv or generates it from extracted raw data if missing."""
    if os.path.exists(CSV_DATA_PATH):
        df = pd.read_csv(CSV_DATA_PATH)
        return df

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Neither {CSV_DATA_PATH} nor {RAW_DATA_PATH} was found.")

    # Reconstruct from raw german.data
    raw_columns = [
        "checking_status", "duration", "credit_history", "purpose", "credit_amount",
        "savings_status", "employment", "installment_rate", "personal_status_sex",
        "other_parties", "residence_since", "property_magnitude", "age",
        "other_payment_plans", "housing", "existing_credits", "job",
        "num_dependents", "telephone", "foreign_worker", "class"
    ]
    df_raw = pd.read_csv(RAW_DATA_PATH, sep=r'\s+', header=None, names=raw_columns)

    value_mappings = {
        "checking_status": {
            "A11": "< 0 DM", "A12": "0 to < 200 DM", "A13": ">= 200 DM", "A14": "no checking account"
        },
        "credit_history": {
            "A30": "no credits taken / all paid duly", "A31": "all credits at this bank paid duly",
            "A32": "existing credits paid duly till now", "A33": "delay in paying off in the past",
            "A34": "critical account / other credits existing"
        },
        "purpose": {
            "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment",
            "A43": "radio/television", "A44": "domestic appliances", "A45": "repairs",
            "A46": "education", "A47": "vacation", "A48": "retraining",
            "A49": "business", "A410": "others"
        },
        "savings_status": {
            "A61": "< 100 DM", "A62": "100 to < 500 DM", "A63": "500 to < 1000 DM",
            "A64": ">= 1000 DM", "A65": "unknown / no savings account"
        },
        "employment": {
            "A71": "unemployed", "A72": "< 1 year", "A73": "1 to < 4 years",
            "A74": "4 to < 7 years", "A75": ">= 7 years"
        },
        "personal_status_sex": {
            "A91": "male : divorced/separated", "A92": "female : divorced/separated/married",
            "A93": "male : single", "A94": "male : married/widowed", "A95": "female : single"
        },
        "other_parties": {
            "A101": "none", "A102": "co-applicant", "A103": "guarantor"
        },
        "property_magnitude": {
            "A121": "real estate", "A122": "building society savings / life insurance",
            "A123": "car or other", "A124": "unknown / no property"
        },
        "other_payment_plans": {
            "A141": "bank", "A142": "stores", "A143": "none"
        },
        "housing": {
            "A151": "rent", "A152": "own", "A153": "for free"
        },
        "job": {
            "A171": "unemployed/unskilled - non-resident", "A172": "unskilled - resident",
            "A173": "skilled employee/official", "A174": "management/self-employed/highly qualified"
        },
        "telephone": {"A191": "none", "A192": "yes"},
        "foreign_worker": {"A201": "yes", "A202": "no"}
    }

    df_clean = df_raw.copy()
    for col, mapping in value_mappings.items():
        df_clean[col] = df_clean[col].map(mapping).fillna(df_clean[col])

    # Convert class 1 (Good) -> 0, class 2 (Bad/Default) -> 1
    df_clean["credit_risk"] = df_clean["class"].apply(lambda x: 1 if x == 2 else 0)
    df_clean.drop(columns=["class"], inplace=True)
    df_clean.to_csv(CSV_DATA_PATH, index=False)
    return df_clean

def load_data() -> pd.DataFrame:
    """Loads the preprocessed credit data."""
    return ensure_dataset_exists()

def get_dataset_profile(df: pd.DataFrame) -> dict:
    """
    Computes dataset statistics: shape, missing values, duplicates, class distribution.
    """
    return {
        "total_records": int(df.shape[0]),
        "total_features": int(df.shape[1] - (1 if TARGET_COL in df.columns else 0)),
        "numerical_count": len(NUMERICAL_COLS),
        "categorical_count": len(CATEGORICAL_COLS),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_records": int(df.duplicated().sum()),
        "class_distribution": df[TARGET_COL].value_counts().to_dict() if TARGET_COL in df.columns else {},
        "default_rate": float(df[TARGET_COL].mean()) if TARGET_COL in df.columns else 0.0,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for consistent feature engineering.
    Uses StandardScaler on numerical attributes and OneHotEncoder on categoricals.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)
        ]
    )
    return preprocessor

def prepare_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits data into train and test sets with stratification to preserve target distribution.
    Prevents data leakage by fitting transformers strictly on train split.
    """
    X = df[ALL_FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_names = NUMERICAL_COLS + cat_names

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "preprocessor": preprocessor,
        "feature_names": feature_names
    }

def transform_single_applicant(applicant_data: dict, preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Transforms a single applicant dict or pandas row into the model-compatible feature vector.
    """
    df_applicant = pd.DataFrame([applicant_data])
    return preprocessor.transform(df_applicant[ALL_FEATURE_COLS])
