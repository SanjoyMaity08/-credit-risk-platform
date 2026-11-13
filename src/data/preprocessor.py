"""
Preprocessing pipeline — cleaning, encoding, imputation, feature engineering.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

CATEGORICAL_THRESHOLD = 20  # columns with ≤ this many unique values → label-encode


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: replace inf, drop near-constant columns."""
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop columns where >80% values are missing
    threshold = 0.8
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > threshold].index.tolist()
    if drop_cols:
        logger.info(f"Dropping {len(drop_cols)} high-missing columns.")
        df = df.drop(columns=drop_cols)
    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True) -> tuple:
    """Label-encode low-cardinality string columns."""
    if encoders is None:
        encoders = {}
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                df[col] = -1
    return df, encoders


def impute(df: pd.DataFrame, medians: dict = None, fit: bool = True) -> tuple:
    """Median imputation for numeric columns."""
    if medians is None:
        medians = {}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if fit:
            medians[col] = df[col].median()
        df[col] = df[col].fillna(medians.get(col, 0))
    return df, medians


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven feature engineering."""
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
        df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
        df["EMPLOYMENT_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1)
    return df


def preprocess(df: pd.DataFrame, state: dict = None, fit: bool = True) -> tuple:
    """Full pipeline: clean → feature engineering → encode → impute."""
    if state is None:
        state = {}
    df = clean(df)
    df = feature_engineer(df)
    df, encoders = encode_categoricals(df, state.get("encoders"), fit=fit)
    df, medians = impute(df, state.get("medians"), fit=fit)
    state["encoders"] = encoders
    state["medians"] = medians
    return df, state
