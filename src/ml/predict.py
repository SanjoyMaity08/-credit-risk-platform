"""
Inference pipeline — load model and predict risk score + risk band.
"""
import os
import pickle
import numpy as np
import pandas as pd

from src.data.preprocessor import preprocess
from src.utils.config import MODEL_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

_models = None
_prep_state = None
_feature_cols = None


def _load_artifacts():
    global _models, _prep_state, _feature_cols
    if _models is None:
        model_path = os.path.join(MODEL_DIR, "lgbm_models.pkl")
        prep_path = os.path.join(MODEL_DIR, "preprocess_state.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model not found. Please train the model first by running: python -m src.ml.train"
            )
        with open(model_path, "rb") as f:
            _models = pickle.load(f)
        with open(prep_path, "rb") as f:
            data = pickle.load(f)
            _prep_state = data["state"]
            _feature_cols = data["feature_cols"]
    return _models, _prep_state, _feature_cols


def predict_proba(input_dict: dict) -> dict:
    """
    Given a dict of feature values, return risk probability, score and band.
    input_dict keys should match the original feature columns.
    """
    models, state, feature_cols = _load_artifacts()

    # Build a single-row dataframe
    row = {col: input_dict.get(col, np.nan) for col in feature_cols}
    df = pd.DataFrame([row])

    df, _ = preprocess(df, state=dict(state), fit=False)
    X = df.values

    # Average across all fold models
    proba = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)[0]
    score = int(round(proba * 1000))  # 0-1000 risk score

    if proba < 0.3:
        band = "Low"
    elif proba < 0.6:
        band = "Medium"
    else:
        band = "High"

    return {
        "default_probability": round(float(proba), 4),
        "risk_score": score,
        "risk_band": band,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predict for a dataframe of applicants."""
    models, state, feature_cols = _load_artifacts()
    feature_cols_present = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    for col in missing:
        df[col] = np.nan

    X_df = df[feature_cols].copy()
    X_df, _ = preprocess(X_df, state=dict(state), fit=False)
    X = X_df.values

    proba = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    df = df.copy()
    df["DEFAULT_PROBABILITY"] = np.round(proba, 4)
    df["RISK_SCORE"] = (proba * 1000).astype(int)
    df["RISK_BAND"] = pd.cut(
        proba,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    return df
