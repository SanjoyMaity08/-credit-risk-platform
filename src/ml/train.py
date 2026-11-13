"""
ML training pipeline — LightGBM with class-imbalance handling.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from src.data.loader import load_main_tables
from src.data.preprocessor import preprocess
from src.utils.logger import get_logger
from src.utils.config import MODEL_DIR

logger = get_logger(__name__)

TARGET = "TARGET"
DROP_COLS = ["SK_ID_CURR", TARGET]

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 11,  # handles ~9% default rate imbalance
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load and preprocess
    df = load_main_tables()
    y = df[TARGET].values
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X_df = df[feature_cols].copy()

    X_df, state = preprocess(X_df, fit=True)
    X = X_df.values

    # Save preprocessing state + feature columns
    with open(os.path.join(MODEL_DIR, "preprocess_state.pkl"), "wb") as f:
        pickle.dump({"state": state, "feature_cols": feature_cols}, f)

    # 5-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    models = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Training fold {fold + 1}/5 ...")
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        models.append(model)

    oof_auc = roc_auc_score(y, oof_preds)
    logger.info(f"OOF AUC: {oof_auc:.4f}")

    # Save all fold models
    with open(os.path.join(MODEL_DIR, "lgbm_models.pkl"), "wb") as f:
        pickle.dump(models, f)

    # Save feature importances
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": np.mean([m.feature_importances_ for m in models], axis=0),
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)

    logger.info(f"Training complete. OOF AUC = {oof_auc:.4f}")
    return oof_auc


if __name__ == "__main__":
    train()
