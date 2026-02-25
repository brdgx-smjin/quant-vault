"""ML model definitions for signal prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier

from config.settings import MODEL_DIR

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "return_1", "return_3", "return_6", "return_12",
    "volatility_12", "volatility_24",
    "high_low_range", "body_ratio", "upper_wick", "lower_wick",
    "rsi", "rsi_delta",
    "macd", "macd_hist", "macd_hist_delta",
    "bb_position", "bb_width",
    "ema_20_ratio", "ema_50_ratio", "ema_cross",
    "atr_norm",
    "volume_ratio", "volume_trend",
    "higher_high", "lower_low", "higher_close",
    "consec_up", "consec_down",
]


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""

    accuracy: float
    f1: float
    precision_1: float
    recall_1: float
    feature_importance: dict[str, float]


class SignalPredictor:
    """XGBoost-based trade signal predictor."""

    def __init__(self, model_name: str = "xgb_signal") -> None:
        self.model_name = model_name
        self.model: Optional[XGBClassifier] = None
        self.feature_cols = [c for c in FEATURE_COLS]  # copy

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        n_estimators: int = 300,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        scale_pos_weight: float | None = None,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_weight: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.7,
        gamma: float = 0.1,
        early_stopping_rounds: int = 50,
    ) -> ModelMetrics:
        """Train XGBoost classifier with regularization and early stopping.

        Args:
            train_df: DataFrame with feature columns and 'target'.
            val_df: Optional validation DataFrame for early stopping.
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth (reduced to prevent overfitting).
            learning_rate: Boosting learning rate.
            scale_pos_weight: Balance for imbalanced classes.
            reg_alpha: L1 regularization on weights.
            reg_lambda: L2 regularization on weights.
            min_child_weight: Minimum sum of instance weight in a child.
            subsample: Row subsampling ratio per tree.
            colsample_bytree: Column subsampling ratio per tree.
            gamma: Minimum loss reduction to make a split.
            early_stopping_rounds: Stop if val metric doesn't improve.

        Returns:
            ModelMetrics on training data.
        """
        available = [c for c in self.feature_cols if c in train_df.columns]
        self.feature_cols = available

        X = train_df[available].values
        y = train_df["target"].values

        if scale_pos_weight is None:
            neg = (y == 0).sum()
            pos = (y == 1).sum()
            scale_pos_weight = neg / pos if pos > 0 else 1.0

        callbacks = []
        if val_df is not None and early_stopping_rounds > 0:
            from xgboost.callback import EarlyStopping
            callbacks.append(EarlyStopping(
                rounds=early_stopping_rounds,
                save_best=True,
                metric_name="logloss",
            ))

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            callbacks=callbacks if callbacks else None,
        )

        fit_params: dict = {"verbose": False}
        if val_df is not None:
            X_val = val_df[available].values
            y_val = val_df["target"].values
            fit_params["eval_set"] = [(X_val, y_val)]

        self.model.fit(X, y, **fit_params)

        return self._evaluate(train_df, "train")

    def evaluate(self, test_df: pd.DataFrame) -> ModelMetrics:
        """Evaluate model on test data.

        Args:
            test_df: Test DataFrame with features and target.

        Returns:
            ModelMetrics on test data.
        """
        return self._evaluate(test_df, "test")

    def _evaluate(self, df: pd.DataFrame, split: str) -> ModelMetrics:
        X = df[self.feature_cols].values
        y_true = df["target"].values
        y_pred = self.model.predict(X)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        precision_1 = report.get("1", {}).get("precision", 0)
        recall_1 = report.get("1", {}).get("recall", 0)

        importances = dict(zip(
            self.feature_cols,
            self.model.feature_importances_.tolist(),
        ))
        top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:10])

        logger.info("[%s] Accuracy: %.3f | F1: %.3f | Precision(1): %.3f | Recall(1): %.3f",
                    split, acc, f1, precision_1, recall_1)

        return ModelMetrics(
            accuracy=acc, f1=f1,
            precision_1=precision_1, recall_1=recall_1,
            feature_importance=top_features,
        )

    def select_features(
        self,
        train_df: pd.DataFrame,
        max_features: int = 15,
        importance_threshold: float = 0.01,
    ) -> list[str]:
        """Select top features by training a quick model and ranking importance.

        Args:
            train_df: Training DataFrame with features and 'target'.
            max_features: Maximum number of features to keep.
            importance_threshold: Minimum importance to include a feature.

        Returns:
            List of selected feature column names.
        """
        available = [c for c in FEATURE_COLS if c in train_df.columns]
        X = train_df[available].values
        y = train_df["target"].values

        quick_model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.7,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        quick_model.fit(X, y, verbose=False)

        importances = dict(zip(available, quick_model.feature_importances_.tolist()))
        ranked = sorted(importances.items(), key=lambda x: -x[1])

        selected = [
            name for name, imp in ranked
            if imp >= importance_threshold
        ][:max_features]

        logger.info("Feature selection: %d/%d features kept (threshold=%.3f)",
                     len(selected), len(available), importance_threshold)
        for name, imp in ranked[:max_features]:
            logger.info("  %-20s %.4f%s", name, imp,
                        " *" if name in selected else "")

        self.feature_cols = selected
        return selected

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability of class 1 (profitable long).

        Args:
            df: DataFrame with feature columns.

        Returns:
            Array of probabilities.
        """
        X = df[self.feature_cols].values
        return self.model.predict_proba(X)[:, 1]

    def save(self) -> str:
        """Save model to disk."""
        path = Path(MODEL_DIR) / f"{self.model_name}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_cols}, path)
        logger.info("Model saved to %s", path)
        return str(path)

    def load(self) -> None:
        """Load model from disk."""
        path = Path(MODEL_DIR) / f"{self.model_name}.joblib"
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_cols = data["features"]
        logger.info("Model loaded from %s", path)
