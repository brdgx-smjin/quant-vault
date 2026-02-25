"""Cross-validated ML training pipeline with purged time-series CV."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.ml.features import build_features, purged_kfold_split
from src.ml.models import FEATURE_COLS, ModelMetrics, SignalPredictor

logger = logging.getLogger(__name__)


@dataclass
class CVReport:
    """Results from cross-validated training pipeline."""

    fold_metrics: list[ModelMetrics]
    mean_accuracy: float
    mean_f1: float
    std_f1: float
    selected_features: list[str]
    avg_best_iteration: int
    final_train_metrics: ModelMetrics | None = None
    final_test_metrics: ModelMetrics | None = None


def train_with_cv(
    df: pd.DataFrame,
    model_name: str = "xgb_signal_4h",
    future_bars: int = 12,
    n_folds: int = 5,
    purge_bars: int = 12,
    max_features: int = 15,
    n_estimators: int = 300,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    test_ratio: float = 0.2,
) -> tuple[SignalPredictor, CVReport]:
    """Orchestrate feature selection, purged CV evaluation, and final model training.

    Pipeline:
        1. Build features from OHLCV data
        2. Hold out final test set (chronological)
        3. Feature selection on training portion
        4. Purged k-fold CV on training portion
        5. Train final model using average best_iteration from CV

    Args:
        df: OHLCV DataFrame with indicators.
        model_name: Name for the saved model.
        future_bars: Look-ahead bars for target label.
        n_folds: Number of CV folds.
        purge_bars: Gap between train/test in CV (should match future_bars).
        max_features: Maximum features to select.
        n_estimators: Max boosting rounds.
        max_depth: Tree depth.
        learning_rate: Boosting learning rate.
        test_ratio: Fraction of data held out for final evaluation.

    Returns:
        Tuple of (trained SignalPredictor, CVReport).
    """
    logger.info("=" * 60)
    logger.info("  ML Training Pipeline (Purged CV)")
    logger.info("=" * 60)

    # 1. Build features
    feat = build_features(df, future_bars=future_bars)
    logger.info("Features built: %d samples, %d features",
                len(feat), len(feat.columns) - 2)

    # 2. Hold out final test set (chronological)
    split_idx = int(len(feat) * (1 - test_ratio))
    train_full = feat.iloc[:split_idx]
    test_final = feat.iloc[split_idx:]
    logger.info("Train: %d | Final test: %d", len(train_full), len(test_final))

    # 3. Feature selection
    predictor = SignalPredictor(model_name)
    selected = predictor.select_features(train_full, max_features=max_features)
    logger.info("Selected %d features: %s", len(selected), selected)

    # 4. Purged k-fold CV
    folds = purged_kfold_split(train_full, n_folds=n_folds, purge_bars=purge_bars)
    logger.info("Purged CV: %d folds (purge_bars=%d)", len(folds), purge_bars)

    fold_metrics: list[ModelMetrics] = []
    best_iterations: list[int] = []

    for i, (fold_train, fold_val) in enumerate(folds):
        fold_predictor = SignalPredictor(f"{model_name}_fold{i}")
        fold_predictor.feature_cols = list(selected)

        fold_predictor.train(
            fold_train,
            val_df=fold_val,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            early_stopping_rounds=50,
        )

        val_metrics = fold_predictor.evaluate(fold_val)
        fold_metrics.append(val_metrics)

        # Record best iteration from early stopping
        best_iter = getattr(fold_predictor.model, "best_iteration", n_estimators)
        best_iterations.append(best_iter)

        logger.info("  Fold %d: Acc=%.3f F1=%.3f (best_iter=%d, train=%d, val=%d)",
                     i + 1, val_metrics.accuracy, val_metrics.f1,
                     best_iter, len(fold_train), len(fold_val))

    # CV summary
    cv_accuracies = [m.accuracy for m in fold_metrics]
    cv_f1s = [m.f1 for m in fold_metrics]
    avg_best_iter = int(np.mean(best_iterations)) if best_iterations else n_estimators

    logger.info("")
    logger.info("  CV Summary:")
    logger.info("    Mean Accuracy: %.3f (+/- %.3f)", np.mean(cv_accuracies), np.std(cv_accuracies))
    logger.info("    Mean F1:       %.3f (+/- %.3f)", np.mean(cv_f1s), np.std(cv_f1s))
    logger.info("    Avg best_iter: %d", avg_best_iter)

    # 5. Train final model on full training set with avg best_iteration
    logger.info("")
    logger.info("  Training final model (n_estimators=%d)...", avg_best_iter)

    final_predictor = SignalPredictor(model_name)
    final_predictor.feature_cols = list(selected)

    final_train_metrics = final_predictor.train(
        train_full,
        n_estimators=avg_best_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    final_test_metrics = final_predictor.evaluate(test_final)

    logger.info("  Final Train — Acc: %.3f | F1: %.3f", final_train_metrics.accuracy, final_train_metrics.f1)
    logger.info("  Final Test  — Acc: %.3f | F1: %.3f", final_test_metrics.accuracy, final_test_metrics.f1)
    logger.info("  Gap: %.1f%% (train - test accuracy)",
                (final_train_metrics.accuracy - final_test_metrics.accuracy) * 100)

    final_predictor.save()

    report = CVReport(
        fold_metrics=fold_metrics,
        mean_accuracy=float(np.mean(cv_accuracies)),
        mean_f1=float(np.mean(cv_f1s)),
        std_f1=float(np.std(cv_f1s)),
        selected_features=selected,
        avg_best_iteration=avg_best_iter,
        final_train_metrics=final_train_metrics,
        final_test_metrics=final_test_metrics,
    )

    return final_predictor, report
