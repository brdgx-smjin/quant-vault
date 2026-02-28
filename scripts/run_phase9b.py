#!/usr/bin/env python3
"""Phase 9b — ML Meta-Model: Regime Classifier for Strategy Selection.

System prompt task 4: "전략 자체를 ML로 만들지 말고, 어떤 전략을 쓸지
결정하는 메타 모델로 접근"

Tests whether ML-based regime classification can improve strategy selection
over simple equal-weight portfolio.

Prior art:
  - Simple ADX threshold (Phase 8): 86% WF rob but Full -1.82% — UNRELIABLE
  - ML Regime Strategy (Phase 10c): 57% rob at 7w — worse than ADX
  - Simple RSI+DC 50/50 portfolio: 71% rob (7w)
  - Cross-TF 33/33/34: 88% rob (9w) — gold standard
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config.settings import DATA_DIR, SYMBOL
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.walk_forward import WalkForwardAnalyzer
from src.indicators.basic import BasicIndicators
from src.strategy.base import BaseStrategy, Signal, TradeSignal
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.monitoring.logger import setup_logging

logger = setup_logging("phase9b")

logging.getLogger("src.backtest.engine").setLevel(logging.WARNING)

SYMBOL_FILE = SYMBOL.replace("/", "_").replace(":", "_")
N_WINDOWS = 7


def load_data(timeframe: str) -> pd.DataFrame:
    path = f"{DATA_DIR}/processed/{SYMBOL_FILE}_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    df.dropna(inplace=True)
    return df


def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features for regime classification."""
    feat = pd.DataFrame(index=df.index)
    feat["adx"] = df["ADX_14"]
    feat["atr_norm"] = df["atr_14"] / df["close"]
    bbu = df.get("BBU_20_2.0_2.0")
    bbl = df.get("BBL_20_2.0_2.0")
    if bbu is not None and bbl is not None:
        feat["bb_width"] = (bbu - bbl) / df["close"]
    else:
        feat["bb_width"] = 0.0
    feat["ema_trend"] = (df["ema_20"] - df["ema_50"]) / df["close"]
    feat["atr_change"] = df["atr_14"].pct_change(6)
    feat["adx_change"] = df["ADX_14"].diff(6)
    feat["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    feat.dropna(inplace=True)
    return feat


def build_regime_labels(df: pd.DataFrame, future_bars: int = 12) -> pd.Series:
    """Label: 1=trending (|return|>2%), 0=ranging."""
    future_ret = df["close"].pct_change(future_bars).shift(-future_bars)
    return (future_ret.abs() > 0.02).astype(int)


class MLRegimeMetaStrategy(BaseStrategy):
    """Select DC or RSI MR based on ML regime prediction."""

    name = "ml_regime_meta"

    def __init__(self, model, feature_cols, dc_strategy, rsi_strategy):
        self.model = model
        self.feature_cols = feature_cols
        self.dc_strategy = dc_strategy
        self.rsi_strategy = rsi_strategy
        self.symbol = SYMBOL

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 50:
            return self._hold(df)
        feat = build_regime_features(df)
        if len(feat) == 0:
            return self._hold(df)
        try:
            row = feat.iloc[[-1]][self.feature_cols]
            if row.isna().any().any():
                return self._hold(df)
            pred = self.model.predict(row)[0]
        except Exception:
            return self._hold(df)
        if pred == 1:
            sig = self.dc_strategy.generate_signal(df)
        else:
            sig = self.rsi_strategy.generate_signal(df)
        if sig.signal != Signal.HOLD:
            sig.metadata["regime_pred"] = "trend" if pred == 1 else "range"
        return sig

    def _hold(self, df):
        return TradeSignal(
            signal=Signal.HOLD, symbol=self.symbol,
            price=float(df["close"].iloc[-1]), timestamp=df.index[-1],
        )

    def get_required_indicators(self):
        return list(set(
            self.dc_strategy.get_required_indicators()
            + self.rsi_strategy.get_required_indicators() + ["ADX_14"]
        ))


def print_result(name: str, r: BacktestResult) -> None:
    logger.info(
        "  %-40s %+8.2f%% | Shp %5.2f | DD %5.1f%% | WR %5.1f%% | %4d tr | PF %5.2f",
        name, r.total_return, r.sharpe_ratio, r.max_drawdown,
        r.win_rate * 100, r.total_trades, r.profit_factor,
    )


def main() -> None:
    logger.info("=" * 72)
    logger.info("  PHASE 9b — ML Meta-Model: Regime Classifier")
    logger.info("=" * 72)
    logger.info("")

    df_1h = load_data("1h")
    df_4h = load_data("4h")
    logger.info("1h data: %d bars (%s ~ %s)",
                len(df_1h), df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("")

    engine = BacktestEngine(initial_capital=10_000, max_hold_bars=48, freq="1h")

    feature_cols = ["adx", "atr_norm", "bb_width", "ema_trend",
                    "atr_change", "adx_change", "vol_ratio"]

    # ================================================================
    # ML Regime WF Test
    # ================================================================
    logger.info("─" * 72)
    logger.info("  ML Regime Classifier + Strategy Selection (%dw)", N_WINDOWS)
    logger.info("─" * 72)
    logger.info("")

    n = len(df_1h)
    window_size = n // N_WINDOWS
    test_size = int(window_size * 0.3)
    if test_size < 50:
        test_size = 50

    ml_oos_returns = []
    ml_oos_trades = 0

    for i in range(N_WINDOWS):
        test_end_idx = n - (N_WINDOWS - 1 - i) * test_size
        test_start_idx = test_end_idx - test_size
        # Expanding window: train from start
        train_df = df_1h.iloc[:test_start_idx]
        test_df = df_1h.iloc[test_start_idx:test_end_idx]

        if len(train_df) < 200 or len(test_df) < 30:
            ml_oos_returns.append(0.0)
            continue

        feat_train = build_regime_features(train_df)
        labels_train = build_regime_labels(train_df)
        common_idx = feat_train.index.intersection(labels_train.dropna().index)
        feat_train = feat_train.loc[common_idx]
        labels_train = labels_train.loc[common_idx]

        if len(feat_train) < 100:
            logger.info("  W%d: insufficient data (%d)", i + 1, len(feat_train))
            ml_oos_returns.append(0.0)
            continue

        X_train = feat_train[feature_cols]
        y_train = labels_train

        model = XGBClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, gamma=0.2,
            min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric="logloss", verbosity=0,
        )
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)

        # Predict on test regime
        feat_test = build_regime_features(test_df)
        if len(feat_test) > 0:
            test_preds = model.predict(feat_test[feature_cols])
            pct_trend = test_preds.mean() * 100
        else:
            pct_trend = 50.0

        def ml_factory(m=model):
            dc = DonchianTrendStrategy(
                entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
                vol_mult=0.8, cooldown_bars=6,
            )
            rsi = RSIMeanReversionStrategy(
                rsi_oversold=35.0, rsi_overbought=65.0,
                atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
            )
            return MultiTimeframeFilter(
                MLRegimeMetaStrategy(m, feature_cols, dc, rsi)
            )

        oos_result = engine.run(ml_factory(), test_df, htf_df=df_4h)

        logger.info(
            "  W%d: acc=%.3f trend=%.0f%% | OOS %+7.2f%% (WR %.0f%%, %d tr)",
            i + 1, train_acc, pct_trend,
            oos_result.total_return, oos_result.win_rate * 100,
            oos_result.total_trades,
        )

        ml_oos_returns.append(oos_result.total_return)
        ml_oos_trades += oos_result.total_trades

    compounded = 1.0
    for r in ml_oos_returns:
        compounded *= (1 + r / 100)
    ml_total = (compounded - 1) * 100
    ml_profitable = sum(1 for r in ml_oos_returns if r > 0)
    ml_robustness = ml_profitable / len(ml_oos_returns) if ml_oos_returns else 0

    logger.info("")
    logger.info("  ML Regime Meta: OOS %+.2f%% | Robustness: %.0f%% (%d/%d) | Trades: %d",
                ml_total, ml_robustness * 100, ml_profitable,
                len(ml_oos_returns), ml_oos_trades)

    # ================================================================
    # Reference: Simple RSI+DC 50/50
    # ================================================================
    logger.info("")
    logger.info("─" * 72)
    logger.info("  Reference: RSI+DC 50/50 (%dw)", N_WINDOWS)
    logger.info("─" * 72)
    logger.info("")

    wf = WalkForwardAnalyzer(train_ratio=0.7, n_windows=N_WINDOWS, engine=engine)

    def rsi_factory():
        base = RSIMeanReversionStrategy(
            rsi_oversold=35.0, rsi_overbought=65.0,
            atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    def dc_factory():
        base = DonchianTrendStrategy(
            entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
            vol_mult=0.8, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    wf_rsi = wf.run(rsi_factory, df_1h, htf_df=df_4h)
    wf_dc = wf.run(dc_factory, df_1h, htf_df=df_4h)

    n_win = min(len(wf_rsi.windows), len(wf_dc.windows))
    port_oos = []
    port_prof = 0
    for i in range(n_win):
        r_ret = wf_rsi.windows[i].out_of_sample.total_return
        d_ret = wf_dc.windows[i].out_of_sample.total_return
        combined = 0.5 * r_ret + 0.5 * d_ret
        port_oos.append(combined)
        if combined > 0:
            port_prof += 1
        logger.info("  W%d: RSI %+6.2f%% + DC %+6.2f%% -> %+6.2f%%",
                    i + 1, r_ret, d_ret, combined)

    compounded = 1.0
    for r in port_oos:
        compounded *= (1 + r / 100)
    port_total = (compounded - 1) * 100
    port_rob = port_prof / n_win if n_win > 0 else 0

    logger.info("  RSI+DC 50/50: OOS %+.2f%% | Robustness: %.0f%% (%d/%d)",
                port_total, port_rob * 100, port_prof, n_win)

    # ================================================================
    # FINAL COMPARISON
    # ================================================================
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 9b — FINAL COMPARISON")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  %-30s %8s %6s %5s", "Approach", "OOS Ret", "WF Rob", "Trades")
    logger.info("  " + "-" * 55)
    logger.info("  %-30s %+7.2f%% %5.0f%% %5d",
                "ML Regime Meta-Model", ml_total, ml_robustness * 100, ml_oos_trades)
    logger.info("  %-30s %+7.2f%% %5.0f%% %5s",
                "Simple RSI+DC 50/50", port_total, port_rob * 100, "-")
    logger.info("  %-30s %+7.2f%% %5.0f%% %5s",
                "Cross-TF 33/33/34 (9w ref)", 17.44, 88, "162")
    logger.info("")

    if ml_robustness > port_rob + 0.05:
        logger.info("  CONCLUSION: ML Meta-Model IMPROVES over simple portfolio.")
    else:
        logger.info("  CONCLUSION: ML regime classification adds NO value.")
        logger.info("  Simple equal-weight portfolio is better or equal.")
        logger.info("  Recommendation: Use Cross-TF 33/33/34 (88%% rob) for production.")

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 9b complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
