#!/usr/bin/env python3
"""Phase 28 — Fisher Transform & Z-Score Mean Reversion: Untested Indicators.

Both indicators use fundamentally different mathematics from existing components:
  - RSI: 14-bar close-to-close gain/loss ratio (bounded 0-100)
  - WillR: 14-bar close position in H-L range (bounded -100 to 0)
  - CCI: 20-bar TP deviation from SMA (unbounded, linear)
  - Fisher: arctanh-normalized median price (unbounded, NONLINEAR)
  - Z-Score: standard deviations from rolling mean (unbounded, statistical)

Fisher Transform amplifies extremes via nonlinear transformation.
Z-Score is a pure statistical measure of price deviation.

Plan:
  PART 1: Fisher Transform MR + MTF (1h) — period + threshold grid
  PART 2: Z-Score MR + MTF (1h) — period + threshold grid
  PART 3: Best new indicator on 15m (if 1h >= 55%)
  PART 4: Best as 5th component or WillR replacement in cross-TF
  PART 5: Summary
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pandas_ta as ta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators.basic import BasicIndicators
from src.backtest.engine import BacktestEngine
from src.backtest.walk_forward import (
    CrossTFComponent,
    CrossTFReport,
    WalkForwardAnalyzer,
)
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.willr_mean_reversion import WilliamsRMeanReversionStrategy
from src.strategy.fisher_mean_reversion import (
    FisherMeanReversionStrategy,
    add_fisher,
)
from src.strategy.base import BaseStrategy, Signal, TradeSignal

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase28")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase28.log", mode="w")
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
))
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)

for name in [
    "src.backtest.engine",
    "src.strategy.mtf_filter",
]:
    logging.getLogger(name).setLevel(logging.WARNING)

wf_logger = logging.getLogger("src.backtest.walk_forward")
wf_logger.setLevel(logging.INFO)
wf_logger.handlers.clear()
wf_logger.addHandler(fh)
wf_logger.addHandler(sh)


# ─── Z-Score MR Strategy ────────────────────────────────────────

class ZScoreMeanReversionStrategy(BaseStrategy):
    """Fade extremes using Z-Score of close price + BB confirmation.

    Z-Score = (close - SMA(close, N)) / STDEV(close, N)

    Entry rules:
      LONG:  z-score < -threshold AND close <= BB_lower
      SHORT: z-score > +threshold AND close >= BB_upper

    Exit: ATR-based SL/TP.
    """

    name = "zscore_mean_reversion"

    def __init__(
        self,
        zscore_period: int = 20,
        threshold: float = 2.0,
        bb_column_lower: str = "BBL_20_2.0_2.0",
        bb_column_upper: str = "BBU_20_2.0_2.0",
        bb_proximity: float = 1.01,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        cooldown_bars: int = 6,
        symbol: str = "BTC/USDT:USDT",
    ) -> None:
        self.zscore_period = zscore_period
        self.threshold = threshold
        self.bb_lower = bb_column_lower
        self.bb_upper = bb_column_upper
        self.bb_proximity = bb_proximity
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.cooldown_bars = cooldown_bars
        self._last_entry_idx = -999
        self.symbol = symbol
        self._zscore_col = f"ZSCORE_{zscore_period}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 30:
            return self._hold(df)

        last = df.iloc[-1]
        close = float(last["close"])
        ts = df.index[-1]

        zscore = last.get(self._zscore_col)
        atr = last.get("atr_14")
        bb_low = last.get(self.bb_lower)
        bb_up = last.get(self.bb_upper)

        if any(pd.isna(v) for v in [zscore, atr, bb_low, bb_up]):
            return self._hold(df)

        zscore = float(zscore)
        atr = float(atr)
        bb_low = float(bb_low)
        bb_up = float(bb_up)

        current_idx = len(df)
        if current_idx - self._last_entry_idx < self.cooldown_bars:
            return self._hold(df)

        # LONG: z-score oversold + near lower BB
        if zscore < -self.threshold and close <= bb_low * self.bb_proximity:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            confidence = min(1.0, 0.55 + abs(zscore + self.threshold) / 5.0)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.LONG,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={"strategy": self.name, "zscore": zscore},
            )

        # SHORT: z-score overbought + near upper BB
        if zscore > self.threshold and close >= bb_up / self.bb_proximity:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            confidence = min(1.0, 0.55 + abs(zscore - self.threshold) / 5.0)
            self._last_entry_idx = current_idx
            return TradeSignal(
                signal=Signal.SHORT,
                symbol=self.symbol,
                price=close,
                timestamp=ts,
                confidence=confidence,
                stop_loss=sl,
                take_profit=tp,
                metadata={"strategy": self.name, "zscore": zscore},
            )

        return self._hold(df)

    def _hold(self, df: pd.DataFrame) -> TradeSignal:
        return TradeSignal(
            signal=Signal.HOLD,
            symbol=self.symbol,
            price=float(df["close"].iloc[-1]),
            timestamp=df.index[-1],
        )

    def get_required_indicators(self) -> list[str]:
        return [self._zscore_col, "atr_14", self.bb_lower, self.bb_upper]


# ─── Data ─────────────────────────────────────────────────────────

def load_data(timeframe: str) -> pd.DataFrame:
    path = ROOT / f"data/processed/BTC_USDT_USDT_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


def add_zscore(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    col = f"ZSCORE_{period}"
    if col not in df.columns:
        result = ta.zscore(df["close"], length=period)
        if result is not None:
            df[col] = result
    return df


def add_willr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    col = f"WILLR_{period}"
    if col not in df.columns:
        df[col] = ta.willr(df["high"], df["low"], df["close"], length=period)
    return df


# ─── Strategy Factories ──────────────────────────────────────────

def make_fisher_1h(
    length: int = 9, threshold: float = 2.0, cooldown: int = 6,
) -> MultiTimeframeFilter:
    base = FisherMeanReversionStrategy(
        fisher_length=length, threshold=threshold,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


def make_fisher_15m(
    length: int = 9, threshold: float = 2.0, cooldown: int = 12,
) -> MultiTimeframeFilter:
    base = FisherMeanReversionStrategy(
        fisher_length=length, threshold=threshold,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


def make_zscore_1h(
    period: int = 20, threshold: float = 2.0, cooldown: int = 6,
) -> MultiTimeframeFilter:
    base = ZScoreMeanReversionStrategy(
        zscore_period=period, threshold=threshold,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


def make_zscore_15m(
    period: int = 20, threshold: float = 2.0, cooldown: int = 12,
) -> MultiTimeframeFilter:
    base = ZScoreMeanReversionStrategy(
        zscore_period=period, threshold=threshold,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=cooldown,
    )
    return MultiTimeframeFilter(base)


def make_rsi_1h() -> MultiTimeframeFilter:
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def make_dc_1h() -> MultiTimeframeFilter:
    base = DonchianTrendStrategy(
        entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
        vol_mult=0.8, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def make_rsi_15m() -> MultiTimeframeFilter:
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
    )
    return MultiTimeframeFilter(base)


def make_willr_1h() -> MultiTimeframeFilter:
    base = WilliamsRMeanReversionStrategy(
        willr_period=14, oversold_level=90.0, overbought_level=90.0,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


# ─── Logging Helpers ──────────────────────────────────────────────

def log_wf_detail(name: str, report) -> None:
    for w in report.windows:
        marker = "+" if w.out_of_sample.total_return > 0 else "-"
        logger.info(
            "    W%d [%s ~ %s]: IS %+.2f%% | OOS %+.2f%% | %d trades %s",
            w.window_id, w.test_start, w.test_end,
            w.in_sample.total_return, w.out_of_sample.total_return,
            w.out_of_sample.total_trades, marker,
        )
    logger.info(
        "  %s: OOS %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d | "
        "Sharpe: %.2f | MaxDD: %.1f%%",
        name, report.oos_total_return,
        int(report.robustness_score * 100),
        sum(1 for w in report.windows if w.out_of_sample.total_return > 0),
        report.total_windows,
        report.oos_total_trades,
        report.oos_avg_sharpe,
        max(
            (abs(w.out_of_sample.max_drawdown) for w in report.windows),
            default=0,
        ),
    )


def log_cross_tf_detail(name: str, report: CrossTFReport) -> None:
    for w in report.windows:
        parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
        marker = "+" if w.weighted_return > 0 else "-"
        logger.info(
            "    W%d [%s ~ %s]: %s -> %+.2f%% %s",
            w.window_id, w.test_start, w.test_end,
            " | ".join(parts), w.weighted_return, marker,
        )
    logger.info(
        "  %s: OOS %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
        name, report.oos_total_return,
        int(report.robustness_score * 100),
        report.oos_profitable_windows, report.total_windows,
        report.total_trades,
    )


# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 28 — Fisher Transform & Z-Score Mean Reversion")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Fisher: arctanh nonlinear transform — amplifies extremes")
    logger.info("  Z-Score: (close - SMA) / STDEV — pure statistical deviation")
    logger.info("  Both are mathematically distinct from RSI/WillR/CCI.")
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")

    # Add Fisher Transform columns for different periods
    for length in [5, 9, 13]:
        df_1h = add_fisher(df_1h, length=length)
    df_15m = add_fisher(df_15m, length=9)

    # Add Z-Score columns for different periods
    for period in [14, 20, 30]:
        df_1h = add_zscore(df_1h, period=period)
    df_15m = add_zscore(df_15m, period=20)

    # Add WillR for portfolio comparison
    df_1h = add_willr(df_1h, 14)

    logger.info("1h data:  %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("15m data: %d bars (%s ~ %s)", len(df_15m),
                df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")
    wf_1h = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
    wf_15m = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)
    wf_xtf = WalkForwardAnalyzer(n_windows=9)

    # Track best results
    all_results: dict[str, tuple] = {}  # name -> (rob%, oos%, trades)
    best_fisher_1h = ("", 0.0, None)   # (name, rob%, report)
    best_zscore_1h = ("", 0.0, None)

    # ═════════════════════════════════════════════════════════════
    #   PART 1: Fisher Transform MR + MTF (1h) — Grid Search
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 1: Fisher Transform MR + MTF on 1h")
    logger.info("-" * 72)
    logger.info("  Testing: periods=[5,9,13] x thresholds=[1.5,2.0,2.5,3.0]")
    logger.info("  Config: SL=2.0ATR, TP=3.0ATR, cool=6, BB confirm")
    logger.info("")

    for length in [5, 9, 13]:
        for threshold in [1.5, 2.0, 2.5, 3.0]:
            name = f"Fisher_p{length}_t{threshold}"
            factory = lambda l=length, t=threshold: make_fisher_1h(
                length=l, threshold=t,
            )
            report = wf_1h.run(factory, df_1h, htf_df=df_4h)
            rob = report.robustness_score * 100
            log_wf_detail(name, report)
            logger.info("")

            all_results[name] = (rob, report.oos_total_return, report.oos_total_trades)
            if rob > best_fisher_1h[1] or (
                rob == best_fisher_1h[1] and report.oos_total_return >
                (best_fisher_1h[2].oos_total_return if best_fisher_1h[2] else -999)
            ):
                best_fisher_1h = (name, rob, report)

    logger.info("  PART 1 BEST: %s (%.0f%% rob, %+.2f%% OOS)",
                best_fisher_1h[0], best_fisher_1h[1],
                best_fisher_1h[2].oos_total_return if best_fisher_1h[2] else 0)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: Z-Score MR + MTF (1h) — Grid Search
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 2: Z-Score MR + MTF on 1h")
    logger.info("-" * 72)
    logger.info("  Testing: periods=[14,20,30] x thresholds=[1.5,2.0,2.5,3.0]")
    logger.info("  Config: SL=2.0ATR, TP=3.0ATR, cool=6, BB confirm")
    logger.info("")

    for period in [14, 20, 30]:
        for threshold in [1.5, 2.0, 2.5, 3.0]:
            name = f"ZScore_p{period}_t{threshold}"
            factory = lambda p=period, t=threshold: make_zscore_1h(
                period=p, threshold=t,
            )
            report = wf_1h.run(factory, df_1h, htf_df=df_4h)
            rob = report.robustness_score * 100
            log_wf_detail(name, report)
            logger.info("")

            all_results[name] = (rob, report.oos_total_return, report.oos_total_trades)
            if rob > best_zscore_1h[1] or (
                rob == best_zscore_1h[1] and report.oos_total_return >
                (best_zscore_1h[2].oos_total_return if best_zscore_1h[2] else -999)
            ):
                best_zscore_1h = (name, rob, report)

    logger.info("  PART 2 BEST: %s (%.0f%% rob, %+.2f%% OOS)",
                best_zscore_1h[0], best_zscore_1h[1],
                best_zscore_1h[2].oos_total_return if best_zscore_1h[2] else 0)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Best indicators on 15m
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 3: Best indicators on 15m")
    logger.info("-" * 72)

    best_overall = max(
        [best_fisher_1h, best_zscore_1h],
        key=lambda x: (x[1], x[2].oos_total_return if x[2] else -999),
    )

    if best_overall[1] >= 55:
        logger.info("  Best 1h = %s (%.0f%% rob) — testing on 15m",
                     best_overall[0], best_overall[1])
        logger.info("")

        # Test Fisher on 15m if it was best
        if best_fisher_1h[1] >= 55:
            fisher_name = best_fisher_1h[0]
            fp = int(fisher_name.split("_p")[1].split("_")[0])
            ft = float(fisher_name.split("_t")[1])

            for t_adj in [-0.5, 0, +0.5]:
                t = round(ft + t_adj, 1)
                if t < 1.0:
                    continue
                name = f"Fisher_15m_p{fp}_t{t}"
                factory = lambda l=fp, th=t: make_fisher_15m(length=l, threshold=th)
                report = wf_15m.run(factory, df_15m, htf_df=df_4h)
                log_wf_detail(name, report)
                logger.info("")

        # Test Z-Score on 15m if it was best
        if best_zscore_1h[1] >= 55:
            zscore_name = best_zscore_1h[0]
            zp = int(zscore_name.split("_p")[1].split("_")[0])
            zt = float(zscore_name.split("_t")[1])

            for t_adj in [-0.5, 0, +0.5]:
                t = round(zt + t_adj, 1)
                if t < 1.0:
                    continue
                name = f"ZScore_15m_p{zp}_t{t}"
                factory = lambda p=zp, th=t: make_zscore_15m(period=p, threshold=th)
                report = wf_15m.run(factory, df_15m, htf_df=df_4h)
                log_wf_detail(name, report)
                logger.info("")
    else:
        logger.info("  SKIP — best 1h robustness %.0f%% < 55%%", best_overall[1])
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Cross-TF Portfolio Integration
    # ═════════════════════════════════════════════════════════════
    logger.info("-" * 72)
    logger.info("  PART 4: Cross-TF Portfolio Integration")
    logger.info("-" * 72)

    # Use the single best indicator (Fisher or Z-Score)
    if best_overall[1] >= 55:
        logger.info("  Baseline: 4-comp RSI/DC/RSI15/WR = 88%% rob, +23.98%% OOS")
        logger.info("  Testing: %s as 5th component / WillR replacement",
                     best_overall[0])
        logger.info("")

        # Parse best config to build factory
        best_name = best_overall[0]
        if best_name.startswith("Fisher"):
            bp = int(best_name.split("_p")[1].split("_")[0])
            bt = float(best_name.split("_t")[1])
            new_factory = lambda: make_fisher_1h(length=bp, threshold=bt)
            new_label = "1hFisher"
        else:
            bp = int(best_name.split("_p")[1].split("_")[0])
            bt = float(best_name.split("_t")[1])
            new_factory = lambda: make_zscore_1h(period=bp, threshold=bt)
            new_label = "1hZScore"

        # 5-component test
        logger.info("  --- As 5th Component ---")
        logger.info("")

        weight_configs_5 = [
            (10, 40, 10, 20, 20),
            (10, 35, 10, 25, 20),
            (15, 35, 10, 20, 20),
        ]
        for w_rsi, w_dc, w_rsi15, w_wr, w_new in weight_configs_5:
            name = f"5comp_{w_rsi}/{w_dc}/{w_rsi15}/{w_wr}/{w_new}"
            report = wf_xtf.run_cross_tf([
                CrossTFComponent(
                    strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
                    engine=engine_1h, weight=w_rsi / 100, label="1hRSI",
                ),
                CrossTFComponent(
                    strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
                    engine=engine_1h, weight=w_dc / 100, label="1hDC",
                ),
                CrossTFComponent(
                    strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
                    engine=engine_15m, weight=w_rsi15 / 100, label="15mRSI",
                ),
                CrossTFComponent(
                    strategy_factory=make_willr_1h, df=df_1h, htf_df=df_4h,
                    engine=engine_1h, weight=w_wr / 100, label="1hWillR",
                ),
                CrossTFComponent(
                    strategy_factory=new_factory, df=df_1h, htf_df=df_4h,
                    engine=engine_1h, weight=w_new / 100, label=new_label,
                ),
            ])
            log_cross_tf_detail(name, report)
            logger.info("")

        # WillR replacement test (4-comp)
        logger.info("  --- Replacing WillR (4-comp) ---")
        logger.info("")

        replace_configs = [
            (15, 50, 10, 25),
            (20, 40, 15, 25),
            (15, 40, 15, 30),
        ]
        for w_rsi, w_dc, w_rsi15, w_new in replace_configs:
            name = f"4comp_new_{w_rsi}/{w_dc}/{w_rsi15}/{w_new}"
            report = wf_xtf.run_cross_tf([
                CrossTFComponent(
                    strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
                    engine=engine_1h, weight=w_rsi / 100, label="1hRSI",
                ),
                CrossTFComponent(
                    strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
                    engine=engine_1h, weight=w_dc / 100, label="1hDC",
                ),
                CrossTFComponent(
                    strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
                    engine=engine_15m, weight=w_rsi15 / 100, label="15mRSI",
                ),
                CrossTFComponent(
                    strategy_factory=new_factory, df=df_1h, htf_df=df_4h,
                    engine=engine_1h, weight=w_new / 100, label=new_label,
                ),
            ])
            log_cross_tf_detail(name, report)
            logger.info("")
    else:
        logger.info("  SKIP — no indicator reached 55%% robustness")
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 5: Summary
    # ═════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("  PHASE 28 SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    logger.info("  Part 1 — Fisher Transform MR (1h standalone):")
    for name, (rob, oos, trades) in sorted(all_results.items()):
        if name.startswith("Fisher"):
            logger.info("    %s: OOS %+.2f%% | Rob %d%% | Trades %d",
                         name, oos, int(rob), trades)
    logger.info("  Best: %s (%.0f%% rob)", best_fisher_1h[0], best_fisher_1h[1])
    logger.info("")

    logger.info("  Part 2 — Z-Score MR (1h standalone):")
    for name, (rob, oos, trades) in sorted(all_results.items()):
        if name.startswith("ZScore"):
            logger.info("    %s: OOS %+.2f%% | Rob %d%% | Trades %d",
                         name, oos, int(rob), trades)
    logger.info("  Best: %s (%.0f%% rob)", best_zscore_1h[0], best_zscore_1h[1])
    logger.info("")

    logger.info("  Reference (current production):")
    logger.info("    4-comp 15/50/10/25: 88%% rob, +23.98%% OOS")
    logger.info("    WillR standalone: 77%% rob, +19.17%% OOS")
    logger.info("    RSI 1h standalone: 66%% rob, +13.29%% OOS")
    logger.info("    CCI standalone: 66%% rob, +13.48%% OOS")
    logger.info("    Ichimoku standalone: 66%% rob, +2.06%% OOS")
    logger.info("")
    logger.info("  Phase 28 complete.")


if __name__ == "__main__":
    main()
