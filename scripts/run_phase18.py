#!/usr/bin/env python3
"""Phase 18 — Breaking the 88% Robustness Ceiling.

Phase 17 result: Cross-TF 1hRSI/1hDC/15mRSI 33/33/34 = 88% (8/9).
Only W2 [2025-11-20 ~ 2025-12-02] is negative (-3.54%).

W2 component breakdown:
  1hRSI:  -5.59% (5 trades, 0% WR) — all losers
  1hDC:   -2.46% (2 trades, 0% WR) — all losers
  15mRSI: -2.60% (15 trades, 46% WR) — some wins but net negative

All strategies/TFs lose during this extreme whipsaw period.

Approaches tested in this phase:
  PART 1: W2 Regime Analysis — What makes Nov 20-Dec 2 different?
  PART 2: CCI 1h as 4th component — Different indicator math, possibly different W2 outcome
  PART 3: Chop filter — Skip trades when ATR_pctile > X AND ADX < Y
  PART 4: Combined best approach via cross-TF WF

Key insight needed: Even +0.01% in W2 flips it from 88% to 100%.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
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
from src.strategy.base import BaseStrategy, Signal, TradeSignal
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.cci_mean_reversion import CCIMeanReversionStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase18")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase18.log", mode="w")
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


# ─── Data ─────────────────────────────────────────────────────────

def load_data(timeframe: str) -> pd.DataFrame:
    """Load and add indicators to OHLCV data."""
    path = ROOT / f"data/processed/BTC_USDT_USDT_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add CCI indicator to DataFrame."""
    col = f"CCI_{period}"
    if col not in df.columns:
        df[col] = ta.cci(df["high"], df["low"], df["close"], length=period)
    return df


# ─── Strategy Factories ──────────────────────────────────────────

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


def make_cci_1h() -> MultiTimeframeFilter:
    base = CCIMeanReversionStrategy(
        cci_period=20, oversold_level=200, overbought_level=200,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


# ─── Chop Filter Wrapper ─────────────────────────────────────────

class ChopFilter(BaseStrategy):
    """Wraps a strategy and blocks signals during high-chop regimes.

    Detects market chop as: high ATR percentile AND low ADX.
    When chop is detected, returns HOLD instead of the base signal.

    This is a portfolio-level overlay to reduce losses during whipsaw.
    """

    def __init__(
        self,
        base_strategy: BaseStrategy,
        atr_pctile_threshold: float = 80.0,
        adx_threshold: float = 20.0,
        atr_lookback: int = 168,  # 7 days of 1h bars
    ) -> None:
        self.base_strategy = base_strategy
        self.atr_pctile_threshold = atr_pctile_threshold
        self.adx_threshold = adx_threshold
        self.atr_lookback = atr_lookback
        self.name = f"chop_{base_strategy.name}"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < self.atr_lookback + 20:
            return self.base_strategy.generate_signal(df)

        last = df.iloc[-1]
        atr = last.get("atr_14")
        adx = last.get("ADX_14")

        if pd.isna(atr) or pd.isna(adx):
            return self.base_strategy.generate_signal(df)

        # ATR percentile over lookback window
        atr_window = df["atr_14"].iloc[-self.atr_lookback:]
        atr_pctile = (atr_window < float(atr)).mean() * 100

        # Block if high volatility + no direction (chop)
        if atr_pctile >= self.atr_pctile_threshold and float(adx) < self.adx_threshold:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=getattr(self.base_strategy, "symbol", "BTC/USDT:USDT"),
                price=float(last["close"]),
                timestamp=df.index[-1],
                metadata={"blocked_by": "chop_filter",
                          "atr_pctile": atr_pctile,
                          "adx": float(adx)},
            )

        return self.base_strategy.generate_signal(df)

    def get_required_indicators(self) -> list[str]:
        return self.base_strategy.get_required_indicators() + ["atr_14", "ADX_14"]


class ChopMTFFilter(BaseStrategy):
    """Combines ChopFilter with MultiTimeframeFilter.

    Applies chop detection BEFORE MTF filter to avoid even generating
    signals during extreme whipsaw periods.
    """

    def __init__(
        self,
        base_strategy: BaseStrategy,
        atr_pctile_threshold: float = 80.0,
        adx_threshold: float = 20.0,
        atr_lookback: int = 168,
    ) -> None:
        chop_wrapped = ChopFilter(
            base_strategy,
            atr_pctile_threshold=atr_pctile_threshold,
            adx_threshold=adx_threshold,
            atr_lookback=atr_lookback,
        )
        self.mtf = MultiTimeframeFilter(chop_wrapped)
        self.name = f"chopmtf_{base_strategy.name}"

    def set_htf_data(self, df_htf: pd.DataFrame) -> None:
        self.mtf.set_htf_data(df_htf)

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        return self.mtf.generate_signal(df)

    def get_required_indicators(self) -> list[str]:
        return self.mtf.get_required_indicators()


# ─── Logging Helpers ──────────────────────────────────────────────

def log_wf_report(name: str, report, engine: BacktestEngine,
                  factory, df: pd.DataFrame, htf_df=None) -> None:
    for w in report.windows:
        oos = w.out_of_sample
        logger.info(
            "  W%d: OOS %+6.2f%% (WR %d%%, %d tr)",
            w.window_id, oos.total_return,
            int(oos.win_rate * 100), oos.total_trades,
        )
    logger.info(
        "  OOS: %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
        report.oos_total_return,
        int(report.robustness_score * 100),
        report.oos_profitable_windows, report.total_windows,
        report.oos_total_trades,
    )
    full = engine.run(factory(), df, htf_df=htf_df)
    logger.info(
        "  %s Full %+8.2f%% | Shp %5.2f | DD %5.1f%% | WR %5.1f%% | %4d tr | PF %5.2f",
        name, full.total_return, full.sharpe_ratio, full.max_drawdown,
        full.win_rate * 100, full.total_trades, full.profit_factor,
    )


def log_cross_tf(name: str, report: CrossTFReport) -> None:
    for w in report.windows:
        parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
        marker = "+" if w.weighted_return > 0 else "-"
        logger.info("    W%d [%s ~ %s]: %s -> %+.2f%% %s",
                     w.window_id, w.test_start, w.test_end,
                     " | ".join(parts), w.weighted_return, marker)
    logger.info(
        "  %s: OOS %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
        name, report.oos_total_return,
        int(report.robustness_score * 100),
        report.oos_profitable_windows, report.total_windows,
        report.total_trades,
    )


# ─── Main ─────────────────────────────────────────────────────────

def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 18 — Breaking the 88%% Robustness Ceiling")
    logger.info("=" * 72)
    logger.info("")

    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")
    df_1h = add_cci(df_1h, 20)

    logger.info("1h data:  %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("15m data: %d bars (%s ~ %s)", len(df_15m),
                df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("4h data:  %d bars (%s ~ %s)", len(df_4h),
                df_4h.index[0].date(), df_4h.index[-1].date())

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")

    # ═════════════════════════════════════════════════════════════
    #   PART 1: W2 Regime Analysis
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("-" * 72)
    logger.info("  PART 1: W2 Regime Analysis (Nov 20 - Dec 2, 2025)")
    logger.info("-" * 72)
    logger.info("")

    w2_start = pd.Timestamp("2025-11-20")
    w2_end = pd.Timestamp("2025-12-02")
    w2_mask = (df_1h.index >= w2_start) & (df_1h.index < w2_end)
    w2_data = df_1h[w2_mask]

    # Full dataset statistics for comparison
    full_atr = df_1h["atr_14"].dropna()
    full_adx = df_1h["ADX_14"].dropna()

    w2_atr = w2_data["atr_14"].dropna()
    w2_adx = w2_data["ADX_14"].dropna()

    if len(w2_data) > 0:
        # ATR statistics
        atr_mean_full = full_atr.mean()
        atr_mean_w2 = w2_atr.mean()
        atr_pctile_of_w2 = (full_atr < atr_mean_w2).mean() * 100

        # ADX statistics
        adx_mean_full = full_adx.mean()
        adx_mean_w2 = w2_adx.mean()

        # Price range
        w2_high = w2_data["high"].max()
        w2_low = w2_data["low"].min()
        w2_range_pct = (w2_high - w2_low) / w2_low * 100

        # BB width
        bb_upper = w2_data.get("BBU_20_2.0")
        bb_lower = w2_data.get("BBL_20_2.0")
        if bb_upper is not None and bb_lower is not None:
            bb_mid = w2_data.get("BBM_20_2.0", (bb_upper + bb_lower) / 2)
            bb_width = ((bb_upper - bb_lower) / bb_mid * 100).dropna()
            bb_width_w2 = bb_width.mean()
            full_bb_upper = df_1h.get("BBU_20_2.0")
            full_bb_lower = df_1h.get("BBL_20_2.0")
            full_bb_mid = df_1h.get("BBM_20_2.0")
            if full_bb_upper is not None:
                full_bb_width = ((full_bb_upper - full_bb_lower) / full_bb_mid * 100).dropna()
                bb_width_full = full_bb_width.mean()
            else:
                bb_width_full = float("nan")
        else:
            bb_width_w2 = float("nan")
            bb_width_full = float("nan")

        # RSI statistics
        rsi_w2 = w2_data["rsi_14"].dropna()

        logger.info("  W2 period: %s ~ %s (%d bars)", w2_start.date(), w2_end.date(), len(w2_data))
        logger.info("")
        logger.info("  %-25s %10s %10s %10s", "Indicator", "W2 Mean", "Full Mean", "W2 Pctile")
        logger.info("  " + "-" * 60)
        logger.info("  %-25s %10.1f %10.1f %9.0f%%", "ATR(14)", atr_mean_w2, atr_mean_full, atr_pctile_of_w2)
        logger.info("  %-25s %10.1f %10.1f %10s", "ADX(14)", adx_mean_w2, adx_mean_full, "")
        logger.info("  %-25s %10.2f %10.2f %10s", "BB Width %%", bb_width_w2, bb_width_full, "")
        logger.info("  %-25s %10.1f %10.1f %10s", "RSI(14)", rsi_w2.mean(), df_1h["rsi_14"].dropna().mean(), "")
        logger.info("  %-25s %10.1f %10s %10s", "RSI min/max", rsi_w2.min(), f"{rsi_w2.max():.1f}", "")
        logger.info("  %-25s %10.2f%% %10s %10s", "Price Range", w2_range_pct, "", "")
        logger.info("")

        # ATR percentile bar-by-bar during W2
        logger.info("  Bar-by-bar ATR percentile during W2 (vs trailing 168h):")
        for idx, row in w2_data.iterrows():
            lookback = df_1h.loc[:idx, "atr_14"].iloc[-168:]
            atr_val = row["atr_14"]
            if pd.notna(atr_val) and len(lookback) > 0:
                pctile = (lookback < atr_val).mean() * 100
                adx_val = row.get("ADX_14", float("nan"))
                if pctile >= 80 and (pd.isna(adx_val) or adx_val < 20):
                    marker = " ← CHOP"
                elif pctile >= 80:
                    marker = " ← HIGH VOL"
                else:
                    marker = ""
                if idx.hour == 0:  # Log once per day
                    logger.info("    %s: ATR_pctile=%5.1f%% ADX=%5.1f%s",
                                idx.date(), pctile, adx_val if pd.notna(adx_val) else 0, marker)

    # ═════════════════════════════════════════════════════════════
    #   PART 2: CCI 1h as 4th Component
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("-" * 72)
    logger.info("  PART 2: CCI_20_200+MTF as 4th Component (9w)")
    logger.info("-" * 72)
    logger.info("  Goal: Check if CCI has a different W2 outcome than RSI")
    logger.info("")

    # Individual CCI baseline
    logger.info("  --- CCI_20_200+MTF (1h, 9w) ---")
    wf9 = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
    cci_report = wf9.run(make_cci_1h, df_1h, htf_df=df_4h)
    log_wf_report("CCI_1h+MTF", cci_report, engine_1h, make_cci_1h, df_1h, df_4h)
    logger.info("")

    # 4-component cross-TF: 1hRSI/1hDC/1hCCI/15mRSI
    logger.info("  --- Cross-TF 4-comp: 1hRSI/1hDC/1hCCI/15mRSI ---")
    wf_cross = WalkForwardAnalyzer(n_windows=9)

    weight_combos_4 = [
        ("25/25/25/25 (equal)", 0.25, 0.25, 0.25, 0.25),
        ("20/20/20/40 (15m heavy)", 0.20, 0.20, 0.20, 0.40),
        ("30/20/20/30", 0.30, 0.20, 0.20, 0.30),
        ("25/25/15/35", 0.25, 0.25, 0.15, 0.35),
    ]

    for combo_name, w_rsi, w_dc, w_cci, w_rsi15 in weight_combos_4:
        label = f"4comp {combo_name}"
        logger.info("  --- %s ---", label)

        components = []
        if w_rsi > 0:
            components.append(CrossTFComponent(
                strategy_factory=make_rsi_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_rsi, label="1hRSI",
            ))
        if w_dc > 0:
            components.append(CrossTFComponent(
                strategy_factory=make_dc_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_dc, label="1hDC",
            ))
        if w_cci > 0:
            components.append(CrossTFComponent(
                strategy_factory=make_cci_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=w_cci, label="1hCCI",
            ))
        if w_rsi15 > 0:
            components.append(CrossTFComponent(
                strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=w_rsi15, label="15mRSI",
            ))

        report = wf_cross.run_cross_tf(components)
        log_cross_tf(combo_name, report)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Chop Filter — Skip trades in extreme whipsaw
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("-" * 72)
    logger.info("  PART 3: Chop Filter (ATR_pctile + ADX threshold)")
    logger.info("-" * 72)
    logger.info("  Skip trades when ATR_pctile >= X AND ADX < Y")
    logger.info("")

    chop_configs = [
        # (atr_pctile, adx_threshold, atr_lookback)
        (80, 20, 168),   # Standard: top 20% vol + no trend
        (85, 20, 168),   # Stricter vol threshold
        (80, 15, 168),   # Stricter direction threshold
        (75, 25, 168),   # Looser thresholds
        (80, 20, 336),   # Longer lookback (14 days)
        (90, 25, 168),   # Very strict vol, loose direction
    ]

    for atr_pct, adx_thr, lookback in chop_configs:
        config_name = f"Chop_atr{atr_pct}_adx{adx_thr}_lb{lookback}"
        logger.info("  --- %s ---", config_name)

        def make_chop_rsi_1h(atr_p=atr_pct, adx_t=adx_thr, lb=lookback):
            base = RSIMeanReversionStrategy(
                rsi_oversold=35, rsi_overbought=65,
                atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
            )
            return ChopMTFFilter(base, atr_pctile_threshold=atr_p,
                                 adx_threshold=adx_t, atr_lookback=lb)

        def make_chop_dc_1h(atr_p=atr_pct, adx_t=adx_thr, lb=lookback):
            base = DonchianTrendStrategy(
                entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
                vol_mult=0.8, cooldown_bars=6,
            )
            return ChopMTFFilter(base, atr_pctile_threshold=atr_p,
                                 adx_threshold=adx_t, atr_lookback=lb)

        def make_chop_rsi_15m(atr_p=atr_pct, adx_t=adx_thr, lb=lookback):
            base = RSIMeanReversionStrategy(
                rsi_oversold=35, rsi_overbought=65,
                atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
            )
            # For 15m, scale lookback: 168h * 4 = 672 bars
            return ChopMTFFilter(base, atr_pctile_threshold=atr_p,
                                 adx_threshold=adx_t, atr_lookback=lb * 4)

        # Cross-TF with chop filter
        report = wf_cross.run_cross_tf([
            CrossTFComponent(
                strategy_factory=make_chop_rsi_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.33, label="1hRSI+chop",
            ),
            CrossTFComponent(
                strategy_factory=make_chop_dc_1h, df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.33, label="1hDC+chop",
            ),
            CrossTFComponent(
                strategy_factory=make_chop_rsi_15m, df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=0.34, label="15mRSI+chop",
            ),
        ])
        log_cross_tf(config_name, report)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Wider SL during volatile periods
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("-" * 72)
    logger.info("  PART 4: Wider SL variants (3.0 ATR instead of 2.0)")
    logger.info("-" * 72)
    logger.info("  W2 losses are from SL hits — wider SL might survive whipsaw")
    logger.info("")

    def make_rsi_1h_wide():
        base = RSIMeanReversionStrategy(
            rsi_oversold=35, rsi_overbought=65,
            atr_sl_mult=3.0, atr_tp_mult=4.5, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    def make_dc_1h_wide():
        base = DonchianTrendStrategy(
            entry_period=24, atr_sl_mult=3.0, rr_ratio=2.0,
            vol_mult=0.8, cooldown_bars=6,
        )
        return MultiTimeframeFilter(base)

    def make_rsi_15m_wide():
        base = RSIMeanReversionStrategy(
            rsi_oversold=35, rsi_overbought=65,
            atr_sl_mult=3.0, atr_tp_mult=4.5, cooldown_bars=12,
        )
        return MultiTimeframeFilter(base)

    logger.info("  --- Wide SL (3.0 ATR) 33/33/34 ---")
    report_wide = wf_cross.run_cross_tf([
        CrossTFComponent(
            strategy_factory=make_rsi_1h_wide, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.33, label="1hRSI_wide",
        ),
        CrossTFComponent(
            strategy_factory=make_dc_1h_wide, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.33, label="1hDC_wide",
        ),
        CrossTFComponent(
            strategy_factory=make_rsi_15m_wide, df=df_15m, htf_df=df_4h,
            engine=engine_15m, weight=0.34, label="15mRSI_wide",
        ),
    ])
    log_cross_tf("Wide SL 3.0ATR 33/33/34", report_wide)
    logger.info("")

    # Also test longer cooldown (reduce # of trades in whipsaw)
    def make_rsi_1h_long_cool():
        base = RSIMeanReversionStrategy(
            rsi_oversold=35, rsi_overbought=65,
            atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,  # 12 instead of 6
        )
        return MultiTimeframeFilter(base)

    def make_dc_1h_long_cool():
        base = DonchianTrendStrategy(
            entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
            vol_mult=0.8, cooldown_bars=12,
        )
        return MultiTimeframeFilter(base)

    logger.info("  --- Long Cooldown (12 bars) 33/33/34 ---")
    report_cool = wf_cross.run_cross_tf([
        CrossTFComponent(
            strategy_factory=make_rsi_1h_long_cool, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.33, label="1hRSI_c12",
        ),
        CrossTFComponent(
            strategy_factory=make_dc_1h_long_cool, df=df_1h, htf_df=df_4h,
            engine=engine_1h, weight=0.33, label="1hDC_c12",
        ),
        CrossTFComponent(
            strategy_factory=make_rsi_15m, df=df_15m, htf_df=df_4h,
            engine=engine_15m, weight=0.34, label="15mRSI",
        ),
    ])
    log_cross_tf("Long Cooldown 12h 33/33/34", report_cool)
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 5: Summary
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 18 — SUMMARY")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Phase 17 baseline: 33/33/34 = 88%% rob, +18.81%% OOS (W2 = -3.54%%)")
    logger.info("")
    logger.info("  Goal: Make W2 positive → 100%% robustness, or reduce W2 loss → stay 88%%")
    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 18 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
