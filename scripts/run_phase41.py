#!/usr/bin/env python3
"""Phase 41 — MTF Filter Response Speed Optimization.

40 phases tested indicators, weights, exits, directions — but NEVER varied
the MTF EMA periods themselves.  The current 4h EMA_20 vs EMA_50 reacts
too slowly: 62K→73K (+18%) rally took ~2 weeks for an EMA cross, meaning
the filter stayed BEARISH the entire move and only allowed SHORT entries.

Tests 5 MTF configurations with the production 4-comp portfolio
(1hRSI/1hDC/15mRSI/1hWillR, 15/50/10/25, 9-window Walk-Forward):

| Config | Setting                 | Reaction Speed    | HTF Data |
|--------|-------------------------|-------------------|----------|
| 1      | 4h EMA_20 vs EMA_50     | Slow (baseline)   | df_4h    |
| 2      | 4h EMA_10 vs EMA_20     | ~2x faster        | df_4h    |
| 3      | 4h close > EMA_20       | Fastest            | df_4h    |
| 4      | 4h close > EMA_50       | Fast              | df_4h    |
| 5      | 1h EMA_20 vs EMA_50     | ~4x faster        | df_1h    |

Baseline: 4-comp 15/50/10/25, 88% rob, +23.98% OOS, 236 trades.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas_ta as ta
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators.basic import BasicIndicators
from src.backtest.engine import BacktestEngine
from src.backtest.walk_forward import (
    CrossTFComponent,
    WalkForwardAnalyzer,
)
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.willr_mean_reversion import WilliamsRMeanReversionStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase41")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase41.log", mode="w")
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
))
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)

for name in ["src.backtest.engine", "src.strategy.mtf_filter"]:
    logging.getLogger(name).setLevel(logging.WARNING)

wf_logger = logging.getLogger("src.backtest.walk_forward")
wf_logger.setLevel(logging.INFO)
wf_logger.handlers.clear()
wf_logger.addHandler(fh)
wf_logger.addHandler(sh)


# ─── Data ─────────────────────────────────────────────────────────

def load_data(timeframe: str) -> pd.DataFrame:
    path = ROOT / f"data/processed/BTC_USDT_USDT_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


def add_willr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    col = f"WILLR_{period}"
    if col not in df.columns:
        df[col] = ta.willr(df["high"], df["low"], df["close"], length=period)
    return df


def add_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Add EMA column if not already present."""
    col = f"ema_{period}"
    if col not in df.columns:
        df[col] = ta.ema(df["close"], length=period)
    return df


# ─── Strategy Factories ──────────────────────────────────────────

def make_mtf_strategy(
    base_factory,
    trend_mode: str = "ema_cross",
    fast_ema: int = 20,
    slow_ema: int = 50,
    price_ema: int = 20,
    extreme_oversold_rsi: float = 20.0,
    extreme_overbought_rsi: float = 70.0,
    extreme_oversold_willr: float = -97.0,
    extreme_overbought_willr: float = -10.0,
):
    """Wrap a base strategy factory with parameterized MTF filter."""
    def factory():
        return MultiTimeframeFilter(
            base_factory(),
            trend_mode=trend_mode,
            fast_ema_period=fast_ema,
            slow_ema_period=slow_ema,
            price_ema_period=price_ema,
            extreme_oversold_rsi=extreme_oversold_rsi,
            extreme_overbought_rsi=extreme_overbought_rsi,
            extreme_oversold_willr=extreme_oversold_willr,
            extreme_overbought_willr=extreme_overbought_willr,
        )
    return factory


def make_rsi_1h_base():
    return RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )


def make_dc_1h_base():
    return DonchianTrendStrategy(
        entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
        vol_mult=0.8, cooldown_bars=6,
    )


def make_rsi_15m_base():
    return RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
    )


def make_willr_1h_base():
    return WilliamsRMeanReversionStrategy(
        willr_period=14, oversold_level=90.0, overbought_level=90.0,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )


def log_window_details(report, logger_fn=logger.info):
    """Log per-window breakdown."""
    for w in report.windows:
        if hasattr(w, "components") and w.components:
            parts = [f"{cr.label} {cr.oos_return:+.2f}%" for cr in w.components]
            marker = "+" if w.weighted_return > 0 else "-"
            logger_fn("    W%d: %s -> %+.2f%% %s",
                       w.window_id, " | ".join(parts), w.weighted_return, marker)
        else:
            ret = w.oos_return if hasattr(w, "oos_return") else w.weighted_return
            marker = "+" if ret > 0 else "-"
            logger_fn("    W%d: %+.2f%% %s", w.window_id, ret, marker)


# ─── Build 4-comp portfolio for a given MTF config ───────────────

def build_components(
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    htf_df: pd.DataFrame,
    engine_1h: BacktestEngine,
    engine_15m: BacktestEngine,
    trend_mode: str = "ema_cross",
    fast_ema: int = 20,
    slow_ema: int = 50,
    price_ema: int = 20,
    label_prefix: str = "",
) -> list[CrossTFComponent]:
    """Build 4-comp CrossTFComponent list with given MTF params."""

    # Common MTF kwargs (extreme override = Phase 40 deployed thresholds)
    mtf_kw = dict(
        trend_mode=trend_mode,
        fast_ema=fast_ema,
        slow_ema=slow_ema,
        price_ema=price_ema,
        extreme_oversold_rsi=20.0,
        extreme_overbought_rsi=70.0,
    )

    # RSI strategies get RSI extreme override only
    mtf_kw_rsi = {**mtf_kw, "extreme_oversold_willr": -100.0, "extreme_overbought_willr": 0.0}

    # WillR strategy gets WillR extreme override only
    mtf_kw_willr = {**mtf_kw, "extreme_oversold_rsi": 0.0, "extreme_overbought_rsi": 100.0,
                    "extreme_oversold_willr": -97.0, "extreme_overbought_willr": -10.0}

    # DC gets no extreme override
    mtf_kw_dc = {**mtf_kw, "extreme_oversold_rsi": 0.0, "extreme_overbought_rsi": 100.0,
                 "extreme_oversold_willr": -100.0, "extreme_overbought_willr": 0.0}

    pfx = f"{label_prefix}_" if label_prefix else ""

    return [
        CrossTFComponent(
            strategy_factory=make_mtf_strategy(make_rsi_1h_base, **mtf_kw_rsi),
            df=df_1h, htf_df=htf_df,
            engine=engine_1h, weight=0.15, label=f"{pfx}1hRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_mtf_strategy(make_dc_1h_base, **mtf_kw_dc),
            df=df_1h, htf_df=htf_df,
            engine=engine_1h, weight=0.50, label=f"{pfx}1hDC",
        ),
        CrossTFComponent(
            strategy_factory=make_mtf_strategy(make_rsi_15m_base, **mtf_kw_rsi),
            df=df_15m, htf_df=htf_df,
            engine=engine_15m, weight=0.10, label=f"{pfx}15mRSI",
        ),
        CrossTFComponent(
            strategy_factory=make_mtf_strategy(make_willr_1h_base, **mtf_kw_willr),
            df=df_1h, htf_df=htf_df,
            engine=engine_1h, weight=0.25, label=f"{pfx}1hWillR",
        ),
    ]


# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 41 — MTF Filter Response Speed Optimization")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  40 phases tested indicators — NEVER varied MTF EMA periods.")
    logger.info("  Current 4h EMA20 vs EMA50 too slow: 62K→73K rally stayed BEARISH.")
    logger.info("  Testing 5 MTF configs with 4-comp portfolio (15/50/10/25).")
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")
    df_1h = add_willr(df_1h, 14)

    # Config 2 needs ema_10 on 4h
    df_4h = add_ema(df_4h, 10)

    logger.info("  1h data:  %d bars (%s ~ %s)",
                len(df_1h), df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("  15m data: %d bars (%s ~ %s)",
                len(df_15m), df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("  4h data:  %d bars (%s ~ %s)",
                len(df_4h), df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")

    # ═════════════════════════════════════════════════════════════
    #   5 MTF Configs
    # ═════════════════════════════════════════════════════════════

    configs = [
        {
            "name": "4h_EMA20v50 (baseline)",
            "htf_df": df_4h,
            "trend_mode": "ema_cross",
            "fast_ema": 20,
            "slow_ema": 50,
            "price_ema": 20,
        },
        {
            "name": "4h_EMA10v20",
            "htf_df": df_4h,
            "trend_mode": "ema_cross",
            "fast_ema": 10,
            "slow_ema": 20,
            "price_ema": 20,
        },
        {
            "name": "4h_close>EMA20",
            "htf_df": df_4h,
            "trend_mode": "price_vs_ema",
            "fast_ema": 20,
            "slow_ema": 50,
            "price_ema": 20,
        },
        {
            "name": "4h_close>EMA50",
            "htf_df": df_4h,
            "trend_mode": "price_vs_ema",
            "fast_ema": 20,
            "slow_ema": 50,
            "price_ema": 50,
        },
        {
            "name": "1h_EMA20v50",
            "htf_df": df_1h,
            "trend_mode": "ema_cross",
            "fast_ema": 20,
            "slow_ema": 50,
            "price_ema": 20,
        },
    ]

    results = []

    for i, cfg in enumerate(configs, 1):
        logger.info("-" * 72)
        logger.info("  Config %d: %s", i, cfg["name"])
        logger.info("-" * 72)
        logger.info("")

        components = build_components(
            df_1h=df_1h,
            df_15m=df_15m,
            htf_df=cfg["htf_df"],
            engine_1h=engine_1h,
            engine_15m=engine_15m,
            trend_mode=cfg["trend_mode"],
            fast_ema=cfg["fast_ema"],
            slow_ema=cfg["slow_ema"],
            price_ema=cfg["price_ema"],
            label_prefix=f"C{i}",
        )

        wf = WalkForwardAnalyzer(n_windows=9)
        report = wf.run_cross_tf(components)
        rob = int(report.robustness_score * 100)

        # Find W2 return (window 2 — the hard one)
        w2_ret = None
        for w in report.windows:
            if w.window_id == 2:
                w2_ret = w.weighted_return
                break

        results.append({
            "name": cfg["name"],
            "rob": rob,
            "oos": report.oos_total_return,
            "trades": report.total_trades,
            "w2_ret": w2_ret,
            "report": report,
        })

        logger.info("  %s: Rob=%d%%, OOS=%+.2f%%, Trades=%d, W2=%s",
                    cfg["name"], rob, report.oos_total_return,
                    report.total_trades,
                    f"{w2_ret:+.2f}%" if w2_ret is not None else "N/A")
        log_window_details(report)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   Summary
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 41 SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    baseline = results[0]

    header = f"{'Config':<22s} | {'Rob%':>4s} | {'OOS Return':>11s} | {'Trades':>6s} | {'W2 Return':>10s} | {'Delta':>7s}"
    logger.info("  %s", header)
    logger.info("  %s", "-" * len(header))

    for r in results:
        delta = r["oos"] - baseline["oos"]
        w2_str = f"{r['w2_ret']:+.2f}%" if r["w2_ret"] is not None else "N/A"
        marker = " *" if r["rob"] >= baseline["rob"] and delta > 0 else ""
        logger.info(
            "  %-22s | %3d%% | %+10.2f%% | %6d | %10s | %+6.2f%%%s",
            r["name"], r["rob"], r["oos"], r["trades"],
            w2_str, delta, marker,
        )

    logger.info("")

    # Find best
    maintained = [r for r in results if r["rob"] >= baseline["rob"]]
    if maintained:
        best = max(maintained, key=lambda x: x["oos"])
        logger.info("  BEST (rob >= %d%%): %s → Rob=%d%%, OOS=%+.2f%%",
                    baseline["rob"], best["name"], best["rob"], best["oos"])
        delta = best["oos"] - baseline["oos"]
        if delta > 0:
            logger.info("  ✓ OOS improved by %+.2f%% — DEPLOY CANDIDATE", delta)
        elif best["name"] == baseline["name"]:
            logger.info("  = Baseline remains best — NO CHANGE NEEDED")
        else:
            logger.info("  ≈ Similar performance — marginal improvement")
    else:
        logger.info("  ✗ All configs below baseline robustness — NO DEPLOY")

    improved = [r for r in results[1:] if r["rob"] > baseline["rob"]]
    if improved:
        logger.info("")
        logger.info("  BREAKTHROUGH — robustness exceeded baseline!")
        for r in improved:
            logger.info("    %s: Rob=%d%% (baseline=%d%%)",
                        r["name"], r["rob"], baseline["rob"])

    logger.info("")
    logger.info("  Phase 41 complete.")


if __name__ == "__main__":
    main()
