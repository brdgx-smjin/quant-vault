#!/usr/bin/env python3
"""Phase 20 — Parameter Stability Analysis.

Validates that the 88% cross-TF portfolio (1hRSI/1hDC/15mRSI 33/33/34)
is robust to parameter perturbation. Tests each key parameter ±1 step from
optimal to verify no cliff-edge degradation.

Methodology:
  1. Compute date-aligned WF windows (same as run_cross_tf)
  2. Run baseline components and cache per-window OOS returns
  3. For each perturbation, re-run ONLY the changed component
  4. Combine perturbed returns with cached baseline returns
  5. Report robustness and return for each perturbation

Pass criterion: robustness ≥66% (6/9) for ALL perturbations.
If any perturbation drops below 55%, the parameter is fragile.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators.basic import BasicIndicators
from src.backtest.engine import BacktestEngine
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase20")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase20.log", mode="w")
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
    "src.backtest.walk_forward",
]:
    logging.getLogger(name).setLevel(logging.WARNING)


# ─── Data ─────────────────────────────────────────────────────────

def load_data(timeframe: str) -> pd.DataFrame:
    """Load and add indicators to OHLCV data."""
    path = ROOT / f"data/processed/BTC_USDT_USDT_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    return df


# ─── Strategy Factories ──────────────────────────────────────────

RSI_1H_DEFAULTS = dict(
    rsi_oversold=35.0, rsi_overbought=65.0,
    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
)

DC_1H_DEFAULTS = dict(
    entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
    vol_mult=0.8, cooldown_bars=6,
)

RSI_15M_DEFAULTS = dict(
    rsi_oversold=35.0, rsi_overbought=65.0,
    atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
)


def make_rsi_1h(**overrides) -> MultiTimeframeFilter:
    params = {**RSI_1H_DEFAULTS, **overrides}
    return MultiTimeframeFilter(RSIMeanReversionStrategy(**params))


def make_dc_1h(**overrides) -> MultiTimeframeFilter:
    params = {**DC_1H_DEFAULTS, **overrides}
    return MultiTimeframeFilter(DonchianTrendStrategy(**params))


def make_rsi_15m(**overrides) -> MultiTimeframeFilter:
    params = {**RSI_15M_DEFAULTS, **overrides}
    return MultiTimeframeFilter(RSIMeanReversionStrategy(**params))


# ─── Window Computation (matches run_cross_tf exactly) ────────────

@dataclass
class WindowBounds:
    """Date-aligned WF window boundaries."""
    window_id: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    is_last: bool


def compute_windows(
    dfs: list[pd.DataFrame],
    n_windows: int = 9,
    train_ratio: float = 0.7,
) -> list[WindowBounds]:
    """Compute date-aligned window boundaries matching run_cross_tf logic."""
    start_date = max(df.index[0] for df in dfs)
    end_date = min(df.index[-1] for df in dfs)
    total_td = end_date - start_date
    test_td = total_td * (1 - train_ratio) / n_windows

    windows = []
    for i in range(n_windows):
        test_end = end_date - (n_windows - 1 - i) * test_td
        test_start = test_end - test_td
        windows.append(WindowBounds(
            window_id=i + 1,
            test_start=test_start,
            test_end=test_end,
            is_last=(i == n_windows - 1),
        ))
    return windows


# ─── Component Runner ─────────────────────────────────────────────

def run_component_windows(
    factory: Callable,
    df: pd.DataFrame,
    htf_df: pd.DataFrame,
    engine: BacktestEngine,
    windows: list[WindowBounds],
) -> list[float]:
    """Run a single component through all windows, return per-window OOS returns."""
    returns = []
    for w in windows:
        if w.is_last:
            test_df = df[df.index >= w.test_start]
        else:
            test_df = df[(df.index >= w.test_start) & (df.index < w.test_end)]

        if len(test_df) < 30:
            returns.append(0.0)
            continue

        strategy = factory()
        result = engine.run(strategy, test_df, htf_df=htf_df)
        returns.append(result.total_return)
    return returns


def compute_portfolio_metrics(
    component_returns: dict[str, list[float]],
    weights: dict[str, float],
    n_windows: int,
) -> tuple[float, int, float]:
    """Compute weighted portfolio metrics from per-component returns.

    Returns:
        (compounded_return, profitable_windows, robustness)
    """
    weighted = []
    for i in range(n_windows):
        wr = sum(
            component_returns[label][i] * weights[label]
            for label in component_returns
        )
        weighted.append(wr)

    profitable = sum(1 for r in weighted if r > 0)
    compounded = 1.0
    for r in weighted:
        compounded *= (1 + r / 100)
    total_ret = (compounded - 1) * 100
    robustness = profitable / n_windows if n_windows > 0 else 0

    return total_ret, profitable, robustness


# ─── Perturbation Definitions ────────────────────────────────────

PERTURBATIONS: dict[str, list[tuple[str, dict]]] = {
    "1hRSI": [
        ("oversold=30", dict(rsi_oversold=30.0)),
        ("oversold=40", dict(rsi_oversold=40.0)),
        ("overbought=60", dict(rsi_overbought=60.0)),
        ("overbought=70", dict(rsi_overbought=70.0)),
        ("sl=1.5", dict(atr_sl_mult=1.5)),
        ("sl=2.5", dict(atr_sl_mult=2.5)),
        ("tp=2.5", dict(atr_tp_mult=2.5)),
        ("tp=3.5", dict(atr_tp_mult=3.5)),
        ("cool=4", dict(cooldown_bars=4)),
        ("cool=8", dict(cooldown_bars=8)),
    ],
    "1hDC": [
        ("period=20", dict(entry_period=20)),
        ("period=28", dict(entry_period=28)),
        ("sl=1.5", dict(atr_sl_mult=1.5)),
        ("sl=2.5", dict(atr_sl_mult=2.5)),
        ("rr=1.5", dict(rr_ratio=1.5)),
        ("rr=2.5", dict(rr_ratio=2.5)),
        ("cool=4", dict(cooldown_bars=4)),
        ("cool=8", dict(cooldown_bars=8)),
    ],
    "15mRSI": [
        ("oversold=30", dict(rsi_oversold=30.0)),
        ("oversold=40", dict(rsi_oversold=40.0)),
        ("overbought=60", dict(rsi_overbought=60.0)),
        ("overbought=70", dict(rsi_overbought=70.0)),
        ("sl=1.5", dict(atr_sl_mult=1.5)),
        ("sl=2.5", dict(atr_sl_mult=2.5)),
        ("tp=2.5", dict(atr_tp_mult=2.5)),
        ("tp=3.5", dict(atr_tp_mult=3.5)),
        ("cool=8", dict(cooldown_bars=8)),
        ("cool=16", dict(cooldown_bars=16)),
    ],
}


# ─── Main Analysis ────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 20 — Parameter Stability Analysis")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Goal: Verify cross-TF portfolio (88%% rob) is parameter-stable.")
    logger.info("  Method: Perturb each param ±1 step, measure robustness change.")
    logger.info("  Pass: All perturbations maintain ≥66%% robustness (6/9).")
    logger.info("")

    # Load data
    logger.info("  Loading data...")
    df_1h = load_data("1h")
    df_15m = load_data("15m")
    df_4h = load_data("4h")
    logger.info("  1h: %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("  15m: %d bars (%s ~ %s)", len(df_15m),
                df_15m.index[0].date(), df_15m.index[-1].date())
    logger.info("")

    # Engines
    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")

    # Compute date-aligned windows
    n_windows = 9
    windows = compute_windows([df_1h, df_15m], n_windows=n_windows)
    logger.info("  Date-aligned windows:")
    for w in windows:
        logger.info("    W%d: %s ~ %s%s", w.window_id,
                    w.test_start.date(), w.test_end.date(),
                    " (last)" if w.is_last else "")
    logger.info("")

    # Weights
    weights = {"1hRSI": 0.33, "1hDC": 0.33, "15mRSI": 0.34}

    # ── Step 1: Baseline ──────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("  STEP 1: Baseline (optimal parameters)")
    logger.info("=" * 72)
    logger.info("")

    logger.info("  Running 1hRSI baseline...")
    baseline_1h_rsi = run_component_windows(
        make_rsi_1h, df_1h, df_4h, engine_1h, windows,
    )
    logger.info("  Running 1hDC baseline...")
    baseline_1h_dc = run_component_windows(
        make_dc_1h, df_1h, df_4h, engine_1h, windows,
    )
    logger.info("  Running 15mRSI baseline...")
    baseline_15m_rsi = run_component_windows(
        make_rsi_15m, df_15m, df_4h, engine_15m, windows,
    )

    baseline_returns = {
        "1hRSI": baseline_1h_rsi,
        "1hDC": baseline_1h_dc,
        "15mRSI": baseline_15m_rsi,
    }

    total_ret, profitable, robustness = compute_portfolio_metrics(
        baseline_returns, weights, n_windows,
    )

    logger.info("")
    logger.info("  Baseline per-window returns:")
    for i, w in enumerate(windows):
        wr = sum(
            baseline_returns[lbl][i] * weights[lbl]
            for lbl in baseline_returns
        )
        parts = [f"{lbl} {baseline_returns[lbl][i]:+.2f}%" for lbl in baseline_returns]
        marker = "+" if wr > 0 else "-"
        logger.info("    W%d: %s -> %+.2f%% %s", w.window_id,
                    " | ".join(parts), wr, marker)

    logger.info("")
    logger.info("  ★ BASELINE: OOS %+.2f%% | Robustness: %d%% (%d/%d)",
                total_ret, int(robustness * 100), profitable, n_windows)
    logger.info("")

    # ── Step 2: Parameter Perturbations ───────────────────────────
    logger.info("=" * 72)
    logger.info("  STEP 2: Parameter Perturbations")
    logger.info("=" * 72)
    logger.info("")

    # Factory and data mapping
    factory_map = {
        "1hRSI": (make_rsi_1h, df_1h, df_4h, engine_1h),
        "1hDC": (make_dc_1h, df_1h, df_4h, engine_1h),
        "15mRSI": (make_rsi_15m, df_15m, df_4h, engine_15m),
    }

    all_results: list[dict] = []
    total_perturbs = sum(len(v) for v in PERTURBATIONS.values())
    run_count = 0

    for component_label, perturbations in PERTURBATIONS.items():
        logger.info("  ─── %s perturbations ───", component_label)
        logger.info("")

        make_fn, df, htf_df, engine = factory_map[component_label]

        for perturb_label, overrides in perturbations:
            run_count += 1
            logger.info("  [%d/%d] %s: %s",
                        run_count, total_perturbs, component_label, perturb_label)

            # Create factory with overrides
            def perturbed_factory(_overrides=overrides, _make_fn=make_fn):
                return _make_fn(**_overrides)

            # Run only this component with perturbed params
            perturbed_returns = run_component_windows(
                perturbed_factory, df, htf_df, engine, windows,
            )

            # Combine with cached baseline for other components
            combined = dict(baseline_returns)
            combined[component_label] = perturbed_returns

            p_ret, p_profitable, p_robustness = compute_portfolio_metrics(
                combined, weights, n_windows,
            )

            rob_delta = (p_robustness - robustness) * 100
            ret_delta = p_ret - total_ret

            status = "OK" if p_robustness >= 0.66 else "WARN" if p_robustness >= 0.55 else "FAIL"

            logger.info(
                "    → OOS %+.2f%% (Δ%+.2f) | Rob %d%% (Δ%+.0f) | %s",
                p_ret, ret_delta, int(p_robustness * 100), rob_delta, status,
            )

            all_results.append({
                "component": component_label,
                "perturbation": perturb_label,
                "oos_return": p_ret,
                "robustness": p_robustness,
                "rob_pct": int(p_robustness * 100),
                "profitable": p_profitable,
                "ret_delta": ret_delta,
                "rob_delta": rob_delta,
                "status": status,
                "per_window": perturbed_returns,
            })

        logger.info("")

    # ── Step 3: Summary ───────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("  STEP 3: Parameter Stability Summary")
    logger.info("=" * 72)
    logger.info("")

    # Overall stats
    all_robs = [r["robustness"] for r in all_results]
    min_rob = min(all_robs)
    avg_rob = np.mean(all_robs)
    fail_count = sum(1 for r in all_results if r["status"] == "FAIL")
    warn_count = sum(1 for r in all_results if r["status"] == "WARN")
    ok_count = sum(1 for r in all_results if r["status"] == "OK")

    logger.info("  Baseline: OOS %+.2f%% | Robustness %d%%",
                total_ret, int(robustness * 100))
    logger.info("")
    logger.info("  %-10s %-16s  %8s  %5s  %8s  %5s  %6s",
                "Component", "Perturbation", "OOS Ret", "Rob", "ΔReturn", "ΔRob", "Status")
    logger.info("  " + "-" * 70)

    for r in all_results:
        logger.info(
            "  %-10s %-16s  %+7.2f%%  %4d%%  %+7.2f%%  %+4.0f%%  %6s",
            r["component"], r["perturbation"],
            r["oos_return"], r["rob_pct"],
            r["ret_delta"], r["rob_delta"], r["status"],
        )

    logger.info("")
    logger.info("  " + "-" * 70)
    logger.info("  Total: %d perturbations tested", len(all_results))
    logger.info("    OK (≥66%%):   %d", ok_count)
    logger.info("    WARN (55-65%%): %d", warn_count)
    logger.info("    FAIL (<55%%):  %d", fail_count)
    logger.info("")
    logger.info("  Min robustness: %d%%", int(min_rob * 100))
    logger.info("  Avg robustness: %d%%", int(avg_rob * 100))
    logger.info("")

    # Per-component stability score
    logger.info("  Per-component stability:")
    for comp_label in PERTURBATIONS:
        comp_results = [r for r in all_results if r["component"] == comp_label]
        comp_robs = [r["robustness"] for r in comp_results]
        comp_min = min(comp_robs)
        comp_avg = np.mean(comp_robs)
        comp_fails = sum(1 for r in comp_results if r["status"] == "FAIL")
        logger.info(
            "    %s: min=%d%%, avg=%d%%, fails=%d/%d",
            comp_label, int(comp_min * 100), int(comp_avg * 100),
            comp_fails, len(comp_results),
        )

    logger.info("")

    # Most fragile parameters
    fragile = [r for r in all_results if r["rob_delta"] < -11]
    if fragile:
        logger.info("  ⚠ Fragile parameters (robustness drop >11%%):")
        for r in sorted(fragile, key=lambda x: x["rob_delta"]):
            logger.info(
                "    %s %s: rob %d%% (Δ%+.0f%%)",
                r["component"], r["perturbation"],
                r["rob_pct"], r["rob_delta"],
            )
        logger.info("")

    # Verdict
    if fail_count == 0 and warn_count == 0:
        verdict = "EXCELLENT — All perturbations maintain ≥66% robustness"
    elif fail_count == 0:
        verdict = f"GOOD — No failures, {warn_count} borderline parameters"
    elif fail_count <= 2:
        verdict = f"ACCEPTABLE — {fail_count} fragile parameter(s)"
    else:
        verdict = f"CONCERNING — {fail_count} fragile parameters, possible overfitting"

    logger.info("  ★ VERDICT: %s", verdict)
    logger.info("")

    # Also check: does OI data exist and could be tested?
    oi_path = ROOT / "data/processed/BTC_USDT_USDT_open_interest_1h.parquet"
    if oi_path.exists():
        oi_df = pd.read_parquet(oi_path)
        logger.info("  ─── Open Interest Data Assessment ───")
        logger.info("  OI data: %d bars (%s ~ %s)",
                    len(oi_df), oi_df.index[0].date(), oi_df.index[-1].date())
        logger.info("  Price data: 1h=%d bars (~%d days)",
                    len(df_1h), len(df_1h) / 24)
        logger.info("  ⚠ OI covers only %d days — insufficient for WF (need ~365 days).",
                    (oi_df.index[-1] - oi_df.index[0]).days)
        logger.info("  Binance OI API limits lookback to ~30 days.")
        logger.info("  Recommendation: data-engineer should set up continuous OI collection.")
        logger.info("")

    elapsed = time.time() - t0
    logger.info("=" * 72)
    logger.info("  PHASE 20 — COMPLETE (%.1f minutes)", elapsed / 60)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
