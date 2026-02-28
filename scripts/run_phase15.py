#!/usr/bin/env python3
"""Phase 15 — Keltner Channel Mean Reversion + New Strategy Exploration.

Phase 14 findings:
  - Best 9w single: RSI_35_65+MTF (66%), CCI_20_200+MTF (66%)
  - Best 9w portfolio: RSI+DC 50/50 (77% rob, +20.27% OOS)
  - 77% robustness is the structural ceiling at 9w

Phase 15 goals:
  1. Keltner Channel MR+MTF — parameter sweep (kc_mult: 1.5, 2.0, 2.5, 3.0)
  2. Best Keltner variant at 5w, 7w, 9w WF
  3. Portfolio: Keltner + existing strategies (if Keltner > 60% robustness)
  4. Can a 5th uncorrelated strategy break the 77% ceiling?
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
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.walk_forward import WalkForwardAnalyzer
from src.strategy.mtf_filter import MultiTimeframeFilter
from src.strategy.keltner_mean_reversion import KeltnerMeanReversionStrategy
from src.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategy.donchian_trend import DonchianTrendStrategy
from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
from src.strategy.cci_mean_reversion import CCIMeanReversionStrategy

# ─── Logging ──────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase15")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase15.log", mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)

for name in ["src.backtest.engine", "src.strategy.mtf_filter", "src.strategy.portfolio"]:
    logging.getLogger(name).setLevel(logging.WARNING)


def load_data(timeframe: str) -> pd.DataFrame:
    """Load parquet data and add indicators."""
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


def log_wf(name: str, report, engine: BacktestEngine,
           strategy_factory, df: pd.DataFrame, htf_df=None) -> BacktestResult:
    """Log WF results and run full backtest."""
    for w in report.windows:
        oos = w.out_of_sample
        is_ = w.in_sample
        logger.info(
            "  W%d: IS %+6.2f%% (WR %d%%, %d tr) | OOS %+6.2f%% (WR %d%%, %d tr)",
            w.window_id,
            is_.total_return, int(is_.win_rate * 100), is_.total_trades,
            oos.total_return, int(oos.win_rate * 100), oos.total_trades,
        )
    logger.info(
        "  OOS: %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
        report.oos_total_return,
        int(report.robustness_score * 100),
        report.oos_profitable_windows, report.total_windows,
        report.oos_total_trades,
    )
    full = engine.run(strategy_factory(), df, htf_df=htf_df)
    logger.info(
        "  %s Full %+8.2f%% | Shp %5.2f | DD %5.1f%% | WR %5.1f%% | %4d tr | PF %5.2f",
        name, full.total_return, full.sharpe_ratio, full.max_drawdown,
        full.win_rate * 100, full.total_trades, full.profit_factor,
    )
    return full


def compute_portfolio(name: str, reports_weights: list[tuple]) -> tuple[float, float, int]:
    """Compute portfolio OOS from per-window weighted returns."""
    n_windows = reports_weights[0][0].total_windows
    portfolio_oos = []
    total_trades = 0

    for w_idx in range(n_windows):
        weighted_return = 0.0
        w_trades = 0
        label_parts = []

        for report, weight, label in reports_weights:
            if w_idx < len(report.windows):
                oos_ret = report.windows[w_idx].out_of_sample.total_return
                oos_tr = report.windows[w_idx].out_of_sample.total_trades
            else:
                oos_ret = 0.0
                oos_tr = 0
            weighted_return += oos_ret * weight
            w_trades += oos_tr
            label_parts.append(f"{label} {oos_ret:+.2f}%")

        portfolio_oos.append(weighted_return)
        total_trades += w_trades
        logger.info("  W%d: %s → Port %+.2f%%",
                     w_idx + 1, " + ".join(label_parts), weighted_return)

    # Compounded OOS
    compounded = 1.0
    for r in portfolio_oos:
        compounded *= (1 + r / 100)
    oos_total = (compounded - 1) * 100
    profitable = sum(1 for r in portfolio_oos if r > 0)
    rob = profitable / n_windows if n_windows > 0 else 0
    logger.info(
        "  %s OOS: %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
        name, oos_total, int(rob * 100), profitable, n_windows, total_trades,
    )
    return oos_total, rob, total_trades


# ─── Strategy factories ────────────────────────────────────────────
def make_keltner_mtf(kc_mult: float = 2.0, sl: float = 2.0,
                     tp: float = 3.0, cool: int = 6):
    """Create Keltner MR + MTF strategy."""
    def factory():
        base = KeltnerMeanReversionStrategy(
            center_period=20, kc_mult=kc_mult,
            atr_sl_mult=sl, atr_tp_mult=tp, cooldown_bars=cool,
        )
        return MultiTimeframeFilter(base)
    return factory


def make_rsi_mtf():
    base = RSIMeanReversionStrategy(
        rsi_oversold=35, rsi_overbought=65,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def make_dc_mtf():
    base = DonchianTrendStrategy(
        entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
        vol_mult=0.8, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def make_vwap_mtf():
    base = VWAPMeanReversionStrategy(
        vwap_period=24, band_mult=2.0, rsi_threshold=35.0,
        atr_sl_mult=2.0, cooldown_bars=4,
    )
    return MultiTimeframeFilter(base)


def make_cci_mtf():
    base = CCIMeanReversionStrategy(
        cci_period=20, oversold_level=200, overbought_level=200,
        atr_sl_mult=2.0, atr_tp_mult=3.0, vol_mult=0.0, cooldown_bars=6,
    )
    return MultiTimeframeFilter(base)


def main() -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 15 — Keltner Channel MR + New Strategy Exploration")
    logger.info("=" * 72)
    logger.info("")

    # ─── Load data ────────────────────────────────────────────────
    df_1h = load_data("1h")
    df_4h = load_data("4h")
    df_1h = add_cci(df_1h, 20)

    logger.info("1h data: %d bars (%s ~ %s)", len(df_1h),
                df_1h.index[0].date(), df_1h.index[-1].date())
    logger.info("4h data: %d bars (%s ~ %s)", len(df_4h),
                df_4h.index[0].date(), df_4h.index[-1].date())
    logger.info("")

    engine = BacktestEngine(max_hold_bars=48)

    # ═════════════════════════════════════════════════════════════
    #   PART 1: Keltner MR Parameter Sweep at 5w
    # ═════════════════════════════════════════════════════════════
    logger.info("─" * 72)
    logger.info("  PART 1: Keltner MR Parameter Sweep (5w)")
    logger.info("─" * 72)
    logger.info("")

    kc_results_5w = {}
    best_kc_config = None
    best_kc_rob = -1.0

    # Parameter sweep: kc_mult, atr_sl, atr_tp, cooldown
    configs = [
        # kc_mult, sl, tp, cool, label
        (1.5, 2.0, 3.0, 6, "KC_1.5_SL2_TP3"),
        (2.0, 2.0, 3.0, 6, "KC_2.0_SL2_TP3"),
        (2.5, 2.0, 3.0, 6, "KC_2.5_SL2_TP3"),
        (3.0, 2.0, 3.0, 6, "KC_3.0_SL2_TP3"),
        # Tighter SL/TP variants for best kc_mults
        (2.0, 1.5, 2.5, 6, "KC_2.0_SL1.5_TP2.5"),
        (2.0, 2.0, 2.0, 6, "KC_2.0_SL2_TP2_RR1"),
        (2.5, 1.5, 2.5, 6, "KC_2.5_SL1.5_TP2.5"),
        (2.5, 2.0, 4.0, 6, "KC_2.5_SL2_TP4"),
        # Cooldown variants
        (2.0, 2.0, 3.0, 4, "KC_2.0_cool4"),
        (2.5, 2.0, 3.0, 4, "KC_2.5_cool4"),
    ]

    for kc_mult, sl, tp, cool, label in configs:
        full_label = f"{label}+MTF_5w"
        logger.info("  --- %s ---", full_label)
        factory = make_keltner_mtf(kc_mult=kc_mult, sl=sl, tp=tp, cool=cool)
        wf = WalkForwardAnalyzer(n_windows=5, engine=engine)
        report = wf.run(factory, df_1h, htf_df=df_4h)
        full = log_wf(full_label, report, engine, factory, df_1h, htf_df=df_4h)
        rob = report.robustness_score
        kc_results_5w[full_label] = (
            report.oos_total_return, rob, report.oos_total_trades,
            full.total_return, full.sharpe_ratio, full.max_drawdown,
        )
        if rob > best_kc_rob or (rob == best_kc_rob and report.oos_total_return >
                                  (best_kc_config[1] if best_kc_config else -999)):
            best_kc_rob = rob
            best_kc_config = (label, report.oos_total_return, rob, kc_mult, sl, tp, cool)
        logger.info("")

    logger.info("  ═══ 5w Summary ═══")
    logger.info("  %-30s %8s %6s %6s %8s %5s %5s", "Config", "OOS Ret", "Rob", "Trades",
                "FullRet", "Sharpe", "DD")
    logger.info("  " + "-" * 75)
    for name, (oos, rob, tr, fret, shp, dd) in sorted(
            kc_results_5w.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.6 else ""
        logger.info("  %-30s %+7.2f%% %5d%% %6d %+7.2f%% %5.2f %5.1f%%%s",
                     name, oos, int(rob * 100), tr, fret, shp, dd, marker)

    if best_kc_config is None or best_kc_rob < 0.4:
        logger.info("")
        logger.info("  !!! Keltner MR does not meet minimum 40%% robustness at 5w.")
        logger.info("  !!! Stopping Keltner exploration.")
        _print_final_summary(kc_results_5w, {}, {}, {}, {})
        return

    # ═════════════════════════════════════════════════════════════
    #   PART 2: Best Keltner variants at 7w and 9w
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─" * 72)
    logger.info("  PART 2: Best Keltner variants at 7w and 9w")
    logger.info("─" * 72)
    logger.info("")

    # Pick top 3 configs by robustness then OOS
    top_configs = sorted(kc_results_5w.items(),
                         key=lambda x: (x[1][1], x[1][0]), reverse=True)[:3]

    kc_results_7w = {}
    kc_results_9w = {}
    best_kc_report_9w = None
    best_kc_name_9w = None
    best_kc_factory_9w = None

    # Extract original params from label
    label_to_params = {label: (kc, sl, tp, cool)
                       for kc, sl, tp, cool, label in configs}

    for top_name, _ in top_configs:
        # Parse the label to get params
        base_label = top_name.replace("+MTF_5w", "")
        if base_label not in label_to_params:
            continue
        kc, sl, tp, cool = label_to_params[base_label]
        factory = make_keltner_mtf(kc_mult=kc, sl=sl, tp=tp, cool=cool)

        # 7w
        label_7w = f"{base_label}+MTF_7w"
        logger.info("  --- %s ---", label_7w)
        wf7 = WalkForwardAnalyzer(n_windows=7, engine=engine)
        report7 = wf7.run(factory, df_1h, htf_df=df_4h)
        full7 = log_wf(label_7w, report7, engine, factory, df_1h, htf_df=df_4h)
        kc_results_7w[label_7w] = (
            report7.oos_total_return, report7.robustness_score,
            report7.oos_total_trades, full7.total_return,
            full7.sharpe_ratio, full7.max_drawdown,
        )
        logger.info("")

        # 9w
        label_9w = f"{base_label}+MTF_9w"
        logger.info("  --- %s ---", label_9w)
        wf9 = WalkForwardAnalyzer(n_windows=9, engine=engine)
        report9 = wf9.run(factory, df_1h, htf_df=df_4h)
        full9 = log_wf(label_9w, report9, engine, factory, df_1h, htf_df=df_4h)
        kc_results_9w[label_9w] = (
            report9.oos_total_return, report9.robustness_score,
            report9.oos_total_trades, full9.total_return,
            full9.sharpe_ratio, full9.max_drawdown,
        )
        if best_kc_report_9w is None or report9.robustness_score > best_kc_report_9w.robustness_score:
            best_kc_report_9w = report9
            best_kc_name_9w = base_label
            best_kc_factory_9w = factory
        logger.info("")

    logger.info("  ═══ 7w Summary ═══")
    logger.info("  %-30s %8s %6s %6s", "Config", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 55)
    for name, (oos, rob, tr, *_) in sorted(
            kc_results_7w.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.6 else ""
        logger.info("  %-30s %+7.2f%% %5d%% %6d%s", name, oos, int(rob * 100), tr, marker)

    logger.info("")
    logger.info("  ═══ 9w Summary ═══")
    logger.info("  %-30s %8s %6s %6s", "Config", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 55)
    for name, (oos, rob, tr, *_) in sorted(
            kc_results_9w.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.6 else ""
        logger.info("  %-30s %+7.2f%% %5d%% %6d%s", name, oos, int(rob * 100), tr, marker)

    # Check if any Keltner meets 60% at 9w
    best_9w_rob = max((v[1] for v in kc_results_9w.values()), default=0)
    if best_9w_rob < 0.55:
        logger.info("")
        logger.info("  !!! No Keltner variant meets 55%% robustness at 9w.")
        logger.info("  !!! Skipping portfolio tests.")
        _print_final_summary(kc_results_5w, kc_results_7w, kc_results_9w, {}, {})
        return

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Portfolio tests (Keltner + existing strategies)
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─" * 72)
    logger.info("  PART 3: Portfolio Tests (Keltner + existing, 9w)")
    logger.info("─" * 72)
    logger.info("")

    # Run existing strategies at 9w for portfolio composition
    existing_reports = {}
    for bname, factory in [
        ("RSI", make_rsi_mtf),
        ("DC", make_dc_mtf),
        ("VWAP", make_vwap_mtf),
        ("CCI", make_cci_mtf),
    ]:
        label = f"{bname}+MTF_9w"
        logger.info("  --- %s ---", label)
        wf = WalkForwardAnalyzer(n_windows=9, engine=engine)
        report = wf.run(factory, df_1h, htf_df=df_4h)
        log_wf(label, report, engine, factory, df_1h, htf_df=df_4h)
        existing_reports[bname] = report
        logger.info("")

    portfolio_results = {}

    # Test Keltner with each existing strategy
    kc_report = best_kc_report_9w
    kc_name = best_kc_name_9w

    portfolio_combos = [
        (f"KC+RSI_50_50_9w", [(kc_report, 0.5, "KC"), (existing_reports["RSI"], 0.5, "RSI")]),
        (f"KC+DC_50_50_9w", [(kc_report, 0.5, "KC"), (existing_reports["DC"], 0.5, "DC")]),
        (f"KC+VWAP_50_50_9w", [(kc_report, 0.5, "KC"), (existing_reports["VWAP"], 0.5, "VWAP")]),
        (f"KC+CCI_50_50_9w", [(kc_report, 0.5, "KC"), (existing_reports["CCI"], 0.5, "CCI")]),
        # 3-strategy combos with KC
        (f"KC+RSI+DC_equal_9w", [
            (kc_report, 0.33, "KC"),
            (existing_reports["RSI"], 0.33, "RSI"),
            (existing_reports["DC"], 0.34, "DC"),
        ]),
        (f"KC+VWAP+DC_equal_9w", [
            (kc_report, 0.33, "KC"),
            (existing_reports["VWAP"], 0.33, "VWAP"),
            (existing_reports["DC"], 0.34, "DC"),
        ]),
        # Compare with Phase 14 best: RSI+DC 50/50
        (f"RSI+DC_50_50_9w_baseline", [
            (existing_reports["RSI"], 0.5, "RSI"),
            (existing_reports["DC"], 0.5, "DC"),
        ]),
    ]

    for pname, components in portfolio_combos:
        logger.info("  --- %s ---", pname)
        oos, rob, trades = compute_portfolio(pname, components)
        portfolio_results[pname] = (oos, rob, trades)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Standalone Keltner (no MTF) — sanity check
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─" * 72)
    logger.info("  PART 4: Keltner Standalone (no MTF) — sanity check")
    logger.info("─" * 72)
    logger.info("")

    standalone_results = {}
    if best_kc_config:
        _, _, _, kc, sl, tp, cool = best_kc_config
        label = f"KC_{kc}_standalone_5w"
        logger.info("  --- %s ---", label)

        def standalone_factory():
            return KeltnerMeanReversionStrategy(
                center_period=20, kc_mult=kc,
                atr_sl_mult=sl, atr_tp_mult=tp, cooldown_bars=cool,
            )
        wf = WalkForwardAnalyzer(n_windows=5, engine=engine)
        report = wf.run(standalone_factory, df_1h, htf_df=None)
        full = log_wf(label, report, engine, standalone_factory, df_1h, htf_df=None)
        standalone_results[label] = (
            report.oos_total_return, report.robustness_score,
            report.oos_total_trades, full.total_return,
        )
        logger.info("")

    _print_final_summary(kc_results_5w, kc_results_7w, kc_results_9w,
                          portfolio_results, standalone_results)


def _print_final_summary(kc_5w, kc_7w, kc_9w, portfolios, standalone):
    """Print final summary."""
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 15 — SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    logger.info("  Keltner MR + MTF (5w):")
    logger.info("  %-30s %8s %6s %6s %8s %5s %5s", "Config", "OOS Ret", "Rob",
                "Trades", "FullRet", "Sharpe", "DD")
    logger.info("  " + "-" * 75)
    for name, (oos, rob, tr, fret, shp, dd) in sorted(
            kc_5w.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
        marker = " ★" if rob >= 0.6 else ""
        logger.info("  %-30s %+7.2f%% %5d%% %6d %+7.2f%% %5.2f %5.1f%%%s",
                     name, oos, int(rob * 100), tr, fret, shp, dd, marker)

    if kc_7w:
        logger.info("")
        logger.info("  Keltner MR + MTF (7w):")
        logger.info("  %-30s %8s %6s %6s", "Config", "OOS Ret", "Rob", "Trades")
        logger.info("  " + "-" * 55)
        for name, (oos, rob, tr, *_) in sorted(
                kc_7w.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
            marker = " ★" if rob >= 0.6 else ""
            logger.info("  %-30s %+7.2f%% %5d%% %6d%s", name, oos, int(rob * 100), tr, marker)

    if kc_9w:
        logger.info("")
        logger.info("  Keltner MR + MTF (9w):")
        logger.info("  %-30s %8s %6s %6s", "Config", "OOS Ret", "Rob", "Trades")
        logger.info("  " + "-" * 55)
        for name, (oos, rob, tr, *_) in sorted(
                kc_9w.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
            marker = " ★" if rob >= 0.6 else ""
            logger.info("  %-30s %+7.2f%% %5d%% %6d%s", name, oos, int(rob * 100), tr, marker)

    if portfolios:
        logger.info("")
        logger.info("  Portfolio Tests (9w):")
        logger.info("  %-35s %8s %6s %6s", "Portfolio", "OOS Ret", "Rob", "Trades")
        logger.info("  " + "-" * 60)
        for name, (oos, rob, trades) in sorted(
                portfolios.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True):
            marker = " ★" if rob >= 0.77 else ""
            logger.info("  %-35s %+7.2f%% %5d%% %6d%s",
                         name, oos, int(rob * 100), trades, marker)

    if standalone:
        logger.info("")
        logger.info("  Standalone (no MTF) — sanity check:")
        for name, (oos, rob, tr, fret) in standalone.items():
            logger.info("  %s: OOS %+.2f%%, Rob %d%%, Full %+.2f%%",
                         name, oos, int(rob * 100), fret)

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 15 complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
