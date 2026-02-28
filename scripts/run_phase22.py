#!/usr/bin/env python3
"""Phase 22 — Multi-Asset Diversification Research.

Goal: Break the 88% robustness ceiling by adding ETH and/or SOL.

Hypothesis: BTC strategies (RSI+MTF, DC+MTF) may work on other crypto assets.
Cross-ASSET diversification should provide uncorrelated return streams,
similar to how cross-TIMEFRAME diversification lifted BTC from 66% → 88%.

Steps:
  1. Collect ETH/SOL data (1h, 15m, 4h) if not already present
  2. Validate BTC strategies on ETH and SOL independently (9w WF)
  3. Analyze cross-asset return correlations
  4. Build multi-asset portfolios and test if robustness > 88%

Data requirement: ~365 days of 1h, 15m, 4h OHLCV for each symbol.

Usage:
    # Full run (collect data if needed + backtest)
    python scripts/run_phase22.py

    # Skip data collection (data must already exist)
    python scripts/run_phase22.py --skip-collect

    # Test specific symbols only
    python scripts/run_phase22.py --symbols ETH/USDT:USDT
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
from src.data.preprocessor import DataPreprocessor

# ─── Constants ───────────────────────────────────────────────────

BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
SOL = "SOL/USDT:USDT"
ALL_SYMBOLS = [BTC, ETH, SOL]

# Timeframes needed for strategy validation
REQUIRED_TFS = ["1h", "15m", "4h"]

DATA_DIR = ROOT / "data" / "processed"

# ─── Logging ─────────────────────────────────────────────────────

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("phase22")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_DIR / "phase22.log", mode="w")
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
))
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)

# Suppress noisy sub-loggers
for name in [
    "src.backtest.engine",
    "src.strategy.mtf_filter",
    "src.backtest.walk_forward",
]:
    logging.getLogger(name).setLevel(logging.WARNING)

# Redirect WF cross-TF logs to our logger
wf_logger = logging.getLogger("src.backtest.walk_forward")
wf_logger.setLevel(logging.INFO)
wf_logger.handlers.clear()
wf_logger.addHandler(fh)
wf_logger.addHandler(sh)


# ─── Data Utilities ──────────────────────────────────────────────

def symbol_to_filename(symbol: str) -> str:
    """Convert symbol like 'ETH/USDT:USDT' to 'ETH_USDT_USDT'."""
    return symbol.replace("/", "_").replace(":", "_")


def data_path(symbol: str, tf: str) -> Path:
    """Get parquet file path for a symbol and timeframe."""
    return DATA_DIR / f"{symbol_to_filename(symbol)}_{tf}.parquet"


def data_exists(symbol: str) -> bool:
    """Check if all required timeframe data exists for a symbol."""
    return all(data_path(symbol, tf).exists() for tf in REQUIRED_TFS)


async def collect_symbol_data(symbol: str, days: int = 365) -> None:
    """Collect historical OHLCV data for a symbol.

    Fetches 1h, 15m, 4h data and saves to parquet files.
    Uses public API (no auth required for OHLCV data).
    Skips timeframes that already have sufficient data.
    """
    import ccxt.async_support as ccxt_async
    from datetime import datetime, timedelta, timezone

    # Use public API without auth keys — OHLCV is public data
    exchange = ccxt_async.binanceusdm({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    preprocessor = DataPreprocessor()

    try:
        for tf in REQUIRED_TFS:
            path = data_path(symbol, tf)

            # Check if we already have enough data
            if path.exists():
                existing = pd.read_parquet(path)
                span_days = (existing.index[-1] - existing.index[0]).days
                if span_days >= days * 0.9:  # Allow 10% tolerance
                    logger.info(
                        "  %s %s: already have %d days — skipping",
                        symbol, tf, span_days,
                    )
                    continue

            logger.info("  Collecting %s %s (%d days)...", symbol, tf, days)

            all_candles: list[list] = []
            since = int(
                (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
                * 1000
            )

            while True:
                candles = await exchange.fetch_ohlcv(
                    symbol, timeframe=tf, since=since, limit=1500
                )
                if not candles:
                    break
                all_candles.extend(candles)
                since = candles[-1][0] + 1
                logger.info(
                    "    %s %s: fetched %d candles (total %d)",
                    symbol, tf, len(candles), len(all_candles),
                )
                await asyncio.sleep(exchange.rateLimit / 1000)

            df = pd.DataFrame(
                all_candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.drop_duplicates(subset=["timestamp"]).set_index(
                "timestamp"
            ).sort_index()
            df = preprocessor.clean_ohlcv(df)

            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            logger.info(
                "  %s %s: %d bars (%s ~ %s)",
                symbol, tf, len(df),
                df.index[0].date(), df.index[-1].date(),
            )
    finally:
        await exchange.close()


def load_data(symbol: str, tf: str) -> pd.DataFrame:
    """Load OHLCV data and add indicators."""
    path = data_path(symbol, tf)
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_parquet(path)
    df = BasicIndicators.add_all(df)
    df.attrs["symbol"] = symbol
    return df


# ─── Strategy Factories ──────────────────────────────────────────

def make_rsi_1h(symbol: str = BTC) -> MultiTimeframeFilter:
    """1h RSI MR with validated BTC params."""
    return MultiTimeframeFilter(RSIMeanReversionStrategy(
        rsi_oversold=35.0, rsi_overbought=65.0,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=6,
        symbol=symbol,
    ))


def make_dc_1h(symbol: str = BTC) -> MultiTimeframeFilter:
    """1h Donchian with validated BTC params."""
    return MultiTimeframeFilter(DonchianTrendStrategy(
        entry_period=24, atr_sl_mult=2.0, rr_ratio=2.0,
        vol_mult=0.8, cooldown_bars=6,
        symbol=symbol,
    ))


def make_rsi_15m(symbol: str = BTC) -> MultiTimeframeFilter:
    """15m RSI MR mid-config from Phase 16."""
    return MultiTimeframeFilter(RSIMeanReversionStrategy(
        rsi_oversold=35.0, rsi_overbought=65.0,
        atr_sl_mult=2.0, atr_tp_mult=3.0, cooldown_bars=12,
        symbol=symbol,
    ))


# ─── Analysis Functions ──────────────────────────────────────────

def log_wf_report(
    name: str,
    report,
    engine: BacktestEngine,
    factory,
    df: pd.DataFrame,
    htf_df: pd.DataFrame,
) -> None:
    """Log WF window-by-window details and full-period backtest."""
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
    full = engine.run(factory(), df, htf_df=htf_df)
    logger.info(
        "  %s Full %+8.2f%% | Shp %5.2f | DD %5.1f%% | WR %5.1f%% | %4d tr | PF %5.2f",
        name, full.total_return, full.sharpe_ratio, full.max_drawdown,
        full.win_rate * 100, full.total_trades, full.profit_factor,
    )


def log_cross_tf(name: str, report: CrossTFReport) -> None:
    """Log cross-TF report summary."""
    logger.info(
        "  %s: OOS %+.2f%% | Robustness: %d%% (%d/%d) | Trades: %d",
        name, report.oos_total_return,
        int(report.robustness_score * 100),
        report.oos_profitable_windows, report.total_windows,
        report.total_trades,
    )


def compute_window_returns(
    report: CrossTFReport,
) -> list[float]:
    """Extract per-window weighted returns from a cross-TF report."""
    return [w.weighted_return for w in report.windows]


def analyze_correlation(
    symbol_returns: dict[str, list[float]],
) -> pd.DataFrame:
    """Compute pairwise correlation matrix of per-window returns."""
    df = pd.DataFrame(symbol_returns)
    return df.corr()


# ─── Main ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 22: Multi-Asset Research")
    parser.add_argument(
        "--skip-collect", action="store_true",
        help="Skip data collection (assumes data exists)",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to test (default: ETH/USDT:USDT SOL/USDT:USDT)",
    )
    args = parser.parse_args()

    symbols_to_test = args.symbols or [ETH, SOL]

    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 22 — Multi-Asset Diversification Research")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  BTC baseline: 88%% robustness (8/9), +18.81%% OOS")
    logger.info("  Goal: Break 88%% via cross-asset diversification")
    logger.info("  Test symbols: %s", ", ".join(symbols_to_test))
    logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 0: Data Collection
    # ═════════════════════════════════════════════════════════════

    if not args.skip_collect:
        logger.info("-" * 72)
        logger.info("  PART 0: Data Collection")
        logger.info("-" * 72)
        logger.info("")

        for symbol in symbols_to_test:
            if data_exists(symbol):
                logger.info("  %s: data already exists — skipping", symbol)
            else:
                logger.info("  Collecting %s data (365 days, 1h/15m/4h)...", symbol)
                try:
                    asyncio.run(collect_symbol_data(symbol, days=365))
                    logger.info("  %s: collection complete", symbol)
                except Exception as e:
                    logger.error("  %s: collection FAILED — %s", symbol, e)
                    logger.error(
                        "  To fix: run 'python scripts/collect_data.py' style "
                        "for %s, or check API connectivity", symbol,
                    )
                    symbols_to_test.remove(symbol)
            logger.info("")

    # Verify all data exists
    available_symbols = []
    for symbol in symbols_to_test:
        if data_exists(symbol):
            available_symbols.append(symbol)
        else:
            logger.warning("  %s: data not available — SKIPPING", symbol)

    if not available_symbols:
        logger.error("No symbol data available. Run without --skip-collect first.")
        sys.exit(1)

    # ═════════════════════════════════════════════════════════════
    #   PART 1: Per-Symbol Strategy Validation (9w WF)
    # ═════════════════════════════════════════════════════════════

    logger.info("")
    logger.info("-" * 72)
    logger.info("  PART 1: Per-Symbol Strategy Validation (9w WF)")
    logger.info("-" * 72)
    logger.info("  Testing BTC's validated strategies on new symbols")
    logger.info("")

    engine_1h = BacktestEngine(max_hold_bars=48, freq="1h")
    engine_15m = BacktestEngine(max_hold_bars=96, freq="15m")

    # Store per-symbol results for later correlation analysis
    # {symbol: {strategy_name: WF_report}}
    symbol_results: dict[str, dict] = {}
    # {symbol: {strategy: per-window returns}}
    symbol_window_returns: dict[str, dict[str, list[float]]] = {}

    for symbol in available_symbols:
        short_name = symbol.split("/")[0]  # "ETH", "SOL"
        logger.info("  ═══ %s ═══", symbol)
        logger.info("")

        df_1h = load_data(symbol, "1h")
        df_15m = load_data(symbol, "15m")
        df_4h = load_data(symbol, "4h")

        logger.info("  1h:  %d bars (%s ~ %s)", len(df_1h),
                     df_1h.index[0].date(), df_1h.index[-1].date())
        logger.info("  15m: %d bars (%s ~ %s)", len(df_15m),
                     df_15m.index[0].date(), df_15m.index[-1].date())
        logger.info("  4h:  %d bars (%s ~ %s)", len(df_4h),
                     df_4h.index[0].date(), df_4h.index[-1].date())
        logger.info("")

        symbol_results[symbol] = {}
        symbol_window_returns[symbol] = {}

        # --- 1h RSI+MTF ---
        logger.info("  --- %s 1h RSI_35_65+MTF (9w) ---", short_name)
        wf9 = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
        factory_rsi_1h = lambda s=symbol: make_rsi_1h(s)
        rpt = wf9.run(factory_rsi_1h, df_1h, htf_df=df_4h)
        log_wf_report(f"{short_name}_RSI_1h", rpt, engine_1h, factory_rsi_1h, df_1h, df_4h)
        symbol_results[symbol]["rsi_1h"] = rpt
        logger.info("")

        # --- 1h DC+MTF ---
        logger.info("  --- %s 1h DC_24+MTF (9w) ---", short_name)
        wf9_dc = WalkForwardAnalyzer(n_windows=9, engine=engine_1h)
        factory_dc_1h = lambda s=symbol: make_dc_1h(s)
        rpt_dc = wf9_dc.run(factory_dc_1h, df_1h, htf_df=df_4h)
        log_wf_report(f"{short_name}_DC_1h", rpt_dc, engine_1h, factory_dc_1h, df_1h, df_4h)
        symbol_results[symbol]["dc_1h"] = rpt_dc
        logger.info("")

        # --- 15m RSI+MTF (cool=12, hold=96) ---
        logger.info("  --- %s 15m RSI_35_65_mid+MTF (9w) ---", short_name)
        wf9_15m = WalkForwardAnalyzer(n_windows=9, engine=engine_15m)
        factory_rsi_15m = lambda s=symbol: make_rsi_15m(s)
        rpt_15m = wf9_15m.run(factory_rsi_15m, df_15m, htf_df=df_4h)
        log_wf_report(f"{short_name}_RSI_15m", rpt_15m, engine_15m, factory_rsi_15m, df_15m, df_4h)
        symbol_results[symbol]["rsi_15m"] = rpt_15m
        logger.info("")

        # --- Cross-TF portfolio (same as BTC config) ---
        logger.info("  --- %s Cross-TF 33/33/34 (9w) ---", short_name)
        wf_cross = WalkForwardAnalyzer(n_windows=9)
        cross_rpt = wf_cross.run_cross_tf([
            CrossTFComponent(
                strategy_factory=lambda s=symbol: make_rsi_1h(s),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.33, label=f"{short_name}_1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=lambda s=symbol: make_dc_1h(s),
                df=df_1h, htf_df=df_4h,
                engine=engine_1h, weight=0.33, label=f"{short_name}_1hDC",
            ),
            CrossTFComponent(
                strategy_factory=lambda s=symbol: make_rsi_15m(s),
                df=df_15m, htf_df=df_4h,
                engine=engine_15m, weight=0.34, label=f"{short_name}_15mRSI",
            ),
        ])
        log_cross_tf(f"{short_name} Cross-TF 33/33/34", cross_rpt)
        symbol_results[symbol]["cross_tf"] = cross_rpt
        symbol_window_returns[symbol]["cross_tf"] = compute_window_returns(cross_rpt)

        # Also store individual component window returns
        for w in cross_rpt.windows:
            for cr in w.components:
                key = cr.label
                if key not in symbol_window_returns[symbol]:
                    symbol_window_returns[symbol][key] = []
                symbol_window_returns[symbol][key].append(cr.oos_return)

        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 2: Cross-Asset Correlation Analysis
    # ═════════════════════════════════════════════════════════════

    logger.info("")
    logger.info("-" * 72)
    logger.info("  PART 2: Cross-Asset Correlation Analysis")
    logger.info("-" * 72)
    logger.info("  Low correlation → diversification benefit → can break 88%%")
    logger.info("")

    # Load BTC cross-TF portfolio returns for comparison
    logger.info("  Loading BTC baseline for correlation comparison...")
    btc_1h = load_data(BTC, "1h")
    btc_15m = load_data(BTC, "15m")
    btc_4h = load_data(BTC, "4h")

    wf_btc = WalkForwardAnalyzer(n_windows=9)
    btc_cross_rpt = wf_btc.run_cross_tf([
        CrossTFComponent(
            strategy_factory=lambda: make_rsi_1h(BTC),
            df=btc_1h, htf_df=btc_4h,
            engine=engine_1h, weight=0.33, label="BTC_1hRSI",
        ),
        CrossTFComponent(
            strategy_factory=lambda: make_dc_1h(BTC),
            df=btc_1h, htf_df=btc_4h,
            engine=engine_1h, weight=0.33, label="BTC_1hDC",
        ),
        CrossTFComponent(
            strategy_factory=lambda: make_rsi_15m(BTC),
            df=btc_15m, htf_df=btc_4h,
            engine=engine_15m, weight=0.34, label="BTC_15mRSI",
        ),
    ])
    btc_window_returns = compute_window_returns(btc_cross_rpt)
    logger.info("  BTC Cross-TF: OOS %+.2f%%, Rob %d%%",
                btc_cross_rpt.oos_total_return,
                int(btc_cross_rpt.robustness_score * 100))
    logger.info("")

    # Compute cross-asset correlations
    corr_data: dict[str, list[float]] = {"BTC": btc_window_returns}
    for symbol in available_symbols:
        short_name = symbol.split("/")[0]
        if "cross_tf" in symbol_window_returns.get(symbol, {}):
            corr_data[short_name] = symbol_window_returns[symbol]["cross_tf"]

    if len(corr_data) >= 2:
        logger.info("  Per-window return correlation matrix (cross-TF portfolios):")
        corr_df = analyze_correlation(corr_data)
        logger.info("")
        for row_name in corr_df.index:
            parts = [f"{corr_df.loc[row_name, col]:+.3f}" for col in corr_df.columns]
            logger.info("  %-6s %s", row_name, "  ".join(parts))
        logger.info("")

        # Identify best diversification pairs
        for i, sym_a in enumerate(corr_df.columns):
            for sym_b in corr_df.columns[i + 1:]:
                c = corr_df.loc[sym_a, sym_b]
                quality = "EXCELLENT" if c < 0.2 else "GOOD" if c < 0.5 else "MODERATE" if c < 0.7 else "POOR"
                logger.info("  %s ↔ %s: %.3f (%s diversification)", sym_a, sym_b, c, quality)
        logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 3: Multi-Asset Portfolio WF
    # ═════════════════════════════════════════════════════════════

    logger.info("")
    logger.info("-" * 72)
    logger.info("  PART 3: Multi-Asset Portfolio Walk-Forward (9w)")
    logger.info("-" * 72)
    logger.info("  Combining BTC + other assets to break 88%% ceiling")
    logger.info("")

    # For each available symbol, build BTC+SYMBOL portfolios
    multi_asset_results: dict[str, CrossTFReport] = {}

    for symbol in available_symbols:
        short_name = symbol.split("/")[0]
        sym_1h = load_data(symbol, "1h")
        sym_15m = load_data(symbol, "15m")
        sym_4h = load_data(symbol, "4h")

        # ── BTC Cross-TF + SYMBOL Cross-TF (50/50) ──
        # Each asset gets its own cross-TF portfolio with halved weights
        label = f"BTC+{short_name} 50/50 cross-TF"
        logger.info("  --- %s ---", label)

        wf_multi = WalkForwardAnalyzer(n_windows=9)
        components = [
            # BTC components (50% total = 16.5/16.5/17)
            CrossTFComponent(
                strategy_factory=lambda: make_rsi_1h(BTC),
                df=btc_1h, htf_df=btc_4h,
                engine=engine_1h, weight=0.165, label="BTC_1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=lambda: make_dc_1h(BTC),
                df=btc_1h, htf_df=btc_4h,
                engine=engine_1h, weight=0.165, label="BTC_1hDC",
            ),
            CrossTFComponent(
                strategy_factory=lambda: make_rsi_15m(BTC),
                df=btc_15m, htf_df=btc_4h,
                engine=engine_15m, weight=0.17, label="BTC_15mRSI",
            ),
            # SYMBOL components (50% total = 16.5/16.5/17)
            CrossTFComponent(
                strategy_factory=lambda s=symbol: make_rsi_1h(s),
                df=sym_1h, htf_df=sym_4h,
                engine=engine_1h, weight=0.165, label=f"{short_name}_1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=lambda s=symbol: make_dc_1h(s),
                df=sym_1h, htf_df=sym_4h,
                engine=engine_1h, weight=0.165, label=f"{short_name}_1hDC",
            ),
            CrossTFComponent(
                strategy_factory=lambda s=symbol: make_rsi_15m(s),
                df=sym_15m, htf_df=sym_4h,
                engine=engine_15m, weight=0.17, label=f"{short_name}_15mRSI",
            ),
        ]
        rpt = wf_multi.run_cross_tf(components)
        multi_asset_results[label] = rpt
        log_cross_tf(label, rpt)

        # Per-window breakdown
        for w in rpt.windows:
            btc_ret = sum(
                cr.oos_return * c.weight
                for cr, c in zip(w.components, components)
                if cr.label.startswith("BTC_")
            )
            sym_ret = sum(
                cr.oos_return * c.weight
                for cr, c in zip(w.components, components)
                if cr.label.startswith(f"{short_name}_")
            )
            marker = "+" if w.weighted_return > 0 else "-"
            logger.info(
                "    W%d [%s ~ %s]: BTC %+.2f%% | %s %+.2f%% -> %+.2f%% %s",
                w.window_id, w.test_start, w.test_end,
                btc_ret, short_name, sym_ret, w.weighted_return, marker,
            )
        logger.info("")

        # ── BTC 70% + SYMBOL 30% ──
        label_7030 = f"BTC+{short_name} 70/30 cross-TF"
        logger.info("  --- %s ---", label_7030)

        components_7030 = [
            CrossTFComponent(
                strategy_factory=lambda: make_rsi_1h(BTC),
                df=btc_1h, htf_df=btc_4h,
                engine=engine_1h, weight=0.231, label="BTC_1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=lambda: make_dc_1h(BTC),
                df=btc_1h, htf_df=btc_4h,
                engine=engine_1h, weight=0.231, label="BTC_1hDC",
            ),
            CrossTFComponent(
                strategy_factory=lambda: make_rsi_15m(BTC),
                df=btc_15m, htf_df=btc_4h,
                engine=engine_15m, weight=0.238, label="BTC_15mRSI",
            ),
            CrossTFComponent(
                strategy_factory=lambda s=symbol: make_rsi_1h(s),
                df=sym_1h, htf_df=sym_4h,
                engine=engine_1h, weight=0.099, label=f"{short_name}_1hRSI",
            ),
            CrossTFComponent(
                strategy_factory=lambda s=symbol: make_dc_1h(s),
                df=sym_1h, htf_df=sym_4h,
                engine=engine_1h, weight=0.099, label=f"{short_name}_1hDC",
            ),
            CrossTFComponent(
                strategy_factory=lambda s=symbol: make_rsi_15m(s),
                df=sym_15m, htf_df=sym_4h,
                engine=engine_15m, weight=0.102, label=f"{short_name}_15mRSI",
            ),
        ]
        rpt_7030 = wf_multi.run_cross_tf(components_7030)
        multi_asset_results[label_7030] = rpt_7030
        log_cross_tf(label_7030, rpt_7030)
        logger.info("")

    # ── 3-asset portfolio if we have both ETH and SOL ──
    if len(available_symbols) >= 2:
        for alloc_name, btc_pct, alt_pct in [
            ("equal 33/33/33", 1/3, 1/3),
            ("BTC-heavy 50/25/25", 0.5, 0.25),
        ]:
            sym_a = available_symbols[0]
            sym_b = available_symbols[1]
            short_a = sym_a.split("/")[0]
            short_b = sym_b.split("/")[0]

            label_3 = f"BTC+{short_a}+{short_b} {alloc_name}"
            logger.info("  --- %s ---", label_3)

            a_1h = load_data(sym_a, "1h")
            a_15m = load_data(sym_a, "15m")
            a_4h = load_data(sym_a, "4h")
            b_1h = load_data(sym_b, "1h")
            b_15m = load_data(sym_b, "15m")
            b_4h = load_data(sym_b, "4h")

            # Distribute weights: each asset gets its alloc split across 3 components
            # (33/33/34 internal split per asset)
            def asset_weights(total: float) -> tuple[float, float, float]:
                return (total * 0.33, total * 0.33, total * 0.34)

            btc_w = asset_weights(btc_pct)
            a_w = asset_weights(alt_pct)
            # Second alt gets the remainder
            b_pct = 1.0 - btc_pct - alt_pct
            b_w = asset_weights(b_pct)

            wf_3asset = WalkForwardAnalyzer(n_windows=9)
            components_3 = [
                # BTC
                CrossTFComponent(
                    strategy_factory=lambda: make_rsi_1h(BTC),
                    df=btc_1h, htf_df=btc_4h,
                    engine=engine_1h, weight=btc_w[0], label="BTC_1hRSI",
                ),
                CrossTFComponent(
                    strategy_factory=lambda: make_dc_1h(BTC),
                    df=btc_1h, htf_df=btc_4h,
                    engine=engine_1h, weight=btc_w[1], label="BTC_1hDC",
                ),
                CrossTFComponent(
                    strategy_factory=lambda: make_rsi_15m(BTC),
                    df=btc_15m, htf_df=btc_4h,
                    engine=engine_15m, weight=btc_w[2], label="BTC_15mRSI",
                ),
                # Asset A
                CrossTFComponent(
                    strategy_factory=lambda s=sym_a: make_rsi_1h(s),
                    df=a_1h, htf_df=a_4h,
                    engine=engine_1h, weight=a_w[0], label=f"{short_a}_1hRSI",
                ),
                CrossTFComponent(
                    strategy_factory=lambda s=sym_a: make_dc_1h(s),
                    df=a_1h, htf_df=a_4h,
                    engine=engine_1h, weight=a_w[1], label=f"{short_a}_1hDC",
                ),
                CrossTFComponent(
                    strategy_factory=lambda s=sym_a: make_rsi_15m(s),
                    df=a_15m, htf_df=a_4h,
                    engine=engine_15m, weight=a_w[2], label=f"{short_a}_15mRSI",
                ),
                # Asset B
                CrossTFComponent(
                    strategy_factory=lambda s=sym_b: make_rsi_1h(s),
                    df=b_1h, htf_df=b_4h,
                    engine=engine_1h, weight=b_w[0], label=f"{short_b}_1hRSI",
                ),
                CrossTFComponent(
                    strategy_factory=lambda s=sym_b: make_dc_1h(s),
                    df=b_1h, htf_df=b_4h,
                    engine=engine_1h, weight=b_w[1], label=f"{short_b}_1hDC",
                ),
                CrossTFComponent(
                    strategy_factory=lambda s=sym_b: make_rsi_15m(s),
                    df=b_15m, htf_df=b_4h,
                    engine=engine_15m, weight=b_w[2], label=f"{short_b}_15mRSI",
                ),
            ]
            rpt_3 = wf_3asset.run_cross_tf(components_3)
            multi_asset_results[label_3] = rpt_3
            log_cross_tf(label_3, rpt_3)
            logger.info("")

    # ═════════════════════════════════════════════════════════════
    #   PART 4: Summary
    # ═════════════════════════════════════════════════════════════

    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE 22 — SUMMARY")
    logger.info("=" * 72)
    logger.info("")

    # Per-symbol standalone results
    logger.info("  Per-Symbol Strategy Results (9w WF):")
    logger.info("  %-30s %8s %6s %6s", "Strategy", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 55)

    # BTC baseline
    logger.info(
        "  %-30s %+7.2f%% %5d%% %6d  *** BASELINE ***",
        "BTC Cross-TF 33/33/34",
        btc_cross_rpt.oos_total_return,
        int(btc_cross_rpt.robustness_score * 100),
        btc_cross_rpt.total_trades,
    )

    for symbol in available_symbols:
        short = symbol.split("/")[0]
        for strat_key, strat_name in [
            ("rsi_1h", f"{short} RSI_1h+MTF"),
            ("dc_1h", f"{short} DC_1h+MTF"),
            ("rsi_15m", f"{short} RSI_15m+MTF"),
            ("cross_tf", f"{short} Cross-TF 33/33/34"),
        ]:
            rpt = symbol_results[symbol].get(strat_key)
            if rpt is None:
                continue
            if hasattr(rpt, "oos_total_return"):
                logger.info(
                    "  %-30s %+7.2f%% %5d%% %6d",
                    strat_name,
                    rpt.oos_total_return,
                    int(rpt.robustness_score * 100),
                    getattr(rpt, "total_trades", getattr(rpt, "oos_total_trades", 0)),
                )

    logger.info("")

    # Multi-asset portfolio results
    logger.info("  Multi-Asset Portfolio Results (9w WF):")
    logger.info("  %-45s %8s %6s %6s", "Portfolio", "OOS Ret", "Rob", "Trades")
    logger.info("  " + "-" * 70)

    # Sort by robustness then OOS return
    sorted_results = sorted(
        multi_asset_results.items(),
        key=lambda x: (x[1].robustness_score, x[1].oos_total_return),
        reverse=True,
    )

    for label, report in sorted_results:
        rob = int(report.robustness_score * 100)
        marker = " ★" if rob > 88 else " ***" if rob >= 88 else ""
        logger.info(
            "  %-45s %+7.2f%% %5d%% %6d%s",
            label, report.oos_total_return,
            rob, report.total_trades, marker,
        )

    logger.info("")

    # Check if we broke the ceiling
    best_multi = sorted_results[0] if sorted_results else None
    if best_multi:
        best_rob = int(best_multi[1].robustness_score * 100)
        if best_rob > 88:
            logger.info("  ★★★ BREAKTHROUGH: %d%% robustness > 88%% ceiling! ★★★", best_rob)
            logger.info("  Best: %s", best_multi[0])
        elif best_rob == 88:
            logger.info("  Same ceiling (88%%). Multi-asset diversification did NOT help.")
            logger.info("  Possible causes:")
            logger.info("    - Crypto assets are too correlated (all crash together)")
            logger.info("    - W2 (Nov 20-Dec 2) is a SYSTEMIC event affecting all crypto")
        else:
            logger.info("  WARNING: Multi-asset portfolio WORSE than BTC-only (%d%% < 88%%)", best_rob)
            logger.info("  Adding other assets may introduce noise or different risk profiles")

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Phase 22 complete. Results in logs/phase22.log")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
