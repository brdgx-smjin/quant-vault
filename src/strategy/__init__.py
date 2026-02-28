"""Strategy module — AI Quant Trading System.

Strategy status (Phase 36, 2026-02-27):

PRODUCTION (use with MultiTimeframeFilter):
  - RSIMeanReversionStrategy          77% rob (15m, 9w), 66% (1h, 9w)
  - DonchianTrendStrategy             55% rob (1h, 9w), portfolio diversifier
  - WilliamsRMeanReversionStrategy    77% rob (1h, 9w), 4th component — Phase 25
  - CCIMeanReversionStrategy          66% rob (1h, 9w), alternative to RSI
  - VWAPMeanReversionStrategy         55% rob (1h, 9w), portfolio component
  - BBSqueezeBreakoutStrategy         44% rob (1h, 9w), minor component only

INFRASTRUCTURE:
  - BaseStrategy, Signal, TradeSignal
  - MultiTimeframeFilter (decorator)
  - PortfolioStrategy (composite)
  - CrossTimeframePortfolio (cross-TF live trading)
  - CrossTFComponent / CrossTFReport (walk_forward) — Phase 17

DEPRECATED: see individual files for WF results (21 files, all audited).

Best portfolio (Phase 25-36, date-aligned 9w):
  4-comp Cross-TF 1hRSI/1hDC/15mRSI/1hWR 15/50/10/25 = 88% rob, OOS +23.98%.
  303/375 weight combos (80.8%) achieve 88% — MORE robust than 3-comp (13/28=46.4%).
  263 fine-grained combos (5% step, 10% min) at 88% — Phase 31 confirmed.
  11/12 param perturbations at 88% — STRONG stability, NOT overfit.
  Previous 3-comp best: 1hRSI/1hDC/15mRSI 33/33/34 = 88% rob, +18.81% OOS.
  88% is the ABSOLUTE CEILING — W2 (Nov 20-Dec 2) is unsolvable (ALL 4 components negative).
  Vol-scaling option: Sharpe 1.011, MaxDD -2.09% (vs 0.631/3.34%) at cost of -2.6% return.
  Fixed weights remain optimal — ALL adaptive schemes tested and inferior (Phase 31).
  Tested & failed to break ceiling (Phase 18-36):
    Phase 18: CCI 4th comp, chop filter, wide SL, long cooldown
    Phase 19: 2h/30m/5m timeframes, time-of-day filter
    Phase 22: Multi-asset ETH/SOL — ALL WORSE (77% best vs 88%)
    Phase 27: Ichimoku Kijun MR 5th comp — drops to 77%
    Phase 28: Fisher Transform (66% best), Z-Score MR (66% best)
    Phase 29: Funding Rate filter — too rare (0.6%) to coincide with signals
    Phase 30: CMF (-7.89% OOS), StochRSI (max 44%), EFI (67% but low return)
    Phase 31: Adaptive weights ALL worse — momentum, inv-vol, best-recent, defensive
    Phase 32: Hurst/ER regime filters — W2 NOT detectable, all filters WORSE
    Phase 33: Aroon (44% max), DPO (77% corrected, +6%), TSI (77%, +16%) — none improve portfolio
    Phase 34: Exit optimization — breakeven, signal-driven exit, TP decay ALL worse or same
    Phase 35: Supertrend (77% standalone, drops portfolio to 77%) — 12th indicator ALL fail
    Phase 36: PSAR (77%, +27.62% standalone), TRIX (66%, +9.40%) — 13th-14th ALL fail
"""

from .base import BaseStrategy, Signal, TradeSignal
from .mtf_filter import MultiTimeframeFilter
from .portfolio import PortfolioStrategy
from .cross_tf_portfolio import CrossTimeframePortfolio

# Production strategies
from .rsi_mean_reversion import RSIMeanReversionStrategy
from .donchian_trend import DonchianTrendStrategy
from .willr_mean_reversion import WilliamsRMeanReversionStrategy
from .cci_mean_reversion import CCIMeanReversionStrategy
from .vwap_mean_reversion import VWAPMeanReversionStrategy
from .bb_squeeze_breakout import BBSqueezeBreakoutStrategy

__all__ = [
    # Infrastructure
    "BaseStrategy",
    "Signal",
    "TradeSignal",
    "MultiTimeframeFilter",
    "PortfolioStrategy",
    "CrossTimeframePortfolio",
    # Production strategies
    "RSIMeanReversionStrategy",
    "DonchianTrendStrategy",
    "WilliamsRMeanReversionStrategy",
    "CCIMeanReversionStrategy",
    "VWAPMeanReversionStrategy",
    "BBSqueezeBreakoutStrategy",
]
