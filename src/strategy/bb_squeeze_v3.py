"""BB Squeeze Breakout v3 — DEPRECATED, use BBSqueezeBreakoutStrategy instead.

All v3 entry-quality filters degraded OOS performance vs baseline v1/v2:
  - Body>40%: OOS -0.70%, Robustness 40% (worse than baseline +5.71%, 60%)
  - Body>50%: OOS -3.10%, Robustness 20%
  - Body>60%: OOS +1.27%, Robustness 20%
  - NoAsianLate: OOS +1.01%, Robustness 60% (same as baseline, not better)
  - USEuroOnly: OOS +0.41%, Robustness 60%
  - Combined Body+Session+BE: all worse than baseline

This file is kept for backward compatibility with existing phase scripts.
Use BBSqueezeBreakoutStrategy from bb_squeeze_breakout.py instead.
"""

from src.strategy.bb_squeeze_breakout import BBSqueezeBreakoutStrategy


# Backward-compat alias — existing scripts import BBSqueezeV3Strategy
BBSqueezeV3Strategy = BBSqueezeBreakoutStrategy
