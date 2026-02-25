"""BB Squeeze Breakout v2 — DEPRECATED, use BBSqueezeBreakoutStrategy instead.

This file is kept for backward compatibility with existing phase scripts.
All v2 features (ADX filter, disable_tp) have been merged into the canonical
bb_squeeze_breakout.py. Import BBSqueezeBreakoutStrategy directly.

Walk-Forward results: see bb_squeeze_breakout.py docstring.
"""

from src.strategy.bb_squeeze_breakout import BBSqueezeBreakoutStrategy


# Backward-compat alias — existing scripts import BBSqueezeV2Strategy
BBSqueezeV2Strategy = BBSqueezeBreakoutStrategy
