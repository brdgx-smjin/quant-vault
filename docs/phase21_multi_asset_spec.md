# Phase 21 — Multi-Asset Diversification Research Spec

**Author**: strategy-researcher
**Date**: 2026-02-26
**Status**: BLOCKED — waiting for data-engineer to collect ETH/SOL data
**Prerequisites**: Phase 20 complete (88% robustness confirmed, parameter-stable)

## Background

BTC single-asset cross-TF portfolio has reached a structural ceiling:
- **88% robustness (8/9 windows)** — only W2 (Nov 20-Dec 2) remains negative
- W2 is an extreme macro event: 44.71% price range, ATR 84th percentile
- All attempts to break 88% have FAILED (Phase 18-19): CCI 4th component,
  chop filter, wide SL, timeframe sweep (5m/30m/2h), time-of-day filter
- Phase 20 confirms strategy is NOT parameter-overfit (28/28 perturbations pass)

**Next frontier**: cross-ASSET diversification (ETH, SOL, etc.)

## Hypothesis

BTC W2 (Nov 20-Dec 2, 2025) was a violent crash-recovery across all crypto.
However, **different assets may have different negative windows** in WF,
because:
- ETH has different volatility profile and funding rate dynamics
- Altcoins may trend/range in different periods than BTC
- If ETH's worst WF window is W5 or W6 (not W2), portfolio diversification
  across assets could break the 88% ceiling

**Risk**: Crypto assets are highly correlated during extreme macro events (>0.85 correlation).
W2 may be negative for ALL crypto assets simultaneously, maintaining the 88% ceiling.
This is the most likely outcome but must be empirically tested.

## Data Requirements (for data-engineer)

### Priority 1: ETH/USDT:USDT
```
Symbol: ETH/USDT:USDT
Timeframes: 15m, 1h, 4h
Period: same as BTC (~2025-02-25 to present, 1 year)
Format: data/processed/ETH_USDT_USDT_{timeframe}.parquet
Schema: same as BTC (datetime index, open/high/low/close/volume columns)
```

### Priority 2: SOL/USDT:USDT (if ETH shows promise)
```
Symbol: SOL/USDT:USDT
Timeframes: 15m, 1h, 4h
Period: same as BTC
Format: data/processed/SOL_USDT_USDT_{timeframe}.parquet
```

### Priority 3: Continuous Open Interest Collection
```
Symbol: BTC/USDT:USDT (and ETH if added)
Endpoint: Binance OI API (1h granularity)
Problem: API only provides ~30 days lookback
Solution: Daily cron job to append new OI data
Current: data/processed/BTC_USDT_USDT_open_interest_1h.parquet (20 days only)
Need: 365+ days for WF testing
```

### Config Changes (for data-engineer)
Update `config/symbols.yaml`:
```yaml
symbols:
  - symbol: "BTC/USDT:USDT"
    timeframes: ["15m", "1h", "4h"]
    leverage: 5
    enabled: true

  - symbol: "ETH/USDT:USDT"
    timeframes: ["15m", "1h", "4h"]
    leverage: 3
    enabled: true
```

## Research Plan (strategy-researcher, once data available)

### Step 1: ETH Strategy Validation (~Phase 21a)
1. Load ETH 1h + 4h data
2. Run RSI_MR+MTF on ETH with BTC optimal params (oversold=35, overbought=65, sl=2.0, tp=3.0, cool=6)
3. Run DC+MTF on ETH with BTC optimal params (period=24, sl=2.0, rr=2.0, vol=0.8, cool=6)
4. 9-window WF for each
5. Compare per-window returns with BTC — identify overlap in negative windows

**Key question**: Does ETH W2 also fail? If yes, cross-asset won't break 88%.

### Step 2: ETH Cross-TF Portfolio (~Phase 21b)
If Step 1 shows ≥55% robustness for any ETH strategy:
1. Run ETH 15m RSI_MR+MTF (cool=12, max_hold=96)
2. Build ETH-only cross-TF portfolio (same architecture as BTC)
3. Measure ETH robustness independently

### Step 3: BTC+ETH Cross-Asset Portfolio (~Phase 21c)
1. Define portfolio: e.g., BTC 60% + ETH 40% cross-TF
2. Extend `run_cross_tf()` to support multi-asset components:
   - Each component has its own asset's data
   - Weighted returns combined across assets AND timeframes
3. Measure combined robustness — target: >88%

### Step 4: Correlation Analysis
1. Compute BTC-ETH per-window return correlation
2. If correlation > 0.8 in W2, multi-asset diversification won't help for W2
3. Document which windows ARE decorrelated — this determines potential

## Architecture Considerations

### Current Code Compatibility
- **Strategies**: Already symbol-agnostic (`symbol` param in constructor)
- **BacktestEngine**: Symbol-independent (operates on DataFrames)
- **WalkForwardAnalyzer.run_cross_tf()**: Already supports multi-component
  - Each `CrossTFComponent` has its own `df` and `engine`
  - Just add ETH components alongside BTC components
- **Data format**: `{SYMBOL}_{timeframe}.parquet` — consistent naming

### Required Code Changes (minimal)
1. `config/settings.py`: Add `SYMBOLS` list (currently single `SYMBOL`)
2. `run_cross_tf()`: No changes needed — already supports N components
3. New script: `scripts/run_phase21.py` for multi-asset WF tests
4. Risk: `max_open_positions: 3` in risk.yaml may need adjustment for 2+ assets

### Live Trading Changes (for execution-engineer, later)
- `CrossTimeframePortfolio` needs to handle multiple symbols
- `TradingEngine` needs concurrent streams for BTC and ETH 15m data
- Position management per symbol

## Success Criteria

| Outcome | Robustness | Action |
|---|---|---|
| BTC+ETH portfolio > 88% | ≥ 1 extra positive window | Adopt multi-asset |
| BTC+ETH portfolio = 88% | Same ceiling, same W2 | No benefit, stop |
| ETH standalone < 55% | Strategy doesn't transfer | BTC-only, explore other assets |

## Estimated Timeline
- Data collection (data-engineer): 1-2 days for historical, ongoing for OI
- Phase 21a (ETH validation): ~1 hour compute time
- Phase 21b (ETH cross-TF): ~30 min compute time
- Phase 21c (BTC+ETH combined): ~1 hour compute time
- Total strategy-researcher work: ~1 session once data is ready

## References
- Phase 17 log: formal cross-TF validation methodology
- Phase 18 log: 88% ceiling breaking attempts (all failed)
- Phase 19 log: timeframe sweep (only 1h and 15m viable)
- Phase 20 log: parameter stability (28/28 pass)
