"""Backtesting engine using vectorbt with proper LONG/SHORT support."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.strategy.base import BaseStrategy, Signal

from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeLog:
    """Single trade record for detailed logging."""

    trade_id: int
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    exit_reason: str  # "sl", "tp", "timeout"
    pnl: float
    return_pct: float
    bars_held: int
    metadata: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Backtest performance metrics."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    avg_trade_return: float
    portfolio: object  # vbt.Portfolio
    trade_logs: list[TradeLog] = field(default_factory=list)


class BacktestEngine:
    """Run backtests on strategies using vectorbt.

    Properly handles both LONG and SHORT positions by passing
    separate long_entries/short_entries to vectorbt.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        commission: float = 0.0004,  # Binance futures taker fee
        slippage: float = 0.0001,
        max_hold_bars: int = 48,  # Auto-exit after N bars if no signal
        trailing_atr_mult: float = 0.0,  # 0 = disabled; >0 = ATR trailing stop
        breakeven_at_r: float = 0.0,  # 0 = disabled; >0 = move SL to entry after N*R profit
        freq: str = "1h",  # Data frequency for Sharpe ratio annualization
    ) -> None:
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_hold_bars = max_hold_bars
        self.trailing_atr_mult = trailing_atr_mult
        self.breakeven_at_r = breakeven_at_r
        self.freq = freq

    def run(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        step: int = 1,
        htf_df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Run backtest for a strategy on historical data.

        Generates separate LONG and SHORT entry/exit arrays and
        uses direction-correct SL/TP logic.

        Args:
            strategy: Strategy instance.
            df: OHLCV DataFrame.
            step: Evaluate every N bars (1 = every bar, faster with higher).
            htf_df: Optional higher-timeframe OHLCV DataFrame (e.g. 4h).
                If provided and strategy has ``set_htf_data``, the engine
                slices HTF data up to the current bar's timestamp before
                each signal generation to prevent look-ahead bias.

        Returns:
            BacktestResult with performance metrics.
        """
        has_htf = htf_df is not None and hasattr(strategy, "set_htf_data")
        n = len(df)
        long_entries = np.zeros(n, dtype=bool)
        long_exits = np.zeros(n, dtype=bool)
        short_entries = np.zeros(n, dtype=bool)
        short_exits = np.zeros(n, dtype=bool)

        in_position = False
        position_side = ""  # "long" or "short"
        entry_bar = 0
        current_sl = np.nan
        current_tp = np.nan
        original_sl = np.nan  # Original SL for breakeven calculation
        best_price = np.nan  # For trailing stop: best price since entry
        entry_atr = np.nan   # ATR at entry for trailing stop distance
        warmup = 50

        # Detailed trade logging
        trade_logs: list[TradeLog] = []
        trade_id = 0
        entry_price = 0.0
        entry_time = ""

        has_atr = "atr_14" in df.columns

        for i in range(warmup, n, step):
            close_i = df["close"].iloc[i]
            high_i = df["high"].iloc[i]
            low_i = df["low"].iloc[i]

            # Update trailing stop if enabled
            if in_position and self.trailing_atr_mult > 0 and not np.isnan(entry_atr):
                if position_side == "long":
                    best_price = max(best_price, high_i)
                    trail_sl = best_price - entry_atr * self.trailing_atr_mult
                    if not np.isnan(current_sl):
                        current_sl = max(current_sl, trail_sl)
                    else:
                        current_sl = trail_sl
                elif position_side == "short":
                    best_price = min(best_price, low_i)
                    trail_sl = best_price + entry_atr * self.trailing_atr_mult
                    if not np.isnan(current_sl):
                        current_sl = min(current_sl, trail_sl)
                    else:
                        current_sl = trail_sl

            # Breakeven SL: move SL to entry price after reaching N*R profit
            if in_position and self.breakeven_at_r > 0 and not np.isnan(original_sl):
                risk_dist = abs(entry_price - original_sl)
                be_target = risk_dist * self.breakeven_at_r
                if position_side == "long":
                    if high_i - entry_price >= be_target:
                        current_sl = max(current_sl, entry_price) if not np.isnan(current_sl) else entry_price
                elif position_side == "short":
                    if entry_price - low_i >= be_target:
                        current_sl = min(current_sl, entry_price) if not np.isnan(current_sl) else entry_price

            # Check SL/TP exit with direction-correct logic
            if in_position:
                exit_reason = ""
                exit_price = close_i

                if position_side == "long":
                    # LONG: SL triggers when price drops below SL
                    if not np.isnan(current_sl) and low_i <= current_sl:
                        exit_reason = "sl"
                        exit_price = current_sl
                    # LONG: TP triggers when price rises above TP
                    elif not np.isnan(current_tp) and high_i >= current_tp:
                        exit_reason = "tp"
                        exit_price = current_tp
                elif position_side == "short":
                    # SHORT: SL triggers when price rises above SL
                    if not np.isnan(current_sl) and high_i >= current_sl:
                        exit_reason = "sl"
                        exit_price = current_sl
                    # SHORT: TP triggers when price drops below TP
                    elif not np.isnan(current_tp) and low_i <= current_tp:
                        exit_reason = "tp"
                        exit_price = current_tp

                # Timeout exit (same for both sides)
                if not exit_reason and i - entry_bar >= self.max_hold_bars:
                    exit_reason = "timeout"
                    exit_price = close_i

                if exit_reason:
                    if position_side == "long":
                        long_exits[i] = True
                    else:
                        short_exits[i] = True
                    in_position = False

                    # Calculate PnL
                    if position_side == "long":
                        pnl_pct = (exit_price / entry_price - 1) * 100
                    else:
                        pnl_pct = (1 - exit_price / entry_price) * 100
                    pnl_usdt = self.initial_capital * 0.1 * pnl_pct / 100

                    trade_logs.append(TradeLog(
                        trade_id=trade_id,
                        entry_time=entry_time,
                        exit_time=str(df.index[i]),
                        side=position_side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        stop_loss=current_sl if not np.isnan(current_sl) else 0.0,
                        take_profit=current_tp if not np.isnan(current_tp) else 0.0,
                        exit_reason=exit_reason,
                        pnl=pnl_usdt,
                        return_pct=pnl_pct,
                        bars_held=i - entry_bar,
                    ))
                    continue

            if not in_position:
                window = df.iloc[: i + 1]

                # Slice HTF data up to current bar to prevent look-ahead bias
                if has_htf:
                    current_time = df.index[i]
                    htf_slice = htf_df[htf_df.index <= current_time]
                    strategy.set_htf_data(htf_slice)

                sig = strategy.generate_signal(window)

                if sig.signal == Signal.LONG:
                    long_entries[i] = True
                    in_position = True
                    position_side = "long"
                    entry_bar = i
                    current_sl = sig.stop_loss if sig.stop_loss else np.nan
                    current_tp = sig.take_profit if sig.take_profit else np.nan
                    original_sl = current_sl
                    best_price = high_i
                    entry_atr = float(df["atr_14"].iloc[i]) if has_atr else np.nan
                    trade_id += 1
                    entry_price = close_i
                    entry_time = str(df.index[i])
                elif sig.signal == Signal.SHORT:
                    short_entries[i] = True
                    in_position = True
                    position_side = "short"
                    entry_bar = i
                    current_sl = sig.stop_loss if sig.stop_loss else np.nan
                    current_tp = sig.take_profit if sig.take_profit else np.nan
                    original_sl = current_sl
                    best_price = low_i
                    entry_atr = float(df["atr_14"].iloc[i]) if has_atr else np.nan
                    trade_id += 1
                    entry_price = close_i
                    entry_time = str(df.index[i])

        long_entries_s = pd.Series(long_entries, index=df.index)
        long_exits_s = pd.Series(long_exits, index=df.index)
        short_entries_s = pd.Series(short_entries, index=df.index)
        short_exits_s = pd.Series(short_exits, index=df.index)

        pf = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=long_entries_s,
            exits=long_exits_s,
            short_entries=short_entries_s,
            short_exits=short_exits_s,
            init_cash=self.initial_capital,
            fees=self.commission,
            slippage=self.slippage,
            freq=self.freq,
        )

        result = self._extract_results(pf)
        result.trade_logs = trade_logs

        # Log all trades summary
        self._log_trades(strategy.name, trade_logs, result)

        return result

    def run_vectorized(
        self,
        entries: pd.Series,
        exits: pd.Series,
        close: pd.Series,
        freq: str = "1h",
    ) -> BacktestResult:
        """Run backtest with pre-computed entry/exit signals (long-only).

        Args:
            entries: Boolean series of entry signals.
            exits: Boolean series of exit signals.
            close: Close price series.
            freq: Data frequency.

        Returns:
            BacktestResult.
        """
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.commission,
            slippage=self.slippage,
            freq=freq,
        )
        return self._extract_results(pf)

    def _extract_results(self, pf: vbt.Portfolio) -> BacktestResult:
        """Extract metrics from a vectorbt portfolio."""
        stats = pf.stats()
        trades = pf.trades.records_readable if len(pf.trades) > 0 else pd.DataFrame()

        win_rate = 0.0
        profit_factor = 0.0
        avg_return = 0.0

        if len(trades) > 0 and "PnL" in trades.columns:
            wins = trades[trades["PnL"] > 0]
            losses = trades[trades["PnL"] < 0]
            win_rate = len(wins) / len(trades)
            total_profit = wins["PnL"].sum() if len(wins) > 0 else 0
            total_loss = abs(losses["PnL"].sum()) if len(losses) > 0 else 1
            profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
            avg_return = float(trades["PnL"].mean())

        return BacktestResult(
            total_return=float(stats.get("Total Return [%]", 0)),
            sharpe_ratio=float(stats.get("Sharpe Ratio", 0)),
            max_drawdown=float(stats.get("Max Drawdown [%]", 0)),
            win_rate=win_rate,
            total_trades=len(trades),
            profit_factor=profit_factor,
            avg_trade_return=avg_return,
            portfolio=pf,
        )

    def _log_trades(
        self,
        strategy_name: str,
        trade_logs: list[TradeLog],
        result: BacktestResult,
    ) -> None:
        """Log detailed trade-by-trade breakdown."""
        if not trade_logs:
            logger.info("[%s] No trades executed.", strategy_name)
            return

        logger.info("")
        logger.info("[%s] ── Trade Log (%d trades) ──", strategy_name, len(trade_logs))
        logger.info(
            "  %4s  %-5s  %-19s  %-19s  %10s  %10s  %10s  %10s  %7s  %7s  %5s",
            "#", "Side", "Entry Time", "Exit Time",
            "Entry$", "Exit$", "SL$", "TP$",
            "PnL%", "PnL$", "Exit",
        )
        logger.info("  " + "-" * 130)

        cumulative_pnl = 0.0
        wins = 0
        losses = 0
        sl_exits = 0
        tp_exits = 0
        timeout_exits = 0

        for t in trade_logs:
            cumulative_pnl += t.pnl
            if t.return_pct > 0:
                wins += 1
            elif t.return_pct < 0:
                losses += 1
            if t.exit_reason == "sl":
                sl_exits += 1
            elif t.exit_reason == "tp":
                tp_exits += 1
            elif t.exit_reason == "timeout":
                timeout_exits += 1

            logger.info(
                "  %4d  %-5s  %-19s  %-19s  %10.2f  %10.2f  %10.2f  %10.2f  %+6.2f%%  %+7.2f  %-5s",
                t.trade_id, t.side.upper(),
                t.entry_time[:19], t.exit_time[:19],
                t.entry_price, t.exit_price,
                t.stop_loss, t.take_profit,
                t.return_pct, t.pnl,
                t.exit_reason,
            )

        logger.info("  " + "-" * 130)
        logger.info(
            "  Summary: %d W / %d L | SL:%d TP:%d Timeout:%d | Cum PnL: %+.2f",
            wins, losses, sl_exits, tp_exits, timeout_exits, cumulative_pnl,
        )
        logger.info(
            "  Result:  Return %+.2f%% | Sharpe %.2f | MaxDD %.1f%% | WR %.1f%% | PF %.2f",
            result.total_return, result.sharpe_ratio, result.max_drawdown,
            result.win_rate * 100, result.profit_factor,
        )
        logger.info("")
