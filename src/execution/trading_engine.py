"""Central trading engine coordinator for live/paper trading."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import pandas as pd

from config.settings import SYMBOL, TESTNET
from src.data.collector import BinanceCollector
from src.data.preprocessor import DataPreprocessor
from src.data.stream import BinanceStream
from src.execution.binance_executor import BinanceExecutor
from src.execution.order_manager import OrderManager, OrderSide
from src.execution.position_manager import PositionManager
from src.execution.risk_manager import RiskManager
from src.indicators.basic import BasicIndicators
from src.monitoring.alerter import AlertLevel, Alerter
from src.monitoring.dashboard import DashboardProvider
from src.strategy.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)

MAX_DF_BARS = 500


class TradingEngine:
    """Central coordinator that wires all components for live trading.

    Lifecycle:
        initialize() -> fetch warmup bars
        run()        -> WebSocket stream -> on_candle_close() -> signal -> risk -> execute
        shutdown()   -> close connections, alert status
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        symbol: str = SYMBOL,
        timeframe: str = "4h",
        testnet: bool = TESTNET,
        warmup_bars: int = 200,
        htf_timeframe: Optional[str] = None,
    ) -> None:
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.testnet = testnet
        self.warmup_bars = warmup_bars
        self.htf_timeframe = htf_timeframe

        # Components
        self.executor = BinanceExecutor(testnet=testnet)
        self.order_manager = OrderManager()
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager()
        self.alerter = Alerter()
        self.dashboard = DashboardProvider(initial_equity=Decimal("5000"))
        self.collector = BinanceCollector(symbol=symbol, testnet=testnet)

        # State
        self.df: Optional[pd.DataFrame] = None
        self.df_htf: Optional[pd.DataFrame] = None
        self._running = False
        self._last_daily_reset: Optional[datetime] = None
        self._stream: Optional[BinanceStream] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._trade_count = 0
        self._closing = False  # Guard against concurrent close attempts

        # Partial close config from risk.yaml
        tp_cfg = self.risk_manager.config.get("take_profit", {})
        self._partial_close_pcts = tp_cfg.get("partial_close_pcts", [1.0])
        self._tp_rr_levels = tp_cfg.get("levels", [1.0])

    async def initialize(self) -> None:
        """Fetch warmup bars and prepare indicators."""
        logger.info("Initializing TradingEngine: %s %s (testnet=%s)",
                     self.symbol, self.timeframe, self.testnet)

        self.df = await self.collector.fetch_ohlcv(
            timeframe=self.timeframe,
            limit=self.warmup_bars,
        )
        self.df = BasicIndicators.add_all(self.df)
        logger.info("Warmup complete: %d bars loaded (%s ~ %s)",
                     len(self.df), self.df.index[0], self.df.index[-1])

        # Build HTF data by resampling if strategy supports set_htf_data
        if self.htf_timeframe and hasattr(self.strategy, "set_htf_data"):
            self._rebuild_htf()
            logger.info(
                "HTF (%s) warmup: %d bars (%s ~ %s)",
                self.htf_timeframe, len(self.df_htf),
                self.df_htf.index[0], self.df_htf.index[-1],
            )

        balance = await self.executor.get_balance()
        logger.info("Account balance: %s USDT", balance)

        await self.alerter.alert(
            f"🚀 **TradingEngine Started**\n"
            f"Symbol: {self.symbol} | TF: {self.timeframe}\n"
            f"Strategy: {self.strategy.name}\n"
            f"Balance: {balance} USDT\n"
            f"Mode: {'TESTNET' if self.testnet else 'LIVE'}"
        )

    async def run(self) -> None:
        """Main loop: subscribe to WebSocket and process candles."""
        self._running = True
        self._last_daily_reset = datetime.now(timezone.utc)

        self._stream = BinanceStream(
            symbol=self.symbol,
            testnet=self.testnet,
            on_candle=self._on_kline,
        )

        logger.info("Starting WebSocket stream for %s@kline_%s",
                     self.symbol, self.timeframe)

        # Cancel any stale heartbeat before creating a new one
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

        self._heartbeat_task = asyncio.create_task(self._heartbeat())
        try:
            await self._stream.start_kline_stream(timeframe=self.timeframe)
        finally:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            self._heartbeat_task = None

    async def _on_kline(self, kline: dict) -> None:
        """Handle incoming kline from WebSocket.

        Only processes when a candle closes (x=True).
        Errors are caught to prevent crashing the WebSocket stream.
        """
        try:
            if not kline.get("x"):
                # Candle not closed yet — monitor SL/TP on every tick
                await self._check_sl_tp(float(kline["h"]), float(kline["l"]))
                return

            # Candle closed — process it
            await self._on_candle_close(kline)
        except Exception:
            logger.exception("Error processing kline (stream continues)")

    def _rebuild_htf(self) -> None:
        """Resample base-TF data to higher timeframe and add indicators."""
        self.df_htf = DataPreprocessor.resample(self.df, self.htf_timeframe)
        htf_ohlcv = self.df_htf[["open", "high", "low", "close", "volume"]]
        self.df_htf = BasicIndicators.add_all(htf_ohlcv)

    async def _on_candle_close(self, kline: dict) -> None:
        """Process a closed candle: update data, generate signal, execute."""
        # Append new candle to DataFrame (or update if timestamp already exists)
        ts = pd.Timestamp(int(kline["t"]), unit="ms")
        row_data = {
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
        }

        # Strip to OHLCV first — prevents stale indicator columns and
        # duplicate column names from prior add_all() pd.concat calls
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        self.df = self.df[ohlcv_cols].copy()

        # Deduplicate index to prevent pd.concat InvalidIndexError
        if self.df.index.duplicated().any():
            logger.warning("Duplicate index detected, deduplicating (keeping last)")
            self.df = self.df[~self.df.index.duplicated(keep="last")]

        # Update existing row or append new one
        if ts in self.df.index:
            for col, val in row_data.items():
                self.df.at[ts, col] = val
        else:
            new_row = pd.DataFrame([row_data], index=[ts])
            self.df = pd.concat([self.df, new_row])

        # Bound DataFrame to prevent memory growth
        if len(self.df) > MAX_DF_BARS:
            self.df = self.df.iloc[-MAX_DF_BARS:]

        # Recalculate all indicators on clean OHLCV data
        self.df = BasicIndicators.add_all(self.df)

        close = float(kline["c"])
        logger.info("Candle closed: %s | close=%.2f | bars=%d",
                     ts, close, len(self.df))

        # Daily PnL reset at UTC midnight
        await self._check_daily_reset()

        # Update unrealized PnL for open positions
        self.position_manager.update_unrealized_pnl(
            self.symbol, Decimal(str(close))
        )

        # Log open position status
        if self.symbol in self.position_manager.positions:
            pos = self.position_manager.positions[self.symbol]
            logger.info(
                "[POSITION] %s %s | entry=%.2f | uPnL=%.2f | SL=%s TP=%s",
                pos.side.upper(), self.symbol, float(pos.entry_price),
                float(pos.unrealized_pnl),
                f"{float(pos.stop_loss):.2f}" if pos.stop_loss else "N/A",
                f"{float(pos.take_profit):.2f}" if pos.take_profit else "N/A",
            )

        # Rebuild HTF data and pass to strategy
        if self.htf_timeframe and hasattr(self.strategy, "set_htf_data"):
            self._rebuild_htf()
            self.strategy.set_htf_data(self.df_htf)
            trend = "bullish" if self.df_htf["ema_20"].iloc[-1] > self.df_htf["ema_50"].iloc[-1] else "bearish"
            logger.info("[MTF] 4h trend: %s (EMA20=%.2f, EMA50=%.2f)",
                        trend,
                        self.df_htf["ema_20"].iloc[-1],
                        self.df_htf["ema_50"].iloc[-1])

        # Generate signal
        signal = self.strategy.generate_signal(self.df)
        logger.info(
            "[SIGNAL] %s | confidence=%.3f | strategy=%s | meta=%s",
            signal.signal.value, signal.confidence,
            signal.metadata.get("strategy", self.strategy.name),
            {k: v for k, v in signal.metadata.items() if k != "strategy"},
        )

        if signal.signal in (Signal.LONG, Signal.SHORT):
            await self._execute_signal(signal)

    async def _execute_signal(self, signal) -> None:
        """Check risk, execute order, record position."""
        balance = await self.executor.get_balance()
        risk_check = self.risk_manager.check_trade(signal, balance)

        if not risk_check.approved:
            logger.warning("Trade rejected by RiskManager: %s", risk_check.reason)
            return

        # Close any existing position first
        if self.symbol in self.position_manager.positions:
            await self._close_position(float(signal.price), "new_signal")

        side = OrderSide.BUY if signal.signal == Signal.LONG else OrderSide.SELL
        order = self.order_manager.create_market_order(
            self.symbol, side, risk_check.position_size, risk_check.leverage,
        )

        try:
            result = await self.executor.execute(order)
            logger.info("Order filled: %s", result.get("id"))
        except Exception:
            logger.exception("Order execution failed")
            await self.alerter.alert(
                f"❌ **Order Failed**\n{signal.signal.value} {self.symbol}"
            )
            return

        # Record position
        self._trade_count += 1
        pos_side = "long" if signal.signal == Signal.LONG else "short"
        self.position_manager.open_position(
            symbol=self.symbol,
            side=pos_side,
            entry_price=Decimal(str(signal.price)),
            amount=risk_check.position_size,
            leverage=risk_check.leverage,
            stop_loss=Decimal(str(signal.stop_loss)) if signal.stop_loss else None,
            take_profit=Decimal(str(signal.take_profit)) if signal.take_profit else None,
        )
        self.risk_manager.open_positions += 1

        # Build partial TP levels for the position
        pos = self.position_manager.positions[self.symbol]
        if signal.stop_loss and signal.take_profit and len(self._tp_rr_levels) > 1:
            entry = Decimal(str(signal.price))
            sl = Decimal(str(signal.stop_loss))
            risk_dist = abs(entry - sl)
            tp_levels = []
            for rr, pct in zip(self._tp_rr_levels, self._partial_close_pcts):
                if pos_side == "long":
                    tp_price = entry + risk_dist * Decimal(str(rr))
                else:
                    tp_price = entry - risk_dist * Decimal(str(rr))
                tp_levels.append((tp_price, Decimal(str(pct))))
            pos.tp_levels = tp_levels
            pos.next_tp_idx = 0
            logger.info(
                "[PARTIAL TP] levels: %s",
                [(f"{float(p):.2f}", f"{float(f)*100:.0f}%%") for p, f in tp_levels],
            )

        metrics = self.dashboard.get_metrics()
        logger.info(
            "[ENTRY #%d] %s %s @ %.2f | size=%.4f | lev=%dx | SL=%.2f TP=%.2f | conf=%.3f | bal=%.2f | cumPnL=%.2f",
            self._trade_count, pos_side.upper(), self.symbol, signal.price,
            float(risk_check.position_size), risk_check.leverage,
            signal.stop_loss if signal.stop_loss else 0,
            signal.take_profit if signal.take_profit else 0,
            signal.confidence, float(balance), float(metrics.total_pnl),
        )

        # Calculate risk % (SL distance / entry price)
        risk_pct = 0.0
        if signal.stop_loss and signal.price > 0:
            risk_pct = abs(signal.price - signal.stop_loss) / signal.price * 100

        await self.alerter.alert_entry(
            trade_num=self._trade_count,
            side=pos_side,
            symbol=self.symbol,
            entry_price=signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            size=float(risk_check.position_size),
            confidence=signal.confidence,
            strategy=signal.metadata.get("strategy", self.strategy.name),
            reason=getattr(signal, "reason", ""),
            risk_pct=risk_pct,
        )

    async def _check_sl_tp(self, high: float, low: float) -> None:
        """Monitor SL/TP on every candle tick with partial close support."""
        if self._closing or self.symbol not in self.position_manager.positions:
            return

        pos = self.position_manager.positions[self.symbol]

        # Check stop-loss first
        if pos.stop_loss is not None:
            sl = float(pos.stop_loss)
            if (pos.side == "long" and low <= sl) or (pos.side == "short" and high >= sl):
                logger.info("Stop-loss triggered at %.2f", sl)
                await self._close_position(sl, "stop_loss")
                return

        # Check partial TP levels
        if pos.tp_levels and pos.next_tp_idx < len(pos.tp_levels):
            tp_price, fraction = pos.tp_levels[pos.next_tp_idx]
            tp_f = float(tp_price)
            hit = (pos.side == "long" and high >= tp_f) or (pos.side == "short" and low <= tp_f)

            if hit:
                is_last_tp = pos.next_tp_idx >= len(pos.tp_levels) - 1
                if is_last_tp:
                    # Last TP level — close entire remaining position
                    logger.info("Final TP%d triggered at %.2f — closing remaining", pos.next_tp_idx + 1, tp_f)
                    await self._close_position(tp_f, "take_profit")
                else:
                    # Partial close
                    await self._partial_close(tp_f, fraction, pos.next_tp_idx + 1)
                return

        # Fallback: single TP (no partial levels configured)
        elif pos.take_profit is not None and not pos.tp_levels:
            tp = float(pos.take_profit)
            if (pos.side == "long" and high >= tp) or (pos.side == "short" and low <= tp):
                logger.info("Take-profit triggered at %.2f", tp)
                await self._close_position(tp, "take_profit")
                return

    async def _partial_close(self, exit_price: float, fraction: Decimal, tp_num: int) -> None:
        """Execute partial close at a TP level."""
        if self._closing or self.symbol not in self.position_manager.positions:
            return

        self._closing = True
        pos = self.position_manager.positions[self.symbol]
        close_amount = pos.amount * fraction

        # Execute partial closing order
        close_side = OrderSide.SELL if pos.side == "long" else OrderSide.BUY
        order = self.order_manager.create_market_order(
            self.symbol, close_side, close_amount, pos.leverage,
        )
        order.reduce_only = True

        try:
            await self.executor.execute(order)
        except Exception:
            logger.exception("Partial close order failed")
            self._closing = False
            return

        pnl = self.position_manager.partial_close(
            self.symbol, Decimal(str(exit_price)), fraction,
        )
        self.risk_manager.update_daily_pnl(pnl)
        self.dashboard.record_trade(pnl, {
            "side": pos.side,
            "entry": float(pos.entry_price),
            "exit": exit_price,
            "reason": f"partial_tp{tp_num}",
        })

        # Advance to next TP level
        if self.symbol in self.position_manager.positions:
            self.position_manager.positions[self.symbol].next_tp_idx = tp_num

            # Move SL to entry (breakeven) after first partial TP
            if tp_num == 1:
                old_sl = float(pos.stop_loss) if pos.stop_loss else 0
                self.position_manager.positions[self.symbol].stop_loss = pos.entry_price
                logger.info(
                    "[BREAKEVEN] SL moved to entry %.2f after TP1",
                    float(pos.entry_price),
                )
                await self.alerter.alert_sl_adjustment(
                    symbol=self.symbol,
                    side=pos.side,
                    old_sl=old_sl,
                    new_sl=float(pos.entry_price),
                    reason=f"TP{tp_num} 달성 → 손익분기 SL",
                )

        entry_price = float(pos.entry_price)
        if entry_price > 0:
            ret_pct = ((exit_price / entry_price) - 1) * 100 if pos.side == "long" else ((1 - exit_price / entry_price)) * 100
        else:
            ret_pct = 0.0

        remaining = self.position_manager.positions[self.symbol].amount if self.symbol in self.position_manager.positions else Decimal("0")
        logger.info(
            "[PARTIAL TP%d] %s %s | %.0f%% closed @ %.2f | PnL=%+.2f | remaining=%s",
            tp_num, pos.side.upper(), self.symbol,
            float(fraction) * 100, exit_price, float(pnl), float(remaining),
        )

        self._closing = False

        msg = (
            f"📊 **Partial TP{tp_num}** | {pos.side.upper()} {self.symbol}\n"
            f"{float(fraction)*100:.0f}% 청산 @ {exit_price:,.2f} | PnL: {float(pnl):+,.2f} USDT ({ret_pct:+.2f}%)\n"
            f"잔여: {float(remaining):.4f}"
        )
        await self.alerter.alert(msg)

    async def _close_position(self, exit_price: float, reason: str) -> None:
        """Close the current position and record PnL."""
        if self._closing or self.symbol not in self.position_manager.positions:
            return

        self._closing = True
        pos = self.position_manager.positions[self.symbol]

        # Execute closing order
        close_side = OrderSide.SELL if pos.side == "long" else OrderSide.BUY
        order = self.order_manager.create_market_order(
            self.symbol, close_side, pos.amount, pos.leverage,
        )
        order.reduce_only = True

        try:
            await self.executor.execute(order)
        except Exception:
            logger.exception("Close order execution failed — local state NOT updated")
            self._closing = False
            return

        entry_price = float(pos.entry_price)
        pnl = self.position_manager.close_position(
            self.symbol, Decimal(str(exit_price))
        )
        self.risk_manager.update_daily_pnl(pnl)
        self.risk_manager.open_positions = max(0, self.risk_manager.open_positions - 1)
        self.dashboard.record_trade(pnl, {
            "side": pos.side,
            "entry": entry_price,
            "exit": exit_price,
            "reason": reason,
        })

        if entry_price > 0:
            ret_pct = ((exit_price / entry_price) - 1) * 100 if pos.side == "long" else ((1 - exit_price / entry_price)) * 100
        else:
            ret_pct = 0.0

        metrics = self.dashboard.get_metrics()
        logger.info(
            "[EXIT #%d] %s %s | entry=%.2f -> exit=%.2f | PnL=%+.2f USDT (%+.2f%%) | reason=%s | cumPnL=%.2f | equity=%.2f | DD=%.2f%%",
            self._trade_count, pos.side.upper(), self.symbol,
            entry_price, exit_price,
            float(pnl), ret_pct, reason,
            float(metrics.total_pnl), metrics.current_equity,
            metrics.current_drawdown * 100,
        )

        self._closing = False

        await self.alerter.alert_exit(
            trade_num=self._trade_count,
            side=pos.side,
            symbol=self.symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=float(pnl),
            ret_pct=ret_pct,
            reason=reason,
            cum_pnl=float(metrics.total_pnl),
            equity=metrics.current_equity,
        )

        # Check daily loss severity and alert/halt if needed
        await self._check_daily_loss_level()

    async def _check_daily_loss_level(self) -> None:
        """Check daily loss level and send alerts or halt trading."""
        try:
            balance = await self.executor.get_balance()
        except Exception:
            return
        level = self.risk_manager.get_daily_loss_level(balance)

        if level == "critical":
            logger.critical(
                "CRITICAL: Daily loss >5%% (PnL=%s, balance=%s) — HALTING trading",
                self.risk_manager.daily_pnl, balance,
            )
            await self.alerter.alert(
                f"일일 손실 5% 초과! PnL: {self.risk_manager.daily_pnl} | 잔고: {balance}\n"
                f"자동 거래 중지 — 수동 재시작 필요",
                level=AlertLevel.CRITICAL,
            )
            self._running = False
            if self._stream:
                self._stream.stop()
        elif level == "warn":
            logger.warning(
                "WARN: Daily loss >3%% (PnL=%s, balance=%s)",
                self.risk_manager.daily_pnl, balance,
            )
            await self.alerter.alert(
                f"일일 손실 3% 초과 주의! PnL: {self.risk_manager.daily_pnl} | 잔고: {balance}",
                level=AlertLevel.WARN,
            )

    async def _check_daily_reset(self) -> None:
        """Reset daily PnL at UTC midnight and send summary."""
        now = datetime.now(timezone.utc)
        if self._last_daily_reset is None or now.date() > self._last_daily_reset.date():
            metrics = self.dashboard.get_metrics()
            await self.alerter.alert(
                f"📊 **Daily Summary** ({self._last_daily_reset.date() if self._last_daily_reset else 'N/A'})\n"
                f"Daily PnL: {metrics.daily_pnl}\n"
                f"Total PnL: {metrics.total_pnl}\n"
                f"Trades: {metrics.total_trades} | WR: {metrics.win_rate:.1%}\n"
                f"Max DD: {metrics.max_drawdown:.2%}"
            )
            self.risk_manager.reset_daily()
            self.dashboard.reset_daily()
            self._last_daily_reset = now
            logger.info("Daily PnL reset at %s", now)

    async def _heartbeat(self) -> None:
        """Periodic heartbeat log."""
        try:
            while self._running:
                await asyncio.sleep(300)  # 5 minutes
                metrics = self.dashboard.get_metrics()
                logger.info(
                    "Heartbeat — PnL: %s | Trades: %d | Positions: %d",
                    metrics.total_pnl, metrics.total_trades,
                    len(self.position_manager.positions),
                )
        except asyncio.CancelledError:
            logger.debug("Heartbeat task cancelled")
            return

    async def shutdown(self) -> None:
        """Graceful shutdown: alert status and close connections."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self._stream:
            self._stream.stop()

        # Report any open positions
        if self.position_manager.positions:
            pos_info = []
            for sym, pos in self.position_manager.positions.items():
                pos_info.append(f"  {sym}: {pos.side} @ {pos.entry_price} (uPnL: {pos.unrealized_pnl})")
            await self.alerter.alert(
                f"⚠️ **TradingEngine Shutting Down**\n"
                f"Open positions:\n" + "\n".join(pos_info)
            )
        else:
            await self.alerter.alert("🛑 **TradingEngine Stopped** — no open positions")

        await self.alerter.close()
        await self.executor.close()
        await self.collector.close()
        logger.info("TradingEngine shutdown complete.")
