"""Discord and Telegram alert sender with structured trade notifications."""

from __future__ import annotations

import asyncio
import logging
import os
import ssl
from enum import Enum
from typing import Optional

import aiohttp
import certifi

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


class Alerter:
    """Sends trading alerts via Discord webhook and Telegram bot."""

    def __init__(
        self,
        discord_webhook: Optional[str] = None,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ) -> None:
        self.discord_webhook = discord_webhook or os.getenv("DISCORD_WEBHOOK_URL")
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a shared aiohttp session."""
        if self._session is None or self._session.closed:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            conn = aiohttp.TCPConnector(ssl=ssl_ctx)
            self._session = aiohttp.ClientSession(connector=conn)
        return self._session

    async def send_discord(self, message: str) -> None:
        """Send alert to Discord webhook."""
        if not self.discord_webhook:
            return
        try:
            session = await self._get_session()
            await session.post(
                self.discord_webhook,
                json={"content": message},
            )
        except Exception:
            logger.exception("Failed to send Discord alert")

    async def send_telegram(self, message: str) -> None:
        """Send alert to Telegram."""
        if not self.telegram_token or not self.telegram_chat_id:
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        try:
            session = await self._get_session()
            await session.post(url, json={
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown",
            })
        except Exception:
            logger.exception("Failed to send Telegram alert")

    async def alert(self, message: str, level: AlertLevel = AlertLevel.INFO) -> None:
        """Send alert to all configured channels in parallel.

        Args:
            message: The alert message text.
            level: Alert severity level.
        """
        if level == AlertLevel.CRITICAL:
            message = f"🚨 **CRITICAL** 🚨\n{message}"
        elif level == AlertLevel.WARN:
            message = f"⚠️ **WARNING**\n{message}"

        await asyncio.gather(
            self.send_discord(message),
            self.send_telegram(message),
            return_exceptions=True,
        )

    async def alert_entry(
        self,
        trade_num: int,
        side: str,
        symbol: str,
        entry_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        size: float,
        confidence: float,
        strategy: str,
        reason: str = "",
        risk_pct: float = 0.0,
    ) -> None:
        """Send a structured position entry alert.

        Args:
            trade_num: Sequential trade number.
            side: "long" or "short".
            symbol: Trading pair symbol.
            entry_price: Entry price.
            stop_loss: Stop-loss price.
            take_profit: Take-profit price.
            size: Position size in asset units.
            confidence: Signal confidence (0-1).
            strategy: Strategy name.
            reason: Human-readable rationale for the entry.
            risk_pct: Risk percentage of account.
        """
        emoji = "📈" if side == "long" else "📉"
        sl_str = f"{stop_loss:,.2f}" if stop_loss else "N/A"
        tp_str = f"{take_profit:,.2f}" if take_profit else "N/A"

        msg = (
            f"{emoji} **{side.upper()} 진입 #{trade_num}** | {symbol} @ {entry_price:,.2f}\n"
        )
        if reason:
            msg += f"근거: {reason}\n"
        msg += (
            f"SL: {sl_str} | TP: {tp_str}"
        )
        if risk_pct > 0:
            msg += f" | Risk: {risk_pct:.1f}%"
        msg += (
            f"\nSize: {size:.4f} | Confidence: {confidence:.2f}\n"
            f"Strategy: {strategy}"
        )
        await self.alert(msg)

    async def alert_exit(
        self,
        trade_num: int,
        side: str,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        ret_pct: float,
        reason: str,
        hold_bars: int = 0,
        cum_pnl: float = 0.0,
        equity: float = 0.0,
    ) -> None:
        """Send a structured position exit alert.

        Args:
            trade_num: Sequential trade number.
            side: "long" or "short".
            symbol: Trading pair symbol.
            entry_price: Original entry price.
            exit_price: Exit price.
            pnl: Realized PnL in USDT.
            ret_pct: Return percentage.
            reason: Exit reason (stop_loss, take_profit, new_signal, etc.).
            hold_bars: Number of bars the position was held.
            cum_pnl: Cumulative PnL.
            equity: Current equity.
        """
        emoji = "✅" if pnl > 0 else "🔴"

        reason_labels = {
            "stop_loss": "SL 도달",
            "take_profit": "TP 도달",
            "new_signal": "반대 시그널 발생",
            "daily_limit": "일일 손실 한도 도달",
            "manual": "수동 종료",
        }
        reason_text = reason_labels.get(reason, reason)

        msg = (
            f"{emoji} **{side.upper()} 종료 #{trade_num}** | PnL: {pnl:+,.2f} USDT ({ret_pct:+.2f}%)\n"
            f"근거: {reason_text}\n"
            f"진입: {entry_price:,.2f} → 종료: {exit_price:,.2f}"
        )
        if hold_bars > 0:
            msg += f" | 보유: {hold_bars}봉"
        if cum_pnl != 0 or equity != 0:
            msg += f"\nCum PnL: {cum_pnl:+,.2f} | Equity: {equity:,.2f}"
        await self.alert(msg)

    async def alert_sl_adjustment(
        self,
        symbol: str,
        side: str,
        old_sl: float,
        new_sl: float,
        reason: str = "",
    ) -> None:
        """Send alert when stop-loss is adjusted.

        Args:
            symbol: Trading pair symbol.
            side: "long" or "short".
            old_sl: Previous stop-loss price.
            new_sl: New stop-loss price.
            reason: Rationale for the adjustment.
        """
        msg = (
            f"🔄 **SL 조정** | {symbol} {side.upper()}\n"
            f"SL: {old_sl:,.2f} → {new_sl:,.2f}"
        )
        if reason:
            msg += f"\n근거: {reason}"
        await self.alert(msg)

    async def close(self) -> None:
        """Close the shared session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
