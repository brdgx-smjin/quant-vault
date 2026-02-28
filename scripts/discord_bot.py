#!/usr/bin/env python3
"""Discord bot for quant-vault remote command & control.

Provides builtin commands for monitoring and controlling the trading bot,
plus free-form Claude Code chat with session continuity.

Messages without '!' prefix are sent to Claude Code as conversation.
Use !new to start a fresh Claude session.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Fix SSL cert for venv (same pattern as live_trading.py)
os.environ.setdefault(
    "SSL_CERT_FILE",
    str(Path(__file__).resolve().parent.parent / ".venv/lib/python3.12/site-packages/certifi/cacert.pem"),
)

import discord
from discord.ext import tasks

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
BOT_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN_QUANT", "")
CHANNEL_ID: int = int(os.getenv("DISCORD_QUANT_CHANNEL_ID", "0"))
ALLOWED_USER_IDS: set[int] = {
    int(uid.strip())
    for uid in os.getenv("DISCORD_ALLOWED_USER_IDS", "").split(",")
    if uid.strip()
}

LOG_DIR = PROJECT_ROOT / "logs"
LIVE_LOG = LOG_DIR / "live.log"
PHASE3_LOG = LOG_DIR / "phase3.log"
CLAUDE_PATH = "/opt/homebrew/bin/claude"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "discord_bot.log"),
    ],
)
logger = logging.getLogger("discord_bot")

# ---------------------------------------------------------------------------
# Discord client
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Concurrency lock for claude calls
_claude_lock = asyncio.Lock()

# Session ID for conversation continuity
_claude_session_id: Optional[str] = None


# ===========================================================================
# Helpers
# ===========================================================================

def _tail(path: Path, n: int = 20) -> str:
    """Read last N lines from a file."""
    if not path.exists():
        return f"`{path.name}` not found."
    lines = path.read_text(errors="replace").splitlines()
    tail_lines = lines[-n:]
    return "\n".join(tail_lines) if tail_lines else "(empty)"


async def _send_long(channel: discord.abc.Messageable, text: str) -> None:
    """Send text, splitting into <=2000-char chunks if necessary."""
    if not text:
        text = "(no output)"
    while text:
        chunk = text[:1990]
        text = text[1990:]
        await channel.send(f"```\n{chunk}\n```")


# ===========================================================================
# Daily team report
# ===========================================================================

ROLE_PATHS: dict[str, list[str]] = {
    "data-engineer": ["src/data/", "scripts/collect_data.py"],
    "strategy-researcher": ["src/indicators/", "src/strategy/", "src/backtest/"],
    "ml-engineer": ["src/ml/", "scripts/train_model.py"],
    "execution-engineer": ["src/execution/", "src/monitoring/", "scripts/live_trading.py"],
}

PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")


def _build_daily_report() -> str:
    """Build the daily team report message."""
    now_kst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    header = f"\U0001f4cb **일일 팀 보고** | {now_kst:%Y-%m-%d %H:%M} KST\n"
    sections: list[str] = []

    for role, paths in ROLE_PATHS.items():
        lines: list[str] = []

        # --- git log (24h) ---
        git_args = ["git", "log", "--since=24 hours ago", "--oneline", "--"]
        git_args.extend(paths)
        try:
            proc = subprocess.run(
                git_args, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=15,
            )
            commits = [l for l in proc.stdout.strip().splitlines() if l]
        except Exception:
            commits = []

        if commits:
            lines.append(f"커밋 (24h): {len(commits)}건")
            for c in commits[:5]:
                lines.append(f"  - {c}")
            if len(commits) > 5:
                lines.append(f"  ... 외 {len(commits) - 5}건")
        else:
            lines.append("커밋 (24h): 0건")

        # --- uncommitted changed files owned by this role ---
        owned: list[str] = []
        try:
            staged = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10,
            ).stdout.strip().splitlines()
            unstaged = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10,
            ).stdout.strip().splitlines()
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10,
            ).stdout.strip().splitlines()
            all_changed = set(staged + unstaged + untracked)
            owned = [f for f in all_changed if any(f.startswith(p) or f == p for p in paths)]
            lines.append(f"변경 파일: {len(owned)}개 (uncommitted)")
        except Exception:
            lines.append("변경 파일: (확인 실패)")

        # --- review.py --role ---
        try:
            proc = subprocess.run(
                [PYTHON, str(PROJECT_ROOT / "scripts" / "review.py"), "--role", role],
                capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=60,
            )
            output = proc.stdout + proc.stderr
            # Parse summary line: PASS=N | WARN=N | FAIL=N
            m = re.search(r"PASS.*?=\s*(\d+).*WARN.*?=\s*(\d+).*FAIL.*?=\s*(\d+)", output)
            if m:
                p, w, f = int(m.group(1)), int(m.group(2)), int(m.group(3))
                review_str = f"리뷰: PASS={p} | WARN={w} | FAIL={f}"
                if f > 0:
                    review_str += " \u26a0\ufe0f"
                lines.append(review_str)
            else:
                lines.append("리뷰: (변경 없음)")
        except Exception:
            lines.append("리뷰: (실행 실패)")

        # --- assemble section ---
        if not commits and not owned:
            sections.append(f"\u2501\u2501\u2501 {role} \u2501\u2501\u2501\n(활동 없음)")
        else:
            sections.append(f"\u2501\u2501\u2501 {role} \u2501\u2501\u2501\n" + "\n".join(lines))

    return header + "\n\n".join(sections)


@tasks.loop(time=datetime.time(hour=0, minute=0, tzinfo=datetime.timezone.utc))
async def daily_team_report() -> None:
    """Send daily team report at 00:00 UTC (09:00 KST)."""
    channel = client.get_channel(CHANNEL_ID)
    if channel is None:
        logger.warning("daily_team_report: channel %s not found", CHANNEL_ID)
        return

    logger.info("Generating daily team report...")
    try:
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, _build_daily_report)
        # Split if needed (Discord 2000 char limit)
        while report:
            chunk = report[:1990]
            report = report[1990:]
            await channel.send(chunk)
        logger.info("Daily team report sent.")
    except Exception:
        logger.exception("Failed to send daily team report")


# ===========================================================================
# Builtin commands
# ===========================================================================

async def cmd_status(channel: discord.abc.Messageable) -> None:
    """Trading bot status from live log."""
    if not LIVE_LOG.exists():
        await channel.send("```\nlive.log not found — bot may not have run yet.\n```")
        return

    lines = LIVE_LOG.read_text(errors="replace").splitlines()
    last_200 = lines[-200:]

    # Extract key info with specific patterns to avoid cross-matching
    heartbeat = None
    position = None
    entry = None
    exit_line = None
    error = None
    mtf_trend = None

    for line in reversed(last_200):
        if not heartbeat and "Heartbeat" in line:
            heartbeat = line.strip()
        if not position and "[POSITION]" in line:
            position = line.strip()
        if not entry and "[ENTRY" in line:
            entry = line.strip()
        if not exit_line and "[EXIT" in line:
            exit_line = line.strip()
        if not mtf_trend and "[MTF]" in line and "trend" in line.lower():
            mtf_trend = line.strip()
        if not error and "[ERROR]" in line:
            error = line.strip()

    # Parse heartbeat for clean display
    hb_display = "not found (bot may be stopped)"
    if heartbeat:
        m = re.search(r"Heartbeat.*PnL:\s*([^\|]+)\|\s*Trades:\s*(\d+)\s*\|\s*Positions:\s*(\d+)", heartbeat)
        if m:
            hb_display = f"PnL: {m.group(1).strip()} | Trades: {m.group(2)} | Positions: {m.group(3)}"
        else:
            hb_display = heartbeat[heartbeat.find("Heartbeat"):][:80]

    parts = ["**Trading Bot Status**"]
    parts.append(f"Heartbeat: `{hb_display}`")

    if mtf_trend:
        # Extract just the MTF info portion
        mtf_part = mtf_trend[mtf_trend.find("[MTF]"):][:80] if "[MTF]" in mtf_trend else mtf_trend[-80:]
        parts.append(f"MTF: `{mtf_part}`")

    if position:
        pos_part = position[position.find("[POSITION]"):][:100] if "[POSITION]" in position else position[-100:]
        parts.append(f"Position: `{pos_part}`")
    else:
        parts.append("Position: No open position")

    if entry:
        entry_part = entry[entry.find("[ENTRY"):][:120] if "[ENTRY" in entry else entry[-120:]
        parts.append(f"Last Entry: `{entry_part}`")

    if exit_line:
        exit_part = exit_line[exit_line.find("[EXIT"):][:120] if "[EXIT" in exit_line else exit_line[-120:]
        parts.append(f"Last Exit: `{exit_part}`")

    if error:
        parts.append(f"Last Error: `{error[-120:]}`")

    # Last log timestamp
    if last_200:
        parts.append(f"Last log: `{last_200[-1][:50]}`")

    await channel.send("\n".join(parts))


async def cmd_logs(channel: discord.abc.Messageable, n: int = 20) -> None:
    """Show recent N lines from live trading log."""
    n = min(n, 50)  # cap at 50
    tail = _tail(LIVE_LOG, n)
    await _send_long(channel, tail)


async def cmd_balance(channel: discord.abc.Messageable) -> None:
    """Fetch Binance demo balance via ccxt."""
    try:
        import ccxt.async_support as ccxt

        exchange = ccxt.binanceusdm({
            "apiKey": os.getenv("BINANCE_API_KEY"),
            "secret": os.getenv("BINANCE_SECRET_KEY"),
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        exchange.enable_demo_trading(True)

        balance = await exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        free = usdt.get("free", 0)
        used = usdt.get("used", 0)
        total = usdt.get("total", 0)

        await channel.send(
            f"**Binance Demo Balance (USDT)**\n"
            f"```\n"
            f"Free:  {float(free):,.2f}\n"
            f"Used:  {float(used):,.2f}\n"
            f"Total: {float(total):,.2f}\n"
            f"```"
        )
        await exchange.close()
    except Exception as e:
        await channel.send(f"```\nBalance fetch error: {e}\n```")


async def cmd_position(channel: discord.abc.Messageable) -> None:
    """Show current positions from Binance demo."""
    try:
        import ccxt.async_support as ccxt

        exchange = ccxt.binanceusdm({
            "apiKey": os.getenv("BINANCE_API_KEY"),
            "secret": os.getenv("BINANCE_SECRET_KEY"),
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        exchange.enable_demo_trading(True)

        positions = await exchange.fetch_positions()
        open_pos = [p for p in positions if abs(float(p.get("contracts", 0))) > 0]

        if not open_pos:
            await channel.send("```\nNo open positions.\n```")
        else:
            parts = ["**Open Positions**"]
            for p in open_pos:
                side = p.get("side", "?")
                symbol = p.get("symbol", "?")
                contracts = p.get("contracts", 0)
                entry = p.get("entryPrice", 0)
                unrealized = p.get("unrealizedPnl", 0)
                parts.append(
                    f"```\n"
                    f"{symbol} {side.upper()}\n"
                    f"  Contracts:     {contracts}\n"
                    f"  Entry Price:   {float(entry):,.2f}\n"
                    f"  Unrealized PnL: {float(unrealized):+,.2f}\n"
                    f"```"
                )
            await channel.send("\n".join(parts))
        await exchange.close()
    except Exception as e:
        await channel.send(f"```\nPosition fetch error: {e}\n```")


async def cmd_signal(channel: discord.abc.Messageable) -> None:
    """Show recent signal info from live log."""
    if not LIVE_LOG.exists():
        await channel.send("```\nlive.log not found.\n```")
        return

    lines = LIVE_LOG.read_text(errors="replace").splitlines()
    signal_lines = [l for l in lines if "signal" in l.lower() or "LONG" in l or "SHORT" in l]
    recent = signal_lines[-10:] if signal_lines else ["No signal entries found."]
    await _send_long(channel, "\n".join(recent))


async def cmd_backtest(channel: discord.abc.Messageable) -> None:
    """Show Phase 3 walk-forward summary."""
    if not PHASE3_LOG.exists():
        await channel.send("```\nphase3.log not found.\n```")
        return

    lines = PHASE3_LOG.read_text(errors="replace").splitlines()

    # Extract summary sections
    summary_lines: list[str] = []
    capture = False
    for line in lines:
        if "Walk-Forward Summary" in line or "Robustness Score" in line:
            capture = True
        if capture:
            summary_lines.append(line)
            if len(summary_lines) > 8:
                capture = False
        # Also grab Fib WF summary line
        if "Fib WF:" in line:
            summary_lines.append(line)

    if not summary_lines:
        # Fallback: show last 20 lines
        summary_lines = lines[-20:]

    await _send_long(channel, "\n".join(summary_lines))


async def cmd_team(channel: discord.abc.Messageable) -> None:
    """Show agent team status from tmux."""
    agents = {
        "data": "Data Engineer",
        "backtest": "Strategy Researcher",
        "live": "Execution Engineer",
    }
    parts = ["**Agent Team Status**\n```"]

    for window, label in agents.items():
        try:
            # Get pane PID
            r = subprocess.run(
                ["tmux", "list-panes", "-t", f"quant:{window}", "-F", "#{pane_pid}"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode != 0:
                parts.append(f"  {label:<22} (no window)")
                continue

            pane_pid = r.stdout.strip().split("\n")[0]

            # Check if claude is running under this pane
            r2 = subprocess.run(
                ["pgrep", "-P", pane_pid, "-a"],
                capture_output=True, text=True, timeout=5,
            )
            claude_procs = [l for l in r2.stdout.splitlines() if "claude" in l]

            if claude_procs:
                # Get last 3 lines of pane output for context
                r3 = subprocess.run(
                    ["tmux", "capture-pane", "-t", f"quant:{window}", "-p"],
                    capture_output=True, text=True, timeout=5,
                )
                last_lines = [l.strip() for l in r3.stdout.splitlines() if l.strip()]
                preview = last_lines[-1][:60] if last_lines else ""
                parts.append(f"  {label:<22} RUNNING")
                if preview:
                    parts.append(f"    > {preview}")
            else:
                parts.append(f"  {label:<22} idle")

        except Exception:
            parts.append(f"  {label:<22} (error)")

    parts.append("```")
    await channel.send("\n".join(parts))


async def cmd_start(channel: discord.abc.Messageable) -> None:
    """Start live trading bot via tmux."""
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", "quant:live",
             f"cd {PROJECT_ROOT} && .venv/bin/python scripts/live_trading.py", "Enter"],
            check=True,
            capture_output=True,
        )
        await channel.send("Trading bot start command sent to `tmux quant:live`.")
    except subprocess.CalledProcessError as e:
        await channel.send(f"```\nFailed to send start command: {e}\n```")
    except FileNotFoundError:
        await channel.send("```\ntmux not found.\n```")


async def cmd_stop(channel: discord.abc.Messageable) -> None:
    """Stop live trading bot via tmux C-c."""
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", "quant:live", "C-c", ""],
            check=True,
            capture_output=True,
        )
        await channel.send("Stop signal (Ctrl-C) sent to `tmux quant:live`.")
    except subprocess.CalledProcessError as e:
        await channel.send(f"```\nFailed to send stop command: {e}\n```")
    except FileNotFoundError:
        await channel.send("```\ntmux not found.\n```")


# ===========================================================================
# Claude passthrough
# ===========================================================================

def _run_claude_sync(
    prompt: str, session_id: Optional[str], new_session: bool,
) -> tuple[str, str, str]:
    """Run Claude Code subprocess synchronously (called from thread).

    Returns:
        (result_text, session_id or "", stderr)
    """
    cmd = [
        CLAUDE_PATH, "-p", prompt,
        "--output-format", "json",
        "--model", os.getenv("DISCORD_CLAUDE_MODEL", "opus"),
        "--max-turns", os.getenv("DISCORD_CLAUDE_MAX_TURNS", "15"),
        "--dangerously-skip-permissions",
    ]

    if session_id and not new_session:
        cmd.extend(["--resume", session_id])

    # Remove Claude env vars to avoid nested-session detection
    _strip_keys = {"CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"}
    env = {k: v for k, v in os.environ.items() if k not in _strip_keys}

    try:
        r = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
            timeout=600,  # 10 min
        )
    except subprocess.TimeoutExpired:
        return ("", "", "Timeout (10 min limit)")

    if r.returncode != 0:
        return ("", "", r.stderr.strip() or f"Exit code {r.returncode}")

    # Parse JSON result
    stdout = r.stdout.strip()
    if not stdout:
        return ("", "", r.stderr.strip() or "Empty output")

    try:
        data = json.loads(stdout)
        result_text = data.get("result", "")
        new_session_id = data.get("session_id", "")
        return (result_text, new_session_id, "")
    except json.JSONDecodeError:
        return (stdout[:2000], "", "")


async def _run_claude_chat(
    channel: discord.abc.Messageable, prompt: str, *, new_session: bool = False,
) -> None:
    """Run Claude Code and return the response.

    Spawns Claude CLI in a thread to avoid asyncio pipe issues.
    Shows a periodic progress indicator in Discord while waiting.

    Args:
        channel: Discord channel to respond in.
        prompt: User message.
        new_session: Force a new session.
    """
    global _claude_session_id

    if _claude_lock.locked():
        await channel.send("> Another request is processing. Please wait.")
        return

    async with _claude_lock:
        status_msg = await channel.send("> Thinking...")

        try:
            logger.info("Running claude for: %s", prompt[:80])

            # Run subprocess in thread pool (avoids asyncio pipe issues)
            loop = asyncio.get_event_loop()
            start_time = loop.time()

            future = loop.run_in_executor(
                None, _run_claude_sync, prompt, _claude_session_id, new_session,
            )

            # Update status in Discord while waiting
            while not future.done():
                await asyncio.sleep(4)
                elapsed_s = int(loop.time() - start_time)
                try:
                    await status_msg.edit(content=f"> Processing... ({elapsed_s}s)")
                except discord.HTTPException:
                    pass

            result_text, sid, err = future.result()

            if sid:
                _claude_session_id = sid
                logger.info("Claude session: %s", _claude_session_id)

            if err:
                logger.error("Claude CLI error: %s", err)
                await status_msg.edit(content=f"> Error: {err}")
                return

            if not result_text:
                result_text = "(no output)"

            await status_msg.delete()
            await _send_long(channel, result_text)

        except FileNotFoundError:
            await status_msg.edit(content="> `claude` CLI not found on this machine.")
        except Exception as e:
            logger.exception("Claude chat error")
            await status_msg.edit(content=f"> Error: {e}")


# ===========================================================================
# Event handlers
# ===========================================================================

@client.event
async def on_ready() -> None:
    logger.info("Discord bot online: %s (ID: %s)", client.user, client.user.id)
    logger.info("Listening on channel: %s", CHANNEL_ID)
    logger.info("Allowed users: %s", ALLOWED_USER_IDS)
    if not daily_team_report.is_running():
        daily_team_report.start()
        logger.info("daily_team_report started (00:00 UTC / 09:00 KST)")


@client.event
async def on_message(message: discord.Message) -> None:
    # Ignore own messages
    if message.author == client.user:
        return

    # Security: channel filter
    if message.channel.id != CHANNEL_ID:
        return

    # Security: user filter
    if message.author.id not in ALLOWED_USER_IDS:
        return

    content = message.content.strip()
    if not content:
        return

    # '!' commands — builtin shortcuts
    if content.startswith("!"):
        parts = content.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        logger.info("Command from %s: %s", message.author, content[:100])

        if cmd == "!status":
            await cmd_status(message.channel)
        elif cmd == "!logs":
            n = 20
            if arg.isdigit():
                n = int(arg)
            await cmd_logs(message.channel, n)
        elif cmd == "!balance":
            await cmd_balance(message.channel)
        elif cmd == "!position":
            await cmd_position(message.channel)
        elif cmd == "!signal":
            await cmd_signal(message.channel)
        elif cmd == "!backtest":
            await cmd_backtest(message.channel)
        elif cmd == "!team":
            await cmd_team(message.channel)
        elif cmd == "!start":
            await cmd_start(message.channel)
        elif cmd == "!stop":
            await cmd_stop(message.channel)
        elif cmd == "!report":
            await message.channel.send("> Generating team report...")
            try:
                loop = asyncio.get_event_loop()
                report = await loop.run_in_executor(None, _build_daily_report)
                while report:
                    chunk = report[:1990]
                    report = report[1990:]
                    await message.channel.send(chunk)
            except Exception as e:
                await message.channel.send(f"```\nReport error: {e}\n```")
        elif cmd == "!new":
            global _claude_session_id
            _claude_session_id = None
            await message.channel.send("> New Claude session started.")
        elif cmd == "!help":
            await message.channel.send(
                "**Quant Vault Bot**\n"
                "일반 메시지 → Claude Code 대화 (세션 유지)\n"
                "```\n"
                "!new       — 새 Claude 세션 시작\n"
                "!status    — 트레이딩 봇 상태\n"
                "!logs [N]  — 최근 N줄 로그 (기본 20, 최대 50)\n"
                "!balance   — Binance 데모 잔고\n"
                "!position  — 현재 포지션\n"
                "!signal    — 최근 시그널\n"
                "!backtest  — Walk-Forward 결과\n"
                "!team      — 에이전트 팀 상태\n"
                "!report    — 팀원 일일 보고 (매일 09:00 KST 자동)\n"
                "!start     — 봇 시작 (tmux)\n"
                "!stop      — 봇 중지 (tmux Ctrl-C)\n"
                "!help      — 이 메시지\n"
                "```"
            )
        return

    # Free-form message → Claude Code chat with session continuity
    logger.info("Chat from %s: %s", message.author, content[:100])
    await _run_claude_chat(message.channel, content)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    if not BOT_TOKEN:
        logger.error("DISCORD_BOT_TOKEN_QUANT not set in .env")
        sys.exit(1)
    if CHANNEL_ID == 0:
        logger.error("DISCORD_QUANT_CHANNEL_ID not set in .env")
        sys.exit(1)
    if not ALLOWED_USER_IDS:
        logger.error("DISCORD_ALLOWED_USER_IDS not set in .env")
        sys.exit(1)

    logger.info("Starting quant-vault Discord bot...")
    client.run(BOT_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
