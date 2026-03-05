#!/bin/bash
# Live trading monitor — shows balance, PnL, positions, recent trades
LOG="logs/live.log"
POS_FILE="data/position_state.json"
TRADE_FILE="data/trade_history.json"

while true; do
    clear
    echo "============================================"
    echo "  QUANT VAULT — Live Trading Monitor"
    echo "  $(date '+%Y-%m-%d %H:%M:%S KST')"
    echo "============================================"
    echo ""

    # Latest heartbeat
    HB=$(grep 'Heartbeat' "$LOG" 2>/dev/null | tail -1)
    if [ -n "$HB" ]; then
        echo "  $HB" | sed 's/.*Heartbeat/Heartbeat/'
    else
        echo "  Heartbeat: (waiting...)"
    fi
    echo ""

    # Current position from position_state.json
    echo "--- Current Position ---"
    if [ -s "$POS_FILE" ] && [ "$(cat "$POS_FILE")" != "{}" ]; then
        python3 -c "
import json
with open('$POS_FILE') as f:
    pos = json.load(f)
if not pos:
    print('  No open position')
else:
    for k, v in pos.items():
        side = v.get('side','?').upper()
        entry = float(v.get('entry_price',0))
        sl = float(v.get('stop_loss',0))
        tp = float(v.get('take_profit',0))
        amt = float(v.get('amount',0))
        comp = v.get('component_id','?')
        print(f'  {side} @ {entry:,.2f} | SL={sl:,.2f} TP={tp:,.2f}')
        print(f'  size={amt:.4f} | component={comp}')
" 2>/dev/null
    else
        echo "  No open position"
    fi

    # Latest position update from log
    LAST_POS=$(grep '\[POSITION\]' "$LOG" 2>/dev/null | tail -1)
    if [ -n "$LAST_POS" ]; then
        echo ""
        echo "  Last update:"
        echo "  $LAST_POS" | sed 's/.*\[POSITION\]/  [POS]/'
    fi
    echo ""

    # Account info from latest entry/exit
    echo "--- Account ---"
    LAST_BAL=$(grep 'bal=' "$LOG" 2>/dev/null | tail -1 | sed 's/.*bal=/bal=/' | sed 's/ .*//')
    LAST_CUM=$(grep 'cumPnL=' "$LOG" 2>/dev/null | tail -1 | sed 's/.*cumPnL=/cumPnL=/' | sed 's/ .*//')
    LAST_EQ=$(grep 'equity=' "$LOG" 2>/dev/null | tail -1 | sed 's/.*equity=/equity=/' | sed 's/ .*//')
    LAST_DD=$(grep 'DD=' "$LOG" 2>/dev/null | tail -1 | sed 's/.*DD=/DD=/' | sed 's/ .*//')
    echo "  ${LAST_BAL:-bal=?}  |  ${LAST_CUM:-cumPnL=?}"
    echo "  ${LAST_EQ:-equity=?}  |  ${LAST_DD:-DD=?}"
    echo ""

    # Recent trades (last 5)
    echo "--- Recent Trades (last 5) ---"
    grep '\[EXIT' "$LOG" 2>/dev/null | tail -5 | while read -r line; do
        echo "$line" | sed 's/.*\[EXIT/  [EXIT/' | sed 's/| cumPnL.*//'
    done

    TOTAL_TRADES=$(grep -c '\[EXIT' "$LOG" 2>/dev/null)
    WINS=$(grep '\[EXIT' "$LOG" 2>/dev/null | grep -c 'PnL=+')
    if [ "$TOTAL_TRADES" -gt 0 ]; then
        WR=$((WINS * 100 / TOTAL_TRADES))
        echo ""
        echo "  Total: ${TOTAL_TRADES} trades | Wins: ${WINS} | WR: ${WR}%"
    fi
    echo ""

    # MTF trend
    LAST_MTF=$(grep '\[MTF\] 1h trend' "$LOG" 2>/dev/null | tail -1)
    if [ -n "$LAST_MTF" ]; then
        echo "--- MTF Filter ---"
        echo "$LAST_MTF" | sed 's/.*\[MTF\]/  [MTF]/'
        echo ""
    fi

    sleep 30
done
