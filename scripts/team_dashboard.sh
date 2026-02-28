#!/usr/bin/env bash
# Team member dashboard — shows per-role review status
# Usage: scripts/team_dashboard.sh [role]
# Roles: strategy-researcher, execution-engineer, data-engineer, ml-engineer, all

set -euo pipefail
cd "$(dirname "$0")/.."

ROLE="${1:-all}"
PYTHON=".venv/bin/python"

BOLD="\033[1m"
RESET="\033[0m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
CYAN="\033[36m"
DIM="\033[2m"

clear

# Header
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════╗${RESET}"
if [ "$ROLE" = "all" ]; then
    echo -e "${BOLD}${CYAN}║   Tech-Lead Review Dashboard             ║${RESET}"
else
    printf "${BOLD}${CYAN}║   %-38s ║${RESET}\n" "$ROLE"
fi
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════╝${RESET}"
echo -e "${DIM}$(date '+%Y-%m-%d %H:%M:%S')${RESET}"
echo ""

if [ "$ROLE" = "all" ]; then
    # Tech-lead overview: run full review
    $PYTHON scripts/review.py 2>&1
else
    # Per-role view
    echo -e "${BOLD}[ Changed Files — $ROLE ]${RESET}"
    echo ""

    # Show files owned by this role
    case "$ROLE" in
        strategy-researcher)
            PATHS="src/indicators/ src/strategy/ src/backtest/"
            ;;
        execution-engineer)
            PATHS="src/execution/ src/monitoring/ scripts/live_trading.py"
            ;;
        data-engineer)
            PATHS="src/data/ scripts/collect_data.py"
            ;;
        ml-engineer)
            PATHS="src/ml/ scripts/train_model.py"
            ;;
    esac

    # Show changed files for this role
    CHANGED=$(git diff --name-only 2>/dev/null; git diff --cached --name-only 2>/dev/null; git ls-files --others --exclude-standard 2>/dev/null)
    COUNT=0
    for prefix in $PATHS; do
        MATCHED=$(echo "$CHANGED" | grep "^${prefix}" 2>/dev/null | sort -u || true)
        if [ -n "$MATCHED" ]; then
            while IFS= read -r f; do
                echo -e "  ${GREEN}M${RESET} $f"
                COUNT=$((COUNT + 1))
            done <<< "$MATCHED"
        fi
    done

    if [ $COUNT -eq 0 ]; then
        echo -e "  ${DIM}(no changes)${RESET}"
    else
        echo ""
        echo -e "${DIM}$COUNT file(s) changed${RESET}"
    fi

    echo ""
    echo -e "${BOLD}[ Review Status ]${RESET}"
    echo ""
    $PYTHON scripts/review.py --role "$ROLE" 2>&1
fi
