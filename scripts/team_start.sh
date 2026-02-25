#!/bin/bash
# scripts/team_start.sh — Claude Code Agent Team 자동 시작
#
# 사용법:
#   ./scripts/team_start.sh              # 전체 팀 시작
#   ./scripts/team_start.sh backtest     # backtest 에이전트만 시작
#   ./scripts/team_start.sh stop         # 전체 팀 중지
#   ./scripts/team_start.sh status       # 상태 확인

set -euo pipefail

SESSION="quant"
PROJECT_DIR="$HOME/quant/quant-vault"
CLAUDE="/opt/homebrew/bin/claude"
MODEL="${CLAUDE_TEAM_MODEL:-opus}"
PROMPT_DIR="$PROJECT_DIR/.claude/team-prompts"

mkdir -p "$PROMPT_DIR"

# Prompt files are managed separately in .claude/team-prompts/*.md
# Do NOT overwrite them here — edit them directly to update agent instructions.

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

start_agent() {
    local window="$1"
    local prompt_file="$PROMPT_DIR/$window.md"
    local label="$2"

    if [ ! -f "$prompt_file" ]; then
        echo "Error: prompt file not found: $prompt_file"
        return 1
    fi

    # Check if window exists
    if ! tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -qx "$window"; then
        echo "  Creating window: $window"
        tmux new-window -t "$SESSION" -n "$window" -c "$PROJECT_DIR"
        sleep 1
    fi

    # Kill any existing claude process in this window
    tmux send-keys -t "$SESSION:$window" C-c "" 2>/dev/null || true
    sleep 0.5
    tmux send-keys -t "$SESSION:$window" C-c "" 2>/dev/null || true
    sleep 0.5

    echo "  Starting $label in $SESSION:$window (loop mode)..."

    # Unset CLAUDECODE env vars and run claude in a loop (restarts after completion)
    tmux send-keys -t "$SESSION:$window" \
        "unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT && cd $PROJECT_DIR && while true; do echo '>>> [$window] Starting run at '\$(date); $CLAUDE -p \"\$(cat $prompt_file)\" --model $MODEL --max-turns 50 --dangerously-skip-permissions; echo '>>> [$window] Run completed at '\$(date)'. Restarting in 30s...'; sleep 30; done" \
        Enter
}

stop_all() {
    echo "Stopping all agents..."
    for window in backtest data live; do
        tmux send-keys -t "$SESSION:$window" C-c "" 2>/dev/null || true
        sleep 0.3
        tmux send-keys -t "$SESSION:$window" C-c "" 2>/dev/null || true
    done
    echo "Done."
}

show_status() {
    echo "=== Agent Team Status ==="
    echo ""
    for window in claude data backtest live discord; do
        if ! tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -qx "$window"; then
            printf "  %-12s (no window)\n" "[$window]"
            continue
        fi
        local pane_pid
        pane_pid=$(tmux list-panes -t "$SESSION:$window" -F '#{pane_pid}' 2>/dev/null | head -1)
        local claude_proc
        claude_proc=$(ps -o pid,command -p $(pgrep -P "$pane_pid" 2>/dev/null || echo 0) 2>/dev/null | grep claude | head -1 || echo "")
        if [ -n "$claude_proc" ]; then
            printf "  %-12s RUNNING  %s\n" "[$window]" "$(echo "$claude_proc" | awk '{print $2, $3, $4}')"
        else
            printf "  %-12s idle\n" "[$window]"
        fi
    done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Error: tmux session '$SESSION' not found."
    echo "Run: tmux new-session -d -s $SESSION -n claude -c $PROJECT_DIR"
    exit 1
fi

case "${1:-all}" in
    backtest)
        start_agent "backtest" "Strategy Researcher"
        ;;
    data)
        start_agent "data" "Data Engineer"
        ;;
    live|execution)
        start_agent "live" "Execution Engineer"
        ;;
    all)
        echo "Starting agent team (model: $MODEL)..."
        echo ""
        start_agent "backtest" "Strategy Researcher"
        sleep 2
        start_agent "data" "Data Engineer"
        sleep 2
        start_agent "live" "Execution Engineer"
        echo ""
        echo "All agents started."
        echo ""
        echo "  Ctrl+b 0  → claude (lead)"
        echo "  Ctrl+b 1  → data agent"
        echo "  Ctrl+b 2  → backtest agent"
        echo "  Ctrl+b 3  → live agent"
        ;;
    stop)
        stop_all
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 [all|backtest|data|live|stop|status]"
        exit 1
        ;;
esac
