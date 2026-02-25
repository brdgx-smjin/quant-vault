#!/bin/bash
# scripts/tmux_setup.sh — Quant Trading tmux 환경 세팅

SESSION="quant"
PROJECT_DIR="$HOME/quant/quant-vault"

# 기존 세션이 있으면 접속
tmux has-session -t $SESSION 2>/dev/null && {
    echo "Session '$SESSION' already exists. Attaching..."
    tmux attach -t $SESSION
    exit 0
}

# 새 세션 생성 — 윈도우 0: Claude Code
tmux new-session -d -s $SESSION -n "claude" -c $PROJECT_DIR

# 윈도우 1: 데이터 모니터링
tmux new-window -t $SESSION -n "data" -c $PROJECT_DIR
tmux split-window -h -t $SESSION:data -c $PROJECT_DIR
# 왼쪽: 데이터 수집 로그
# 오른쪽: Redis 모니터 또는 DB 쿼리

# 윈도우 2: 백테스트/분석
tmux new-window -t $SESSION -n "backtest" -c $PROJECT_DIR
tmux split-window -h -t $SESSION:backtest -c $PROJECT_DIR
# 왼쪽: 백테스트 실행
# 오른쪽: 결과 확인

# 윈도우 3: 라이브 트레이딩
tmux new-window -t $SESSION -n "live" -c $PROJECT_DIR
tmux split-window -h -t $SESSION:live -c $PROJECT_DIR
tmux split-window -v -t $SESSION:live.1 -c $PROJECT_DIR
# 레이아웃:
# ┌─────────────┬──────────────┐
# │             │   포지션 모니터 │
# │  트레이딩 봇 ├──────────────┤
# │             │   알림 로그    │
# └─────────────┴──────────────┘

# 윈도우 0 (Claude Code)으로 돌아감
tmux select-window -t $SESSION:claude

# Python 가상환경 활성화 명령 전송
for i in 0 1 2 3; do
    for pane in $(tmux list-panes -t $SESSION:$i -F '#{pane_index}'); do
        tmux send-keys -t $SESSION:$i.$pane "cd $PROJECT_DIR && source .venv/bin/activate" Enter
    done
done

echo "tmux session '$SESSION' created with 4 windows."
echo "  0: claude   — Claude Code (메인)"
echo "  1: data     — 데이터 수집/모니터링"
echo "  2: backtest — 백테스트/분석"
echo "  3: live     — 라이브 트레이딩"
echo ""
echo "Attaching..."

# 접속
tmux attach -t $SESSION
