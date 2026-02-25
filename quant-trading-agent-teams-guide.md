# AI Quant Trading System — Claude Code Agent Teams 구축 가이드

## ⚠️ 주의사항
> 이 시스템은 교육 및 연구 목적입니다. 실제 자금 투자 시 손실 가능성이 있습니다.
> 반드시 Binance Testnet에서 충분한 검증 후 실전에 적용하세요.
> 선물거래는 원금 이상의 손실이 발생할 수 있습니다.

---

## 1. 전체 아키텍처 개요

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code Agent Teams                │
│                                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Lead     │  │ Data     │  │ Strategy │  │ Execution│ │
│  │ Agent    │←→│ Engineer │  │ Researcher│  │ Engineer │ │
│  │ (조율자)  │  │ (데이터)  │  │ (전략연구)│  │ (실행엔진)│ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│       │              │              │              │       │
│       └──────────────┴──────────────┴──────────────┘       │
│                          │                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐      ┌─────▼─────┐     ┌─────▼─────┐
   │ Binance │      │ PostgreSQL│     │   Redis   │
   │ API     │      │ /InfluxDB │     │ (realtime)│
   └─────────┘      └───────────┘     └───────────┘
```

---

## 2. 기술 스택

| 분류 | 도구 | 용도 |
|------|------|------|
| **거래소 연결** | ccxt | Binance 선물 API 통합 |
| **데이터 처리** | pandas, numpy | OHLCV 데이터 처리 |
| **기술적 지표** | pandas-ta, TA-Lib | RSI, MACD, 볼린저밴드 등 |
| **패턴 인식** | 커스텀 모듈 | 하모닉패턴, 피보나치 되돌림 |
| **백테스팅** | vectorbt | 고속 벡터화 백테스팅 |
| **ML/DL** | scikit-learn, pytorch | 패턴 학습, 시그널 예측 |
| **DB** | PostgreSQL + TimescaleDB | 시계열 데이터 저장 |
| **캐시/실시간** | Redis | 실시간 데이터, 상태 관리 |
| **스케줄링** | APScheduler / cron | 주기적 데이터 수집, 리밸런싱 |
| **모니터링** | Grafana + Prometheus | 성과 대시보드 |
| **알림** | Discord/Telegram Bot | 매매 신호, 체결 알림 |

---

## 3. 프로젝트 디렉토리 구조

```
quant-trading/
├── CLAUDE.md                    # Claude Code 프로젝트 컨텍스트
├── .env                         # API 키 (절대 git에 올리지 않음)
├── .env.example                 # 환경변수 템플릿
├── pyproject.toml               # Python 프로젝트 설정
├── requirements.txt
│
├── config/
│   ├── settings.py              # 전역 설정
│   ├── symbols.yaml             # 거래 대상 심볼 설정
│   └── risk.yaml                # 리스크 관리 파라미터
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                    # 📊 데이터 수집/처리 모듈
│   │   ├── __init__.py
│   │   ├── collector.py         # Binance OHLCV 수집기
│   │   ├── preprocessor.py      # 데이터 전처리
│   │   ├── storage.py           # DB 저장/조회
│   │   └── stream.py            # WebSocket 실시간 스트림
│   │
│   ├── indicators/              # 📈 기술적 지표 모듈
│   │   ├── __init__.py
│   │   ├── basic.py             # MA, RSI, MACD, BB 등
│   │   ├── fibonacci.py         # 피보나치 되돌림/확장
│   │   ├── harmonic.py          # 하모닉 패턴 (XABCD)
│   │   ├── ichimoku.py          # 일목균형표
│   │   ├── volume_profile.py    # 볼륨 프로파일
│   │   └── patterns.py          # 캔들스틱 패턴 인식
│   │
│   ├── strategy/                # 🎯 전략 모듈
│   │   ├── __init__.py
│   │   ├── base.py              # 전략 베이스 클래스
│   │   ├── fibonacci_retracement.py
│   │   ├── harmonic_strategy.py
│   │   ├── multi_timeframe.py   # 멀티 타임프레임 분석
│   │   ├── ensemble.py          # 앙상블 전략 (여러 전략 조합)
│   │   └── ml_strategy.py       # ML 기반 전략
│   │
│   ├── backtest/                # 🔬 백테스팅 모듈
│   │   ├── __init__.py
│   │   ├── engine.py            # vectorbt 기반 백테스트 엔진
│   │   ├── optimizer.py         # 파라미터 최적화
│   │   ├── walk_forward.py      # Walk-forward 분석
│   │   └── report.py            # 백테스트 리포트 생성
│   │
│   ├── ml/                      # 🤖 머신러닝 모듈
│   │   ├── __init__.py
│   │   ├── features.py          # 피처 엔지니어링
│   │   ├── models.py            # 모델 정의 (LSTM, XGBoost 등)
│   │   ├── trainer.py           # 모델 학습 파이프라인
│   │   ├── predictor.py         # 실시간 예측
│   │   └── evaluation.py        # 모델 성능 평가
│   │
│   ├── execution/               # ⚡ 주문 실행 모듈
│   │   ├── __init__.py
│   │   ├── order_manager.py     # 주문 생성/관리
│   │   ├── position_manager.py  # 포지션 관리
│   │   ├── risk_manager.py      # 리스크 관리 (손절/익절/최대손실)
│   │   └── binance_executor.py  # Binance 선물 실행기
│   │
│   ├── monitoring/              # 📱 모니터링/알림 모듈
│   │   ├── __init__.py
│   │   ├── dashboard.py         # 성과 대시보드
│   │   ├── alerter.py           # Discord/Telegram 알림
│   │   └── logger.py            # 트레이딩 로그
│   │
│   └── utils/                   # 🔧 유틸리티
│       ├── __init__.py
│       ├── time_utils.py
│       └── math_utils.py
│
├── tests/                       # 테스트
│   ├── test_data/
│   ├── test_indicators/
│   ├── test_strategy/
│   ├── test_backtest/
│   └── test_execution/
│
├── notebooks/                   # 분석 노트북
│   ├── 01_data_exploration.ipynb
│   ├── 02_indicator_analysis.ipynb
│   ├── 03_backtest_results.ipynb
│   └── 04_ml_training.ipynb
│
├── scripts/                     # 실행 스크립트
│   ├── collect_data.py          # 히스토리컬 데이터 수집
│   ├── run_backtest.py          # 백테스트 실행
│   ├── train_model.py           # 모델 학습
│   ├── live_trading.py          # 실시간 매매
│   └── tmux_setup.sh            # tmux 환경 세팅
│
├── models/                      # 학습된 모델 저장
│   └── .gitkeep
│
├── data/                        # 로컬 데이터 캐시
│   ├── raw/
│   ├── processed/
│   └── .gitkeep
│
└── logs/                        # 로그
    └── .gitkeep
```

---

## 4. 환경 구축 — 단계별 가이드

### Step 1: 기본 환경 설치

```bash
# 1. 프로젝트 디렉토리 생성
mkdir -p ~/quant-trading && cd ~/quant-trading
git init

# 2. Python 가상환경 (pyenv + 3.11 권장)
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate

# 3. 핵심 패키지 설치
pip install ccxt pandas numpy pandas-ta scikit-learn
pip install vectorbt[full]
pip install torch torchvision          # PyTorch (GPU 있으면 CUDA 버전)
pip install xgboost lightgbm
pip install python-dotenv pyyaml
pip install redis psycopg2-binary      # DB
pip install apscheduler
pip install discord.py python-telegram-bot  # 알림
pip install pytest pytest-asyncio      # 테스트
pip install jupyter plotly             # 분석

# 4. TA-Lib 설치 (시스템 레벨 C 라이브러리 필요)
# macOS
brew install ta-lib
pip install TA-Lib

# Ubuntu
sudo apt install -y build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib
./configure --prefix=/usr && make && sudo make install
pip install TA-Lib

# 5. ccxt 성능 최적화
pip install orjson coincurve
```

### Step 2: 환경 변수 설정

```bash
# .env 파일 생성
cat > .env << 'EOF'
# Binance API (Testnet으로 시작!)
BINANCE_API_KEY=your_testnet_api_key
BINANCE_SECRET_KEY=your_testnet_secret_key
BINANCE_TESTNET=true

# Database
DATABASE_URL=postgresql://quant:password@localhost:5432/quant_trading

# Redis
REDIS_URL=redis://localhost:6379/0

# Discord 알림 (선택)
DISCORD_WEBHOOK_URL=your_webhook_url

# Telegram 알림 (선택)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
EOF

# .gitignore에 추가
echo ".env" >> .gitignore
echo "models/*.pt" >> .gitignore
echo "data/raw/" >> .gitignore
```

### Step 3: Claude Code 설정

```bash
# Claude Code Agent Teams 활성화
cat > ~/.claude/settings.json << 'EOF'
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "permissions": {
    "allow": [
      "Read(**)",
      "Write(~/quant-trading/**)",
      "Bash(python *)",
      "Bash(pip install *)",
      "Bash(pytest *)",
      "Bash(git *)"
    ]
  }
}
EOF
```

### Step 4: CLAUDE.md 작성 (핵심!)

```markdown
# CLAUDE.md — AI Quant Trading System

## 프로젝트 개요
Binance BTC/USDT 선물거래를 위한 AI 퀀트 트레이딩 시스템.
하모닉 패턴, 피보나치 되돌림 등 기술적 분석 + ML 기반 시그널 생성.

## 기술 스택
- Python 3.11, ccxt (Binance), pandas-ta, vectorbt
- ML: PyTorch, XGBoost, scikit-learn
- DB: PostgreSQL + TimescaleDB, Redis
- 백테스팅: vectorbt

## 프로젝트 구조
- src/data/       → 데이터 수집/전처리
- src/indicators/ → 기술적 지표 (하모닉, 피보나치 등)
- src/strategy/   → 매매 전략 로직
- src/backtest/   → vectorbt 기반 백테스팅
- src/ml/         → ML 모델 학습/예측
- src/execution/  → Binance 주문 실행
- src/monitoring/ → 알림, 대시보드

## 코딩 컨벤션
- Type hints 필수
- docstring은 Google style
- 모든 모듈에 단위 테스트 작성
- async/await 패턴 사용 (ccxt async)
- 금액 계산은 Decimal 사용

## 중요 규칙
- .env의 API 키를 절대 코드에 하드코딩하지 않음
- 모든 주문은 risk_manager를 통과해야 함
- 최대 손실 한도: 계좌 잔고의 2% per trade
- Testnet 환경에서 먼저 검증

## tmux 환경
- 세션 1개, 윈도우 4개 사용
- 윈도우 0: Claude Code (메인)
- 윈도우 1: 데이터 수집/모니터링
- 윈도우 2: 백테스트/분석
- 윈도우 3: 라이브 트레이딩 로그

## 팀 작업 시 파일 소유권
- data-engineer  → src/data/, scripts/collect_data.py
- strategy-researcher → src/indicators/, src/strategy/, src/backtest/
- ml-engineer → src/ml/, scripts/train_model.py
- execution-engineer → src/execution/, src/monitoring/, scripts/live_trading.py
→ 팀원 간 같은 파일 수정 금지! 인터페이스로 소통.
```

---

## 5. Agent Teams 구성

### 팀원 역할 정의

| 팀원 | 역할 | 담당 모듈 |
|------|------|-----------|
| **Lead (리드)** | 전체 조율, 태스크 분배, 결과 종합 | — |
| **data-engineer** | 데이터 수집, 전처리, DB 저장, 실시간 스트림 | `src/data/` |
| **strategy-researcher** | 지표 구현, 전략 개발, 백테스팅 | `src/indicators/`, `src/strategy/`, `src/backtest/` |
| **execution-engineer** | 주문 실행, 리스크 관리, 포지션 관리, 모니터링 | `src/execution/`, `src/monitoring/` |

> ML 모듈(`src/ml/`)은 strategy-researcher가 주도하되,
> 피처는 data-engineer, 실시간 예측은 execution-engineer가 협업.

### 팀 실행 방법

```bash
# 1. tmux 세션 시작
tmux new -s quant

# 2. Claude Code 실행
cd ~/quant-trading
claude

# 3. 팀 생성 프롬프트
```

#### Phase 1: 데이터 인프라 구축 프롬프트

```
에이전트 팀을 만들어서 데이터 인프라를 구축해줘.

팀 구성:
1. "data-engineer" — Binance BTC/USDT 선물 데이터 수집기 구현
   - ccxt로 OHLCV 히스토리컬 데이터 수집 (1m, 5m, 15m, 1h, 4h, 1d)
   - WebSocket 실시간 스트림 구현
   - PostgreSQL 저장 레이어 구현
   - 최소 1년치 데이터 수집 스크립트

2. "strategy-researcher" — 기술적 지표 모듈 구현
   - 피보나치 되돌림 계산기 (0.236, 0.382, 0.5, 0.618, 0.786)
   - 하모닉 패턴 감지기 (Gartley, Butterfly, Bat, Crab, Shark)
   - 기본 지표: RSI, MACD, BB, EMA, ATR
   - 멀티 타임프레임 지표 통합

3. "execution-engineer" — 실행 인프라 기초
   - Binance 선물 연결기 (ccxt, testnet 모드)
   - 리스크 관리 모듈 (최대 손실 2%, 포지션 사이징)
   - 기본 주문 인터페이스 (market, limit, stop-loss)

파일 소유권을 철저히 지켜서 충돌 없이 작업하고,
각 모듈 간 인터페이스는 CLAUDE.md에 정의된 구조를 따라줘.
모든 모듈에 pytest 단위 테스트를 포함해.
```

#### Phase 2: 전략 개발 + 백테스팅 프롬프트

```
에이전트 팀을 만들어서 매매 전략을 개발하고 백테스팅해줘.

팀 구성:
1. "strategy-fib" — 피보나치 되돌림 전략 구현 + 백테스트
   - 스윙 하이/로우 자동 감지
   - 되돌림 레벨에서 진입, 확장 레벨에서 익절
   - vectorbt로 1년치 BTC/USDT 백테스트
   - 다양한 타임프레임(15m, 1h, 4h) 비교

2. "strategy-harmonic" — 하모닉 패턴 전략 구현 + 백테스트
   - XABCD 패턴 자동 감지 (Gartley, Butterfly, Bat, Crab)
   - PRZ(Potential Reversal Zone) 기반 진입
   - 패턴 완성도 점수 + 필터링
   - vectorbt 백테스트 + 성과 비교

3. "strategy-ensemble" — 앙상블 전략 + ML 통합
   - 개별 전략 시그널을 통합하는 앙상블 모듈
   - XGBoost/LSTM으로 시그널 강도 예측
   - Walk-forward 분석으로 과적합 방지
   - 최종 백테스트 리포트 (Sharpe, MaxDD, Win Rate 등)

각 팀원이 독립적으로 전략을 개발하고,
결과를 비교해서 최적의 앙상블 조합을 찾아줘.
백테스트 결과는 data/processed/ 에 저장.
```

#### Phase 3: 라이브 트레이딩 프롬프트

```
에이전트 팀을 만들어서 라이브 트레이딩 시스템을 완성해줘.

팀 구성:
1. "live-engine" — 실시간 매매 엔진 구현
   - 실시간 데이터 → 지표 계산 → 시그널 생성 → 주문 실행 파이프라인
   - 비동기 이벤트 루프 기반 아키텍처
   - 장애 복구 (reconnect, 주문 상태 확인)
   - Binance Testnet에서 24시간 드라이런 테스트

2. "risk-monitor" — 리스크 관리 + 모니터링
   - 실시간 PnL 추적, 최대 드로우다운 감시
   - 일일 최대 손실 도달 시 자동 거래 중단
   - Discord/Telegram 알림 (진입/청산/위험신호)
   - Grafana 대시보드 설정

3. "ml-pipeline" — ML 모델 실시간 적용
   - 학습된 모델 로딩 + 실시간 예측
   - 온라인 학습 파이프라인 (주기적 재학습)
   - 모델 성능 모니터링 + 드리프트 감지
   - A/B 테스트 프레임워크 (새 모델 vs 기존 모델)

반드시 Testnet 환경에서 실행되도록 설정하고,
실전 전환은 .env의 BINANCE_TESTNET=false 변경으로만 가능하게 해줘.
```

---

## 6. tmux 환경 구성 스크립트

아래 스크립트를 `scripts/tmux_setup.sh`로 저장:

```bash
#!/bin/bash
# scripts/tmux_setup.sh — Quant Trading tmux 환경 세팅

SESSION="quant"
PROJECT_DIR="$HOME/quant-trading"

# 기존 세션이 있으면 접속
tmux has-session -t $SESSION 2>/dev/null && {
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
# 왼쪽: Jupyter notebook 또는 백테스트 실행
# 오른쪽: 결과 확인

# 윈도우 3: 라이브 트레이딩
tmux new-window -t $SESSION -n "live" -c $PROJECT_DIR
tmux split-window -h -t $SESSION:live -c $PROJECT_DIR
tmux split-window -v -t $SESSION:live.1 -c $PROJECT_DIR
# 왼쪽: 트레이딩 봇 실행
# 오른쪽 상단: 포지션/PnL 모니터
# 오른쪽 하단: 알림 로그

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

# Claude Code 실행
tmux send-keys -t $SESSION:claude "claude" Enter

# 접속
tmux attach -t $SESSION
```

```bash
chmod +x scripts/tmux_setup.sh
```

---

## 7. 개발 순서 (로드맵)

```
Phase 1: 데이터 인프라 (1~2주)
├── Binance 연결 + OHLCV 수집기
├── 1년치 히스토리컬 데이터 수집
├── DB 스키마 + 저장 레이어
├── WebSocket 실시간 스트림
└── 기본 기술적 지표 구현

Phase 2: 전략 연구 (2~3주)
├── 피보나치 되돌림 전략
├── 하모닉 패턴 감지 + 전략
├── 멀티 타임프레임 분석
├── vectorbt 백테스팅 프레임워크
├── 파라미터 최적화
└── Walk-forward 검증

Phase 3: ML 모델 (2~3주)
├── 피처 엔지니어링
├── XGBoost / LightGBM 학습
├── LSTM 시계열 예측
├── 앙상블 모델 구성
├── 교차 검증 + 과적합 방지
└── 모델 성능 평가

Phase 4: 실행 엔진 (1~2주)
├── 주문 관리 시스템
├── 리스크 관리 (포지션 사이징, 손절)
├── 실시간 파이프라인 통합
├── 장애 복구 메커니즘
└── Testnet 드라이런

Phase 5: 모니터링 + 운영 (1주~)
├── 실시간 PnL 대시보드
├── Discord/Telegram 알림
├── 성과 리포트 자동화
├── 온라인 재학습 파이프라인
└── Testnet 안정화 → 실전 전환 검토
```

---

## 8. 핵심 설정 파일 예시

### config/settings.py

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Exchange
EXCHANGE_ID = "binanceusdm"  # Binance USDT-M Futures
API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# Trading
SYMBOL = "BTC/USDT:USDT"
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
DEFAULT_LEVERAGE = 5
MAX_POSITION_SIZE_PCT = 0.1   # 계좌의 10%
MAX_LOSS_PER_TRADE_PCT = 0.02 # 거래당 최대 손실 2%
DAILY_MAX_LOSS_PCT = 0.05     # 일일 최대 손실 5%

# Fibonacci Levels
FIB_RETRACEMENT = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXTENSION = [1.0, 1.272, 1.618, 2.0, 2.618]

# Database
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

# ML
MODEL_DIR = "models/"
RETRAIN_INTERVAL_HOURS = 24
LOOKBACK_PERIODS = 500
```

### config/risk.yaml

```yaml
risk_management:
  max_loss_per_trade_pct: 0.02      # 2% per trade
  daily_max_loss_pct: 0.05          # 5% daily
  max_open_positions: 3
  max_leverage: 10
  default_leverage: 5
  trailing_stop_pct: 0.015          # 1.5%
  
position_sizing:
  method: "kelly"                    # kelly | fixed | volatility_adjusted
  kelly_fraction: 0.25              # quarter-Kelly
  fixed_amount_usdt: 100
  
stop_loss:
  method: "atr"                      # atr | fixed_pct | swing_low
  atr_multiplier: 2.0
  fixed_pct: 0.02
  
take_profit:
  method: "fibonacci_extension"
  levels: [1.0, 1.618, 2.618]
  partial_close_pcts: [0.5, 0.3, 0.2]  # 50% → 30% → 20% 분할 익절
```

---

## 9. 빠른 시작 요약

```bash
# 1. 프로젝트 클론/생성
mkdir ~/quant-trading && cd ~/quant-trading
git init

# 2. 이 가이드의 디렉토리 구조대로 생성
# (Claude Code에게 시키면 됨)

# 3. 환경 설치
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. .env 설정 (Binance Testnet API 키)

# 5. CLAUDE.md 작성

# 6. Claude Code Agent Teams 활성화
# ~/.claude/settings.json에 CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1

# 7. tmux 환경 실행
./scripts/tmux_setup.sh

# 8. Claude Code에서 Phase 1 팀 프롬프트 실행
# → Phase 2 → Phase 3 → Phase 4 → Phase 5 순서로 진행
```

---

## 10. Agent Teams 사용 팁

1. **파일 소유권 명확히** — CLAUDE.md에 팀원별 담당 파일을 명시해야 충돌 방지
2. **권한 사전 설정** — 팀원이 permission prompt에서 멈추지 않도록 allowlist 설정
3. **인터페이스 먼저** — 모듈 간 인터페이스(클래스, 함수 시그니처)를 먼저 정의하고 구현
4. **작은 단위로** — Phase별로 팀을 새로 만드는 게 좋음 (세션 유지 제한 있음)
5. **Plan 먼저** — `claude` 실행 후 `/plan` 모드로 계획 수립 → 팀에 위임
6. **토큰 비용 관리** — Agent Teams는 토큰을 많이 소모하므로, 단순 수정은 단일 세션으로
