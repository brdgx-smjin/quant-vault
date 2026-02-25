# CLAUDE.md — AI Quant Trading System

## 프로젝트 개요
Binance BTC/USDT 선물거래를 위한 AI 퀀트 트레이딩 시스템.
하모닉 패턴, 피보나치 되돌림 등 기술적 분석 + ML 기반 시그널 생성.

## 기술 스택
- Python 3.12, ccxt (Binance), pandas-ta, TA-Lib, vectorbt
- ML: PyTorch, XGBoost, LightGBM, scikit-learn
- DB: PostgreSQL + TimescaleDB, Redis
- 백테스팅: vectorbt

## 프로젝트 구조
- config/         → 설정 파일 (settings.py, risk.yaml, symbols.yaml)
- src/data/       → 데이터 수집/전처리
- src/indicators/ → 기술적 지표 (하모닉, 피보나치 등)
- src/strategy/   → 매매 전략 로직
- src/backtest/   → vectorbt 기반 백테스팅
- src/ml/         → ML 모델 학습/예측
- src/execution/  → Binance 주문 실행
- src/monitoring/ → 알림, 대시보드
- src/utils/      → 유틸리티
- scripts/        → 실행 스크립트
- tests/          → 테스트

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
- 세션 이름: quant
- 윈도우 0: claude — Claude Code (메인)
- 윈도우 1: data — 데이터 수집/모니터링
- 윈도우 2: backtest — 백테스트/분석
- 윈도우 3: live — 라이브 트레이딩 로그

## 팀 작업 시 파일 소유권
- data-engineer       → src/data/, scripts/collect_data.py
- strategy-researcher → src/indicators/, src/strategy/, src/backtest/
- ml-engineer         → src/ml/, scripts/train_model.py
- execution-engineer  → src/execution/, src/monitoring/, scripts/live_trading.py
→ 팀원 간 같은 파일 수정 금지! 인터페이스로 소통.
