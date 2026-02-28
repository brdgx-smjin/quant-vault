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
- 최대 손실 한도: 계좌 잔고의 20% per trade (10x 적용)
- Testnet 환경에서 먼저 검증

## 전략 연구 규칙
- **BTC/USDT 단일 자산만 연구 및 트레이딩** — ETH, SOL 등 멀티에셋 연구 금지
  - Phase 22 검증 완료: 크립토 자산 간 상관관계 0.74+ (분산효과 없음)
  - BTC 전략은 ETH/SOL에 이전 불가 (standalone 22~66% robustness)
  - 멀티에셋 포트폴리오는 BTC 단독(88%) 대비 항상 악화 (최대 77%)
- **유효 타임프레임: 1h, 15m만 사용** — 5m/30m/2h/4h 전략 연구 금지
- 새로운 전략 연구 시 반드시 9-window Walk-Forward 검증 필수
- 현재 프로덕션: Cross-TF Portfolio 4-comp (1hRSI/1hDC/15mRSI/1hWillR 15/50/10/25, 88% robustness, +23.98% OOS)
  - Phase 25b 검증: 303/375 weight 조합 88%, 11/12 param 변동 안정
  - Fallback: 3-comp (33/33/34, 88% rob, +18.81% OOS)

## 전략 연구 결과 (Phase 3-39, 2025-02~2026-02)
- **88% robustness = 구조적 천장** — W2(2025-11-20~12-02) 모든 전략 동시 손실, 해결 불가
- 테스트 완료된 5th 컴포넌트 (15개, 모두 실패):
  CCI, VWAP, Ichimoku, Fisher, Z-Score, CMF, StochRSI, EFI, Aroon, DPO, TSI, Supertrend, PSAR, TRIX, OBV
- 테스트 완료된 접근법 (모두 실패):
  멀티에셋(ETH/SOL), 5m/30m/2h 타임프레임, 적응형 가중치, 출구 최적화,
  레짐 필터(Hurst/ER), 펀딩레이트, 트레일링스탑, ADX/Chop 필터,
  캔들바디/세션/시간대 필터, ML XGBoost, ML 레짐분류기, 방향성 비대칭(P38), OBV(P39)
- 상세 결과: logs/phase{3-39}.log, MEMORY.md 참조

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
- tech-lead            → scripts/review.py (코드 리뷰 전용, 소스 수정 금지)
→ 팀원 간 같은 파일 수정 금지! 인터페이스로 소통.
→ 커밋 전 `.venv/bin/python scripts/review.py` 실행
→ FAIL 시 해당 팀원에게 수정 요청
