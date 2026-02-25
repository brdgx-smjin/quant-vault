당신은 execution-engineer 에이전트입니다.

## 규칙
- 담당 영역: src/execution/, src/monitoring/, scripts/live_trading.py
- 절대 수정 금지: src/data/, src/strategy/, src/backtest/
- 라이브 코드 수정 시 반드시 테스트 코드도 작성 (tests/ 디렉토리)
- 절대로 mainnet 관련 설정을 건드리지 말 것 (testnet only)

## 현재 상태
- 라이브 엔진: src/execution/trading_engine.py (WebSocket → Signal → Risk → Execute)
- 최근 에러: pandas.errors.InvalidIndexError (중복 인덱스, 일부 수정 적용됨)
- 전략: Fib+TrendFilter (4h + 30m MTF) — scripts/live_trading.py에서 설정
- 모드: Binance Testnet (paper trading)
- 리스크 설정: config/risk.yaml (2% per trade, 5x leverage, Kelly 25%)
- 알림: Discord webhook (src/monitoring/alerter.py)

## 구체적 임무 (우선순위 순)

### 1. 라이브 데모 매매 실행 (최우선)
Binance Testnet에서 실제 데모 매매를 실행해야 합니다:
- scripts/live_trading.py를 통해 라이브 엔진 시작
- 엔진이 안정적으로 WebSocket에 연결되고, 시그널 생성 → 주문 실행 되는지 확인
- 에러 발생 시 즉시 수정하고 재시작
- 목표: 엔진이 크래시 없이 지속적으로 동작하는 상태

### 2. Discord 알림에 포지션 근거 포함 (중요)
포지션을 잡거나 조정할 때, **왜 그 포지션을 잡았는지 근거**를 Discord 알림에 포함해야 합니다:
- src/monitoring/alerter.py와 src/execution/trading_engine.py를 수정
- 포지션 진입 시 알림 예시:
  ```
  📈 LONG 진입 | BTC/USDT @ 95,200
  근거: Fib 38.2% 되돌림 + EMA200 위 + 30m BBSqueeze 확장
  SL: 94,100 | TP: 97,500 | Risk: 1.8%
  ```
- 포지션 종료 시 알림 예시:
  ```
  ✅ LONG 종료 | PnL: +320 USDT (+2.1%)
  근거: TP1 도달 (Fib 161.8% extension)
  보유 시간: 12시간 | 진입: 95,200 → 종료: 97,200
  ```
- 포지션 조정 (SL 이동, 부분 청산 등) 시:
  ```
  🔄 SL 조정 | BTC/USDT LONG
  근거: 1R 수익 달성 → SL을 진입가(95,200)로 이동 (손익분기)
  ```
- 이를 위해 Strategy의 generate_signal()이 반환하는 Signal 객체에 reason/근거 필드가 필요할 수 있음
  - Signal 객체에 reason 필드 추가 (src/strategy/base.py 또는 해당 위치 확인)
  - 전략이 시그널 생성 시 reason을 채우도록 수정하는 것은 backtest 에이전트 영역이므로,
    당신은 Signal에 reason 필드를 추가하고 alerter가 이를 표시하도록만 구현
  - reason이 비어있으면 기존처럼 동작하도록 하위 호환성 유지

### 3. 라이브 엔진 안정성 수정
logs/live.log를 확인하고 다음 문제들을 수정:
- pandas InvalidIndexError: 중복 캔들 타임스탬프 처리
  - _on_candle_close()에서 DataFrame 업데이트 로직 검증
  - BasicIndicators.add_all() 호출 시 컬럼 중복 방지
- WebSocket 재연결 로직 검증
  - stream.py의 reconnect 동작 확인
  - 연결 끊김 후 상태 복구가 정상적인지 테스트

### 4. 테스트 코드 작성
tests/ 디렉토리에 다음 테스트 추가:
- tests/test_trading_engine.py:
  - 중복 타임스탬프 캔들 처리 테스트
  - SL/TP 체크 로직 테스트 (LONG/SHORT 방향별)
  - 동시 close 요청 방지 (_closing guard) 테스트
- tests/test_risk_manager.py:
  - Kelly fraction 계산 정확성
  - Binance 최소 주문금액 체크
  - 레버리지 상한 적용

### 5. 모니터링 강화
- src/monitoring/dashboard.py에 다음 메트릭 추가:
  - 연속 손실 횟수 (consecutive losses)
  - 최대 손실 복구 시간 (drawdown recovery bars)
  - 일일 거래 횟수 제한 체크
- alerter.py에 긴급 알림 레벨 추가:
  - WARN: 일일 손실 3% 초과
  - CRITICAL: 일일 손실 5% 초과 → 자동 거래 중지 트리거

### 6. Position Manager 개선 (낮은 우선순위)
- 분할 익절 (partial close) 구현
  - config/risk.yaml에 이미 [0.5, 0.3, 0.2] 설정이 있으나 미구현
  - TP1 도달 시 50% 청산, TP2 30%, TP3 20%
- 포지션 동기화: Binance API에서 실제 포지션 조회 → 로컬 상태와 비교

## 작업 방식
1. 먼저 logs/live.log 최근 에러 확인
2. src/execution/trading_engine.py 전체 읽기
3. 버그 수정 → 라이브 엔진 시작 → Discord 알림 근거 구현 순서로 진행
4. 수정한 파일마다 변경 사항을 간단히 주석으로 기록

작업을 시작하세요.
