# AI Quant Trading — 하이브리드 배포 가이드

## ⚠️ 주의사항
> 선물거래는 원금 이상의 손실이 발생할 수 있습니다.
> 반드시 Binance Testnet에서 충분한 검증 후 실전에 적용하세요.

---

## 1. 배포 전략: 로컬 + 클라우드 하이브리드

```
┌─────────────────────────┐          ┌────────────────────────────┐
│      로컬 PC (개발)       │          │   클라우드 VPS (운영)        │
│                         │          │                            │
│  • Claude Code          │   SSH    │  • 라이브 트레이딩 봇 (24h)   │
│  • Agent Teams          │─────────→│  • 데이터 수집 데몬           │
│  • 백테스팅 (vectorbt)   │   git    │  • WebSocket 실시간 스트림    │
│  • ML 모델 학습 (GPU)    │─────────→│  • 리스크 모니터링            │
│  • 전략 연구/분석        │   deploy │  • Discord/Telegram 알림     │
│  • Jupyter Notebook     │          │  • PostgreSQL + Redis       │
│                         │          │  • systemd 자동 재시작       │
└─────────────────────────┘          └────────────────────────────┘
        개발/연구 환경                         운영/실행 환경
```

### 역할 분담

| 단계 | 환경 | 이유 |
|------|------|------|
| 개발 + Agent Teams | 로컬 PC | Claude Code는 로컬에서 실행 |
| 백테스팅 + ML 학습 | 로컬 PC (GPU) | 연산량 많음, GPU 활용 |
| 라이브 트레이딩 | 클라우드 VPS | 24시간 안정 가동, 낮은 네트워크 레이턴시 |
| 모니터링 | 둘 다 | 서버에서 실행, 로컬에서 SSH/대시보드로 확인 |

### 왜 로컬만으로는 안 되는가

- PC를 끄면 봇도 종료됨 → 오픈 포지션 방치 위험
- 가정용 네트워크 불안정 → 주문 누락, WebSocket 끊김
- Windows 업데이트/재부팅 → 예고 없는 중단
- IP 변동 → Binance API 화이트리스트 관리 어려움

---

## 2. 클라우드 VPS 선택

### 추천: Oracle Cloud Free Tier (비용 0원)

| 항목 | 스펙 |
|------|------|
| Shape | VM.Standard.A1.Flex (ARM) |
| CPU | 4 OCPU (Ampere A1) |
| RAM | 24GB |
| Storage | 200GB (Boot Volume) |
| OS | Ubuntu 22.04 / 24.04 |
| 비용 | **평생 무료** (Always Free) |

> Oracle Cloud Always Free 티어는 ARM 인스턴스 4 OCPU / 24GB RAM을 무기한 제공.
> 트레이딩 봇 + DB + Redis 운영에 충분한 스펙.

### 대안

| 서비스 | 스펙 | 월 비용 | 비고 |
|--------|------|---------|------|
| Oracle Cloud Free | 4 OCPU / 24GB | $0 | 최고 가성비, ARM |
| AWS Lightsail | 2 vCPU / 4GB | $20 | 안정적, 익숙한 환경 |
| Vultr | 2 vCPU / 4GB | $24 | 서울 리전 있음 |
| Hetzner | 4 vCPU / 8GB | €7 | 유럽 기반, 저렴 |
| 자택 미니PC | N100 / 16GB | 전기세만 | 네트워크 관리 필요 |

---

## 3. 클라우드 서버 초기 세팅

### 3.1 Oracle Cloud VM 생성

```
1. cloud.oracle.com 접속 → 계정 생성
2. Compute → Instances → Create Instance
3. Image: Ubuntu 22.04 (또는 24.04)
4. Shape: VM.Standard.A1.Flex
   - OCPU: 4, Memory: 24GB
5. Networking: Public subnet, 공인 IP 할당
6. SSH Key: 공개키 등록
7. Boot Volume: 100GB (무료 범위 내)
8. Create 클릭
```

### 3.2 보안 그룹 (Ingress Rules)

```
# 필수 포트만 개방
SSH (22)        → 내 IP만 허용
PostgreSQL (5432) → 서버 내부만 (127.0.0.1)
Redis (6379)      → 서버 내부만 (127.0.0.1)
Grafana (3000)    → 내 IP만 허용 (선택)
```

### 3.3 서버 기본 설치

```bash
# SSH 접속
ssh -i ~/.ssh/oracle_key ubuntu@<서버-공인-IP>

# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지
sudo apt install -y \
    tmux git curl wget unzip \
    build-essential \
    python3.11 python3.11-venv python3.11-dev \
    postgresql postgresql-contrib \
    redis-server \
    nginx certbot  # 리버스 프록시 (Grafana용, 선택)

# TA-Lib C 라이브러리 (ARM용)
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib
./configure --prefix=/usr && make -j$(nproc) && sudo make install
cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# 방화벽 설정
sudo ufw allow OpenSSH
sudo ufw enable
```

### 3.4 PostgreSQL 설정

```bash
# DB + 유저 생성
sudo -u postgres psql << 'SQL'
CREATE USER quant WITH PASSWORD 'your_secure_password';
CREATE DATABASE quant_trading OWNER quant;
\c quant_trading
CREATE EXTENSION IF NOT EXISTS timescaledb;
SQL

# 외부 접속 차단 확인
# /etc/postgresql/*/main/pg_hba.conf에서
# host 항목이 127.0.0.1만 허용하는지 확인
```

### 3.5 Redis 설정

```bash
# 비밀번호 설정
sudo sed -i 's/# requirepass foobared/requirepass your_redis_password/' \
    /etc/redis/redis.conf

# 외부 접속 차단
sudo sed -i 's/^bind .*/bind 127.0.0.1/' /etc/redis/redis.conf

sudo systemctl restart redis
```

### 3.6 프로젝트 배포

```bash
# 프로젝트 클론
git clone https://github.com/your-username/quant-trading.git ~/quant-trading
cd ~/quant-trading

# Python 가상환경
python3.11 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# ccxt 성능 최적화
pip install orjson coincurve

# 환경변수 설정
cp .env.example .env
nano .env  # API 키, DB 정보 입력
```

---

## 4. 운영 환경 구성

### 4.1 systemd 서비스 (자동 실행 + 자동 복구)

봇이 죽어도 자동 재시작, 서버 재부팅해도 자동 실행됩니다.

#### 데이터 수집 데몬

```bash
sudo tee /etc/systemd/system/quant-collector.service << 'EOF'
[Unit]
Description=Quant Trading Data Collector
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/quant-trading
Environment=PATH=/home/ubuntu/quant-trading/.venv/bin:/usr/bin
EnvironmentFile=/home/ubuntu/quant-trading/.env
ExecStart=/home/ubuntu/quant-trading/.venv/bin/python scripts/collect_data.py
Restart=always
RestartSec=10
StartLimitIntervalSec=60
StartLimitBurst=5

[Install]
WantedBy=multi-user.target
EOF
```

#### 라이브 트레이딩 봇

```bash
sudo tee /etc/systemd/system/quant-bot.service << 'EOF'
[Unit]
Description=Quant Trading Live Bot
After=network.target postgresql.service redis.service quant-collector.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/quant-trading
Environment=PATH=/home/ubuntu/quant-trading/.venv/bin:/usr/bin
EnvironmentFile=/home/ubuntu/quant-trading/.env
ExecStart=/home/ubuntu/quant-trading/.venv/bin/python scripts/live_trading.py
Restart=always
RestartSec=15
StartLimitIntervalSec=300
StartLimitBurst=5

[Install]
WantedBy=multi-user.target
EOF
```

#### 서비스 등록 및 실행

```bash
sudo systemctl daemon-reload

# 부팅 시 자동 시작 등록
sudo systemctl enable quant-collector
sudo systemctl enable quant-bot

# 시작
sudo systemctl start quant-collector
sudo systemctl start quant-bot

# 상태 확인
sudo systemctl status quant-bot
sudo systemctl status quant-collector

# 실시간 로그
journalctl -u quant-bot -f
journalctl -u quant-collector -f
```

### 4.2 tmux로 수동 모니터링 (SSH 접속 시)

```bash
# 서버에 SSH 접속 후
tmux new -s monitor

# 패인 분할
# 왼쪽: 봇 로그
journalctl -u quant-bot -f

# 오른쪽 (prefix + %): 데이터 수집 로그
journalctl -u quant-collector -f

# 아래 (prefix + "): Redis 모니터
redis-cli -a your_redis_password monitor
```

### 4.3 Docker 배포 (선택 — 더 깔끔한 방법)

```dockerfile
# Dockerfile
FROM python:3.11-slim

# TA-Lib 설치
RUN apt-get update && apt-get install -y \
    build-essential wget && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib && \
    ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz && \
    apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install orjson coincurve

COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/

CMD ["python", "scripts/live_trading.py"]
```

```yaml
# docker-compose.yml
version: "3.8"

services:
  postgres:
    image: timescale/timescaledb:latest-pg16
    restart: always
    environment:
      POSTGRES_USER: quant
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: quant_trading
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "127.0.0.1:5432:5432"

  redis:
    image: redis:7-alpine
    restart: always
    command: redis-server --requirepass ${REDIS_PASSWORD}
    ports:
      - "127.0.0.1:6379:6379"

  collector:
    build: .
    restart: always
    env_file: .env
    depends_on:
      - postgres
      - redis
    command: python scripts/collect_data.py

  bot:
    build: .
    restart: always
    env_file: .env
    depends_on:
      - postgres
      - redis
      - collector
    command: python scripts/live_trading.py

  grafana:
    image: grafana/grafana:latest
    restart: always
    ports:
      - "127.0.0.1:3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  pgdata:
  grafana_data:
```

```bash
# 실행
docker compose up -d

# 로그 확인
docker compose logs -f bot
docker compose logs -f collector

# 업데이트 배포
git pull
docker compose build
docker compose up -d
```

---

## 5. 로컬 → 서버 배포 워크플로우

### 일상적인 개발-배포 사이클

```
로컬 PC                              클라우드 서버
─────────                            ──────────
1. Claude Code Agent Teams
   로 전략 개발/수정
        │
2. vectorbt 백테스팅
   → 성과 확인
        │
3. pytest 테스트 통과
        │
4. git commit + push ──────────────→ 5. git pull
                                          │
                                     6. pip install -r requirements.txt
                                        (새 패키지 있을 때만)
                                          │
                                     7. sudo systemctl restart quant-bot
                                          │
                                     8. journalctl -u quant-bot -f
                                        → 정상 동작 확인
```

### 배포 자동화 스크립트

로컬에서 한 번에 배포하는 스크립트:

```bash
#!/bin/bash
# scripts/deploy.sh — 로컬에서 실행

SERVER="ubuntu@<서버-IP>"
KEY="~/.ssh/oracle_key"
PROJECT_DIR="/home/ubuntu/quant-trading"

echo "=== 테스트 실행 ==="
pytest tests/ || { echo "테스트 실패! 배포 중단."; exit 1; }

echo "=== Git Push ==="
git push origin main

echo "=== 서버 배포 ==="
ssh -i $KEY $SERVER << 'REMOTE'
    cd /home/ubuntu/quant-trading
    git pull origin main
    source .venv/bin/activate
    pip install -r requirements.txt -q
    
    # 설정 변경 검증
    python -c "from config.settings import *; print('Config OK')"
    
    # 서비스 재시작
    sudo systemctl restart quant-collector
    sudo systemctl restart quant-bot
    
    # 상태 확인
    sleep 5
    sudo systemctl is-active quant-bot && echo "✅ Bot running" || echo "❌ Bot failed"
    sudo systemctl is-active quant-collector && echo "✅ Collector running" || echo "❌ Collector failed"
REMOTE

echo "=== 배포 완료 ==="
```

```bash
chmod +x scripts/deploy.sh
```

### Docker 사용 시 배포

```bash
#!/bin/bash
# scripts/deploy-docker.sh

SERVER="ubuntu@<서버-IP>"
KEY="~/.ssh/oracle_key"

echo "=== 테스트 ==="
pytest tests/ || exit 1

echo "=== Git Push ==="
git push origin main

echo "=== 서버 배포 ==="
ssh -i $KEY $SERVER << 'REMOTE'
    cd /home/ubuntu/quant-trading
    git pull origin main
    docker compose build
    docker compose up -d
    sleep 10
    docker compose ps
    docker compose logs --tail=20 bot
REMOTE
```

---

## 6. 모니터링 체계

### 6.1 알림 설정

```
트레이딩 봇
    │
    ├─→ Discord Webhook ─→ #trading-signals 채널
    │     • 진입/청산 알림
    │     • 일일 PnL 리포트
    │
    ├─→ Telegram Bot ─→ 개인 알림
    │     • 긴급 알림 (최대손실 도달, 봇 에러)
    │     • 포지션 상태 조회 커맨드
    │
    └─→ Grafana Dashboard
          • 실시간 PnL 차트
          • 포지션 히스토리
          • 승률/Sharpe 추이
```

### 6.2 헬스체크

```bash
# 간단한 헬스체크 크론잡
# crontab -e 에 추가

# 5분마다 봇 상태 확인, 죽었으면 재시작 + 알림
*/5 * * * * systemctl is-active quant-bot || \
    (systemctl restart quant-bot && \
     curl -s -X POST "$DISCORD_WEBHOOK_URL" \
     -H "Content-Type: application/json" \
     -d '{"content":"⚠️ 트레이딩 봇 재시작됨"}')
```

---

## 7. 보안 체크리스트

### API 키 보호
- [ ] .env 파일 600 권한 (`chmod 600 .env`)
- [ ] .gitignore에 .env 포함
- [ ] Binance API 키에 IP 화이트리스트 설정 (서버 IP만)
- [ ] API 키 권한: 선물거래만 허용, 출금 비활성화
- [ ] Testnet 키와 실전 키를 별도 .env 파일로 관리

### 서버 보안
- [ ] SSH 키 인증만 허용 (패스워드 로그인 비활성화)
- [ ] fail2ban 설치 (`sudo apt install fail2ban`)
- [ ] UFW 방화벽 활성화, 필요 포트만 개방
- [ ] PostgreSQL/Redis는 127.0.0.1만 바인딩
- [ ] 정기적 시스템 업데이트 (`unattended-upgrades`)

### 트레이딩 안전장치
- [ ] 일일 최대 손실 한도 설정 (예: 계좌의 5%)
- [ ] 거래당 최대 손실 한도 설정 (예: 계좌의 2%)
- [ ] 최대 포지션 수 제한
- [ ] 긴급 중단 기능 (kill switch)
- [ ] Testnet에서 최소 1주 드라이런 후 실전 전환

---

## 8. 전체 실행 순서 요약

```
1. 로컬 개발 환경 구축
   └─ Python venv, 패키지 설치, CLAUDE.md 작성

2. Claude Code Agent Teams로 Phase 1~3 개발 (로컬)
   ├─ Phase 1: 데이터 인프라
   ├─ Phase 2: 전략 개발 + 백테스팅
   └─ Phase 3: ML 모델 학습

3. Testnet 검증 (로컬에서 먼저)
   └─ Binance Testnet API로 드라이런

4. 클라우드 서버 세팅
   ├─ Oracle Cloud VM 생성
   ├─ PostgreSQL + Redis 설치
   └─ 프로젝트 배포

5. 서버에서 Testnet 드라이런 (최소 1주)
   ├─ systemd 서비스 등록
   ├─ 알림 시스템 연동
   └─ 24시간 안정성 확인

6. 실전 전환
   └─ .env에서 BINANCE_TESTNET=false 변경
      + 실전 API 키 교체
      + IP 화이트리스트 설정

7. 지속 운영
   ├─ 로컬에서 전략 개선 → 서버 배포 반복
   ├─ 주기적 모델 재학습
   └─ 성과 모니터링 + 리스크 관리
```
