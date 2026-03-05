---

kanban-plugin: board

---

## Backlog

- [ ] Vol-scaling 프로덕션 적용 검토 (Sharpe 1.011, MaxDD -1.90%)
- [ ] 데이터 품질 자동 검증 (gap/OHLC/extreme move 탐지)
- [ ] 백테스트 결과 대시보드 (Obsidian or Grafana)
- [ ] 리스크 관리 알림 강화 (DD 임계치 Discord 알림)


## In Progress

- [ ] 라이브 트레이딩 운영 (4-comp Cross-TF, 88% rob, +23.98% OOS)
- [ ] 데이터 수집 자동화 (pm2 cron 30분 주기)
- [ ] Discord 봇 운영 (일일 보고 09:00 KST + 모니터링)
- [ ] 라이브 트레이딩 수익성 모니터링 (3/5 4-comp 복원, cooldown 수정 + override 비활성화 후 관찰)


## On Hold


## Done

- [x] Phase 41: MTF 반응속도 최적화 — close>EMA20이 최적 (77% rob, +21.39% OOS, W2=-0.16%) (3/5)
- [x] 4-comp 복원 — Discord 에이전트가 WillR 제거한 것 복구 (3/5, 15/50/10/25)
- [x] Extreme Override 비활성화 확정 — RSI>70 SHORT 강제진입이 상승장에서 손실 유발 (3/5)
- [x] Cooldown 버그 수정 — len(df) → timestamp 기반 (3/5, SHORT만 10회 잡는 문제 해결)
- [x] pm2 data-collector 복구 (Discord 에이전트가 삭제한 것 재등록, 3/5)
- [x] Discord daily report에 데이터 수집 연동 (09:00 KST 보고 시 자동 수집)
- [x] Phase 1-40 전략 연구 완료 (15개 지표, 모든 접근법 테스트)
- [x] Phase 40: MTF Extreme Override 배포 (RSI<20/RSI>70 우회, 88% 유지)
- [x] ML 접근 실패 확정 (XGBoost, Regime Classifier — W2 감지 불가)
- [x] 멀티에셋 실패 확정 (ETH/SOL 상관관계 0.74+, 최대 77%)
- [x] 88% robustness = 구조적 천장 확정 (Phase 40, W2 해결 불가)
- [x] 4-comp Cross-TF 포트폴리오 프로덕션 배포 (1hRSI/1hDC/15mRSI/1hWillR)
- [x] 데이터 수집 pm2 전환 (claude 에이전트 → pm2 cron)
- [x] Discord 일일 팀 보고 구현 (!report + 자동 스케줄)
- [x] tmux 환경 정리 (불필요 세션/에이전트 정리)
- [x] Walk-Forward 검증 프레임워크 구축 (9-window, cross-TF)
- [x] Risk analytics 분석 (Sharpe, MaxDD, 컴포넌트 상관관계)




%% kanban:settings
```
{"kanban-plugin":"board","list-collapse":[false,false,false,false]}
```
%%