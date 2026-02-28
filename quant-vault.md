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

## On Hold

- [ ] 전략 연구 중단 — 88% robustness = 구조적 천장 (Phase 39 확정)
- [ ] ML 접근 (XGBoost, Regime Classifier 모두 실패)
- [ ] 멀티에셋 ETH/SOL (상관관계 0.74+, 분산효과 없음)

## Done

- [x] Phase 1-39 전략 연구 완료 (15개 지표, 모든 접근법 테스트)
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
