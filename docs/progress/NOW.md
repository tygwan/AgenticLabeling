# NOW

- **시각**: 2026-04-21
- **진행 중**: concept 추가 완료, 다음 지시 대기

## 직전 완료
- ✅ Concept 신설: `docs/concepts/ml/florence-sam-pipeline.md` — grounded detection, promptable segmentation, pipeline composition, class 흐름. 5 Evidence + 5 open follow-ups.
- ✅ concepts README 인덱스 업데이트 (florence-sam-pipeline 상단 등록)
- ✅ OPTIMIZATION-NOTES에 "이중 image encoding" 항목 추가
- ✅ WORKLOG [DEV] entry: 사용자 용어 학습 목적 → concept hub 저장, 향후 연결점 제공

## 서버 상태
- URL: **http://127.0.0.1:8090/** 계속 가동 중 (SAM3 warm)

## 미해결 항목 (baseline cycle 발견)
- sources.file_path absolute path 환경 이전 불가 — DB 4 rows 500 에러 중
- SAM3 cold start 32s (warm 시 0.8s) — eager preload 검토 대기

## 다음 대기
- 사용자 "확인할 내용" — 용어 공부 기반으로 다음 질문·지시 대기
- 선택지: (a) file_path migration 즉시, (b) Wave C, (c) prompt template 실험, (d) 추가 개념 정리

## 축적 feed
- 활동: [WORKLOG.md](WORKLOG.md)
- 일반 원칙: [LEARNINGS.md](LEARNINGS.md)
- 최적화 후보: [OPTIMIZATION-NOTES.md](OPTIMIZATION-NOTES.md)
- 개념 설명: [../concepts/README.md](../concepts/README.md)
