# NOW

- **시각**: 2026-04-21
- **진행 중**: baseline cycle 관찰 기록 완료, 사용자가 다음 확인 항목 제시 대기

## 직전 완료 (방금 업로드 1건 + 관찰 기록)
- ✅ 실사용 업로드: SAM3 warm 상태에서 파이프라인 **791 ms**, Florence-2 + SAM3 정상 동작 확인
- ✅ LEARNING 추가: 파일 경로는 상대 경로/content-addressed 저장 (absolute path 금지)
- ✅ OPTIMIZATION-NOTES 2건 추가: (1) SAM3 cold start 32s → warm 0.8s, (2) sources.file_path absolute path 환경 이전 불가
- ✅ WORKLOG `[DEV]` entry 추가 — baseline cycle 관찰 2건 연결

## 서버 상태
- URL: **http://127.0.0.1:8090/** 계속 가동 중
- SAM3 warm, Florence-2 warm (이후 업로드는 ~800ms)
- 기존 DB 중 4 sources는 절대 경로 문제로 이미지 로드 실패(500) — 신규 업로드엔 영향 없음

## 주목할 관찰 포인트 (기록 완료)
1. **cold start 32s** (`runs.duration_ms` = 32,658): SAM3 모델 로드
2. **warm 791ms**: 두 번째 업로드부터 정상 추론 속도
3. **absolute path 오염**: 과거 Linux 경로가 Windows에서 FileNotFoundError

## 다음 대기
- 사용자가 지시한 "확인할 내용" 수신 대기
- 이후 선택: (a) file_path migration 즉시 처리, (b) Wave C 착수, (c) SAM3 cold start preload 실험

## 축적 feed
- 활동: [WORKLOG.md](WORKLOG.md)
- 일반 원칙: [LEARNINGS.md](LEARNINGS.md)
- 최적화 후보: [OPTIMIZATION-NOTES.md](OPTIMIZATION-NOTES.md)
- 개념 설명: [../concepts/README.md](../concepts/README.md)
