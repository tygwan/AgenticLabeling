# NOW

- **시각**: 2026-04-21
- **진행 중**: baseline cycle 브라우저 검증 계속 (사용자가 테스트 중)

## 직전 완료
- ✅ Review 뷰어 contain-fit + bbox/mask/label 픽셀 정렬 수정
- ✅ concept `frontend/image-viewer-overlay-alignment.md` 작성 (JS-measured overlay layer 패턴)
- ✅ LEARNING 2건 추가: (1) contain-fit overlay JS 측정, (2) SQLite DDL 순서 의존성
- ✅ WORKLOG 2개 [DEV] entry 추가, concept·LEARNING 양방향 링크

## 서버 기동 상태
- URL: **http://127.0.0.1:8090/**
- Config: GPU + BF16, Florence-2-base
- 실시간 가동 중 (process ID 등은 필요 시 `netstat -ano | grep :8090`)

## 남은 cycle 체크리스트 (브라우저)
- [x] Review 워크스페이스 이미지 contain-fit
- [x] bbox/label 정렬
- [ ] 키보드 단축키 (J/K/A/D/B/M/L/←/→/?/U)
- [ ] 오버레이 토글 (B/M/L) 실시간 반응
- [ ] 소스 전환 후 재측정 (←/→)
- [ ] 새 이미지 업로드 (Home 드롭존)
- [ ] Export 실행 + 다운로드
- [ ] Settings 트윅(테마·accent·density)

## 다음 대기 (cycle 후)
- 관찰 결과 → OPTIMIZATION-NOTES 재평가
- Wave C (Batch Curation) 또는 Research follow-up (Florence-2-large / SAM3 심화) 선택

## 축적 feed
- 활동: [WORKLOG.md](WORKLOG.md)
- 일반 원칙: [LEARNINGS.md](LEARNINGS.md)
- 최적화 후보: [OPTIMIZATION-NOTES.md](OPTIMIZATION-NOTES.md)
- 개념 설명: [../concepts/README.md](../concepts/README.md)
