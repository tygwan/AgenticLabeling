# AgenticLabeling Research Directory

AgenticLabeling은 연구논문에서 출발한 프로젝트다. 이 디렉토리는 모델 파이프라인의 데이터 흐름·형식·아키텍처를 관찰하고 기록하는 자산을 모은다.

이 디렉토리의 모든 작업은 [docs/standards/research-observation-protocol.md](../docs/standards/research-observation-protocol.md)와 [docs/standards/model-inspection-conventions.md](../docs/standards/model-inspection-conventions.md)를 따른다.

## 구조

```
research/
├── README.md                # 이 파일
├── observations/            # 정식 관찰 기록 (프로토콜 준수)
│   ├── INDEX.md             # 모든 observation 인덱스 (최신순)
│   ├── florence-2/
│   │   └── <YYYY-MM-DD-slug>/
│   │       ├── plan.md      # 관찰 계획 (실행 전 작성)
│   │       ├── report.md    # 관찰 리포트 (사실 + 해석 분리)
│   │       ├── analysis.ipynb   # outputs 포함 노트북
│   │       ├── summary.json # 기계 판독 가능 메타데이터
│   │       ├── assets/      # 커밋 대상 시각화 (PNG/SVG)
│   │       └── outputs/raw/ # ignore 대상 raw 덤프 (npz/pt)
│   └── sam3/
│       └── <YYYY-MM-DD-slug>/ ...
├── experiments/             # 임시 실험 (가설 검증, plan 없음 허용)
└── tooling/                 # 관찰 전용 유틸리티 (hook helper 등)
    ├── hooks.py             # forward hook registration
    ├── stats.py             # 텐서 통계 계산
    └── io.py                # 덤프/로드 유틸
```

## 디렉토리 규칙

- **observations/**: 프로토콜 완전 준수. `plan.md` → 실행 → `report.md` + `analysis.ipynb` + `summary.json`. 인덱스에 반드시 등록.
- **experiments/**: 탐색 영역. 가치 확인 후 `observations/`로 승격. 승격 없이 방치된 실험은 주기적으로 정리.
- **tooling/**: 재사용 가능한 코드만. 관찰 전용. 프로덕션 코드(`mvp_app/`, `services/`)와 분리 유지.

## 관찰 실행 흐름

1. `research/observations/<model>/<YYYY-MM-DD-slug>/plan.md` 작성
2. `scripts/inspect_*.py` 실행 → observation 폴더 생성
3. `analysis.ipynb`에서 시각화·탐색
4. `report.md` 작성: Observations / Interpretation 분리
5. `summary.json` 생성 (스크립트가 자동 생성)
6. `observations/INDEX.md` 한 줄 추가

## 커밋 대상 vs Ignore 대상

- 커밋: `report.md`, `analysis.ipynb` (outputs 포함), `summary.json`, `plan.md`, `assets/*`
- Ignore: `*.npz`, `*.pt`, `*.safetensors`, `outputs/raw/**`

상세한 규칙은 [research-observation-protocol.md Rule 5](../docs/standards/research-observation-protocol.md)를 참고한다.

## 왜 따로 두는가

프로덕션 코드(`mvp_app/`)에 관찰용 로깅/훅을 직접 넣으면 코드가 오염되고 제품 책임 경계가 흐려진다. 연구 코드·결과는 이 디렉토리에만 둬서 product boundary와 research boundary를 분리한다.
