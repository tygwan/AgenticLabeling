# AgenticLabeling PRD (Product Requirements Document)

## 메타데이터
- **작성자**: Product Manager
- **작성일**: 2025-12-29
- **버전**: v1.0
- **상태**: Draft
- **프로젝트**: AgenticLabeling - AI-Powered Auto-Labeling Platform

---

## 1. 개요 (Overview)

AgenticLabeling은 최첨단 AI 모델(Florence-2, SAM2, DINOv2)을 활용하여 이미지/비디오/텍스트 데이터의 객체를 자동으로 감지, 세그멘테이션, 분류하고 이를 체계적으로 관리하여 고품질 학습 데이터셋을 구축하는 마이크로서비스 기반 플랫폼입니다. 수동 라벨링의 한계를 극복하고, 객체 중심의 데이터 관리를 통해 VLA(Vision-Language-Action) 확장을 지향합니다.

### 1.1 배경 (Background)

현재 머신러닝 프로젝트에서 직면하는 주요 문제점:
- **수동 라벨링의 병목**: 대규모 데이터셋 라벨링에 수백 시간 소요
- **품질 불균일**: 라벨러마다 다른 기준과 실수로 인한 데이터 품질 저하
- **객체 추적 어려움**: 비디오에서 동일 객체를 일관되게 추적하기 어려움
- **재사용성 부족**: 유사 프로젝트에서 기존 라벨링 자산 재활용 불가
- **확장성 제약**: 로봇 행동 데이터, 멀티모달 학습으로 확장 시 기존 시스템 재구축 필요

AgenticLabeling은 최신 Foundation Model의 zero-shot/few-shot 능력을 활용하여 자동 라벨링을 수행하고, 객체 중심의 데이터 레지스트리를 통해 재사용성과 확장성을 확보합니다.

### 1.2 목표 (Goals)

- [ ] **라벨링 속도 10배 향상**: 자동 파이프라인으로 수동 대비 10배 이상 처리량 달성
- [ ] **95% 이상 정확도**: Florence-2 + SAM2 조합으로 주요 클래스 95% 이상 정확도 확보
- [ ] **객체 중심 데이터 관리**: SQLite + ChromaDB 하이브리드로 1M+ 객체 효율적 관리
- [ ] **YOLO/COCO 포맷 지원**: 주요 프레임워크와 즉시 호환 가능한 데이터셋 내보내기
- [ ] **VLA 확장 기반 마련**: Affordance, Action Sequence 등 로봇 학습으로 확장 가능한 아키텍처

### 1.3 비목표 (Non-Goals)

이번 Phase 1에서 제외되는 항목:
- 웹 UI 기반 라벨 검수 도구 (Phase 2)
- 실시간 비디오 스트리밍 처리 (Phase 3)
- 분산 멀티 GPU 클러스터 배포 (Phase 6)
- 상용 라벨링 서비스 제공 (범위 밖)
- 3D 포인트 클라우드 라벨링 (범위 밖)

---

## 2. 문제 정의 (Problem Statement)

### 2.1 현재 상황

머신러닝 엔지니어와 연구자들은 다음과 같은 데이터 라벨링 문제에 직면해 있습니다:

**Pain Points**:
1. **시간 소모**: 1,000장 이미지 바운딩박스 라벨링에 평균 20-40시간 소요
2. **세그멘테이션 비용**: 픽셀 단위 마스크는 바운딩박스 대비 5-10배 작업 시간 필요
3. **일관성 부족**: 같은 객체에 대해 라벨러마다 다른 기준 적용 (예: "person" vs "pedestrian")
4. **비디오 추적 복잡도**: 프레임 간 동일 객체 ID 유지를 위한 수작업 매칭 필요
5. **검색 불가능**: 특정 속성(크기, 포즈, 장면)을 가진 객체 찾기 어려움

### 2.2 해결 방안

AgenticLabeling은 다음 전략으로 문제를 해결합니다:

**솔루션 구성 요소**:

1. **자동 라벨링 파이프라인**
   - Florence-2: 텍스트 프롬프트로 객체 감지 (grounding)
   - SAM2: 바운딩박스 기반 고정밀 세그멘테이션
   - DINOv2: 시각 임베딩 추출 및 few-shot 분류

2. **Object Registry (핵심)**
   - SQLite: 메타데이터, 관계, 프로젝트 관리
   - ChromaDB: 벡터 유사도 검색 (임베딩)
   - Filesystem: 마스크 파일 sharded 저장

3. **품질 관리 워크플로우**
   - 신뢰도 기반 필터링 (threshold 조정 가능)
   - 수동 검증 상태 추적 (is_validated)
   - IoU 기반 중복 제거

4. **유연한 데이터셋 구성**
   - 카테고리/신뢰도/검증 상태 기반 필터링
   - train/val/test 자동 분할
   - YOLO/COCO 포맷 내보내기

---

## 3. 사용자 페르소나 (User Personas)

### 3.1 ML 엔지니어 (Alex)

**목표**: 객체 감지 모델 학습을 위한 YOLO 데이터셋 빠르게 구축

**시나리오**:
- 드론 촬영 영상에서 "person", "vehicle", "building" 감지 모델 학습 필요
- 10,000장 이미지에 대해 바운딩박스 + 마스크 필요
- 기존 수동 라벨링 예상 시간: 200시간 → AgenticLabeling: 20시간

**사용 기능**:
- `/auto_label` API로 배치 처리
- 신뢰도 0.7 이상만 필터링
- YOLO v8 포맷으로 내보내기

### 3.2 컴퓨터 비전 연구자 (Sam)

**목표**: 특정 객체 유형의 다양한 샘플 수집 및 분석

**시나리오**:
- "손" 제스처 연구를 위해 다양한 손 모양 수집
- 기존 데이터셋에서 유사한 손 모양 검색
- 임베딩 기반 클러스터링으로 자세 카테고리 자동 분류

**사용 기능**:
- 임베딩 벡터 검색 (`/objects/search/embedding`)
- 유사도 기반 객체 그룹핑
- 품질 점수 기반 우선순위 정렬

### 3.3 로보틱스 엔지니어 (Jordan)

**목표**: 로봇 조작 학습을 위한 객체-행동 데이터 구축 (Phase 5 목표)

**시나리오**:
- "컵", "병", "상자" 등 조작 가능한 객체 라벨링
- 각 객체에 affordance 속성 추가 (graspable, stackable)
- 로봇 행동 시퀀스와 객체 ID 연결

**사용 기능**:
- Object Registry 확장 테이블 (affordances)
- 비디오 프레임별 객체 추적
- Action-Object 매핑 데이터 내보내기

---

## 4. 핵심 기능 (Core Features)

### 4.1 기능 요구사항 (Functional Requirements)

| ID | 요구사항 | 우선순위 | 담당 서비스 | 상태 |
|----|---------|---------|-------------|------|
| FR-001 | 텍스트 프롬프트 기반 객체 감지 (Florence-2) | P0 | detection-agent | ✅ |
| FR-002 | 바운딩박스 기반 세그멘테이션 (SAM2) | P0 | segmentation-agent | ✅ |
| FR-003 | 시각 임베딩 추출 (DINOv2) | P0 | classification-agent | ✅ |
| FR-004 | 객체 메타데이터 SQLite 저장 | P0 | object-registry | ✅ |
| FR-005 | 임베딩 벡터 ChromaDB 저장 및 검색 | P0 | object-registry | ✅ |
| FR-006 | 마스크 파일 sharded 저장 (PNG) | P0 | object-registry | ✅ |
| FR-007 | 배치 객체 등록 (동시성 안전) | P0 | object-registry | ⚠️ 진행 중 |
| FR-008 | 소스 등록 (이미지/비디오) | P0 | object-registry | ✅ |
| FR-009 | 카테고리 동적 생성 및 관리 | P0 | object-registry | ✅ |
| FR-010 | 자동 라벨링 파이프라인 (detect → segment → register) | P0 | gateway | ✅ |
| FR-011 | 유사 객체 임베딩 검색 (top-k) | P1 | object-registry | ✅ |
| FR-012 | 필터 기반 객체 검색 (category, confidence, validated) | P1 | object-registry | ✅ |
| FR-013 | 데이터셋 생성 및 train/val/test 분할 | P1 | object-registry | ✅ |
| FR-014 | YOLO 포맷 데이터셋 내보내기 | P1 | data-manager | ⬜ 미구현 |
| FR-015 | COCO 포맷 데이터셋 내보내기 | P1 | data-manager | ⬜ 미구현 |
| FR-016 | 객체 검증 상태 업데이트 (is_validated) | P1 | object-registry | ✅ |
| FR-017 | Re-ID 기반 트랙 생성 (비디오용) | P1 | object-registry | ✅ |
| FR-018 | 프레임별 객체 연결 관리 | P1 | object-registry | ✅ |
| FR-019 | 통계 대시보드 (객체 수, 카테고리 분포) | P1 | object-registry | ✅ |
| FR-020 | YOLO 모델 학습 파이프라인 | P2 | training-agent | ⬜ 미구현 |
| FR-021 | 모델 평가 (mAP, confusion matrix) | P2 | evaluation-agent | ⬜ 미구현 |
| FR-022 | MLflow 실험 추적 통합 | P2 | training-agent | ⬜ 미구현 |
| FR-023 | 비디오 프레임 자동 추출 | P2 | data-manager | ⬜ 미구현 |
| FR-024 | Active Learning 샘플 제안 | P2 | evaluation-agent | ⬜ 미구현 |

### 4.2 비기능 요구사항 (Non-Functional Requirements)

| ID | 요구사항 | 기준 | 측정 방법 |
|----|---------|------|----------|
| NFR-001 | 단일 이미지 처리 속도 | < 3초 (detect + segment) | 벤치마크 테스트 (1024x1024) |
| NFR-002 | 배치 처리 처리량 | > 100 이미지/분 | 비동기 파이프라인 |
| NFR-003 | 객체 검색 응답 시간 | < 100ms (1M 객체 기준) | ChromaDB 인덱스 성능 |
| NFR-004 | 임베딩 검색 정확도 | Top-10 Recall > 0.9 | 수동 검증 샘플 |
| NFR-005 | 마스크 저장 효율성 | < 100KB/객체 (PNG 압축) | 디스크 사용량 모니터링 |
| NFR-006 | SQLite WAL 동시성 | 10+ 동시 쓰기 처리 | 부하 테스트 |
| NFR-007 | GPU 메모리 사용률 | < 16GB (A4000 기준) | nvidia-smi 모니터링 |
| NFR-008 | API 가용성 | 99% uptime (로컬 개발) | Health check 엔드포인트 |
| NFR-009 | 데이터 무결성 | 외래 키 제약 100% 준수 | SQLite PRAGMA check |
| NFR-010 | 코드 커버리지 | > 80% (핵심 모듈) | pytest-cov |

---

## 5. 사용자 스토리 (User Stories)

### US-001: 자동 라벨링 파이프라인 실행

**As a** ML 엔지니어
**I want to** 이미지를 업로드하고 원하는 클래스를 지정하여 자동으로 객체를 감지하고 세그멘테이션하고 싶다
**So that** 수동 라벨링 없이 고품질 학습 데이터를 빠르게 얻을 수 있다

**인수 조건 (Acceptance Criteria)**:
- [ ] 이미지와 클래스 리스트를 `/auto_label` API에 제출
- [ ] 신뢰도 threshold 조정 가능 (default 0.5)
- [ ] 반환 결과에 바운딩박스, 라벨, 마스크 수 포함
- [ ] Object Registry에 자동 등록되며 object_id 반환
- [ ] 전체 파이프라인 처리 시간 < 3초 (단일 이미지 기준)

### US-002: 유사 객체 검색

**As a** 컴퓨터 비전 연구자
**I want to** 특정 객체 이미지를 업로드하면 시각적으로 유사한 객체들을 찾고 싶다
**So that** 데이터셋에서 특정 패턴이나 포즈를 가진 샘플들을 빠르게 수집할 수 있다

**인수 조건 (Acceptance Criteria)**:
- [ ] 쿼리 이미지의 임베딩 벡터 추출
- [ ] ChromaDB에서 코사인 유사도 기반 top-k 검색
- [ ] 유사도 점수와 함께 결과 반환 (1.0 = 완전 일치)
- [ ] 카테고리 필터 적용 가능
- [ ] 응답 시간 < 100ms

### US-003: 데이터셋 내보내기 (YOLO)

**As a** ML 엔지니어
**I want to** 특정 카테고리와 신뢰도 기준을 만족하는 객체들을 YOLO 포맷으로 내보내고 싶다
**So that** YOLOv8/v11 모델을 즉시 학습시킬 수 있다

**인수 조건 (Acceptance Criteria)**:
- [ ] 필터 조건 설정 (categories, min_confidence, is_validated)
- [ ] train/val/test 비율 지정 (default 0.8/0.1/0.1)
- [ ] 출력 구조: `images/`, `labels/`, `data.yaml`
- [ ] 라벨 파일은 YOLO 포맷 (normalized x_center, y_center, w, h)
- [ ] `data.yaml`에 클래스 매핑 자동 생성

### US-004: 비디오 객체 추적

**As a** 로보틱스 엔지니어
**I want to** 비디오에서 감지된 동일 객체를 프레임 간 추적하고 하나의 트랙으로 그룹핑하고 싶다
**So that** 시간에 따른 객체 움직임과 상태 변화를 분석할 수 있다

**인수 조건 (Acceptance Criteria)**:
- [ ] 비디오 프레임별 객체 감지 및 등록
- [ ] IoU 기반 프레임 간 객체 매칭
- [ ] Re-ID 기반 트랙 생성 (track_id)
- [ ] 트랙 조회 시 시간순 정렬된 객체 리스트 반환
- [ ] 트랙 통계 (start_frame, end_frame, avg_confidence)

### US-005: 객체 검증 워크플로우

**As a** 데이터 품질 관리자
**I want to** 자동 라벨링 결과를 검토하고 올바른 객체만 검증 표시를 하고 싶다
**So that** 학습 데이터 품질을 확보하고 잘못된 라벨로 인한 모델 성능 저하를 방지할 수 있다

**인수 조건 (Acceptance Criteria)**:
- [ ] 객체 리스트 조회 시 신뢰도 낮은 순 정렬 가능
- [ ] 개별 객체 `is_validated` 상태 업데이트 (PATCH)
- [ ] 검증자 ID 기록 (validated_by)
- [ ] 통계에서 검증율 확인 가능
- [ ] 데이터셋 내보내기 시 검증된 객체만 필터링 옵션

---

## 6. 시스템 아키텍처 (System Architecture)

### 6.1 마이크로서비스 구성

```
┌─────────────────────────────────────────────────────────────────────┐
│                         API Gateway (8000)                          │
│                    HTTP/REST → 내부 서비스 라우팅                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│  Detection    │         │ Segmentation  │         │Classification │
│    Agent      │         │    Agent      │         │    Agent      │
│  (Florence-2) │         │    (SAM2)     │         │   (DINOv2)    │
│   GPU:8001    │         │   GPU:8002    │         │   GPU:8003    │
└───────────────┘         └───────────────┘         └───────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Object Registry (8010)                         │
│              SQLite (메타데이터) + ChromaDB (임베딩)                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 데이터 플로우 (자동 라벨링)

```
[클라이언트]
     │
     │ POST /auto_label (image, classes, confidence)
     ▼
[Gateway:8000]
     │
     ├──> [Detection-Agent:8001] → boxes, labels, scores
     │
     ├──> [Segmentation-Agent:8002] → masks (PNG base64)
     │
     ├──> [Labeling-Agent:8004] → 라벨 저장 (선택)
     │
     └──> [Object-Registry:8010]
           ├─> POST /sources → source_id
           └─> POST /objects/batch → object_ids[]
```

### 6.3 Object Registry 저장소 구조

```
data/registry/
├── registry.db              # SQLite (메타데이터)
│   ├── sources              # 이미지/비디오 소스
│   ├── objects              # 감지된 객체
│   ├── categories           # 클래스 온톨로지
│   ├── tracks               # Re-ID 트랙
│   └── datasets             # 내보내기 구성
│
├── chroma/                  # ChromaDB (임베딩)
│   └── objects_dinov2_base_v1/
│
└── masks/                   # 마스크 파일 (sharded)
    └── {hash[0:2]}/{hash[2:4]}/{object_id}.png
```

---

## 7. 기술 스택 (Technology Stack)

### 7.1 AI/ML 모델

| 모델 | 용도 | 입출력 | 리소스 |
|------|------|--------|--------|
| Florence-2-large | 객체 감지 (grounding) | 이미지 + 텍스트 → bbox + labels | GPU 6GB |
| SAM2 (Hiera B+) | 세그멘테이션 | 이미지 + bbox → masks | GPU 8GB |
| DINOv2-base | 임베딩 추출 | 이미지 → [768d] | GPU 4GB |

### 7.2 백엔드

| 구성요소 | 기술 | 버전 | 용도 |
|----------|------|------|------|
| API 서버 | FastAPI | 0.115+ | REST API |
| DB (메타) | SQLite | 3.40+ | 메타데이터, 관계 (→ Postgres) |
| DB (벡터) | ChromaDB | 0.4+ | 임베딩 검색 (→ Qdrant) |
| 캐시 | Redis | 7.0+ | 결과 캐싱, 작업 큐 |
| 실험추적 | MLflow | 2.10+ | 학습 메트릭, 모델 저장 |

### 7.3 인프라

| 환경 | 도구 | 용도 |
|------|------|------|
| 컨테이너 | Docker + Docker Compose | 로컬 개발 환경 |
| GPU 런타임 | NVIDIA Docker | GPU 컨테이너 |
| 모니터링 | FastAPI `/health` | 서비스 헬스 체크 |

---

## 8. 데이터 모델 (Data Model)

### 8.1 핵심 엔티티

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Sources   │────<│   Objects   │>────│  Categories │
│             │     │             │     │             │
│ source_id   │     │ object_id   │     │ category_id │
│ source_type │     │ bbox        │     │ name        │
│ file_path   │     │ mask_path   │     │ synonyms    │
│ width/height│     │ confidence  │     └─────────────┘
└─────────────┘     │ embedding   │
                    └─────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
       ┌───────────┐ ┌───────────┐ ┌───────────┐
       │  Tracks   │ │ Datasets  │ │ ChromaDB  │
       │           │ │           │ │           │
       │ track_id  │ │dataset_id │ │ embedding │
       │ objects[] │ │ filters   │ │ metadata  │
       └───────────┘ └───────────┘ └───────────┘
```

### 8.2 Object 스키마 (핵심)

```sql
CREATE TABLE objects (
    object_id TEXT PRIMARY KEY,
    category_id INTEGER REFERENCES categories,
    source_id TEXT REFERENCES sources,
    frame_id TEXT REFERENCES frames,
    project_id TEXT,

    -- Geometry
    bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
    mask_path TEXT,
    area REAL,

    -- Detection metadata
    confidence REAL,
    detection_model TEXT,

    -- Quality & validation
    is_validated BOOLEAN DEFAULT FALSE,
    validated_by TEXT,
    quality_score REAL,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 9. API 명세 (API Specification)

### 9.1 Gateway API

| Endpoint | Method | 설명 | 입력 | 출력 |
|----------|--------|------|------|------|
| `/auto_label` | POST | 자동 라벨링 파이프라인 | image, classes, confidence | boxes, labels, masks_count, object_ids |
| `/detect_and_segment` | POST | 감지 + 세그멘테이션 | image, classes, confidence | detections, masks |
| `/detect` | POST | 객체 감지만 | image, classes | boxes, labels, scores |
| `/segment` | POST | 세그멘테이션만 | image, boxes | masks |

### 9.2 Object Registry API

| Endpoint | Method | 설명 | 주요 파라미터 |
|----------|--------|------|--------------|
| `POST /sources` | POST | 소스 등록 | source_type, file_path, width, height |
| `GET /sources/{id}` | GET | 소스 조회 | source_id |
| `POST /objects` | POST | 객체 등록 | source_id, category, bbox, confidence |
| `POST /objects/batch` | POST | 배치 등록 | source_id, objects[] |
| `GET /objects` | GET | 객체 검색 | category, source_id, min_confidence, limit |
| `GET /objects/{id}` | GET | 객체 조회 | object_id |
| `PATCH /objects/{id}` | PATCH | 객체 수정 | is_validated, quality_score |
| `DELETE /objects/{id}` | DELETE | 객체 삭제 | object_id |
| `GET /objects/{id}/mask` | GET | 마스크 이미지 | object_id |
| `POST /objects/search/embedding` | POST | 유사 검색 | embedding, top_k, category |
| `POST /tracks` | POST | 트랙 생성 | source_id, object_ids[] |
| `GET /tracks/{id}` | GET | 트랙 조회 | track_id |
| `POST /datasets` | POST | 데이터셋 생성 | name, format, filter_config, split_config |
| `POST /datasets/{id}/build` | POST | 데이터셋 구축 | dataset_id |
| `POST /datasets/{id}/export` | POST | 내보내기 | dataset_id, output_dir |
| `GET /stats` | GET | 통계 | - |
| `GET /categories` | GET | 카테고리 목록 | - |

---

## 10. 제약사항 (Constraints)

### 10.1 기술적 제약

- **GPU 메모리**: NVIDIA RTX A4000 (16GB) 기준 최적화, 더 작은 GPU에서는 모델 다운그레이드 필요
- **SQLite 동시성**: WAL 모드에서 최대 ~10 동시 쓰기, 초과 시 Postgres 마이그레이션 필요
- **ChromaDB 확장성**: 단일 노드 embedded 모드, 10M+ 벡터 시 Qdrant 필요
- **네트워크 지연**: 마이크로서비스 간 HTTP 통신 오버헤드 (~10-50ms/call), Phase 6에서 gRPC로 전환
- **디스크 I/O**: 마스크 파일 저장 시 SSD 권장 (HDD에서 처리량 50% 감소)

### 10.2 비즈니스 제약

- **라이선스**: Florence-2 (MIT), SAM2 (Apache 2.0), DINOv2 (Apache 2.0) - 상업적 사용 가능
- **데이터 프라이버시**: 로컬 실행 필수 (의료/금융 등 민감 데이터)
- **모델 업데이트**: Foundation Model 업데이트 시 재라벨링 비용 발생 가능
- **검증 리소스**: 자동 라벨링 결과의 10-20%는 수동 검증 권장

### 10.3 Phase 1 범위 제약

- 단일 인스턴스 배포만 지원 (K8s 멀티 노드 미지원)
- 웹 UI 없음 (API만 제공)
- 실시간 비디오 스트리밍 미지원 (프레임 단위 처리만)

---

## 11. 의존성 (Dependencies)

### 11.1 외부 라이브러리

| 라이브러리 | 용도 | 버전 제약 |
|-----------|------|----------|
| transformers | Florence-2 모델 로딩 | >= 4.38.0 |
| torch | PyTorch 백엔드 | >= 2.1.0, CUDA 11.8+ |
| sam2 | SAM2 세그멘테이션 | GitHub main branch |
| chromadb | 벡터 검색 | >= 0.4.0 |
| pillow | 이미지 처리 | >= 10.0.0 |
| opencv-python | 비디오 처리 | >= 4.8.0 |
| fastapi | API 서버 | >= 0.115.0 |
| pydantic | 데이터 검증 | >= 2.0.0 |

### 11.2 시스템 의존성

- **NVIDIA Driver**: >= 525 (CUDA 12.x 지원)
- **Docker**: >= 20.10
- **Docker Compose**: >= 2.0
- **Python**: 3.10 or 3.11
- **디스크 공간**: 최소 100GB (모델 체크포인트 + 데이터)

### 11.3 서비스 간 의존성

```
gateway
  ├─> detection-agent (required)
  ├─> segmentation-agent (required)
  ├─> classification-agent (optional)
  ├─> object-registry (required)
  └─> labeling-agent (legacy, optional)

object-registry
  ├─> SQLite (embedded)
  └─> ChromaDB (embedded)

training-agent
  ├─> data-manager (required)
  └─> mlflow (required)
```

---

## 12. 성공 지표 (Success Metrics / KPIs)

### 12.1 성능 KPI

| 지표 | 목표 | 측정 방법 | 달성 기준 |
|------|------|----------|----------|
| 처리 속도 | < 3초/이미지 | 벤치마크 스크립트 (1024x1024) | 평균 < 3초, p95 < 5초 |
| 배치 처리량 | > 100 이미지/분 | 비동기 파이프라인 (10 동시 요청) | 지속 가능한 100 img/min |
| 검색 응답 시간 | < 100ms | ChromaDB query 벤치마크 (1M 벡터) | p95 < 100ms |
| GPU 활용률 | > 80% | nvidia-smi 모니터링 | 배치 처리 중 평균 > 80% |

### 12.2 품질 KPI

| 지표 | 목표 | 측정 방법 | 달성 기준 |
|------|------|----------|----------|
| 감지 정확도 | mAP > 0.9 | 수동 라벨링 ground truth 비교 (500 샘플) | 주요 클래스 mAP > 0.9 |
| 세그멘테이션 IoU | mIoU > 0.85 | SAM2 마스크 vs 수동 마스크 (100 샘플) | mIoU > 0.85 |
| 임베딩 검색 정확도 | Top-10 Recall > 0.9 | 유사 객체 수동 검증 | 사용자 만족도 > 90% |
| 중복 제거율 | < 5% 중복 | IoU > 0.9 객체 자동 병합 | 중복 < 5% |

### 12.3 사용성 KPI

| 지표 | 목표 | 측정 방법 | 달성 기준 |
|------|------|----------|----------|
| 라벨링 시간 절감 | 10배 향상 | 수동 라벨링 vs 자동 + 검수 시간 비교 | 평균 10배 빠름 |
| API 응답 성공률 | > 99% | 에러 로그 분석 | 500 에러 < 1% |
| 데이터셋 내보내기 성공률 | 100% | YOLO/COCO 포맷 검증 | 검증 스크립트 통과 |

### 12.4 확장성 KPI

| 지표 | 목표 | 측정 방법 | 달성 기준 |
|------|------|----------|----------|
| 객체 관리 용량 | 1M+ 객체 | SQLite + ChromaDB 부하 테스트 | 검색 성능 유지 |
| 동시 사용자 | 10+ 동시 요청 | 부하 테스트 (locust) | 응답 시간 < 5초 유지 |

---

## 13. 마일스톤 (Milestones)

| 단계 | 내용 | 예상일 | 담당자 | 상태 |
|------|------|--------|--------|------|
| M1.1 | Object Registry 스키마 완성 및 배포 | 2025-01-05 | Backend Team | ✅ 완료 |
| M1.2 | Gateway ↔ Registry 통합 (auto_label) | 2025-01-08 | Integration Team | ⚠️ 진행 중 |
| M1.3 | 배치 등록 동시성 이슈 해결 | 2025-01-10 | Backend Team | ⬜ 대기 |
| M1.4 | Phase 1 통합 테스트 (e2e) | 2025-01-12 | QA Team | ⬜ 대기 |
| M1.5 | Phase 1 배포 및 문서화 | 2025-01-15 | All | ⬜ 대기 |
| M2.1 | YOLO/COCO 내보내기 구현 | 2025-01-22 | Backend Team | ⬜ 계획 |
| M2.2 | 품질 검수 워크플로우 API | 2025-01-29 | Backend Team | ⬜ 계획 |
| M2.3 | 유사 객체 검색 UI (기본) | 2025-02-05 | Frontend Team | ⬜ 계획 |
| M2.4 | Phase 2 배포 | 2025-02-12 | All | ⬜ 계획 |
| M3.1 | 비디오 프레임 추출 파이프라인 | 2025-02-26 | Backend Team | ⬜ 계획 |
| M3.2 | Re-ID 기반 트랙 생성 | 2025-03-05 | AI Team | ⬜ 계획 |
| M3.3 | Phase 3 배포 | 2025-03-12 | All | ⬜ 계획 |

---

## 14. 위험 관리 (Risk Management)

| 위험 | 확률 | 영향도 | 완화 전략 | 담당자 |
|------|------|--------|----------|--------|
| GPU 메모리 부족 (모델 로딩 실패) | 중 | 높음 | 모델 다운그레이드 옵션 제공 (Florence-2-base, SAM2 Small) | AI Team |
| SQLite 동시성 한계 (배치 등록 실패) | 높음 | 중 | WAL 모드 + retry 로직, Postgres 마이그레이션 준비 | Backend Team |
| ChromaDB 성능 저하 (10M+ 벡터) | 낮음 | 중 | Qdrant 마이그레이션 계획, 샤딩 전략 | Backend Team |
| Foundation Model API 변경 | 중 | 중 | 버전 고정 (transformers, sam2), 추상화 레이어 유지 | AI Team |
| 자동 라벨링 정확도 부족 | 중 | 높음 | 신뢰도 기반 필터링, 수동 검증 워크플로우 제공 | Product Team |
| 데이터셋 내보내기 포맷 호환성 | 낮음 | 중 | YOLO/COCO 공식 검증 도구 테스트 | QA Team |
| 비디오 추적 정확도 부족 | 중 | 중 | IoU threshold 조정, Re-ID 모델 개선 | AI Team |

---

## 15. Phase별 로드맵 요약 (Roadmap Summary)

### Phase 1: 코어 기능 (현재 - 2025-01-15) ✅ 진행 중
- [x] 마이크로서비스 아키텍처 구축
- [x] GPU 서비스 (Florence-2, SAM2, DINOv2) 통합
- [x] auto_label 파이프라인
- [x] Object Registry 설계 및 기본 구현
- [ ] 배치 등록 동시성 문제 해결
- [ ] gateway ↔ object-registry 완전 통합

### Phase 2: 데이터 관리 (2025-01-22 - 2025-02-12)
- [ ] YOLO/COCO 내보내기 구현
- [ ] 품질 검수 워크플로우
- [ ] 라벨 수정/검증 UI (기본)
- [ ] 임베딩 기반 유사 객체 검색

### Phase 3: 비디오 & 추적 (2025-02-26 - 2025-03-12)
- [ ] 비디오 프레임 추출 및 처리
- [ ] IoU 기반 객체 연결 (단일 비디오)
- [ ] Re-ID 기반 트랙 관리
- [ ] 트랙 시각화 및 수정

### Phase 4: 학습 & 평가 (2025-03-19 - 2025-04-30)
- [ ] YOLO 학습 파이프라인 완성
- [ ] MLflow 실험 추적 통합
- [ ] 모델 평가 메트릭 (mAP, confusion matrix)
- [ ] Active Learning 루프

### Phase 5: VLA 확장 (2025-05-01 - 2025-06-30)
- [ ] Affordance 테이블 추가
- [ ] 행동 시퀀스 기록
- [ ] 로봇 행동 데이터 연동
- [ ] 멀티모달 임베딩 통합

### Phase 6: 프로덕션 배포 (2025-07-01 - 2025-08-31)
- [ ] K8s 매니페스트 작성
- [ ] Postgres + Qdrant 마이그레이션
- [ ] gRPC 서비스 간 통신
- [ ] 메시지 큐 기반 비동기 처리
- [ ] 모니터링/알림 구성

---

## 16. 부록 (Appendix)

### 16.1 용어 정의 (Glossary)

| 용어 | 정의 |
|------|------|
| **Grounding** | 텍스트 프롬프트로 이미지 내 특정 객체 위치 찾기 (Florence-2) |
| **Segmentation** | 픽셀 단위로 객체 영역 분할 (SAM2) |
| **Embedding** | 고차원 시각 특징을 저차원 벡터로 압축 (DINOv2 768d) |
| **Re-ID** | 서로 다른 프레임/카메라에서 동일 객체 재식별 |
| **Affordance** | 객체의 행동 가능성 (예: graspable, pushable) |
| **VLA** | Vision-Language-Action, 비전-언어-행동 통합 모델 |
| **YOLO Format** | `<class_id> <x_center> <y_center> <width> <height>` (normalized) |
| **COCO Format** | JSON 기반 표준 데이터셋 포맷 (categories, images, annotations) |
| **ChromaDB** | 오픈소스 임베딩 데이터베이스 (벡터 유사도 검색) |
| **WAL Mode** | Write-Ahead Logging, SQLite 동시성 향상 모드 |

### 16.2 참고 자료 (References)

- [Florence-2 Paper](https://arxiv.org/abs/2311.06242)
- [SAM2 GitHub](https://github.com/facebookresearch/segment-anything-2)
- [DINOv2 Documentation](https://github.com/facebookresearch/dinov2)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [YOLO Format Specification](https://docs.ultralytics.com/datasets/detect/)
- [COCO Format Specification](https://cocodataset.org/#format-data)

### 16.3 관련 문서 (Related Documents)

- `/home/coffin/dev/AgenticLabeling/docs/PROJECT_SPEC.md` - 프로젝트 전체 사양
- `/home/coffin/dev/AgenticLabeling/docs/plans/2024-12-29-object-registry-design.md` - Object Registry 상세 설계
- `/home/coffin/dev/AgenticLabeling/services/object-registry/app/schema.sql` - SQLite 스키마
- `/home/coffin/dev/AgenticLabeling/docker-compose.yml` - 로컬 개발 환경 설정

---

## 변경 이력 (Change Log)

| 버전 | 날짜 | 작성자 | 변경 내용 |
|------|------|--------|----------|
| v1.0 | 2025-12-29 | Product Manager | 최초 작성 (Phase 1 기준) |

---

**문서 종료**
