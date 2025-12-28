# AgenticLabeling - Project Specification

## 1. 프로젝트 개요

### 1.1 비전
AI 기반 자동 라벨링 시스템으로, 이미지/비디오/텍스트 입력에서 객체를 감지, 분할, 분류하고 이를 체계적으로 관리하여 학습 데이터셋을 구축하는 플랫폼.

### 1.2 핵심 목표
1. **자동 라벨링**: Florence-2, SAM2, DINOv2를 활용한 고품질 자동 라벨링
2. **객체 중심 관리**: 개별 객체 단위의 메타데이터, 임베딩, 추적 관리
3. **데이터셋 구축**: YOLO/COCO 포맷 내보내기, 품질 검수/정제
4. **확장성**: VLA(Vision-Language-Action)로의 확장 가능한 아키텍처

### 1.3 사용 시나리오
```
[입력]                    [처리]                      [출력]
이미지/비디오/텍스트  →  감지/분할/분류/추적  →  라벨링된 데이터셋
                              ↓
                      Object Registry에 저장
                              ↓
                      검색/필터링/검수/내보내기
```

---

## 2. 시스템 아키텍처

### 2.1 마이크로서비스 구성

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
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│   Labeling    │         │    Training   │         │  Evaluation   │
│    Agent      │         │     Agent     │         │    Agent      │
│   CPU:8004    │         │   GPU:8005    │         │   GPU:8007    │
└───────────────┘         └───────────────┘         └───────────────┘
        │                           │
        ▼                           ▼
┌───────────────┐         ┌───────────────┐
│ Data Manager  │         │    MLflow     │
│   CPU:8006    │         │     :5000     │
└───────────────┘         └───────────────┘
```

### 2.2 서비스 상세

| 서비스 | 포트 | GPU | 역할 |
|--------|------|-----|------|
| gateway | 8000 | ❌ | API 라우팅, 파이프라인 조율 |
| detection-agent | 8001 | ✅ | Florence-2 객체 감지 (grounding) |
| segmentation-agent | 8002 | ✅ | SAM2 인스턴스 세그멘테이션 |
| classification-agent | 8003 | ✅ | DINOv2 few-shot 분류 |
| labeling-agent | 8004 | ❌ | 라벨 CRUD (레거시, Registry로 통합 예정) |
| training-agent | 8005 | ✅ | YOLO 학습, MLflow 추적 |
| data-manager | 8006 | ❌ | 데이터셋 관리, YOLO/COCO 내보내기 |
| evaluation-agent | 8007 | ✅ | 모델 평가 (mAP, confusion matrix) |
| object-registry | 8010 | ❌ | 객체 중심 데이터 관리 (핵심) |
| redis | 6379 | ❌ | 캐시, 작업 큐 |
| mlflow | 5000 | ❌ | 실험 추적, 모델 레지스트리 |

---

## 3. 핵심 데이터 모델

### 3.1 Object Registry 스키마

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Sources   │────<│   Objects   │>────│  Categories │
│  (이미지/   │     │  (감지된    │     │   (클래스   │
│   비디오)   │     │    객체)    │     │   온톨로지) │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
       ┌───────────┐ ┌───────────┐ ┌───────────┐
       │  Tracks   │ │ Datasets  │ │ Embeddings│
       │ (Re-ID용) │ │ (내보내기)│ │ (ChromaDB)│
       └───────────┘ └───────────┘ └───────────┘
```

### 3.2 Object 핵심 필드

```python
Object = {
    "object_id": str,           # 고유 식별자
    "source_id": str,           # 원본 소스 참조
    "category_id": int,         # 클래스 참조

    # Geometry
    "bbox": [x, y, w, h],       # 바운딩박스
    "mask_path": str,           # 마스크 파일 경로

    # Detection
    "confidence": float,        # 감지 신뢰도
    "detection_model": str,     # 사용된 모델

    # Quality
    "is_validated": bool,       # 검수 여부
    "quality_score": float,     # 품질 점수

    # Embedding (ChromaDB)
    "visual_embedding": [768d], # DINOv2 시각 특징
}
```

---

## 4. 핵심 파이프라인

### 4.1 자동 라벨링 파이프라인

```
POST /auto_label
     │
     ▼
┌─────────────────┐
│  이미지 입력    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Detection Agent │────>│  boxes, labels  │
│   (Florence-2)  │     │  confidence     │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│Segmentation Agent────>│     masks       │
│     (SAM2)      │     │     (PNG)       │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Object Registry │────>│   저장 완료     │
│   (저장/인덱싱)  │     │   object_ids    │
└─────────────────┘     └─────────────────┘
```

### 4.2 유사 객체 검색 파이프라인

```
POST /objects/search/embedding
     │
     ▼
┌─────────────────┐
│  쿼리 이미지    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│Classification   │────>│   embedding     │
│   (DINOv2)      │     │    [768d]       │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│    ChromaDB     │────>│  similar objs   │
│  벡터 유사도    │     │  + similarity   │
└─────────────────┘     └─────────────────┘
```

### 4.3 데이터셋 내보내기 파이프라인

```
POST /datasets/{id}/build → POST /datasets/{id}/export
     │                            │
     ▼                            ▼
┌─────────────────┐     ┌─────────────────┐
│ 필터 조건 적용  │     │  YOLO/COCO     │
│ - categories    │     │  포맷 변환      │
│ - min_confidence│     │                 │
│ - is_validated  │     │  images/        │
└────────┬────────┘     │  labels/        │
         │              │  data.yaml      │
         ▼              └─────────────────┘
┌─────────────────┐
│  train/val/test │
│    스플릿       │
└─────────────────┘
```

---

## 5. 기술 스택

### 5.1 AI/ML 모델

| 모델 | 용도 | 입력 | 출력 |
|------|------|------|------|
| Florence-2-large | 객체 감지 (grounding) | 이미지 + 텍스트 | bbox + labels |
| SAM2 (Hiera B+) | 인스턴스 세그멘테이션 | 이미지 + bbox | masks |
| DINOv2-base | 시각 특징 추출 | 이미지 | embedding [768d] |
| CLIP ViT-B/32 | 텍스트-이미지 정렬 | 이미지/텍스트 | embedding [512d] |

### 5.2 백엔드

| 구성요소 | 기술 | 용도 |
|----------|------|------|
| API | FastAPI | REST API 서버 |
| DB (메타) | SQLite (→ Postgres) | 메타데이터, 관계 |
| DB (벡터) | ChromaDB (→ Qdrant) | 임베딩 검색 |
| 캐시 | Redis | 결과 캐싱, 작업 큐 |
| 실험추적 | MLflow | 학습 메트릭, 모델 저장 |

### 5.3 인프라

| 환경 | 구성 |
|------|------|
| 로컬 개발 | Docker Compose, SQLite, ChromaDB embedded |
| 프로덕션 (K8s) | Deployment/StatefulSet, Postgres, Qdrant cluster |

---

## 6. 개발 로드맵

### Phase 1: 코어 기능 (현재) ✅
- [x] 마이크로서비스 아키텍처 구축
- [x] GPU 서비스 (Florence-2, SAM2, DINOv2) 통합
- [x] auto_label 파이프라인
- [x] Object Registry 설계 및 기본 구현
- [ ] 배치 등록 동시성 문제 해결
- [ ] gateway ↔ object-registry 통합

### Phase 2: 데이터 관리
- [ ] YOLO/COCO 내보내기 구현
- [ ] 품질 검수 워크플로우
- [ ] 라벨 수정/검증 UI (기본)
- [ ] 임베딩 기반 유사 객체 검색

### Phase 3: 비디오 & 추적
- [ ] 비디오 프레임 추출 및 처리
- [ ] IoU 기반 객체 연결 (단일 비디오)
- [ ] Re-ID 기반 트랙 관리
- [ ] 트랙 시각화 및 수정

### Phase 4: 학습 & 평가
- [ ] YOLO 학습 파이프라인 완성
- [ ] MLflow 실험 추적 통합
- [ ] 모델 평가 메트릭 (mAP, confusion matrix)
- [ ] Active Learning 루프

### Phase 5: VLA 확장
- [ ] Affordance 테이블 추가
- [ ] 행동 시퀀스 기록
- [ ] 로봇 행동 데이터 연동
- [ ] 멀티모달 임베딩 통합

### Phase 6: 프로덕션 배포
- [ ] K8s 매니페스트 작성
- [ ] Postgres + Qdrant 마이그레이션
- [ ] gRPC 서비스 간 통신
- [ ] 메시지 큐 기반 비동기 처리
- [ ] 모니터링/알림 구성

---

## 7. API 명세 (핵심)

### 7.1 Gateway API

```yaml
# 자동 라벨링
POST /auto_label:
  input: image, project_id, image_id, classes, confidence
  output: object_ids, boxes, labels, masks_count

# 감지 + 세그멘테이션
POST /detect_and_segment:
  input: image, classes, confidence
  output: detections, segmentation
```

### 7.2 Object Registry API

```yaml
# 소스 관리
POST /sources: 소스 등록
GET /sources/{id}: 소스 조회

# 객체 관리
POST /objects: 객체 등록
POST /objects/batch: 배치 등록
GET /objects: 필터 검색
GET /objects/{id}: 객체 조회
PATCH /objects/{id}: 객체 수정
DELETE /objects/{id}: 객체 삭제
GET /objects/{id}/mask: 마스크 이미지

# 임베딩 검색
POST /objects/search/embedding: 유사 객체 검색

# 트랙 관리
POST /tracks: 트랙 생성
GET /tracks/{id}: 트랙 조회

# 데이터셋
POST /datasets: 데이터셋 생성
POST /datasets/{id}/build: 데이터셋 구축
POST /datasets/{id}/export: 데이터셋 내보내기

# 유틸리티
GET /stats: 통계
GET /categories: 카테고리 목록
```

---

## 8. 설정 및 환경변수

```bash
# 공통
DATA_DIR=/data                    # 데이터 저장 경로

# GPU 서비스
CUDA_VISIBLE_DEVICES=0            # GPU 선택
TORCH_DTYPE=float32               # 정밀도 (float32/float16)

# Object Registry
REGISTRY_DB_PATH=/data/registry/registry.db
CHROMA_PATH=/data/registry/chroma

# Gateway
DETECTION_URL=http://detection-agent:8001
SEGMENTATION_URL=http://segmentation-agent:8002
CLASSIFICATION_URL=http://classification-agent:8003
REGISTRY_URL=http://object-registry:8010

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
```

---

## 9. 디렉토리 구조

```
AgenticLabeling/
├── docs/
│   ├── PROJECT_SPEC.md          # 이 문서
│   └── plans/                   # 설계 문서
│
├── services/
│   ├── gateway/                 # API 게이트웨이
│   ├── detection-agent/         # Florence-2
│   ├── segmentation-agent/      # SAM2
│   ├── classification-agent/    # DINOv2
│   ├── labeling-agent/          # 라벨 관리 (레거시)
│   ├── training-agent/          # YOLO 학습
│   ├── data-manager/            # 데이터셋 관리
│   ├── evaluation-agent/        # 모델 평가
│   └── object-registry/         # 객체 레지스트리
│
├── data/
│   ├── registry/                # Object Registry 저장소
│   │   ├── registry.db          # SQLite
│   │   ├── chroma/              # ChromaDB
│   │   └── masks/               # 마스크 파일
│   ├── sources/                 # 원본 이미지/비디오
│   └── exports/                 # 내보낸 데이터셋
│
├── docker-compose.yml           # 로컬 개발 환경
├── k8s/                         # Kubernetes 매니페스트 (예정)
└── tests/                       # 테스트
```

---

## 10. 품질 기준

### 10.1 성능 목표

| 메트릭 | 목표 |
|--------|------|
| 단일 이미지 처리 | < 3초 (detect + segment) |
| 객체 검색 응답 | < 100ms (1M 객체 기준) |
| 배치 처리량 | > 100 이미지/분 |

### 10.2 데이터 품질

| 메트릭 | 설명 |
|--------|------|
| confidence threshold | 기본 0.5, 조정 가능 |
| 검증율 | 주요 클래스 90% 이상 수동 검증 권장 |
| 중복 제거 | IoU > 0.9 객체 자동 병합 |

---

## 11. 참고 자료

- [Florence-2](https://huggingface.co/microsoft/Florence-2-large)
- [SAM2](https://github.com/facebookresearch/segment-anything-2)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [ChromaDB](https://docs.trychroma.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
