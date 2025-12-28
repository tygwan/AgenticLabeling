# AgenticLabeling 시스템 아키텍처 기술 설계서

## 메타데이터
- **작성자**: System Architect
- **작성일**: 2025-12-29
- **버전**: v1.0
- **관련 PRD**: [PROJECT_SPEC.md](../PROJECT_SPEC.md)
- **상태**: Review

---

## 1. 개요 (Overview)

### 1.1 목적
AgenticLabeling은 AI 기반 자동 라벨링 시스템으로, 이미지/비디오에서 객체를 감지, 분할, 분류하고 객체 중심으로 데이터를 관리하여 고품질 학습 데이터셋을 구축하는 마이크로서비스 플랫폼입니다. 본 문서는 시스템 아키텍처, 데이터 모델, API 설계 및 구현 상세를 기술합니다.

### 1.2 핵심 특징
- **멀티모델 파이프라인**: Florence-2 (detection) → SAM2 (segmentation) → DINOv2 (classification)
- **객체 중심 관리**: 개별 객체 단위 메타데이터, 임베딩, 트랙 관리
- **하이브리드 스토리지**: SQLite (메타데이터) + ChromaDB (벡터 검색)
- **마이그레이션 준비**: Postgres + Qdrant로 확장 가능한 설계

### 1.3 용어 정의
| 용어 | 정의 |
|------|------|
| Source | 원본 이미지/비디오 파일 |
| Object | 감지된 개별 객체 인스턴스 (bbox, mask, embedding 포함) |
| Track | 비디오 프레임 간 동일 객체의 시퀀스 (Re-ID) |
| Category | 객체 클래스 (예: person, car, dog) |
| Embedding | DINOv2 768차원 시각 특징 벡터 |
| Dataset | 내보내기용 객체 컬렉션 (train/val/test 스플릿) |

---

## 2. 아키텍처 (Architecture)

### 2.1 시스템 구조

```
                        ┌──────────────────────────────────┐
                        │      Gateway (CPU:8000)          │
                        │  - API 라우팅 & 파이프라인 조율   │
                        │  - 헬스체크 프록시                │
                        └─────────────┬────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  Detection      │     │  Segmentation   │     │ Classification  │
    │  (GPU:8001)     │     │  (GPU:8002)     │     │  (GPU:8003)     │
    │  Florence-2     │     │  SAM2 Hiera B+  │     │  DINOv2 base    │
    │  large (0.7B)   │     │  prompt-based   │     │  768d features  │
    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │     Object Registry (CPU:8010)               │
              │                                              │
              │  SQLite DB          ChromaDB                 │
              │  ├─ sources         ├─ objects_dinov2_v1     │
              │  ├─ objects         └─ (HNSW cosine index)   │
              │  ├─ categories                               │
              │  ├─ tracks          Masks Storage            │
              │  ├─ datasets        └─ sharded PNG files     │
              │  └─ embedding_outbox                         │
              └──────────────────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   Labeling      │    │   Training      │    │  Evaluation     │
    │  (CPU:8004)     │    │   (GPU:8005)    │    │  (GPU:8007)     │
    │  CRUD (legacy)  │    │   YOLOv8/v11    │    │  mAP, matrix    │
    └─────────────────┘    └────────┬────────┘    └─────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  MLflow :5000   │
                           │  실험 추적       │
                           └─────────────────┘

    ┌─────────────────┐
    │  Redis :6379    │  (캐시, 작업큐)
    └─────────────────┘
```

### 2.2 컴포넌트 설명

| 컴포넌트 | 역할 | 기술 스택 | 리소스 |
|---------|------|----------|--------|
| Gateway | API 라우팅, 파이프라인 조율 | FastAPI, httpx | CPU |
| Detection Agent | 객체 감지 (grounding) | Florence-2-large (0.7B), PyTorch | GPU (VRAM ~4GB) |
| Segmentation Agent | 인스턴스 세그멘테이션 | SAM2 Hiera B+ (0.1B), PyTorch | GPU (VRAM ~2GB) |
| Classification Agent | Few-shot 분류, 임베딩 추출 | DINOv2-base (0.3B), PyTorch | GPU (VRAM ~2GB) |
| Object Registry | 객체 중심 데이터 관리, 벡터 검색 | SQLite, ChromaDB, FastAPI | CPU, Disk I/O |
| Labeling Agent | 라벨 CRUD (레거시, 통합 예정) | FastAPI, JSON 파일 | CPU |
| Training Agent | YOLO 모델 학습 | Ultralytics YOLOv8/v11, MLflow | GPU (VRAM ~6GB) |
| Evaluation Agent | 모델 평가 (mAP, confusion matrix) | Ultralytics, PyTorch | GPU |
| Data Manager | YOLO/COCO 포맷 내보내기 | FastAPI, Python | CPU |
| Redis | 결과 캐싱, 작업 큐 | Redis 7 | CPU, Memory |
| MLflow | 실험 추적, 모델 레지스트리 | MLflow 2.9 | CPU |

### 2.3 데이터 흐름

#### 자동 라벨링 파이프라인 (POST /auto_label)
1. **요청 수신**: Gateway가 이미지, project_id, classes, confidence 수신
2. **객체 감지**: Detection Agent (Florence-2)가 bbox + labels 반환
3. **인스턴스 세그멘테이션**: Segmentation Agent (SAM2)가 bbox 기반 mask 생성
4. **라벨 저장**: Labeling Agent에 boxes, classes, masks 저장 (optional)
5. **소스 등록**: Object Registry에 source (이미지) 등록
6. **배치 객체 등록**: 모든 감지된 객체를 batch API로 등록
   - SQLite에 메타데이터 저장 (bbox, confidence, category)
   - 마스크 PNG 파일 저장 (sharded directory)
   - 임베딩 outbox에 큐잉 후 ChromaDB 동기화
7. **응답 반환**: object_ids, source_id, 감지 통계 반환

#### 유사 객체 검색 파이프라인 (POST /objects/search/embedding)
1. **임베딩 추출**: Classification Agent (DINOv2)가 쿼리 이미지의 768d 임베딩 생성
2. **벡터 검색**: Object Registry가 ChromaDB에서 cosine similarity 기반 top-k 검색
3. **메타데이터 조인**: 검색된 object_id로 SQLite에서 전체 메타데이터 조회
4. **필터 적용**: category, min_confidence 필터 적용
5. **결과 반환**: 유사 객체 리스트 + similarity 점수

#### 데이터셋 내보내기 파이프라인
1. **데이터셋 생성**: filter_config, split_config로 데이터셋 정의
2. **빌드**: 필터 조건 (categories, min_confidence, is_validated)으로 객체 선택 → train/val/test 스플릿
3. **내보내기**: Data Manager가 YOLO/COCO 포맷으로 변환 (images/, labels/, data.yaml)

---

## 3. 데이터 모델 (Data Model)

### 3.1 엔티티 정의

#### Sources (원본 소스)
| 필드 | 타입 | 설명 | 제약조건 |
|------|------|------|----------|
| source_id | TEXT | 고유 식별자 (src_{12자}) | PK |
| source_type | TEXT | 소스 타입 (image/video/text) | NOT NULL, CHECK |
| file_path | TEXT | 파일 경로 | - |
| file_name | TEXT | 파일명 | - |
| width | INTEGER | 이미지 너비 (px) | - |
| height | INTEGER | 이미지 높이 (px) | - |
| frame_count | INTEGER | 비디오 프레임 수 | - |
| fps | REAL | 비디오 FPS | - |
| metadata | JSON | 커스텀 메타데이터 | - |
| created_at | TIMESTAMP | 생성 시각 | DEFAULT NOW |

**용도**: 이미지/비디오 원본 파일 정보 관리. 모든 객체는 source_id로 연결.

#### Objects (핵심 테이블)
| 필드 | 타입 | 설명 | 제약조건 |
|------|------|------|----------|
| object_id | TEXT | 고유 식별자 (obj_{12자}) | PK |
| source_id | TEXT | 원본 소스 참조 | FK, NOT NULL |
| category_id | INTEGER | 카테고리 참조 | FK |
| frame_id | TEXT | 프레임 참조 (비디오용) | FK, nullable |
| project_id | TEXT | 프로젝트 참조 | - |
| bbox_x, bbox_y | REAL | 바운딩박스 좌상단 좌표 | NOT NULL |
| bbox_w, bbox_h | REAL | 바운딩박스 너비/높이 | NOT NULL |
| mask_path | TEXT | 마스크 파일 상대경로 | - |
| polygon | JSON | 폴리곤 좌표 (선택) | - |
| area | REAL | 객체 면적 (px^2) | - |
| confidence | REAL | 감지 신뢰도 (0.0-1.0) | - |
| detection_model | TEXT | 사용 모델명 (florence2) | - |
| classification_model | TEXT | 분류 모델명 | - |
| classification_confidence | REAL | 분류 신뢰도 | - |
| is_validated | BOOLEAN | 검수 여부 | DEFAULT FALSE |
| validated_by | TEXT | 검수자 ID | - |
| validated_at | TIMESTAMP | 검수 시각 | - |
| quality_score | REAL | 품질 점수 (0.0-1.0) | - |
| is_occluded | BOOLEAN | 가림 여부 | DEFAULT FALSE |
| is_truncated | BOOLEAN | 잘림 여부 | DEFAULT FALSE |
| is_difficult | BOOLEAN | 어려움 플래그 | DEFAULT FALSE |
| created_at | TIMESTAMP | 생성 시각 | DEFAULT NOW |
| updated_at | TIMESTAMP | 수정 시각 | DEFAULT NOW, Trigger |

**용도**: 감지된 개별 객체 인스턴스. 메타데이터, 기하 정보, 검증 상태 관리.

**인덱스**: source_id, category_id, project_id, confidence, is_validated, created_at

#### Categories (클래스 온톨로지)
| 필드 | 타입 | 설명 | 제약조건 |
|------|------|------|----------|
| category_id | INTEGER | 고유 식별자 | PK, AUTOINCREMENT |
| name | TEXT | 클래스명 (예: person, car) | UNIQUE, NOT NULL |
| supercategory | TEXT | 상위 카테고리 (예: vehicle) | - |
| synonyms | JSON | 동의어 리스트 | - |
| color | TEXT | 시각화 색상 (hex) | DEFAULT #808080 |
| description | TEXT | 설명 | - |
| created_at | TIMESTAMP | 생성 시각 | DEFAULT NOW |

**용도**: 클래스 정의 및 온톨로지 관리. YOLO class ID와 매핑.

#### Tracks (Re-ID 추적)
| 필드 | 타입 | 설명 | 제약조건 |
|------|------|------|----------|
| track_id | TEXT | 고유 식별자 (trk_{12자}) | PK |
| source_id | TEXT | 비디오 소스 참조 | FK, NOT NULL |
| category_id | INTEGER | 카테고리 참조 | FK |
| start_frame_idx | INTEGER | 시작 프레임 인덱스 | - |
| end_frame_idx | INTEGER | 종료 프레임 인덱스 | - |
| object_count | INTEGER | 객체 수 (자동 업데이트) | DEFAULT 0, Trigger |
| avg_confidence | REAL | 평균 신뢰도 | - |
| metadata | JSON | 커스텀 메타데이터 | - |
| created_at | TIMESTAMP | 생성 시각 | DEFAULT NOW |

**용도**: 비디오에서 동일 객체의 시간 시퀀스 관리. IoU/임베딩 기반 연결.

**연결 테이블**: track_objects (track_id, object_id, sequence_idx)

#### Datasets (내보내기용)
| 필드 | 타입 | 설명 | 제약조건 |
|------|------|------|----------|
| dataset_id | TEXT | 고유 식별자 (ds_{12자}) | PK |
| name | TEXT | 데이터셋명 | NOT NULL |
| format | TEXT | 포맷 (yolo/coco) | DEFAULT yolo |
| version | TEXT | 버전 | DEFAULT 1.0 |
| split_config | JSON | 스플릿 비율 (train/val/test) | DEFAULT 0.8/0.1/0.1 |
| filter_config | JSON | 필터 조건 (categories, min_confidence) | - |
| category_mapping | JSON | 클래스 ID 매핑 | - |
| object_count | INTEGER | 객체 수 | DEFAULT 0 |
| export_path | TEXT | 내보내기 경로 | - |
| created_at | TIMESTAMP | 생성 시각 | DEFAULT NOW |
| exported_at | TIMESTAMP | 내보내기 시각 | - |

**용도**: 데이터셋 정의 및 내보내기 이력 관리.

**연결 테이블**: dataset_objects (dataset_id, object_id, split)

#### Embedding Outbox (동기화 큐)
| 필드 | 타입 | 설명 | 제약조건 |
|------|------|------|----------|
| id | INTEGER | 자동 증가 ID | PK, AUTOINCREMENT |
| object_id | TEXT | 객체 참조 | FK, NOT NULL |
| version_id | TEXT | 임베딩 버전 (dinov2_base_v1) | FK, NOT NULL |
| operation | TEXT | 작업 유형 (upsert/delete) | CHECK |
| embedding | BLOB | 임베딩 바이트 (768x4 bytes) | - |
| status | TEXT | 상태 (pending/synced/failed) | DEFAULT pending, CHECK |
| retry_count | INTEGER | 재시도 횟수 | DEFAULT 0 |
| error_message | TEXT | 오류 메시지 | - |
| created_at | TIMESTAMP | 생성 시각 | DEFAULT NOW |
| synced_at | TIMESTAMP | 동기화 시각 | - |

**용도**: SQLite → ChromaDB 임베딩 동기화 트랜잭션 아웃박스 패턴.

**인덱스**: status, object_id

### 3.2 관계도

```
┌───────────┐
│  Sources  │───┐
└───────────┘   │
                │ 1:N
                ▼
           ┌─────────┐      N:1     ┌────────────┐
           │ Objects │◄──────────────│ Categories │
           └─────────┘               └────────────┘
                │
                │ N:M (track_objects)
                ▼
           ┌─────────┐
           │ Tracks  │
           └─────────┘

           ┌─────────┐
           │ Objects │
           └─────────┘
                │
                │ N:M (dataset_objects)
                ▼
           ┌──────────┐
           │ Datasets │
           └──────────┘

           ┌─────────┐      1:1      ┌────────────────┐
           │ Objects │──────────────▶│ Embedding      │
           └─────────┘               │ Outbox         │
                                     └────────────────┘

                                     ┌────────────────┐
                                     │ ChromaDB       │
                                     │ objects_dinov2 │
                                     │ (id=object_id) │
                                     └────────────────┘
```

**관계 설명**:
- Source 1 → N Objects: 한 이미지/비디오에 여러 객체
- Category 1 → N Objects: 한 클래스에 여러 객체 인스턴스
- Track N ↔ M Objects: 트랙은 여러 객체 시퀀스 (track_objects 조인 테이블)
- Dataset N ↔ M Objects: 데이터셋은 필터링된 객체 서브셋 (dataset_objects 조인 테이블)
- Object 1 → 1 Embedding: outbox를 거쳐 ChromaDB에 동기화

---

## 4. API 설계 (API Design)

### 4.1 엔드포인트 목록

#### Gateway API (8000)
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | /health | 게이트웨이 및 서비스 헬스체크 |
| POST | /detect | 객체 감지 (Detection Agent 프록시) |
| POST | /segment | 세그멘테이션 (Segmentation Agent 프록시) |
| POST | /detect_and_segment | 감지 + 세그멘테이션 파이프라인 |
| POST | /auto_label | 전체 자동 라벨링 파이프라인 |
| GET | /registry/stats | Registry 통계 (프록시) |
| GET | /registry/objects | 객체 검색 (프록시) |
| GET | /registry/categories | 카테고리 목록 (프록시) |

#### Object Registry API (8010)
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | /sources | 소스 등록 |
| GET | /sources/{id} | 소스 조회 |
| POST | /objects | 단일 객체 등록 |
| POST | /objects/batch | 배치 객체 등록 |
| GET | /objects | 필터 검색 (category, project_id, min_confidence) |
| GET | /objects/{id} | 객체 조회 |
| PATCH | /objects/{id} | 객체 수정 (is_validated, quality_score) |
| DELETE | /objects/{id} | 객체 삭제 |
| GET | /objects/{id}/mask | 마스크 이미지 반환 (PNG) |
| POST | /objects/search/embedding | 임베딩 벡터 유사도 검색 |
| POST | /tracks | 트랙 생성 |
| GET | /tracks/{id} | 트랙 조회 |
| POST | /datasets | 데이터셋 생성 |
| POST | /datasets/{id}/build | 데이터셋 빌드 (필터 적용 + 스플릿) |
| GET | /categories | 카테고리 목록 |
| POST | /categories | 카테고리 생성 |
| POST | /sync/embeddings | 대기 중 임베딩 동기화 |
| GET | /stats | 통계 (객체 수, 카테고리별 분포) |

#### Detection Agent API (8001)
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | /health | 헬스체크 |
| POST | /detect | 객체 감지 (image, classes, confidence) |
| POST | /unload | 모델 언로드 (GPU 메모리 해제) |

#### Segmentation Agent API (8002)
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | /health | 헬스체크 |
| POST | /segment | bbox 기반 세그멘테이션 |
| POST | /segment_points | 포인트 프롬프트 기반 세그멘테이션 |
| POST | /unload | 모델 언로드 |

#### Classification Agent API (8003)
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | /health | 헬스체크 |
| POST | /support_set/load | Few-shot support set 로드 |
| POST | /classify | 이미지 분류 (threshold, top_k) |
| POST | /classify_batch | 배치 분류 |
| POST | /unload | 모델 언로드 |

### 4.2 상세 명세

#### POST /auto_label (Gateway)
**설명**: 전체 자동 라벨링 파이프라인 (감지 → 세그멘테이션 → 저장 → 등록)

**Request** (multipart/form-data)
```
image: UploadFile (이미지 파일)
project_id: str (프로젝트 ID)
image_id: str (이미지 고유 ID)
classes: str (쉼표 구분 클래스명, 예: "person,car,dog")
confidence: float = 0.5 (최소 신뢰도)
save: bool = true (labeling-agent에 저장 여부)
register: bool = true (object-registry에 등록 여부)
```

**Response (200)**
```json
{
  "success": true,
  "data": {
    "image_id": "img_001",
    "detections": 3,
    "boxes": [[10, 20, 100, 150], ...],
    "labels": ["person", "car", "dog"],
    "masks_count": 3,
    "image_size": {"width": 1920, "height": 1080}
  },
  "saved": true,
  "registered": true,
  "registry": {
    "source_id": "src_a1b2c3d4e5f6",
    "object_ids": ["obj_x1y2z3", "obj_a4b5c6", "obj_d7e8f9"]
  }
}
```

**Error Codes**
| 코드 | 설명 |
|------|------|
| 400 | 잘못된 요청 (이미지 누락, classes 형식 오류) |
| 500 | 내부 서버 오류 (모델 추론 실패, DB 오류) |
| 503 | 서비스 연결 불가 (detection/segmentation agent down) |

**처리 흐름**:
1. Gateway가 이미지 바이트 읽기
2. Detection Agent 호출 → boxes, labels, scores 수신
3. 감지 결과 없으면 early return
4. Segmentation Agent 호출 (boxes 전달) → masks (base64) 수신
5. (Optional) Labeling Agent에 저장
6. Object Registry 호출:
   - Source 등록 → source_id
   - Batch Objects 등록 → object_ids
7. 성공 응답 반환

---

#### POST /objects/batch (Object Registry)
**설명**: 동일 소스의 여러 객체를 배치로 등록 (트랜잭션 최적화)

**Request**
```json
{
  "source_id": "src_a1b2c3d4e5f6",
  "project_id": "project_001",
  "objects": [
    {
      "category": "person",
      "bbox": [10, 20, 100, 150],
      "confidence": 0.95,
      "detection_model": "florence2",
      "mask_base64": "iVBORw0KGgoAAAANSUhEUg...",
      "embedding": [0.123, -0.456, ...]
    },
    {
      "category": "car",
      "bbox": [200, 300, 150, 100],
      "confidence": 0.88,
      "detection_model": "florence2",
      "mask_base64": "iVBORw0KGgoAAAANSUhEUg..."
    }
  ]
}
```

**Response (200)**
```json
{
  "success": true,
  "object_ids": ["obj_x1y2z3", "obj_a4b5c6"],
  "count": 2
}
```

**구현 로직** (/home/coffin/dev/AgenticLabeling/services/object-registry/app/registry.py:202-260)
```python
def register_objects_batch(source_id, objects_data, project_id):
    # 1. 사전에 마스크 저장 (filesystem, 트랜잭션 외부)
    # 2. 카테고리 ID 캐싱 (중복 쿼리 방지)
    # 3. 배치 INSERT (단일 트랜잭션)
    # 4. Embedding outbox 큐잉
    # 5. 커밋 후 ChromaDB 동기화 (비동기 가능)
```

**성능 최적화**:
- 카테고리 사전 조회 및 캐싱으로 N+1 쿼리 방지
- 마스크 파일 저장을 트랜잭션 전 처리
- 임베딩 동기화는 outbox 패턴으로 지연 처리

---

#### POST /objects/search/embedding (Object Registry)
**설명**: 쿼리 임베딩과 유사한 객체 검색 (cosine similarity)

**Request**
```json
{
  "embedding": [0.123, -0.456, ..., 0.789],  // 768차원 벡터
  "top_k": 10,
  "category": "person",  // optional
  "min_confidence": 0.7  // optional
}
```

**Response (200)**
```json
{
  "success": true,
  "data": [
    {
      "object_id": "obj_x1y2z3",
      "category_name": "person",
      "bbox_x": 10, "bbox_y": 20, "bbox_w": 100, "bbox_h": 150,
      "confidence": 0.95,
      "similarity": 0.92,  // cosine similarity
      "source_id": "src_a1b2c3",
      "is_validated": true
    },
    ...
  ],
  "count": 10
}
```

**구현 로직** (/home/coffin/dev/AgenticLabeling/services/object-registry/app/registry.py:378-417)
```python
def search_by_embedding(embedding, top_k, category, min_confidence):
    # 1. ChromaDB에서 벡터 검색 (HNSW 인덱스)
    # 2. 필터 조건 적용 (category, min_confidence)
    # 3. SQLite에서 object_id로 전체 메타데이터 조인
    # 4. 유사도 점수 추가 (1 - distance)
```

**벡터 검색 성능**:
- ChromaDB HNSW 인덱스로 ANN (Approximate Nearest Neighbor)
- 1M 객체 기준 < 100ms 응답 목표
- Metadata filter는 ChromaDB에서 사전 필터링

---

#### POST /datasets/{id}/build (Object Registry)
**설명**: 필터 조건으로 객체 선택 후 train/val/test 스플릿

**Request** (데이터셋 생성 시 설정된 filter_config, split_config 사용)
```json
{
  "filter_config": {
    "categories": ["person", "car"],
    "min_confidence": 0.7,
    "is_validated": true,
    "project_id": "project_001"
  },
  "split_config": {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
  }
}
```

**Response (200)**
```json
{
  "dataset_id": "ds_abc123",
  "object_count": 1000,
  "splits": {
    "train": 800,
    "val": 100,
    "test": 100
  }
}
```

**구현 로직** (/home/coffin/dev/AgenticLabeling/services/object-registry/app/registry.py:556-639)
```python
def build_dataset(dataset_id):
    # 1. 필터 조건으로 WHERE 절 구성
    # 2. RANDOM() 정렬로 객체 셔플
    # 3. 스플릿 비율로 object_ids 분할
    # 4. dataset_objects 테이블에 INSERT (split 컬럼)
    # 5. 통계 업데이트
```

---

## 5. 구현 상세 (Implementation Details)

### 5.1 핵심 로직

#### Florence-2 Detection (전체 구현: /home/coffin/dev/AgenticLabeling/services/detection-agent/app/detector.py)
```python
def detect(image_bytes, classes, confidence):
    # 1. Lazy loading (싱글톤 패턴)
    # 2. CAPTION_TO_PHRASE_GROUNDING 태스크
    # 3. 프롬프트: "A photo of person, and car, and dog."
    # 4. 결과 파싱 후 요청 클래스만 필터링
    # 5. bbox는 [x, y, w, h] 포맷
```

**특징**:
- Zero-shot grounding (사전 학습 없이 텍스트로 클래스 지정)
- Florence-2는 신뢰도 점수 미제공 → 1.0으로 고정
- float32 명시적 사용 (dtype mismatch 방지)

#### SAM2 Segmentation (전체 구현: /home/coffin/dev/AgenticLabeling/services/segmentation-agent/app/segmenter.py)
```python
def segment(image_bytes, boxes):
    # 1. Hiera B+ 모델 로드 (bbox prompt 지원)
    # 2. 각 bbox를 prompt로 변환
    # 3. SAM2 predict (고품질 mask 생성)
    # 4. Mask를 PNG 인코딩 후 base64 변환
    # 5. 마스크별 면적 계산
```

**특징**:
- Prompt-based segmentation (bbox/point/mask 프롬프트)
- 인스턴스별 독립 처리 (배치 처리 미지원)
- VRAM 효율적 (Hiera B+ 백본)

#### DINOv2 Embedding (전체 구현: /home/coffin/dev/AgenticLabeling/services/classification-agent/app/classifier.py)
```python
class FeatureExtractor:
    def extract_features(image):
        # 1. AutoImageProcessor로 전처리
        # 2. DINOv2 forward pass
        # 3. pooler_output 추출 (768차원)
        # 4. L2 정규화
        return features.cpu().numpy().flatten()
```

**Few-shot 분류**:
```python
class CosineSimilarityClassifier:
    def classify(image_bytes, threshold, top_k):
        # 1. 쿼리 임베딩 추출
        # 2. Support set과 코사인 유사도 계산
        # 3. 클래스별 평균 유사도
        # 4. threshold 적용 (미만이면 "Unknown")
        # 5. Margin 계산 (confidence level)
```

#### Hybrid Storage Sync (전체 구현: /home/coffin/dev/AgenticLabeling/services/object-registry/app/registry.py:666-743)
```python
def register_object(source_id, category, bbox, embedding):
    # SQLite 트랜잭션
    with conn:
        # 1. Objects 테이블 INSERT
        # 2. Embedding outbox에 큐잉 (BLOB)
        conn.commit()

    # ChromaDB 동기화 (트랜잭션 외부)
    _sync_embedding(object_id, embedding)
```

**Outbox 패턴**:
- SQLite와 ChromaDB 간 eventual consistency 보장
- 실패 시 재시도 메커니즘 (retry_count < 3)
- 배치 동기화 API (/sync/embeddings)로 대량 처리

### 5.2 의존성

#### Detection Agent
| 패키지 | 버전 | 용도 |
|--------|------|------|
| transformers | >=4.37.0 | Florence-2 모델 로드 |
| torch | >=2.0.0 | PyTorch 백엔드 |
| Pillow | >=10.0.0 | 이미지 처리 |
| fastapi | >=0.104.0 | REST API |

#### Segmentation Agent
| 패키지 | 버전 | 용도 |
|--------|------|------|
| sam2 | custom | SAM2 모델 (Facebook Research) |
| torch | >=2.0.0 | PyTorch 백엔드 |
| opencv-python | >=4.8.0 | 이미지 전처리 |

#### Classification Agent
| 패키지 | 버전 | 용도 |
|--------|------|------|
| transformers | >=4.37.0 | DINOv2 모델 |
| torch | >=2.0.0 | PyTorch 백엔드 |
| clip | custom | CLIP 모델 (선택) |
| torchvision | >=0.15.0 | ResNet (선택) |

#### Object Registry
| 패키지 | 버전 | 용도 |
|--------|------|------|
| fastapi | >=0.104.0 | REST API |
| chromadb | >=0.4.22 | 벡터 DB |
| numpy | >=1.24.0 | 임베딩 처리 |
| Pillow | >=10.0.0 | 마스크 이미지 처리 |

---

## 6. 보안 고려사항 (Security Considerations)

### 6.1 체크리스트

- [ ] **인증/인가**: 현재 미구현 (프로덕션 배포 전 JWT/OAuth2 추가 필요)
- [x] **입력값 검증**: Pydantic 모델로 타입 및 제약조건 검증
- [x] **SQL Injection 방지**: 파라미터화된 쿼리 사용 (sqlite3 placeholders)
- [x] **파일 업로드 보안**:
  - 파일 타입 검증 (PIL로 이미지 검증)
  - 파일명 해싱 (MD5 기반 sharding)
  - 업로드 크기 제한 (FastAPI 설정)
- [ ] **Rate Limiting**: 현재 미구현 (Redis 기반 rate limiter 추가 권장)
- [x] **CORS 설정**: Gateway에 CORSMiddleware 적용 (allow_origins=["*"] → 프로덕션에서 제한 필요)
- [ ] **민감 정보 암호화**:
  - 환경변수로 DB 경로 관리
  - ChromaDB 암호화 미지원 (Qdrant 마이그레이션 시 TLS 적용)
- [x] **에러 메시지 정제**: 스택 트레이스 노출 방지 (Exception 핸들러)

### 6.2 프로덕션 권장사항

1. **API Gateway 보안**:
   - JWT 토큰 기반 인증 (프로젝트별 권한 관리)
   - API 키 발급 (서비스 간 통신)
   - TLS/HTTPS 강제

2. **데이터 보안**:
   - SQLite → Postgres (row-level security)
   - ChromaDB → Qdrant (TLS, authentication)
   - 마스크 파일 암호화 (sensitive data)

3. **네트워크 보안**:
   - 내부 네트워크 격리 (Gateway만 외부 노출)
   - 서비스 간 mTLS (mutual TLS)

---

## 7. 확장성 및 성능 (Scalability & Performance)

### 7.1 현재 성능 목표

| 메트릭 | 목표 | 측정 방법 |
|--------|------|----------|
| 단일 이미지 처리 | < 3초 (detect + segment) | Gateway /auto_label 응답 시간 |
| 객체 검색 응답 | < 100ms (1M 객체 기준) | ChromaDB 벡터 검색 |
| 배치 처리량 | > 100 이미지/분 | GPU 병렬 처리 |
| 동시 요청 처리 | 10 concurrent requests | FastAPI async workers |

### 7.2 병목 지점 및 최적화

#### 병목 1: GPU 메모리 경합
**문제**: 여러 모델이 동일 GPU 공유 시 VRAM 부족
**해결책**:
- 모델별 전용 GPU 할당 (Docker Compose device mapping)
- 동적 모델 로딩/언로딩 (/unload 엔드포인트)
- Float16 정밀도 사용 (정확도 허용 범위 내)

#### 병목 2: SQLite 동시성
**문제**: 다중 스레드 쓰기 시 LOCKED 오류
**해결책** (현재 적용):
- WAL 모드 활성화 (readers 병렬 처리)
- busy_timeout=30초 설정
- 배치 INSERT로 트랜잭션 최소화

**향후 마이그레이션**: Postgres로 전환 시 MVCC로 완전 동시성 지원

#### 병목 3: ChromaDB 쓰기 성능
**문제**: 개별 객체마다 upsert 시 느림
**해결책** (현재 적용):
- Outbox 패턴으로 배치 동기화
- 트랜잭션 커밋 후 비동기 처리

**향후 마이그레이션**: Qdrant cluster로 분산 처리

#### 병목 4: 네트워크 레이턴시 (Gateway → Agents)
**문제**: HTTP 오버헤드 누적
**해결책**:
- 파이프라인 내 HTTP 재사용 (httpx.AsyncClient)
- 이미지 재전송 최소화 (메모리 캐싱)

**향후 개선**: gRPC 전환으로 직렬화 오버헤드 감소

### 7.3 수평 확장 전략

#### Kubernetes 배포 (Phase 6)

```yaml
# Detection Agent: HPA (Horizontal Pod Autoscaler)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: detection-agent
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: detection-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70
```

**스케일링 기준**:
- GPU 사용률 > 70% → 레플리카 증가
- 요청 큐 길이 > 100 → 레플리카 증가
- Redis 작업 큐 기반 오토스케일링

#### Object Registry: StatefulSet

```yaml
# Postgres + Qdrant StatefulSet
# 1. Postgres Primary-Replica 구성
# 2. Qdrant 3-node cluster (shard replication)
# 3. PVC (Persistent Volume Claim)로 데이터 영속성
```

### 7.4 캐싱 전략

#### Redis 캐시 계층
```python
# 감지 결과 캐싱 (이미지 해시 기반)
cache_key = f"detect:{image_hash}:{classes}:{confidence}"
if cached := redis.get(cache_key):
    return json.loads(cached)
# ... 모델 추론 ...
redis.setex(cache_key, ttl=3600, value=json.dumps(result))
```

**캐싱 대상**:
- Detection 결과 (이미지 해시 + 파라미터)
- Classification support set features
- 자주 조회되는 객체 메타데이터

---

## 8. 마이그레이션 계획 (Migration Plan)

### 8.1 SQLite → PostgreSQL

#### 목적
- 동시성 처리 향상 (MVCC)
- 트랜잭션 격리 수준 제어
- JSON 쿼리 성능 (JSONB)
- 복제 및 백업 자동화

#### 마이그레이션 단계

**1단계: 스키마 변환**
```sql
-- SQLite schema.sql → PostgreSQL DDL
-- 변경 사항:
-- - AUTOINCREMENT → SERIAL
-- - JSON → JSONB
-- - TIMESTAMP DEFAULT CURRENT_TIMESTAMP → TIMESTAMPTZ DEFAULT NOW()
-- - CHECK 제약조건 유지
```

**2단계: 데이터 마이그레이션**
```python
# 마이그레이션 스크립트: scripts/migrate_sqlite_to_postgres.py
import sqlite3
import psycopg2

def migrate():
    # 1. SQLite에서 테이블별 dump
    # 2. Postgres COPY 명령으로 bulk insert
    # 3. 시퀀스 재설정 (category_id 등)
    # 4. 외래키 제약조건 재생성
    # 5. 인덱스 재생성 (병렬 생성)
```

**3단계: Application 코드 수정**
```python
# Before (SQLite)
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

# After (PostgreSQL)
import psycopg2.extras
conn = psycopg2.connect(dsn)
cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
```

**4단계: 배포 전략**
- Blue-Green 배포
- Postgres replica에 데이터 동기화
- Read-only 모드로 검증
- 트래픽 전환 (DNS/Load Balancer)

#### 롤백 계획
- SQLite 백업 유지 (7일)
- 마이그레이션 중 dual-write (SQLite + Postgres)
- 검증 실패 시 SQLite로 롤백

### 8.2 ChromaDB → Qdrant

#### 목적
- 프로덕션 안정성 (클러스터 지원)
- 성능 향상 (Rust 기반, 더 빠른 HNSW)
- 멀티테넌시 지원
- 정확한 메타데이터 필터링

#### 마이그레이션 단계

**1단계: Qdrant 클러스터 구성**
```yaml
# docker-compose.qdrant.yml
services:
  qdrant-node1:
    image: qdrant/qdrant:v1.7.0
    ports:
      - "6333:6333"
    environment:
      - QDRANT_CLUSTER__ENABLED=true
      - QDRANT_CLUSTER__P2P__PORT=6335
```

**2단계: 컬렉션 생성**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://qdrant:6333")
client.create_collection(
    collection_name="objects_dinov2_v1",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    hnsw_config={"m": 16, "ef_construct": 100},
)
```

**3단계: 데이터 마이그레이션**
```python
# 마이그레이션 스크립트: scripts/migrate_chroma_to_qdrant.py
def migrate():
    # 1. ChromaDB에서 전체 벡터 export
    # 2. Qdrant batch upsert (1000개씩)
    # 3. 메타데이터 매핑 (payload 구조 변환)
    # 4. 인덱스 최적화 (optimize)
```

**4단계: Application 코드 수정**
```python
# Before (ChromaDB)
collection.query(
    query_embeddings=[embedding],
    n_results=10,
    where={"category": "person"}
)

# After (Qdrant)
from qdrant_client.models import Filter, FieldCondition, MatchValue

client.search(
    collection_name="objects_dinov2_v1",
    query_vector=embedding,
    limit=10,
    query_filter=Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="person"))]
    )
)
```

**5단계: 성능 벤치마크**
- Latency: p50, p95, p99 측정
- Throughput: QPS (Queries Per Second)
- Recall@10 정확도 검증

#### 롤백 계획
- ChromaDB 데이터 유지
- Feature flag로 스토리지 선택
- A/B 테스트로 점진적 전환

### 8.3 동시성 마이그레이션 전략

```python
# Dual-write 패턴 (마이그레이션 중)
class HybridVectorStore:
    def __init__(self):
        self.chroma = ChromaClient()
        self.qdrant = QdrantClient()
        self.use_qdrant = os.getenv("USE_QDRANT", "false") == "true"

    def upsert(self, object_id, embedding, metadata):
        if self.use_qdrant:
            self.qdrant.upsert(...)
        else:
            self.chroma.upsert(...)

        # Background sync for validation
        if os.getenv("DUAL_WRITE", "false") == "true":
            self._sync_to_other(object_id, embedding, metadata)
```

---

## 9. 테스트 전략 (Test Strategy)

### 9.1 단위 테스트

| 대상 | 테스트 케이스 | 상태 |
|------|-------------|------|
| Florence2Detector.detect | 빈 이미지 처리 | - [ ] |
| Florence2Detector.detect | 클래스 필터링 정확도 | - [ ] |
| SAM2Segmenter.segment | bbox 경계 케이스 (이미지 밖) | - [ ] |
| FeatureExtractor.extract_features | 임베딩 차원 검증 (768d) | - [ ] |
| CosineSimilarityClassifier.classify | Threshold 경계값 테스트 | - [ ] |
| ObjectRegistry.register_objects_batch | 트랜잭션 롤백 시나리오 | - [ ] |
| ObjectRegistry.search_by_embedding | 빈 결과 처리 | - [ ] |

**테스트 프레임워크**: pytest, pytest-asyncio

### 9.2 통합 테스트

| 시나리오 | 상태 |
|---------|------|
| Gateway → Detection → Segmentation 파이프라인 | - [ ] |
| auto_label 전체 플로우 (등록 포함) | - [ ] |
| 배치 객체 등록 후 검색 | - [ ] |
| 임베딩 동기화 (outbox → ChromaDB) | - [ ] |
| 데이터셋 빌드 및 스플릿 검증 | - [ ] |
| 트랙 생성 및 객체 연결 | - [ ] |

**테스트 환경**: Docker Compose 테스트 스택 (in-memory DB)

### 9.3 성능 테스트

```python
# tests/performance/test_throughput.py
import asyncio
from locust import HttpUser, task, between

class AgenticLabelingUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def auto_label(self):
        files = {"image": open("test_image.jpg", "rb")}
        data = {
            "project_id": "test",
            "image_id": f"img_{uuid.uuid4()}",
            "classes": "person,car",
            "confidence": 0.5
        }
        self.client.post("/auto_label", files=files, data=data)
```

**벤치마크 대상**:
- 동시 요청 처리 (10, 50, 100 users)
- GPU 메모리 사용률
- 객체 검색 레이턴시 (1K, 10K, 100K, 1M 객체)

---

## 10. 배포 계획 (Deployment Plan)

### 10.1 환경별 설정

| 환경 | 데이터베이스 | 벡터 DB | GPU | 복제 |
|------|------------|---------|-----|------|
| Dev | SQLite (로컬 파일) | ChromaDB (embedded) | 1x GPU | 단일 인스턴스 |
| Staging | Postgres (RDS) | Qdrant (3-node) | 2x GPU | 2 replicas |
| Prod | Postgres (Primary-Replica) | Qdrant cluster | 5x GPU | 5 replicas + HPA |

### 10.2 Docker Compose (로컬 개발)

**현재 구성** (/home/coffin/dev/AgenticLabeling/docker-compose.yml):
```yaml
services:
  gateway:
    ports: ["8000:8000"]
    depends_on: [detection-agent, segmentation-agent, object-registry, redis]

  detection-agent:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - model-cache:/root/.cache  # 모델 가중치 캐싱
```

**시작 순서**:
1. redis, mlflow (인프라)
2. object-registry (DB 초기화)
3. detection-agent, segmentation-agent, classification-agent (GPU 모델 로드)
4. gateway (헬스체크 대기)

### 10.3 Kubernetes 배포 (Phase 6)

#### Deployment 예시
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: detection-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: detection-agent
  template:
    spec:
      containers:
      - name: detection
        image: agenticlabeling/detection-agent:v1.0
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 8Gi
          requests:
            memory: 4Gi
        env:
        - name: TORCH_DTYPE
          value: "float16"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
```

#### StatefulSet (Object Registry)
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: object-registry
spec:
  serviceName: object-registry
  replicas: 3
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

### 10.4 마이그레이션 스크립트

#### DB 스키마 마이그레이션 (Alembic)
```bash
# alembic/versions/001_initial_schema.py
alembic upgrade head
```

#### 데이터 백필
```bash
# scripts/backfill_embeddings.py
# SQLite에서 임베딩 누락 객체 추출 → Classification Agent 호출 → 재등록
python scripts/backfill_embeddings.py --batch-size 100
```

---

## 11. 모니터링 및 로깅 (Monitoring & Logging)

### 11.1 메트릭 (Prometheus + Grafana)

**GPU 메트릭** (DCGM Exporter):
- GPU 사용률, 메모리 사용량
- 추론 레이턴시 (히스토그램)

**Application 메트릭** (FastAPI middleware):
- 요청 처리 시간 (endpoint별)
- 에러율 (4xx, 5xx)
- 동시 요청 수

**데이터베이스 메트릭**:
- SQLite: 쿼리 실행 시간, lock wait time
- ChromaDB/Qdrant: 벡터 검색 레이턴시, 인덱스 크기

### 11.2 로깅 (ELK Stack)

**구조화된 로그** (JSON 포맷):
```python
import structlog

logger = structlog.get_logger()
logger.info("object_registered",
            object_id=obj_id,
            category=category,
            confidence=confidence,
            elapsed_ms=elapsed)
```

**로그 레벨**:
- DEBUG: 모델 추론 상세 (개발 환경)
- INFO: 요청 처리 완료
- WARNING: 임베딩 동기화 실패 (재시도 대기)
- ERROR: 예외 발생 (스택 트레이스 포함)

---

## 12. 변경 이력

| 버전 | 날짜 | 작성자 | 변경 내용 |
|------|------|--------|----------|
| v1.0 | 2025-12-29 | System Architect | 최초 작성 |

---

## 부록 A: 디렉토리 구조

```
AgenticLabeling/
├── services/
│   ├── gateway/                   # API 게이트웨이
│   │   ├── app/
│   │   │   └── main.py           # 라우팅, 파이프라인 조율
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── detection-agent/          # Florence-2 감지
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── detector.py       # 핵심 로직
│   │   │   └── schemas.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── segmentation-agent/       # SAM2 세그멘테이션
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── segmenter.py      # 핵심 로직
│   │   │   └── schemas.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── classification-agent/     # DINOv2 분류
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── classifier.py     # Few-shot 로직
│   │   │   └── schemas.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── object-registry/          # 객체 중심 관리
│   │   ├── app/
│   │   │   ├── main.py           # REST API
│   │   │   ├── registry.py       # 핵심 로직
│   │   │   └── schema.sql        # DB 스키마
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── labeling-agent/           # 라벨 CRUD (레거시)
│   ├── training-agent/           # YOLO 학습
│   ├── data-manager/             # 데이터셋 내보내기
│   └── evaluation-agent/         # 모델 평가
│
├── data/
│   ├── registry/
│   │   ├── registry.db           # SQLite
│   │   ├── chroma/               # ChromaDB
│   │   └── masks/                # 마스크 PNG (sharded)
│   │       ├── a1/
│   │       │   └── b2/
│   │       │       └── obj_xyz.png
│   ├── sources/                  # 원본 이미지/비디오
│   └── exports/                  # 내보낸 데이터셋
│
├── docs/
│   ├── tech-specs/
│   │   └── architecture-spec.md  # 이 문서
│   ├── PROJECT_SPEC.md
│   └── plans/
│
├── scripts/
│   ├── migrate_sqlite_to_postgres.py
│   └── migrate_chroma_to_qdrant.py
│
├── k8s/                          # Kubernetes 매니페스트 (예정)
│   ├── deployments/
│   ├── services/
│   └── statefulsets/
│
├── docker-compose.yml            # 로컬 개발 환경
└── tests/
    ├── unit/
    ├── integration/
    └── performance/
```

---

## 부록 B: 참고 자료

- **모델 문서**:
  - [Florence-2](https://huggingface.co/microsoft/Florence-2-large)
  - [SAM2](https://github.com/facebookresearch/segment-anything-2)
  - [DINOv2](https://github.com/facebookresearch/dinov2)

- **인프라**:
  - [ChromaDB 공식 문서](https://docs.trychroma.com/)
  - [Qdrant 마이그레이션 가이드](https://qdrant.tech/documentation/guides/migrate-from-chroma/)
  - [FastAPI Best Practices](https://fastapi.tiangolo.com/deployment/)

- **설계 패턴**:
  - [Transactional Outbox Pattern](https://microservices.io/patterns/data/transactional-outbox.html)
  - [Saga Pattern for Microservices](https://microservices.io/patterns/data/saga.html)

---

**문서 작성**: 2025-12-29
**검토 요청**: Service Architect, Detection Expert, Database Expert
**승인 대기**: Tech Lead
