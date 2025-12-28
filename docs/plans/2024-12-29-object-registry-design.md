# Object Registry 설계 문서

## 개요

객체 중심의 데이터 관리 시스템으로, 감지/세그멘테이션/분류된 객체들을 일관되게 저장하고 검색할 수 있도록 한다.

## Phase 1 목표 (현재)

- SQLite + ChromaDB 하이브리드 저장소
- 로컬 개발 환경 최적화
- 단일 인스턴스 운영

## 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                   Object Registry                        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   SQLite    │  │  ChromaDB   │  │   Filesystem    │ │
│  │  (메타데이터)│  │  (임베딩)    │  │   (마스크)      │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│         │                │                  │           │
│         └────────────────┼──────────────────┘           │
│                          │                              │
│              ┌───────────┴───────────┐                  │
│              │    Sync Manager       │                  │
│              │  (outbox + reconcile) │                  │
│              └───────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

## 저장소 구조

```
data/
├── registry/
│   ├── registry.db           # SQLite 메인 DB
│   ├── chroma/               # ChromaDB 벡터 저장소
│   │   └── visual_v1/        # 버전별 컬렉션
│   └── masks/                # 마스크 파일 (샤딩)
│       └── {hash[0:2]}/{hash[2:4]}/{object_id}.rle
│
├── sources/                  # 원본 소스
│   ├── images/
│   └── videos/
│
└── exports/                  # 내보낸 데이터셋
    └── {dataset_name}/
```

## SQLite 스키마

### 핵심 테이블

```sql
-- 소스 (이미지/비디오)
CREATE TABLE sources (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,  -- 'image', 'video', 'text'
    file_path TEXT,
    width INTEGER,
    height INTEGER,
    frame_count INTEGER,        -- 비디오용
    fps REAL,                   -- 비디오용
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 프레임 (비디오의 개별 프레임)
CREATE TABLE frames (
    frame_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    frame_idx INTEGER NOT NULL,
    timestamp_ms INTEGER,
    UNIQUE(source_id, frame_idx)
);

-- 카테고리 (클래스 온톨로지)
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    supercategory TEXT,
    synonyms JSON,              -- ["person", "human", "사람"]
    color TEXT,                 -- 시각화용 "#FF0000"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 객체 (핵심 테이블)
CREATE TABLE objects (
    object_id TEXT PRIMARY KEY,
    category_id INTEGER REFERENCES categories(category_id),
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    frame_id TEXT REFERENCES frames(frame_id),

    -- Geometry
    bbox_x REAL NOT NULL,
    bbox_y REAL NOT NULL,
    bbox_w REAL NOT NULL,
    bbox_h REAL NOT NULL,
    mask_path TEXT,             -- RLE 마스크 파일 경로
    area REAL,

    -- Detection metadata
    confidence REAL,
    detection_model TEXT,

    -- Quality & validation
    is_validated BOOLEAN DEFAULT FALSE,
    quality_score REAL,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 트랙 (Re-ID용)
CREATE TABLE tracks (
    track_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    category_id INTEGER REFERENCES categories(category_id),
    start_frame INTEGER,
    end_frame INTEGER,
    object_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 객체-트랙 연결
CREATE TABLE track_objects (
    track_id TEXT NOT NULL REFERENCES tracks(track_id),
    object_id TEXT NOT NULL REFERENCES objects(object_id),
    sequence_idx INTEGER NOT NULL,
    PRIMARY KEY (track_id, object_id)
);

-- 임베딩 버전 관리
CREATE TABLE embedding_versions (
    version_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,   -- 'dinov2-base', 'clip-vit-b32'
    model_version TEXT,
    dimension INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 데이터셋 (YOLO/COCO 내보내기용)
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    format TEXT,                -- 'yolo', 'coco'
    split_config JSON,          -- {"train": 0.8, "val": 0.1, "test": 0.1}
    filter_config JSON,         -- {"min_confidence": 0.5, "categories": [...]}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 데이터셋-객체 연결
CREATE TABLE dataset_objects (
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    object_id TEXT NOT NULL REFERENCES objects(object_id),
    split TEXT NOT NULL,        -- 'train', 'val', 'test'
    PRIMARY KEY (dataset_id, object_id)
);

-- ChromaDB 동기화용 outbox
CREATE TABLE embedding_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id TEXT NOT NULL REFERENCES objects(object_id),
    version_id TEXT NOT NULL REFERENCES embedding_versions(version_id),
    operation TEXT NOT NULL,    -- 'upsert', 'delete'
    status TEXT DEFAULT 'pending',  -- 'pending', 'synced', 'failed'
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    synced_at TIMESTAMP
);

-- 인덱스
CREATE INDEX idx_objects_source ON objects(source_id);
CREATE INDEX idx_objects_category ON objects(category_id);
CREATE INDEX idx_objects_confidence ON objects(confidence);
CREATE INDEX idx_objects_validated ON objects(is_validated);
CREATE INDEX idx_track_objects_track ON track_objects(track_id);
CREATE INDEX idx_outbox_status ON embedding_outbox(status);
```

## ChromaDB 컬렉션 구조

```python
# 버전별 컬렉션
collection_name = f"objects_visual_{version_id}"  # e.g., "objects_visual_dinov2_v1"

# 메타데이터 (필터링용)
metadata = {
    "object_id": "obj_abc123",
    "category": "person",
    "source_id": "video_001",
    "confidence": 0.95,
    "is_validated": False
}

# 임베딩
embedding = [0.1, 0.2, ...]  # DINOv2 768d
```

## 핵심 API

### ObjectRegistry 클래스

```python
class ObjectRegistry:
    def __init__(self, data_dir: str):
        self.db = SQLiteConnection(data_dir / "registry.db")
        self.chroma = ChromaClient(data_dir / "chroma")
        self.mask_store = MaskStore(data_dir / "masks")

    # CRUD
    def register_object(self, object_data: ObjectCreate) -> Object
    def get_object(self, object_id: str) -> Object
    def update_object(self, object_id: str, updates: dict) -> Object
    def delete_object(self, object_id: str) -> bool

    # 검색
    def search_by_embedding(self, embedding: list, top_k: int) -> list[Object]
    def search_by_filter(self, filters: dict) -> list[Object]

    # Re-ID
    def create_track(self, source_id: str, object_ids: list) -> Track
    def find_similar_objects(self, object_id: str, threshold: float) -> list[Object]

    # 데이터셋 내보내기
    def create_dataset(self, config: DatasetConfig) -> Dataset
    def export_yolo(self, dataset_id: str, output_dir: str) -> str
    def export_coco(self, dataset_id: str, output_dir: str) -> str

    # 동기화
    def sync_embeddings(self) -> SyncResult
    def reconcile(self) -> ReconcileResult
```

## YAGNI 적용 (Phase 1에서 제외)

- ❌ `clip_text` 임베딩 저장 → 쿼리 시 on-the-fly 계산
- ❌ `affordance` 필드 → VLA 확장 시 별도 테이블로 추가
- ❌ 다중 임베딩 동시 저장 → DINOv2 visual 1개만 저장
- ❌ 분산 처리 → 단일 인스턴스로 시작

## 마이그레이션 경로 (Phase 2)

```
Phase 1 (현재)          →    Phase 2 (K8s)
SQLite                  →    PostgreSQL
ChromaDB (embedded)     →    Qdrant (cluster)
파일시스템 마스크        →    MinIO/S3
동기 HTTP               →    gRPC + 메시지 큐
```

## 구현 순서

1. SQLite 스키마 생성 및 마이그레이션
2. ObjectRegistry 코어 클래스 구현
3. ChromaDB 연동 및 동기화 로직
4. 마스크 저장소 (RLE 인코딩)
5. REST API 엔드포인트
6. 기존 labeling-agent 통합
7. YOLO/COCO 내보내기 기능
