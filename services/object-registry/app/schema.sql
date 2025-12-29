-- Object Registry SQLite Schema
-- Version: 1.0.0

-- 소스 (이미지/비디오)
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL CHECK(source_type IN ('image', 'video', 'text')),
    file_path TEXT,
    file_name TEXT,
    width INTEGER,
    height INTEGER,
    frame_count INTEGER,
    fps REAL,
    duration_ms INTEGER,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 프레임 (비디오의 개별 프레임)
CREATE TABLE IF NOT EXISTS frames (
    frame_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    frame_idx INTEGER NOT NULL,
    timestamp_ms INTEGER,
    UNIQUE(source_id, frame_idx)
);

-- 카테고리 (클래스 온톨로지)
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    supercategory TEXT,
    synonyms JSON,
    color TEXT DEFAULT '#808080',
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 객체 (핵심 테이블)
CREATE TABLE IF NOT EXISTS objects (
    object_id TEXT PRIMARY KEY,
    category_id INTEGER REFERENCES categories(category_id),
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    frame_id TEXT REFERENCES frames(frame_id),
    project_id TEXT,

    -- Geometry (bbox: x, y, width, height)
    bbox_x REAL NOT NULL,
    bbox_y REAL NOT NULL,
    bbox_w REAL NOT NULL,
    bbox_h REAL NOT NULL,
    mask_path TEXT,
    polygon JSON,
    area REAL,

    -- Detection metadata
    confidence REAL,
    detection_model TEXT,

    -- Classification (if classified)
    classification_model TEXT,
    classification_confidence REAL,

    -- Quality & validation
    is_validated BOOLEAN DEFAULT FALSE,
    validated_by TEXT,
    validated_at TIMESTAMP,
    quality_score REAL,

    -- Flags
    is_occluded BOOLEAN DEFAULT FALSE,
    is_truncated BOOLEAN DEFAULT FALSE,
    is_difficult BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 트랙 (Re-ID/객체 추적용)
CREATE TABLE IF NOT EXISTS tracks (
    track_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES categories(category_id),
    start_frame_idx INTEGER,
    end_frame_idx INTEGER,
    object_count INTEGER DEFAULT 0,
    avg_confidence REAL,
    is_validated BOOLEAN DEFAULT FALSE,
    validated_by TEXT,
    validated_at TIMESTAMP,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 객체-트랙 연결
CREATE TABLE IF NOT EXISTS track_objects (
    track_id TEXT NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,
    object_id TEXT NOT NULL REFERENCES objects(object_id) ON DELETE CASCADE,
    sequence_idx INTEGER NOT NULL,
    PRIMARY KEY (track_id, object_id)
);

-- 임베딩 버전 관리
CREATE TABLE IF NOT EXISTS embedding_versions (
    version_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,  -- 'visual', 'text', 'multimodal'
    dimension INTEGER NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 데이터셋
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    format TEXT DEFAULT 'yolo',
    version TEXT DEFAULT '1.0',
    split_config JSON DEFAULT '{"train": 0.8, "val": 0.1, "test": 0.1}',
    filter_config JSON,
    category_mapping JSON,
    object_count INTEGER DEFAULT 0,
    export_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exported_at TIMESTAMP
);

-- 데이터셋-객체 연결
CREATE TABLE IF NOT EXISTS dataset_objects (
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    object_id TEXT NOT NULL REFERENCES objects(object_id) ON DELETE CASCADE,
    split TEXT NOT NULL CHECK(split IN ('train', 'val', 'test')),
    PRIMARY KEY (dataset_id, object_id)
);

-- ChromaDB 동기화용 outbox
CREATE TABLE IF NOT EXISTS embedding_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id TEXT NOT NULL REFERENCES objects(object_id) ON DELETE CASCADE,
    version_id TEXT NOT NULL REFERENCES embedding_versions(version_id),
    operation TEXT NOT NULL CHECK(operation IN ('upsert', 'delete')),
    embedding BLOB,  -- 임시 저장 (동기화 전)
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'synced', 'failed')),
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    synced_at TIMESTAMP
);

-- 프로젝트 (labeling-agent 호환)
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    classes JSON,
    default_model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_objects_source ON objects(source_id);
CREATE INDEX IF NOT EXISTS idx_objects_category ON objects(category_id);
CREATE INDEX IF NOT EXISTS idx_objects_project ON objects(project_id);
CREATE INDEX IF NOT EXISTS idx_objects_confidence ON objects(confidence);
CREATE INDEX IF NOT EXISTS idx_objects_validated ON objects(is_validated);
CREATE INDEX IF NOT EXISTS idx_objects_created ON objects(created_at);
CREATE INDEX IF NOT EXISTS idx_frames_source ON frames(source_id);
CREATE INDEX IF NOT EXISTS idx_track_objects_track ON track_objects(track_id);
CREATE INDEX IF NOT EXISTS idx_dataset_objects_dataset ON dataset_objects(dataset_id);
CREATE INDEX IF NOT EXISTS idx_dataset_objects_split ON dataset_objects(split);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON embedding_outbox(status);
CREATE INDEX IF NOT EXISTS idx_outbox_object ON embedding_outbox(object_id);

-- 트리거: objects 업데이트 시 updated_at 갱신
CREATE TRIGGER IF NOT EXISTS update_objects_timestamp
AFTER UPDATE ON objects
BEGIN
    UPDATE objects SET updated_at = CURRENT_TIMESTAMP WHERE object_id = NEW.object_id;
END;

-- 트리거: track의 object_count 갱신
CREATE TRIGGER IF NOT EXISTS update_track_count_insert
AFTER INSERT ON track_objects
BEGIN
    UPDATE tracks SET object_count = (
        SELECT COUNT(*) FROM track_objects WHERE track_id = NEW.track_id
    ) WHERE track_id = NEW.track_id;
END;

CREATE TRIGGER IF NOT EXISTS update_track_count_delete
AFTER DELETE ON track_objects
BEGIN
    UPDATE tracks SET object_count = (
        SELECT COUNT(*) FROM track_objects WHERE track_id = OLD.track_id
    ) WHERE track_id = OLD.track_id;
END;

-- 기본 임베딩 버전 추가
INSERT OR IGNORE INTO embedding_versions (version_id, model_name, model_type, dimension, description)
VALUES ('dinov2_base_v1', 'facebook/dinov2-base', 'visual', 768, 'DINOv2 base model visual embeddings');
