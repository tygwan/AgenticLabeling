CREATE TABLE IF NOT EXISTS memory_entries (
    memory_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT,
    metadata_json TEXT,
    source_kind TEXT,
    source_ref TEXT,
    module_scope TEXT,
    change_kind TEXT,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_entries_category ON memory_entries(category);
CREATE INDEX IF NOT EXISTS idx_memory_entries_module_scope ON memory_entries(module_scope);
CREATE INDEX IF NOT EXISTS idx_memory_entries_change_kind ON memory_entries(change_kind);

CREATE TABLE IF NOT EXISTS memory_links (
    link_id TEXT PRIMARY KEY,
    from_memory_id TEXT NOT NULL,
    to_memory_id TEXT NOT NULL,
    link_type TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_links_from ON memory_links(from_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_links_to ON memory_links(to_memory_id);

