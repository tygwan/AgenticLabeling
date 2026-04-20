#!/usr/bin/env python3
"""SQLite-backed project memory store for AgenticLabeling."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_DB_PATH = PROJECT_ROOT / os.getenv(
    "PROJECT_MEMORY_DB_PATH",
    "data/project-memory/project-memory.db",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_db_path(raw_path: str | None) -> Path:
    if not raw_path:
        return DEFAULT_DB_PATH
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    conn = connect(db_path)
    try:
        conn.executescript(schema)
        conn.commit()
    finally:
        conn.close()


def parse_json_arg(raw: str | None, *, default: object) -> str:
    if raw is None:
        return json.dumps(default)
    parsed = json.loads(raw)
    return json.dumps(parsed)


def parse_tags(raw_tags: str | None) -> str:
    if not raw_tags:
        return json.dumps([])
    tags = [item.strip() for item in raw_tags.split(",") if item.strip()]
    return json.dumps(tags)


def add_entry(args: argparse.Namespace) -> None:
    db_path = resolve_db_path(args.db)
    conn = connect(db_path)
    now = utc_now()
    memory_id = args.memory_id or f"mem_{uuid.uuid4().hex[:12]}"
    tags_json = parse_tags(args.tags)
    metadata_json = parse_json_arg(args.metadata_json, default={})
    try:
        conn.execute(
            """
            INSERT INTO memory_entries (
                memory_id, category, title, summary, content, tags, metadata_json,
                source_kind, source_ref, module_scope, change_kind,
                created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                args.category,
                args.title,
                args.summary,
                args.content,
                tags_json,
                metadata_json,
                args.source_kind,
                args.source_ref,
                args.module_scope,
                args.change_kind,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    print(memory_id)


def add_link(args: argparse.Namespace) -> None:
    db_path = resolve_db_path(args.db)
    conn = connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO memory_links (
                link_id, from_memory_id, to_memory_id, link_type, created_at_utc
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                args.link_id or f"lnk_{uuid.uuid4().hex[:12]}",
                args.from_memory_id,
                args.to_memory_id,
                args.link_type,
                utc_now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def query_entries(args: argparse.Namespace) -> None:
    db_path = resolve_db_path(args.db)
    conn = connect(db_path)
    conditions: list[str] = []
    params: list[object] = []
    if args.module_scope:
        conditions.append("module_scope = ?")
        params.append(args.module_scope)
    if args.category:
        placeholders = ",".join("?" for _ in args.category)
        conditions.append(f"category IN ({placeholders})")
        params.extend(args.category)
    if args.text:
        conditions.append("(title LIKE ? OR summary LIKE ? OR content LIKE ?)")
        like = f"%{args.text}%"
        params.extend([like, like, like])

    sql = """
        SELECT memory_id, category, title, summary, tags, metadata_json,
               source_kind, source_ref, module_scope, change_kind,
               created_at_utc, updated_at_utc
        FROM memory_entries
    """
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY updated_at_utc DESC"
    if args.limit:
        sql += " LIMIT ?"
        params.append(args.limit)
    try:
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()

    if args.tag:
        wanted = set(args.tag)
        filtered = []
        for row in rows:
            row_tags = set(json.loads(row["tags"] or "[]"))
            if wanted & row_tags:
                filtered.append(row)
        rows = filtered

    for row in rows:
        row["tags"] = json.loads(row["tags"] or "[]")
        row["metadata_json"] = json.loads(row["metadata_json"] or "{}")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project memory store for AgenticLabeling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_cmd = subparsers.add_parser("init", help="Initialize the project memory database")
    init_cmd.add_argument("--db", help="Path to the SQLite database file")
    init_cmd.set_defaults(func=lambda args: init_db(resolve_db_path(args.db)))

    add_cmd = subparsers.add_parser("add-entry", help="Add a memory entry")
    add_cmd.add_argument("--db", help="Path to the SQLite database file")
    add_cmd.add_argument("--memory-id")
    add_cmd.add_argument("--category", required=True)
    add_cmd.add_argument("--title", required=True)
    add_cmd.add_argument("--summary", required=True)
    add_cmd.add_argument("--content", required=True)
    add_cmd.add_argument("--tags", help="Comma-separated tags")
    add_cmd.add_argument("--metadata-json", help="JSON object string")
    add_cmd.add_argument("--source-kind")
    add_cmd.add_argument("--source-ref")
    add_cmd.add_argument("--module-scope")
    add_cmd.add_argument("--change-kind")
    add_cmd.set_defaults(func=add_entry)

    link_cmd = subparsers.add_parser("link", help="Link two memory entries")
    link_cmd.add_argument("--db", help="Path to the SQLite database file")
    link_cmd.add_argument("--link-id")
    link_cmd.add_argument("--from-memory-id", required=True)
    link_cmd.add_argument("--to-memory-id", required=True)
    link_cmd.add_argument("--link-type", required=True)
    link_cmd.set_defaults(func=add_link)

    query_cmd = subparsers.add_parser("query", help="Query memory entries")
    query_cmd.add_argument("--db", help="Path to the SQLite database file")
    query_cmd.add_argument("--module-scope")
    query_cmd.add_argument("--category", action="append")
    query_cmd.add_argument("--tag", action="append")
    query_cmd.add_argument("--text", help="Free-text search over title, summary, and content")
    query_cmd.add_argument("--limit", type=int, default=20)
    query_cmd.set_defaults(func=query_entries)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

