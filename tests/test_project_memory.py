from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from project_memory.memory_store import DEFAULT_DB_PATH, resolve_db_path


ROOT = Path(__file__).resolve().parents[1]


def run_memory(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "project_memory.memory_store", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_project_memory_init_add_and_query(tmp_path: Path) -> None:
    db_path = tmp_path / "project-memory.db"
    run_memory("init", "--db", str(db_path))

    created = run_memory(
        "add-entry",
        "--db",
        str(db_path),
        "--category",
        "decision",
        "--title",
        "Use MVP as default runtime",
        "--summary",
        "The monolith is the default runtime path.",
        "--content",
        "The repository now defaults to the MVP app instead of the full microservice graph.",
        "--tags",
        "mvp,architecture",
        "--module-scope",
        "mvp_app",
        "--source-kind",
        "doc",
        "--source-ref",
        "docs/plans/2026-04-20-mvp-refactor-plan.md",
        "--change-kind",
        "architecture",
    )
    memory_id = created.stdout.strip()
    assert memory_id.startswith("mem_")

    queried = run_memory(
        "query",
        "--db",
        str(db_path),
        "--module-scope",
        "mvp_app",
        "--category",
        "decision",
        "--tag",
        "mvp",
    )
    payload = json.loads(queried.stdout)
    assert len(payload) == 1
    assert payload[0]["memory_id"] == memory_id
    assert payload[0]["module_scope"] == "mvp_app"
    assert payload[0]["tags"] == ["mvp", "architecture"]


def test_project_memory_paths_are_repo_root_relative() -> None:
    assert DEFAULT_DB_PATH == ROOT / "data/project-memory/project-memory.db"
    assert resolve_db_path("data/project-memory/custom.db") == ROOT / "data/project-memory/custom.db"
