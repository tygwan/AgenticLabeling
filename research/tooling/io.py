"""IO helpers for observation artifacts.

Respects docs/standards/research-observation-protocol.md Rule 5:
raw tensor dumps go under outputs/raw/ which is gitignored,
reports/notebooks/summaries live at the observation folder root.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def observation_dir(model_slug: str, slug: str, root: Path | str = "research/observations") -> Path:
    """Return the canonical observation folder path. Does not create it."""
    date = datetime.now().strftime("%Y-%m-%d")
    return Path(root) / model_slug / f"{date}-{slug}"


def ensure_layout(base: Path) -> dict[str, Path]:
    """Create the standard subfolders for an observation. Returns their paths."""
    paths = {
        "root": base,
        "assets": base / "assets",
        "raw": base / "outputs" / "raw",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def current_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def dump_tensor(obj: Any, raw_dir: Path, name: str) -> Path | None:
    """Save a tensor / array to outputs/raw/ (gitignored). Returns the file path."""
    safe = name.replace("/", "__").replace("[", "_").replace("]", "_")
    if isinstance(obj, torch.Tensor):
        path = raw_dir / f"{safe}.pt"
        torch.save(obj.detach().cpu(), path)
        return path
    if isinstance(obj, np.ndarray):
        path = raw_dir / f"{safe}.npz"
        np.savez_compressed(path, array=obj)
        return path
    return None


def write_summary(summary: dict, base: Path) -> Path:
    """Write summary.json at the observation root."""
    path = base / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    return path
