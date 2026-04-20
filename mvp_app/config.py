"""Configuration for the AgenticLabeling app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    app_name: str = "AgenticLabeling"
    data_dir: Path = Path(os.getenv("DATA_DIR", "data/mvp"))
    sqlite_path: Path = Path(os.getenv("SQLITE_PATH", "data/mvp/sqlite/mvp.db"))
    assets_dir: Path = Path(os.getenv("ASSETS_DIR", "data/mvp/assets"))
    masks_dir: Path = Path(os.getenv("MASKS_DIR", "data/mvp/masks"))
    exports_dir: Path = Path(os.getenv("EXPORTS_DIR", "data/mvp/exports"))
    fake_models: bool = os.getenv("FAKE_MODELS", "0") == "1"
    florence_model_id: str = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-large")
    # Florence-2 compute dtype. Defaults to float32 for maximum compatibility.
    # Set to bfloat16 on Ampere/Ada (RTX 30/40, A100, H100) for ~2x speed and
    # half the VRAM without the overflow risk of float16. Valid: float32, bfloat16, float16.
    florence_dtype: str = os.getenv("FLORENCE_DTYPE", "float32")
    sam3_checkpoint: str = os.getenv("SAM3_CHECKPOINT", "")
    sam3_version: str = os.getenv("SAM3_VERSION", "sam3")

    def ensure_dirs(self) -> None:
        """Create runtime directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()


def get_settings() -> Settings:
    """Return a fresh settings snapshot from the current environment."""
    return Settings()
