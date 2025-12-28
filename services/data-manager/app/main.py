"""Data Manager Service - Dataset management and preprocessing."""
import io
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .dataset import DatasetManager
from .exporters import YOLOExporter, COCOExporter

app = FastAPI(
    title="Data Manager",
    description="Dataset management, preprocessing, and versioning service",
    version="0.1.0",
)

# Use env var for Docker, default for local testing
DATA_DIR = os.getenv("DATA_DIR", os.path.expanduser("~/dev/AgenticLabeling/data"))
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8010")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")

dataset_manager = DatasetManager(data_dir=DATA_DIR)


# ==================== Schemas ====================

class ExportRequest(BaseModel):
    """Request schema for dataset export."""
    dataset_name: str
    format: str = "yolo"  # "yolo" or "coco"
    filter_config: Optional[Dict[str, Any]] = None
    split_config: Optional[Dict[str, float]] = None
    include_masks: bool = False


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "data-manager"}


@app.post("/datasets/create")
async def create_dataset(
    name: str = Form(...),
    description: str = Form(""),
    classes: str = Form(...),
):
    """Create a new dataset."""
    try:
        class_list = json.loads(classes)
        result = dataset_manager.create_dataset(name, description, class_list)
        return DatasetResponse(
            success=True,
            message=f"Dataset '{name}' created",
            data=result,
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/datasets")
async def list_datasets():
    """List all datasets."""
    try:
        datasets = dataset_manager.list_datasets()
        return {"success": True, "datasets": datasets}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/datasets/{dataset_name}")
async def get_dataset(dataset_name: str):
    """Get dataset info."""
    try:
        info = dataset_manager.get_dataset_info(dataset_name)
        if info is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"success": True, "dataset": info}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/datasets/{dataset_name}/upload")
async def upload_images(
    dataset_name: str,
    split: str = Form("train"),
    class_name: str = Form(None),
    images: List[UploadFile] = File(...),
):
    """Upload images to dataset."""
    try:
        uploaded = []
        for img in images:
            content = await img.read()
            path = dataset_manager.add_image(
                dataset_name, split, img.filename, content, class_name
            )
            uploaded.append(path)

        return {
            "success": True,
            "message": f"Uploaded {len(uploaded)} images",
            "paths": uploaded,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/datasets/{dataset_name}/import")
async def import_dataset(
    dataset_name: str,
    format: str = Form("coco"),
    file: UploadFile = File(...),
):
    """Import dataset from zip file (COCO, YOLO, VOC formats)."""
    try:
        content = await file.read()
        result = dataset_manager.import_dataset(dataset_name, format, content)
        return {
            "success": True,
            "message": f"Imported {result['images']} images, {result['annotations']} annotations",
            "data": result,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/datasets/{dataset_name}/export")
async def export_dataset(
    dataset_name: str,
    format: str = Form("yolo"),
    splits: str = Form("train,val"),
):
    """Export dataset in specified format."""
    try:
        split_list = [s.strip() for s in splits.split(",")]
        export_path = dataset_manager.export_dataset(dataset_name, format, split_list)
        return FileResponse(
            export_path,
            media_type="application/zip",
            filename=f"{dataset_name}_{format}.zip",
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/datasets/{dataset_name}/split")
async def split_dataset(
    dataset_name: str,
    train_ratio: float = Form(0.8),
    val_ratio: float = Form(0.1),
    test_ratio: float = Form(0.1),
    stratified: bool = Form(True),
):
    """Split dataset into train/val/test sets."""
    try:
        result = dataset_manager.split_dataset(
            dataset_name, train_ratio, val_ratio, test_ratio, stratified
        )
        return {
            "success": True,
            "message": "Dataset split completed",
            "splits": result,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/datasets/{dataset_name}/images")
async def list_images(
    dataset_name: str,
    split: str = None,
    class_name: str = None,
    limit: int = 100,
    offset: int = 0,
):
    """List images in dataset."""
    try:
        images = dataset_manager.list_images(
            dataset_name, split, class_name, limit, offset
        )
        return {"success": True, "images": images, "total": len(images)}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.delete("/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str, confirm: bool = Form(False)):
    """Delete a dataset."""
    if not confirm:
        return {"success": False, "message": "Confirmation required"}

    try:
        dataset_manager.delete_dataset(dataset_name)
        return {"success": True, "message": f"Dataset '{dataset_name}' deleted"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/datasets/{dataset_name}/augment")
async def augment_dataset(
    dataset_name: str,
    augmentations: str = Form(...),
    target_count: int = Form(None),
):
    """Apply data augmentation."""
    try:
        aug_config = json.loads(augmentations)
        result = dataset_manager.augment_dataset(dataset_name, aug_config, target_count)
        return {
            "success": True,
            "message": f"Generated {result['generated']} augmented images",
            "data": result,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


# ==================== Registry Export Endpoints ====================

@app.post("/registry/export")
async def export_from_registry(request: ExportRequest):
    """Export dataset from Object Registry to YOLO or COCO format.

    This endpoint fetches objects from the Object Registry and exports them
    to the specified format (YOLO or COCO).

    Args:
        request: Export configuration including dataset name, format, filters, and split ratios

    Returns:
        Export result with paths and statistics
    """
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)

        if request.format.lower() == "yolo":
            exporter = YOLOExporter(REGISTRY_URL, EXPORT_DIR)
        elif request.format.lower() == "coco":
            exporter = COCOExporter(REGISTRY_URL, EXPORT_DIR)
        else:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Unsupported format: {request.format}"},
            )

        result = await exporter.export_dataset(
            dataset_name=request.dataset_name,
            filter_config=request.filter_config,
            split_config=request.split_config,
            include_masks=request.include_masks,
        )

        return {"success": True, "data": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/registry/export/{dataset_name}/download")
async def download_exported_dataset(dataset_name: str, format: str = "yolo"):
    """Download exported dataset as zip file.

    Args:
        dataset_name: Name of the exported dataset
        format: Export format (yolo or coco)

    Returns:
        Zip file of the exported dataset
    """
    zip_path = Path(EXPORT_DIR) / f"{dataset_name}.zip"

    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Exported dataset not found")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{dataset_name}_{format}.zip",
    )


@app.get("/registry/exports")
async def list_exports():
    """List all exported datasets."""
    try:
        export_path = Path(EXPORT_DIR)
        if not export_path.exists():
            return {"success": True, "exports": []}

        exports = []
        for zip_file in export_path.glob("*.zip"):
            exports.append({
                "name": zip_file.stem,
                "path": str(zip_file),
                "size_mb": round(zip_file.stat().st_size / (1024 * 1024), 2),
            })

        return {"success": True, "exports": exports}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )
