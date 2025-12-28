"""Labeling Agent Service - Manages labels and annotations."""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .schemas import (
    LabelData,
    LabelResponse,
    ExportFormat,
    ExportResponse,
)
from .label_manager import LabelManager


app = FastAPI(
    title="Labeling Agent",
    description="Label management and annotation service",
    version="0.1.0",
)

import os

# Initialize label manager (use env var for Docker, default for local)
DATA_DIR = os.getenv("DATA_DIR", os.path.expanduser("~/dev/AgenticLabeling/data/labels"))
label_manager = LabelManager(data_dir=DATA_DIR)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "labeling-agent"}


@app.post("/labels/create")
async def create_label(
    project_id: str = Form(...),
    image_id: str = Form(...),
    label_data: str = Form(...),
):
    """Create or update label for an image.

    Args:
        project_id: Project identifier
        image_id: Image identifier
        label_data: JSON string with label data (boxes, masks, classes)
    """
    try:
        data = json.loads(label_data)
        result = label_manager.save_label(project_id, image_id, data)
        return LabelResponse(
            success=True,
            message="Label saved",
            data=result,
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/labels/save")
async def save_label(data: dict):
    """Save label data from internal services (JSON body).

    Used by gateway's auto_label pipeline to save detection/segmentation results.
    """
    try:
        project_id = data.get("project_id")
        image_id = data.get("image_id")
        if not project_id or not image_id:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "project_id and image_id required"},
            )

        # Remove project_id and image_id from data as they're passed separately
        label_data = {k: v for k, v in data.items() if k not in ("project_id", "image_id")}
        result = label_manager.save_label(project_id, image_id, label_data)
        return {"success": True, "message": "Label saved", "data": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/labels/{project_id}/{image_id}")
async def get_label(project_id: str, image_id: str):
    """Get label for a specific image."""
    try:
        label = label_manager.get_label(project_id, image_id)
        if label is None:
            raise HTTPException(status_code=404, detail="Label not found")
        return LabelResponse(success=True, data=label)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/labels/{project_id}")
async def list_labels(project_id: str, limit: int = 100, offset: int = 0):
    """List all labels for a project."""
    try:
        labels = label_manager.list_labels(project_id, limit=limit, offset=offset)
        return {
            "success": True,
            "project_id": project_id,
            "total": len(labels),
            "labels": labels,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/labels/import")
async def import_labels(
    project_id: str = Form(...),
    format: str = Form("coco"),
    file: UploadFile = File(...),
):
    """Import labels from file (COCO, YOLO, Pascal VOC formats)."""
    try:
        content = await file.read()
        result = label_manager.import_labels(project_id, format, content)
        return {
            "success": True,
            "message": f"Imported {result['count']} labels",
            "data": result,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/labels/export", response_model=ExportResponse)
async def export_labels(
    project_id: str = Form(...),
    format: str = Form("coco"),
    include_images: bool = Form(False),
):
    """Export labels in specified format."""
    try:
        result = label_manager.export_labels(project_id, format, include_images)
        return ExportResponse(
            success=True,
            message=f"Exported {result['count']} labels in {format} format",
            download_url=result.get("download_url"),
            data=result,
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.delete("/labels/{project_id}/{image_id}")
async def delete_label(project_id: str, image_id: str):
    """Delete label for an image."""
    try:
        success = label_manager.delete_label(project_id, image_id)
        if not success:
            raise HTTPException(status_code=404, detail="Label not found")
        return {"success": True, "message": "Label deleted"}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/labels/validate")
async def validate_labels(project_id: str = Form(...)):
    """Validate all labels in a project."""
    try:
        result = label_manager.validate_labels(project_id)
        return {
            "success": True,
            "valid": result["valid"],
            "total": result["total"],
            "errors": result.get("errors", []),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/projects")
async def list_projects():
    """List all projects."""
    try:
        projects = label_manager.list_projects()
        return {"success": True, "projects": projects}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/projects/create")
async def create_project(
    project_id: str = Form(...),
    name: str = Form(...),
    classes: str = Form(...),
):
    """Create a new project."""
    try:
        class_list = json.loads(classes)
        result = label_manager.create_project(project_id, name, class_list)
        return {"success": True, "project": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )
