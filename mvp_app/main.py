"""FastAPI monolith for the AgenticLabeling MVP."""

from __future__ import annotations

from contextlib import asynccontextmanager
import io
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

from .config import get_settings
from .detector import DetectionService
from .registry import Registry
from .segmenter import SegmentationService
from .storage import init_storage, store_image_bytes

STATIC_DIR = Path(__file__).parent / "static"

CATEGORY_COLORS = {
    "person": "#ef4444",
    "car": "#3b82f6",
    "truck": "#8b5cf6",
    "bicycle": "#f59e0b",
    "motorcycle": "#ec4899",
    "traffic_light": "#10b981",
    "stop_sign": "#f97316",
    "dog": "#14b8a6",
    "backpack": "#a855f7",
    "handbag": "#06b6d4",
    "bench": "#84cc16",
    "pallet": "#eab308",
    "forklift": "#22c55e",
    "worker": "#ef4444",
    "shelf": "#64748b",
    "box": "#f59e0b",
    "road": "#475569",
    "building": "#6366f1",
    "sky": "#0ea5e9",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_storage()
    yield


app = FastAPI(title="AgenticLabeling MVP", version="0.1.0", lifespan=lifespan)
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
registry = Registry()
detector = DetectionService()
segmenter = SegmentationService()

OVERLAY_COLORS = [
    (230, 57, 70),
    (29, 78, 216),
    (34, 197, 94),
    (249, 115, 22),
    (147, 51, 234),
    (8, 145, 178),
]


def _xyxy_to_xywh(box: list[float], image_width: int, image_height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(x1, image_width))
    y1 = max(0.0, min(y1, image_height))
    x2 = max(x1, min(x2, image_width))
    y2 = max(y1, min(y2, image_height))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _render_source_overlay(source: dict, objects: list[dict], *, include_masks: bool) -> bytes:
    source_path = Path(source["file_path"])
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Source image not found")

    base = Image.open(source_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for idx, obj in enumerate(objects):
        color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
        mask_path = obj.get("mask_path")
        if include_masks and mask_path and Path(mask_path).exists():
            mask = Image.open(mask_path).convert("L")
            if mask.size != base.size:
                mask = mask.resize(base.size)
            alpha_mask = mask.point(lambda p: 110 if p > 0 else 0)
            colored = Image.new("RGBA", base.size, color + (0,))
            colored.putalpha(alpha_mask)
            overlay = Image.alpha_composite(overlay, colored)

        x1 = float(obj["bbox_x"])
        y1 = float(obj["bbox_y"])
        x2 = x1 + float(obj["bbox_w"])
        y2 = y1 + float(obj["bbox_h"])
        draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=3)

        label = obj.get("category_name") or "object"
        label_width = max(48, len(label) * 8 + 10)
        label_top = max(0, y1 - 22)
        draw.rectangle((x1, label_top, x1 + label_width, label_top + 20), fill=color + (210,))
        draw.text((x1 + 4, label_top + 3), label, fill=(255, 255, 255, 255))

    composite = Image.alpha_composite(base, overlay).convert("RGB")
    buffer = io.BytesIO()
    composite.save(buffer, format="PNG")
    return buffer.getvalue()


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "healthy",
        "app": settings.app_name,
        "stats": registry.get_stats(),
        "segmentation_backend": segmenter.backend_name,
    }


def _spa_shell() -> HTMLResponse:
    """Serve the React UMD SPA shell from mvp_app/static/index.html.

    Rewrites relative asset hrefs to absolute /static/ paths so the same HTML
    works both when served by FastAPI and when inspected in-place during
    development.
    """
    shell_path = STATIC_DIR / "index.html"
    if not shell_path.is_file():
        raise HTTPException(status_code=500, detail="SPA shell missing (mvp_app/static/index.html)")
    html_text = shell_path.read_text(encoding="utf-8")
    html_text = html_text.replace('href="styles.css"', 'href="/static/styles.css"')
    html_text = html_text.replace('src="components/', 'src="/static/components/')
    return HTMLResponse(html_text)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return _spa_shell()


@app.post("/upload")
async def upload_and_redirect(
    image: UploadFile = File(...),
    project_id: str = Form("default-project"),
    classes: str = Form("person,car,dog"),
) -> RedirectResponse:
    result = await run_auto_label(image=image, project_id=project_id, classes=classes)
    return RedirectResponse(url=f"/review?source_id={result['source_id']}", status_code=303)


@app.post("/api/pipeline/auto-label")
async def run_auto_label(
    image: UploadFile = File(...),
    project_id: str = Form("default-project"),
    classes: str = Form("person,car,dog"),
) -> dict:
    image_bytes = await image.read()
    class_list = [c.strip() for c in classes.split(",") if c.strip()]

    stored_path = store_image_bytes(image.filename or f"{uuid4().hex}.jpg", image_bytes)
    with Image.open(stored_path).convert("RGB") as img:
        width, height = img.width, img.height

    detection = detector.detect(image_bytes, class_list)
    raw_boxes = detection.get("boxes", [])
    labels = detection.get("labels", [])
    scores = detection.get("scores", [])

    segmentation = segmenter.segment(image_bytes, raw_boxes) if raw_boxes else {"masks": []}
    masks = segmentation.get("masks", [])

    source_id = registry.register_source(
        project_id=project_id,
        file_path=str(stored_path.resolve()),
        file_name=stored_path.name,
        width=width,
        height=height,
    )

    objects_data = []
    for idx, raw_box in enumerate(raw_boxes):
        xywh = _xyxy_to_xywh(raw_box, width, height)
        objects_data.append(
            {
                "category": labels[idx] if idx < len(labels) else "object",
                "bbox": xywh,
                "confidence": scores[idx] if idx < len(scores) else None,
                "detection_model": "florence2",
                "mask_base64": masks[idx]["mask"] if idx < len(masks) else None,
            }
        )
    object_ids = registry.register_objects_batch(source_id, objects_data) if objects_data else []
    return {
        "success": True,
        "source_id": source_id,
        "object_ids": object_ids,
        "detections": len(object_ids),
        "file_name": stored_path.name,
        "segmentation_backend": segmenter.backend_name,
    }


@app.get("/api/review/sources")
def api_sources() -> dict:
    return {"success": True, "data": registry.list_sources()}


@app.get("/api/review/objects")
def api_objects(source_id: Optional[str] = None) -> dict:
    return {"success": True, "data": registry.list_objects(source_id=source_id)}


@app.patch("/api/review/objects/{object_id}")
def api_validate_object(object_id: str) -> dict:
    if not registry.validate_object(object_id):
        raise HTTPException(status_code=404, detail="Object not found")
    return {"success": True, "object_id": object_id}


@app.delete("/api/review/objects/{object_id}")
def api_delete_object(object_id: str) -> dict:
    if not registry.delete_object(object_id):
        raise HTTPException(status_code=404, detail="Object not found")
    return {"success": True, "object_id": object_id}


@app.post("/review/objects/{object_id}/approve")
def approve_object(object_id: str, source_id: str = Form(...)) -> RedirectResponse:
    registry.validate_object(object_id)
    return RedirectResponse(url=f"/review?source_id={source_id}", status_code=303)


@app.post("/review/objects/{object_id}/delete")
def delete_object(object_id: str, source_id: str = Form(...)) -> RedirectResponse:
    registry.delete_object(object_id)
    return RedirectResponse(url=f"/review?source_id={source_id}", status_code=303)


@app.get("/api/assets/{source_id}")
def get_asset(source_id: str) -> FileResponse:
    source = registry.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return FileResponse(source["file_path"])


@app.get("/api/masks/{object_id}")
def get_mask(object_id: str) -> FileResponse:
    obj = registry.get_object(object_id)
    if not obj or not obj.get("mask_path"):
        raise HTTPException(status_code=404, detail="Mask not found")
    mask_path = Path(obj["mask_path"])
    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Mask file not found")
    return FileResponse(mask_path, media_type="image/png")


@app.get("/api/assets/{source_id}/overlay")
def get_asset_overlay(source_id: str) -> Response:
    source = registry.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    objects = registry.list_objects(source_id=source_id)
    overlay_bytes = _render_source_overlay(source, objects, include_masks=True)
    return Response(content=overlay_bytes, media_type="image/png")


@app.get("/api/assets/{source_id}/bbox-overlay")
def get_asset_bbox_overlay(source_id: str) -> Response:
    source = registry.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    objects = registry.list_objects(source_id=source_id)
    overlay_bytes = _render_source_overlay(source, objects, include_masks=False)
    return Response(content=overlay_bytes, media_type="image/png")


@app.post("/api/export")
def api_export(
    dataset_name: str = Form("mvp-dataset"),
    export_format: str = Form("yolo"),
    only_validated: bool = Form(True),
) -> dict:
    result = registry.export_dataset(
        dataset_name=dataset_name,
        export_format=export_format.lower(),
        only_validated=only_validated,
    )
    return {
        "success": True,
        "dataset_name": result.dataset_name,
        "zip_path": str(result.zip_path),
        "image_count": result.image_count,
        "object_count": result.object_count,
        "download_url": f"/api/export/download/{result.zip_path.name}",
    }


@app.get("/api/export/download/{filename}")
def download_export(filename: str) -> FileResponse:
    settings = get_settings()
    path = settings.exports_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export not found")
    return FileResponse(path)


@app.post("/review/export")
def export_from_review(
    dataset_name: str = Form("mvp-dataset"),
    export_format: str = Form("yolo"),
    only_validated: bool = Form(True),
) -> RedirectResponse:
    result = registry.export_dataset(
        dataset_name=dataset_name,
        export_format=export_format,
        only_validated=only_validated,
    )
    return RedirectResponse(
        url=f"/review?exported={result.zip_path.name}",
        status_code=303,
    )


@app.get("/review", response_class=HTMLResponse)
def review(source_id: Optional[str] = None, exported: Optional[str] = None) -> HTMLResponse:
    _ = source_id, exported
    return _spa_shell()


@app.get("/api/review/workspace")
def api_workspace() -> dict:
    """Workspace view-model for the SPA. Returns sources with nested objects
    in the shape the reference React components expect.
    """
    settings = get_settings()
    _ = settings
    sources_raw = registry.list_sources()
    stats = registry.get_stats()

    sources: list[dict] = []
    projects_seen: dict[str, dict] = {}

    for src in sources_raw:
        source_id = src["source_id"]
        width = src.get("width") or 1
        height = src.get("height") or 1
        project_id = src.get("project_id") or "default-project"
        objects_raw = registry.list_objects(source_id=source_id)

        classes_set: set[str] = set()
        approved = 0
        for obj in objects_raw:
            classes_set.add(obj.get("category_name") or "object")
            if obj.get("is_validated"):
                approved += 1

        total_objs = len(objects_raw)
        if total_objs == 0:
            status = "pending"
        elif approved == total_objs:
            status = "validated"
        else:
            status = "in_review"

        objects: list[dict] = []
        for obj in objects_raw:
            category = obj.get("category_name") or "object"
            objects.append(
                {
                    "object_id": obj["object_id"],
                    "category": category,
                    "color": CATEGORY_COLORS.get(category, "#64748b"),
                    "bbox": [
                        (obj.get("bbox_x") or 0.0) / width,
                        (obj.get("bbox_y") or 0.0) / height,
                        (obj.get("bbox_w") or 0.0) / width,
                        (obj.get("bbox_h") or 0.0) / height,
                    ],
                    "bbox_px": [
                        obj.get("bbox_x") or 0.0,
                        obj.get("bbox_y") or 0.0,
                        obj.get("bbox_w") or 0.0,
                        obj.get("bbox_h") or 0.0,
                    ],
                    "confidence": obj.get("confidence"),
                    "validated": "approved" if obj.get("is_validated") else None,
                    "mask_url": f"/api/masks/{obj['object_id']}" if obj.get("mask_path") else None,
                }
            )

        sources.append(
            {
                "id": source_id,
                "file_name": src.get("file_name") or source_id,
                "url": f"/api/assets/{source_id}",
                "width": width,
                "height": height,
                "status": status,
                "project": project_id,
                "classes": sorted(classes_set),
                "uploaded_at": src.get("created_at"),
                "objects": objects,
            }
        )

        bucket = projects_seen.setdefault(
            project_id, {"id": project_id, "name": project_id, "sources": 0, "validated": 0, "classes": set()}
        )
        bucket["sources"] += 1
        if status == "validated":
            bucket["validated"] += 1
        bucket["classes"].update(classes_set)

    projects = [
        {"id": p["id"], "name": p["name"], "sources": p["sources"], "validated": p["validated"], "classes": sorted(p["classes"])}
        for p in projects_seen.values()
    ]

    return {
        "success": True,
        "sources": sources,
        "projects": projects,
        "stats": stats,
        "category_colors": CATEGORY_COLORS,
        "segmentation_backend": segmenter.backend_name,
    }
