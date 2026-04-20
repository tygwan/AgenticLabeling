"""FastAPI monolith for the AgenticLabeling MVP."""

from __future__ import annotations

from contextlib import asynccontextmanager
import html
import io
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from PIL import Image, ImageDraw

from .config import get_settings
from .detector import DetectionService
from .registry import Registry
from .segmenter import SegmentationService
from .storage import init_storage, store_image_bytes


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_storage()
    yield


app = FastAPI(title="AgenticLabeling MVP", version="0.1.0", lifespan=lifespan)
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


def _render_page(title: str, body: str) -> HTMLResponse:
    page = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>{html.escape(title)}</title>
      <style>
        body {{ font-family: sans-serif; margin: 2rem; background: #f7f7f7; color: #222; }}
        main {{ max-width: 1200px; margin: 0 auto; }}
        section, form, table {{ background: white; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; }}
        input, button, select {{ padding: 0.6rem; margin: 0.2rem 0; }}
        button {{ cursor: pointer; }}
        .grid {{ display: grid; grid-template-columns: 280px 1fr 360px; gap: 1rem; }}
        .image-compare {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }}
        .image-card {{ background: #fff; border: 1px solid #eee; border-radius: 12px; padding: 0.75rem; }}
        .source-link {{ display: block; padding: 0.4rem 0; }}
        img {{ max-width: 100%; border-radius: 12px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        td, th {{ padding: 0.5rem; border-bottom: 1px solid #ddd; text-align: left; }}
        .pill {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; background: #eee; }}
      </style>
    </head>
    <body>
      <main>
        <h1>{html.escape(title)}</h1>
        {body}
      </main>
    </body>
    </html>
    """
    return HTMLResponse(page)


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


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    body = """
    <section>
      <h2>Upload And Auto Label</h2>
      <form action="/upload" method="post" enctype="multipart/form-data">
        <div><input type="file" name="image" accept="image/*" required /></div>
        <div><input type="text" name="project_id" value="default-project" placeholder="Project ID" /></div>
        <div><input type="text" name="classes" value="person,car,dog" placeholder="Comma separated classes" /></div>
        <div><button type="submit">Run Auto Label</button></div>
      </form>
    </section>
    <section>
      <a href="/review">Open Review UI</a>
    </section>
    """
    return _render_page("AgenticLabeling MVP", body)


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
    sources = registry.list_sources()
    if not source_id and sources:
        source_id = sources[0]["source_id"]
    source = registry.get_source(source_id) if source_id else None
    objects = registry.list_objects(source_id=source_id) if source_id else []
    stats = registry.get_stats()

    sources_html = "".join(
        f'<a class="source-link" href="/review?source_id={src["source_id"]}">{html.escape(src["file_name"])}</a>'
        for src in sources
    ) or "<p>No sources yet.</p>"

    if source:
        image_html = (
            f'<section><h2>{html.escape(source["file_name"])}</h2>'
            f'<p>Segmentation backend: <span class="pill">{html.escape(segmenter.backend_name)}</span></p>'
            '<div class="image-compare">'
            f'<div class="image-card"><h3>Original</h3><img src="/api/assets/{source["source_id"]}" alt="source image" /></div>'
            f'<div class="image-card"><h3>BBox Overlay</h3><img src="/api/assets/{source["source_id"]}/bbox-overlay" alt="bbox overlay" /></div>'
            f'<div class="image-card"><h3>Segmentation Overlay</h3><img src="/api/assets/{source["source_id"]}/overlay" alt="segmentation overlay" /></div>'
            '</div></section>'
        )
    else:
        image_html = "<section><p>No source selected.</p></section>"

    object_rows = []
    for obj in objects:
        status = "validated" if obj["is_validated"] else "pending"
        approve = (
            ""
            if obj["is_validated"]
            else (
                f'<form action="/review/objects/{obj["object_id"]}/approve" method="post">'
                f'<input type="hidden" name="source_id" value="{source_id}" />'
                '<button type="submit">Approve</button></form>'
            )
        )
        delete = (
            f'<form action="/review/objects/{obj["object_id"]}/delete" method="post">'
            f'<input type="hidden" name="source_id" value="{source_id}" />'
            '<button type="submit">Delete</button></form>'
        )
        object_rows.append(
            "<tr>"
            f"<td>{html.escape(obj['category_name'])}</td>"
            f"<td>{obj['confidence'] or 0:.2f}</td>"
            f"<td>{obj['bbox_x']:.0f}, {obj['bbox_y']:.0f}, {obj['bbox_w']:.0f}, {obj['bbox_h']:.0f}</td>"
            f"<td><span class='pill'>{status}</span></td>"
            f"<td>{approve}{delete}</td>"
            "</tr>"
        )
    objects_html = (
        "<section><h2>Objects</h2><table><thead><tr>"
        "<th>Category</th><th>Confidence</th><th>BBox (x,y,w,h)</th><th>Status</th><th>Actions</th>"
        "</tr></thead><tbody>"
        + "".join(object_rows)
        + "</tbody></table></section>"
    ) if objects else "<section><p>No objects for this source.</p></section>"

    export_notice = (
        f'<section><p>Last export ready: <a href="/api/export/download/{html.escape(exported)}">{html.escape(exported)}</a></p></section>'
        if exported
        else ""
    )

    body = f"""
    <section>
      <p>Sources: {stats['sources']} | Objects: {stats['objects']} | Validated: {stats['validated_objects']}</p>
    </section>
    {export_notice}
    <div class="grid">
      <section><h2>Sources</h2>{sources_html}</section>
      {image_html}
      <section>
        <h2>Export</h2>
        <form action="/review/export" method="post">
          <div><input type="text" name="dataset_name" value="mvp-dataset" /></div>
          <div>
            <select name="export_format">
              <option value="yolo">YOLO</option>
              <option value="coco">COCO</option>
            </select>
          </div>
          <div><label><input type="checkbox" name="only_validated" checked /> Only validated</label></div>
          <div><button type="submit">Export dataset</button></div>
        </form>
      </section>
    </div>
    {objects_html}
    """
    return _render_page("Review Workspace", body)
