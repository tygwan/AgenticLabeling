"""API Gateway - Routes requests to appropriate services."""
import os
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AgenticLabeling Gateway",
    description="API Gateway for AgenticLabeling microservices",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs from environment
SERVICES = {
    "detection": os.getenv("DETECTION_URL", "http://detection-agent:8001"),
    "segmentation": os.getenv("SEGMENTATION_URL", "http://segmentation-agent:8002"),
    "classification": os.getenv("CLASSIFICATION_URL", "http://classification-agent:8003"),
    "labeling": os.getenv("LABELING_URL", "http://labeling-agent:8004"),
    "training": os.getenv("TRAINING_URL", "http://training-agent:8005"),
}


@app.get("/health")
async def health():
    """Check gateway and service health."""
    service_status = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in SERVICES.items():
            try:
                resp = await client.get(f"{url}/health")
                service_status[name] = resp.json() if resp.status_code == 200 else "unhealthy"
            except Exception:
                service_status[name] = "unreachable"

    return {"gateway": "healthy", "services": service_status}


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    classes: str = Form(...),
    confidence: float = Form(0.5),
):
    """Forward detection request to detection-agent."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        files = {"image": (image.filename, await image.read(), image.content_type)}
        data = {"classes": classes, "confidence": str(confidence)}

        resp = await client.post(f"{SERVICES['detection']}/detect", files=files, data=data)
        return resp.json()


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    boxes: str = Form(...),
):
    """Forward segmentation request to segmentation-agent."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        files = {"image": (image.filename, await image.read(), image.content_type)}
        data = {"boxes": boxes}

        resp = await client.post(f"{SERVICES['segmentation']}/segment", files=files, data=data)
        return resp.json()


@app.post("/detect_and_segment")
async def detect_and_segment(
    image: UploadFile = File(...),
    classes: str = Form(...),
    confidence: float = Form(0.5),
):
    """Pipeline: detect objects then segment each one."""
    import json

    image_bytes = await image.read()

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Step 1: Detection
        files = {"image": (image.filename, image_bytes, image.content_type)}
        data = {"classes": classes, "confidence": str(confidence)}

        detect_resp = await client.post(
            f"{SERVICES['detection']}/detect", files=files, data=data
        )
        detect_result = detect_resp.json()

        if not detect_result.get("success"):
            return detect_result

        boxes = detect_result["data"]["boxes"]
        if not boxes:
            return {"success": True, "data": {"detections": [], "masks": []}}

        # Step 2: Segmentation
        files = {"image": (image.filename, image_bytes, image.content_type)}
        data = {"boxes": json.dumps(boxes)}

        segment_resp = await client.post(
            f"{SERVICES['segmentation']}/segment", files=files, data=data
        )
        segment_result = segment_resp.json()

        return {
            "success": True,
            "data": {
                "detections": detect_result["data"],
                "segmentation": segment_result.get("data"),
            },
        }


@app.post("/auto_label")
async def auto_label(
    image: UploadFile = File(...),
    project_id: str = Form(...),
    image_id: str = Form(...),
    classes: str = Form(...),
    confidence: float = Form(0.5),
    save: bool = Form(True),
):
    """Pipeline: detect -> segment -> save labels.

    Complete auto-labeling pipeline that detects objects, segments them,
    and saves the results to labeling-agent.
    """
    import json

    image_bytes = await image.read()
    classes_list = [c.strip() for c in classes.split(",")]

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Step 1: Detection
        files = {"image": (image.filename, image_bytes, image.content_type)}
        data = {"classes": classes, "confidence": str(confidence)}

        detect_resp = await client.post(
            f"{SERVICES['detection']}/detect",
            files=files,
            data=data,
        )
        detect_result = detect_resp.json()

        if not detect_result.get("success"):
            return {"success": False, "error": "Detection failed", "detail": detect_result}

        boxes = detect_result["data"]["boxes"]
        labels = detect_result["data"]["labels"]

        if not boxes:
            return {
                "success": True,
                "data": {"detections": 0, "message": "No objects detected"},
                "saved": False,
            }

        # Step 2: Segmentation
        files = {"image": (image.filename, image_bytes, image.content_type)}
        segment_resp = await client.post(
            f"{SERVICES['segmentation']}/segment",
            files=files,
            data={"boxes": json.dumps(boxes)},
        )
        segment_result = segment_resp.json()

        masks = []
        if segment_result.get("success"):
            masks = [m["mask"] for m in segment_result["data"]["masks"]]

        # Step 3: Save to labeling-agent
        saved = False
        if save:
            label_data = {
                "project_id": project_id,
                "image_id": image_id,
                "boxes": boxes,
                "classes": labels,
                "masks": masks,
            }
            save_resp = await client.post(
                f"{SERVICES['labeling']}/labels/save",
                json=label_data,
            )
            saved = save_resp.json().get("success", False)

        return {
            "success": True,
            "data": {
                "image_id": image_id,
                "detections": len(boxes),
                "boxes": boxes,
                "labels": labels,
                "masks_count": len(masks),
                "image_size": detect_result["data"]["image_size"],
            },
            "saved": saved,
        }
