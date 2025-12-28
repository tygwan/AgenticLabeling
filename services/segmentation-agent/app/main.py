"""Segmentation Agent Service - SAM2 based segmentation."""
import io
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from .segmenter import SAM2Segmenter
from .schemas import SegmentationResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage segmenter lifecycle."""
    app.state.segmenter = SAM2Segmenter()
    yield
    app.state.segmenter.unload()


app = FastAPI(
    title="Segmentation Agent",
    description="SAM2 based image segmentation service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": "sam2"}


@app.post("/segment", response_model=SegmentationResponse)
async def segment(
    image: UploadFile = File(...),
    boxes: str = Form(...),
):
    """Segment objects in an image using bounding boxes.

    Args:
        image: Image file to process
        boxes: JSON string of bounding boxes [[x1,y1,x2,y2], ...]
    """
    import json

    try:
        box_list = json.loads(boxes)
        image_bytes = await image.read()

        result = app.state.segmenter.segment(
            image_bytes=image_bytes,
            boxes=box_list,
        )

        return SegmentationResponse(
            success=True,
            data=result,
            message="Segmentation completed",
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/segment_points")
async def segment_points(
    image: UploadFile = File(...),
    points: str = Form(...),
    labels: str = Form(...),
):
    """Segment objects using point prompts.

    Args:
        image: Image file to process
        points: JSON string of points [[x, y], ...]
        labels: JSON string of labels [1, 0, ...] (1=foreground, 0=background)
    """
    import json

    try:
        point_list = json.loads(points)
        label_list = json.loads(labels)
        image_bytes = await image.read()

        result = app.state.segmenter.segment_with_points(
            image_bytes=image_bytes,
            points=point_list,
            labels=label_list,
        )

        return SegmentationResponse(
            success=True,
            data=result,
            message="Point-based segmentation completed",
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/unload")
async def unload_model():
    """Unload model to free GPU memory."""
    app.state.segmenter.unload()
    return {"status": "unloaded"}
