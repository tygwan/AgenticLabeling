"""Segmentation Agent Service - SAM3 based segmentation."""
import io
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from .segmenter import SAM3Segmenter
from .schemas import SegmentationResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.segmenter = SAM3Segmenter()
    yield
    app.state.segmenter.unload()


app = FastAPI(
    title="Segmentation Agent",
    description="SAM3 based image segmentation service",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "sam3"}


@app.post("/segment", response_model=SegmentationResponse)
async def segment(
    image: UploadFile = File(...),
    boxes: str = Form(...),
):
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
    app.state.segmenter.unload()
    return {"status": "unloaded"}
