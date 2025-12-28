"""Detection Agent Service - Florence-2 based object detection."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from .detector import Florence2Detector
from .schemas import DetectionRequest, DetectionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage detector lifecycle."""
    app.state.detector = Florence2Detector()
    yield
    app.state.detector.unload()


app = FastAPI(
    title="Detection Agent",
    description="Florence-2 based object detection service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": "florence-2"}


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    image: UploadFile = File(...),
    classes: str = Form(...),
    confidence: float = Form(0.5),
):
    """Detect objects in an image.

    Args:
        image: Image file to process
        classes: Comma-separated list of class names to detect
        confidence: Minimum confidence threshold (0.0-1.0)
    """
    try:
        class_list = [c.strip() for c in classes.split(",")]
        image_bytes = await image.read()

        result = app.state.detector.detect(
            image_bytes=image_bytes,
            classes=class_list,
            confidence=confidence,
        )

        return DetectionResponse(
            success=True,
            data=result,
            message="Detection completed",
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/unload")
async def unload_model():
    """Unload model to free GPU memory."""
    app.state.detector.unload()
    return {"status": "unloaded"}
