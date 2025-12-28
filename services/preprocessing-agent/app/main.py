"""Preprocessing Agent - Video processing and frame extraction service."""
import io
import os
import tempfile
import uuid
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .video_processor import VideoProcessor, ObjectTracker

app = FastAPI(
    title="Preprocessing Agent",
    description="Video frame extraction and batch processing service",
    version="0.1.0",
)

# Configuration
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8010")
DATA_DIR = os.getenv("DATA_DIR", "/tmp/preprocessing")

# Initialize processor
video_processor = VideoProcessor(output_dir=os.path.join(DATA_DIR, "frames"))

# In-memory job status (use Redis in production)
processing_jobs = {}


class VideoProcessRequest(BaseModel):
    """Request schema for video processing."""
    video_path: str
    project_id: str
    classes: str  # Comma-separated class names
    frame_interval: int = 30  # Process every Nth frame
    max_frames: Optional[int] = None
    confidence: float = 0.5
    create_tracks: bool = True


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float
    total_frames: int
    processed_frames: int
    detected_objects: int
    tracks_created: int
    error: Optional[str] = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "preprocessing-agent"}


@app.post("/video/info")
async def get_video_info(video: UploadFile = File(...)):
    """Get video metadata."""
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            content = await video.read()
            f.write(content)
            temp_path = f.name

        info = video_processor.get_video_info(temp_path)
        info["filename"] = video.filename

        # Cleanup
        os.unlink(temp_path)

        return {"success": True, "data": info}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/video/process")
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    project_id: str = Form(...),
    classes: str = Form(...),
    frame_interval: int = Form(30),
    max_frames: Optional[int] = Form(None),
    confidence: float = Form(0.5),
    create_tracks: bool = Form(True),
):
    """Process video: extract frames, detect objects, create tracks.

    This is an async operation. Returns a job_id that can be used to check progress.
    """
    try:
        # Save video to temp file
        video_id = f"vid_{uuid.uuid4().hex[:12]}"
        video_dir = os.path.join(DATA_DIR, "videos")
        os.makedirs(video_dir, exist_ok=True)

        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        content = await video.read()
        with open(video_path, "wb") as f:
            f.write(content)

        # Get video info
        info = video_processor.get_video_info(video_path)

        # Create job
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        processing_jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0.0,
            "total_frames": info["frame_count"] // frame_interval,
            "processed_frames": 0,
            "detected_objects": 0,
            "tracks_created": 0,
            "video_id": video_id,
            "source_id": None,
            "error": None,
        }

        # Start background processing
        background_tasks.add_task(
            process_video_task,
            job_id=job_id,
            video_path=video_path,
            video_info=info,
            project_id=project_id,
            classes=classes,
            frame_interval=frame_interval,
            max_frames=max_frames,
            confidence=confidence,
            create_tracks=create_tracks,
        )

        return {
            "success": True,
            "job_id": job_id,
            "message": "Video processing started",
            "video_info": info,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/video/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get video processing job status."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"success": True, "data": processing_jobs[job_id]}


@app.get("/video/jobs")
async def list_jobs(limit: int = 20):
    """List recent processing jobs."""
    jobs = list(processing_jobs.values())[-limit:]
    return {"success": True, "jobs": jobs}


async def process_video_task(
    job_id: str,
    video_path: str,
    video_info: dict,
    project_id: str,
    classes: str,
    frame_interval: int,
    max_frames: Optional[int],
    confidence: float,
    create_tracks: bool,
):
    """Background task for video processing."""
    job = processing_jobs[job_id]
    job["status"] = "processing"

    tracker = ObjectTracker(iou_threshold=0.3) if create_tracks else None

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Register video source
            source_resp = await client.post(
                f"{REGISTRY_URL}/sources",
                json={
                    "source_type": "video",
                    "file_path": video_path,
                    "width": video_info["width"],
                    "height": video_info["height"],
                    "frame_count": video_info["frame_count"],
                    "fps": video_info["fps"],
                    "metadata": {"project_id": project_id},
                },
            )
            source_data = source_resp.json()

            if not source_data.get("success"):
                raise Exception("Failed to register video source")

            source_id = source_data["source_id"]
            job["source_id"] = source_id

            # Process frames
            total_objects = 0
            frame_objects = {}  # frame_idx -> list of objects

            for frame_idx, frame, timestamp_ms in video_processor.extract_frames(
                video_path, frame_interval, max_frames
            ):
                # Convert frame to bytes for API call
                frame_bytes = video_processor.frame_to_bytes(frame)

                # Register frame in registry
                frame_id = f"frm_{uuid.uuid4().hex[:12]}"
                # Note: In production, add frame registration endpoint to registry

                # Call auto_label for this frame
                files = {"image": (f"frame_{frame_idx}.jpg", frame_bytes, "image/jpeg")}
                data = {
                    "project_id": project_id,
                    "image_id": f"{source_id}_frame_{frame_idx}",
                    "classes": classes,
                    "confidence": str(confidence),
                    "save": "false",  # Don't save to labeling-agent
                    "register": "true",  # Register to object registry
                }

                try:
                    resp = await client.post(
                        f"{GATEWAY_URL}/auto_label",
                        files=files,
                        data=data,
                        timeout=60.0,
                    )
                    result = resp.json()

                    if result.get("success"):
                        detections = result.get("data", {}).get("detections", 0)
                        object_ids = result.get("registry", {}).get("object_ids", [])
                        total_objects += detections

                        # Get object details for tracking
                        if tracker and object_ids:
                            objects_data = []
                            for obj_id in object_ids:
                                obj_resp = await client.get(f"{REGISTRY_URL}/objects/{obj_id}")
                                obj_data = obj_resp.json()
                                if obj_data.get("success"):
                                    objects_data.append(obj_data["data"])

                            # Update tracker
                            if objects_data:
                                tracker.update(frame_idx, objects_data)
                                frame_objects[frame_idx] = objects_data

                except Exception as e:
                    # Log but continue processing
                    print(f"Error processing frame {frame_idx}: {e}")

                # Update progress
                job["processed_frames"] += 1
                job["detected_objects"] = total_objects
                job["progress"] = job["processed_frames"] / job["total_frames"]

            # Create tracks in registry
            tracks_created = 0
            if tracker:
                tracks = tracker.get_tracks(min_length=3)  # At least 3 frames
                for track_data in tracks:
                    try:
                        track_resp = await client.post(
                            f"{REGISTRY_URL}/tracks",
                            json={
                                "source_id": source_id,
                                "object_ids": track_data["object_ids"],
                                "category": track_data["category"],
                            },
                        )
                        if track_resp.json().get("success"):
                            tracks_created += 1
                    except Exception:
                        pass

                job["tracks_created"] = tracks_created

            job["status"] = "completed"
            job["progress"] = 1.0

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)


@app.post("/frames/process")
async def process_frames(
    frames: List[UploadFile] = File(...),
    project_id: str = Form(...),
    classes: str = Form(...),
    confidence: float = Form(0.5),
):
    """Process multiple frames in batch.

    For processing pre-extracted frames without video processing.
    """
    results = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        for idx, frame_file in enumerate(frames):
            try:
                frame_bytes = await frame_file.read()

                files = {"image": (frame_file.filename, frame_bytes, "image/jpeg")}
                data = {
                    "project_id": project_id,
                    "image_id": frame_file.filename,
                    "classes": classes,
                    "confidence": str(confidence),
                    "save": "false",
                    "register": "true",
                }

                resp = await client.post(
                    f"{GATEWAY_URL}/auto_label",
                    files=files,
                    data=data,
                    timeout=60.0,
                )
                result = resp.json()
                results.append({
                    "frame": frame_file.filename,
                    "success": result.get("success", False),
                    "detections": result.get("data", {}).get("detections", 0),
                })
            except Exception as e:
                results.append({
                    "frame": frame_file.filename,
                    "success": False,
                    "error": str(e),
                })

    return {
        "success": True,
        "processed": len(results),
        "results": results,
    }


@app.post("/tracks/create")
async def create_tracks_from_objects(
    source_id: str = Form(...),
    iou_threshold: float = Form(0.3),
    min_track_length: int = Form(3),
):
    """Create tracks from existing objects using IoU matching.

    Use this to create tracks after objects are already registered.
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Get all objects for this source
            resp = await client.get(
                f"{REGISTRY_URL}/objects",
                params={"source_id": source_id, "limit": 1000},
            )
            data = resp.json()

            if not data.get("success"):
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": "Failed to get objects"},
                )

            objects = data.get("data", [])

            if not objects:
                return {"success": True, "tracks_created": 0, "message": "No objects found"}

            # Group by frame_id (or use created_at order as proxy)
            # For now, assume objects are ordered by creation time
            tracker = ObjectTracker(iou_threshold=iou_threshold)

            # Simulate frame indices based on object order
            for idx, obj in enumerate(objects):
                tracker.update(idx, [obj])

            # Get tracks and create in registry
            tracks = tracker.get_tracks(min_length=min_track_length)
            tracks_created = 0

            for track_data in tracks:
                track_resp = await client.post(
                    f"{REGISTRY_URL}/tracks",
                    json={
                        "source_id": source_id,
                        "object_ids": track_data["object_ids"],
                        "category": track_data["category"],
                    },
                )
                if track_resp.json().get("success"):
                    tracks_created += 1

            return {
                "success": True,
                "tracks_created": tracks_created,
                "total_objects": len(objects),
            }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )
