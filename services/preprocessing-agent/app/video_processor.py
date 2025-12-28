"""Video processing utilities."""
import os
import tempfile
import uuid
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np


class VideoProcessor:
    """Extract and process frames from video files."""

    def __init__(self, output_dir: str = "/tmp/frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_ms": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) * 1000),
        }

        cap.release()
        return info

    def extract_frames(
        self,
        video_path: str,
        frame_interval: int = 1,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[int, np.ndarray, int], None, None]:
        """Extract frames from video.

        Args:
            video_path: Path to video file
            frame_interval: Extract every Nth frame (default: 1 = all frames)
            max_frames: Maximum number of frames to extract

        Yields:
            Tuple of (frame_index, frame_array, timestamp_ms)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_ms = int(frame_idx / fps * 1000)
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_idx, frame_rgb, timestamp_ms

                extracted_count += 1
                if max_frames and extracted_count >= max_frames:
                    break

            frame_idx += 1

        cap.release()

    def save_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        video_id: str,
    ) -> str:
        """Save frame to disk.

        Args:
            frame: Frame array (RGB)
            frame_idx: Frame index
            video_id: Video identifier

        Returns:
            Path to saved frame
        """
        frame_dir = self.output_dir / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        frame_path = frame_dir / f"frame_{frame_idx:06d}.jpg"

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), frame_bgr)

        return str(frame_path)

    def frame_to_bytes(self, frame: np.ndarray, format: str = "jpg") -> bytes:
        """Convert frame to bytes.

        Args:
            frame: Frame array (RGB)
            format: Output format (jpg, png)

        Returns:
            Frame bytes
        """
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if format.lower() == "png":
            _, buffer = cv2.imencode(".png", frame_bgr)
        else:
            _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return buffer.tobytes()


class ObjectTracker:
    """Simple IoU-based object tracker for connecting objects across frames."""

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.tracks = {}  # track_id -> list of (frame_idx, object_id, bbox)
        self.next_track_id = 1

    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two bounding boxes.

        Args:
            box1: [x, y, w, h]
            box2: [x, y, w, h]

        Returns:
            IoU score
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to x1, y1, x2, y2 format
        box1_x2 = x1 + w1
        box1_y2 = y1 + h1
        box2_x2 = x2 + w2
        box2_y2 = y2 + h2

        # Intersection
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def update(
        self,
        frame_idx: int,
        detections: List[dict],
    ) -> List[Tuple[int, str]]:
        """Update tracks with new detections.

        Args:
            frame_idx: Current frame index
            detections: List of detection dicts with object_id, bbox, category

        Returns:
            List of (track_id, object_id) assignments
        """
        if not detections:
            return []

        assignments = []

        # Get active tracks (tracks that were updated in recent frames)
        active_tracks = {
            tid: track_data
            for tid, track_data in self.tracks.items()
            if track_data and (frame_idx - track_data[-1][0]) <= 5  # Max gap of 5 frames
        }

        # Match detections to existing tracks
        unmatched_detections = list(range(len(detections)))
        matched_tracks = set()

        for det_idx in list(unmatched_detections):
            det = detections[det_idx]
            det_bbox = [det["bbox_x"], det["bbox_y"], det["bbox_w"], det["bbox_h"]]
            det_category = det.get("category_name")

            best_track_id = None
            best_iou = 0

            for track_id, track_data in active_tracks.items():
                if track_id in matched_tracks:
                    continue

                # Get last detection in track
                last_frame_idx, _, last_bbox, last_category = track_data[-1]

                # Only match same category
                if det_category != last_category:
                    continue

                iou = self.compute_iou(det_bbox, last_bbox)
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                # Assign to existing track
                self.tracks[best_track_id].append((
                    frame_idx,
                    det["object_id"],
                    det_bbox,
                    det_category,
                ))
                assignments.append((best_track_id, det["object_id"]))
                matched_tracks.add(best_track_id)
                unmatched_detections.remove(det_idx)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            det_bbox = [det["bbox_x"], det["bbox_y"], det["bbox_w"], det["bbox_h"]]
            det_category = det.get("category_name")

            track_id = self.next_track_id
            self.next_track_id += 1

            self.tracks[track_id] = [(
                frame_idx,
                det["object_id"],
                det_bbox,
                det_category,
            )]
            assignments.append((track_id, det["object_id"]))

        return assignments

    def get_tracks(self, min_length: int = 2) -> List[dict]:
        """Get all tracks with minimum length.

        Args:
            min_length: Minimum number of detections in track

        Returns:
            List of track dicts
        """
        result = []
        for track_id, track_data in self.tracks.items():
            if len(track_data) >= min_length:
                result.append({
                    "track_id": track_id,
                    "object_ids": [d[1] for d in track_data],
                    "frame_indices": [d[0] for d in track_data],
                    "category": track_data[0][3] if track_data else None,
                    "length": len(track_data),
                })
        return result


class ReIDTracker:
    """Re-Identification based object tracker using appearance embeddings.

    This tracker combines IoU-based matching with appearance similarity
    for more robust tracking across occlusions and fast movements.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        embedding_threshold: float = 0.6,
        iou_weight: float = 0.3,
        embedding_weight: float = 0.7,
        max_gap_frames: int = 30,
        max_lost_age: int = 60,
        min_track_hits: int = 3,
    ):
        """Initialize Re-ID tracker.

        Args:
            iou_threshold: Minimum IoU for spatial matching
            embedding_threshold: Minimum embedding cosine similarity
            iou_weight: Weight for IoU in combined score
            embedding_weight: Weight for embedding similarity in combined score
            max_gap_frames: Maximum frames gap to maintain track
            max_lost_age: Maximum age before removing lost track
            min_track_hits: Minimum hits before track is confirmed
        """
        self.iou_threshold = iou_threshold
        self.embedding_threshold = embedding_threshold
        self.iou_weight = iou_weight
        self.embedding_weight = embedding_weight
        self.max_gap_frames = max_gap_frames
        self.max_lost_age = max_lost_age
        self.min_track_hits = min_track_hits

        # Track storage
        self.tracks = {}  # track_id -> TrackState
        self.next_track_id = 1
        self.current_frame = 0

        # Lost tracks (for re-identification)
        self.lost_tracks = {}  # track_id -> TrackState

    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two bounding boxes [x, y, w, h]."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        box1_x2 = x1 + w1
        box1_y2 = y1 + h1
        box2_x2 = x2 + w2
        box2_y2 = y2 + h2

        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area

    def compute_embedding_similarity(
        self, emb1: np.ndarray, emb2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0

        emb1 = np.array(emb1).flatten()
        emb2 = np.array(emb2).flatten()

        if emb1.shape != emb2.shape:
            return 0.0

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def compute_affinity_score(
        self,
        iou: float,
        embedding_sim: float,
    ) -> float:
        """Compute combined affinity score from IoU and embedding similarity."""
        # If embedding is not available, rely more on IoU
        if embedding_sim == 0.0:
            return iou

        # Combined weighted score
        return (self.iou_weight * iou + self.embedding_weight * embedding_sim)

    def predict_position(self, track_state: dict) -> List[float]:
        """Predict next position using simple linear motion model."""
        history = track_state.get("bbox_history", [])
        if len(history) < 2:
            return track_state["last_bbox"]

        # Get last two positions
        prev_bbox = history[-2]
        curr_bbox = history[-1]

        # Calculate velocity
        vx = curr_bbox[0] - prev_bbox[0]
        vy = curr_bbox[1] - prev_bbox[1]

        # Predict next position
        frames_gap = self.current_frame - track_state["last_frame"]
        pred_x = curr_bbox[0] + vx * frames_gap
        pred_y = curr_bbox[1] + vy * frames_gap

        return [pred_x, pred_y, curr_bbox[2], curr_bbox[3]]

    def update(
        self,
        frame_idx: int,
        detections: List[dict],
        embeddings: Optional[List[np.ndarray]] = None,
    ) -> List[Tuple[int, str]]:
        """Update tracks with new detections.

        Args:
            frame_idx: Current frame index
            detections: List of detection dicts with object_id, bbox, category
            embeddings: Optional list of appearance embeddings per detection

        Returns:
            List of (track_id, object_id) assignments
        """
        self.current_frame = frame_idx

        if not detections:
            self._update_lost_tracks()
            return []

        # Ensure embeddings list
        if embeddings is None:
            embeddings = [None] * len(detections)

        assignments = []
        unmatched_detections = list(range(len(detections)))
        matched_tracks = set()

        # Get active tracks
        active_track_ids = list(self.tracks.keys())

        # Build cost matrix (detection x track)
        if active_track_ids:
            cost_matrix = np.zeros((len(detections), len(active_track_ids)))

            for det_idx, det in enumerate(detections):
                det_bbox = [det["bbox_x"], det["bbox_y"], det["bbox_w"], det["bbox_h"]]
                det_category = det.get("category_name")
                det_embedding = embeddings[det_idx]

                for track_idx, track_id in enumerate(active_track_ids):
                    track_state = self.tracks[track_id]

                    # Skip if category mismatch
                    if det_category != track_state.get("category"):
                        cost_matrix[det_idx, track_idx] = -1
                        continue

                    # Predict track position
                    pred_bbox = self.predict_position(track_state)

                    # Compute IoU with predicted position
                    iou = self.compute_iou(det_bbox, pred_bbox)

                    # Compute embedding similarity
                    emb_sim = self.compute_embedding_similarity(
                        det_embedding, track_state.get("embedding")
                    )

                    # Combined affinity score
                    affinity = self.compute_affinity_score(iou, emb_sim)
                    cost_matrix[det_idx, track_idx] = affinity

            # Hungarian matching (greedy for now, could use scipy.optimize.linear_sum_assignment)
            while True:
                # Find best match
                max_val = -1
                best_det = -1
                best_track = -1

                for det_idx in unmatched_detections:
                    for track_idx, track_id in enumerate(active_track_ids):
                        if track_id in matched_tracks:
                            continue

                        score = cost_matrix[det_idx, track_idx]
                        if score > max_val:
                            max_val = score
                            best_det = det_idx
                            best_track = track_idx

                # Check if valid match
                min_threshold = max(self.iou_threshold, self.embedding_threshold) * 0.5
                if max_val < min_threshold or best_det == -1:
                    break

                # Assign detection to track
                track_id = active_track_ids[best_track]
                det = detections[best_det]
                det_bbox = [det["bbox_x"], det["bbox_y"], det["bbox_w"], det["bbox_h"]]
                det_embedding = embeddings[best_det]

                # Update track state
                self._update_track(
                    track_id, frame_idx, det["object_id"],
                    det_bbox, det_embedding, det.get("category_name")
                )

                assignments.append((track_id, det["object_id"]))
                matched_tracks.add(track_id)
                unmatched_detections.remove(best_det)

        # Try to match remaining detections with lost tracks (re-identification)
        for det_idx in list(unmatched_detections):
            det = detections[det_idx]
            det_embedding = embeddings[det_idx]

            if det_embedding is None:
                continue

            best_lost_id = None
            best_similarity = self.embedding_threshold

            for lost_id, lost_state in self.lost_tracks.items():
                if det.get("category_name") != lost_state.get("category"):
                    continue

                sim = self.compute_embedding_similarity(
                    det_embedding, lost_state.get("embedding")
                )

                if sim > best_similarity:
                    best_similarity = sim
                    best_lost_id = lost_id

            if best_lost_id is not None:
                # Reactivate lost track
                det_bbox = [det["bbox_x"], det["bbox_y"], det["bbox_w"], det["bbox_h"]]
                lost_state = self.lost_tracks.pop(best_lost_id)
                self.tracks[best_lost_id] = lost_state

                self._update_track(
                    best_lost_id, frame_idx, det["object_id"],
                    det_bbox, det_embedding, det.get("category_name")
                )

                assignments.append((best_lost_id, det["object_id"]))
                unmatched_detections.remove(det_idx)

        # Create new tracks for remaining unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            det_bbox = [det["bbox_x"], det["bbox_y"], det["bbox_w"], det["bbox_h"]]
            det_embedding = embeddings[det_idx]

            track_id = self._create_track(
                frame_idx, det["object_id"], det_bbox,
                det_embedding, det.get("category_name")
            )
            assignments.append((track_id, det["object_id"]))

        # Update lost tracks and age out old ones
        self._update_lost_tracks()

        return assignments

    def _create_track(
        self,
        frame_idx: int,
        object_id: str,
        bbox: List[float],
        embedding: Optional[np.ndarray],
        category: Optional[str],
    ) -> int:
        """Create a new track."""
        track_id = self.next_track_id
        self.next_track_id += 1

        self.tracks[track_id] = {
            "track_id": track_id,
            "category": category,
            "last_frame": frame_idx,
            "last_bbox": bbox,
            "last_object_id": object_id,
            "embedding": embedding,
            "hits": 1,
            "age": 0,
            "bbox_history": [bbox],
            "object_ids": [object_id],
            "frame_indices": [frame_idx],
        }

        return track_id

    def _update_track(
        self,
        track_id: int,
        frame_idx: int,
        object_id: str,
        bbox: List[float],
        embedding: Optional[np.ndarray],
        category: Optional[str],
    ):
        """Update existing track with new detection."""
        track = self.tracks[track_id]
        track["last_frame"] = frame_idx
        track["last_bbox"] = bbox
        track["last_object_id"] = object_id
        track["hits"] += 1
        track["age"] = 0

        # Update embedding with exponential moving average
        if embedding is not None:
            if track.get("embedding") is not None:
                alpha = 0.3  # Weight for new embedding
                track["embedding"] = (
                    alpha * np.array(embedding) +
                    (1 - alpha) * np.array(track["embedding"])
                )
            else:
                track["embedding"] = embedding

        # Keep last N bboxes for motion prediction
        track["bbox_history"].append(bbox)
        if len(track["bbox_history"]) > 10:
            track["bbox_history"] = track["bbox_history"][-10:]

        track["object_ids"].append(object_id)
        track["frame_indices"].append(frame_idx)

    def _update_lost_tracks(self):
        """Move inactive tracks to lost and age out old lost tracks."""
        # Move inactive tracks to lost
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            gap = self.current_frame - track["last_frame"]
            if gap > self.max_gap_frames:
                if track["hits"] >= self.min_track_hits:
                    # Save to lost tracks for potential re-identification
                    track["age"] = gap
                    self.lost_tracks[track_id] = track
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        # Age out old lost tracks
        lost_to_remove = []
        for track_id, track in self.lost_tracks.items():
            track["age"] = self.current_frame - track["last_frame"]
            if track["age"] > self.max_lost_age:
                lost_to_remove.append(track_id)

        for track_id in lost_to_remove:
            del self.lost_tracks[track_id]

    def get_tracks(self, min_length: int = 2, include_lost: bool = False) -> List[dict]:
        """Get all tracks with minimum length.

        Args:
            min_length: Minimum number of detections in track
            include_lost: Include lost tracks in results

        Returns:
            List of track dicts
        """
        result = []

        all_tracks = dict(self.tracks)
        if include_lost:
            all_tracks.update(self.lost_tracks)

        for track_id, track in all_tracks.items():
            if len(track.get("object_ids", [])) >= min_length:
                result.append({
                    "track_id": track_id,
                    "object_ids": track["object_ids"],
                    "frame_indices": track["frame_indices"],
                    "category": track.get("category"),
                    "length": len(track["object_ids"]),
                    "hits": track.get("hits", 0),
                    "is_active": track_id in self.tracks,
                })

        return result

    def get_active_tracks(self) -> List[dict]:
        """Get currently active (not lost) tracks."""
        return [
            {
                "track_id": track_id,
                "category": track.get("category"),
                "last_bbox": track["last_bbox"],
                "last_object_id": track["last_object_id"],
                "hits": track.get("hits", 0),
            }
            for track_id, track in self.tracks.items()
        ]

    def reset(self):
        """Reset tracker state."""
        self.tracks = {}
        self.lost_tracks = {}
        self.next_track_id = 1
        self.current_frame = 0
