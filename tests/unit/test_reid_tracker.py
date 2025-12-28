"""Unit tests for Re-ID Tracker."""
import os
import sys
import importlib.util
import pytest
import numpy as np

# Check if cv2 is available (required by video_processor)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Skip all tests if cv2 is not available
pytestmark = pytest.mark.skipif(not HAS_CV2, reason="cv2 not installed")

if HAS_CV2:
    # Dynamic import to avoid conflicts with other 'app' packages
    _video_processor_path = os.path.join(
        os.path.dirname(__file__), "../../services/preprocessing-agent/app/video_processor.py"
    )
    _spec = importlib.util.spec_from_file_location("video_processor", _video_processor_path)
    _video_processor_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_video_processor_module)
    ReIDTracker = _video_processor_module.ReIDTracker
else:
    # Dummy class for import to succeed
    class ReIDTracker:
        pass


class TestReIDTrackerBasic:
    """Test basic ReIDTracker functionality."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker instance."""
        return ReIDTracker(
            iou_threshold=0.3,
            embedding_threshold=0.6,
            max_gap_frames=5,
            max_lost_age=10,
        )

    def test_init(self, tracker):
        """Test tracker initialization."""
        assert tracker.iou_threshold == 0.3
        assert tracker.embedding_threshold == 0.6
        assert tracker.tracks == {}
        assert tracker.lost_tracks == {}
        assert tracker.next_track_id == 1

    def test_compute_iou_identical(self, tracker):
        """Test IoU of identical boxes."""
        box = [10, 10, 50, 50]
        iou = tracker.compute_iou(box, box)
        assert pytest.approx(iou, abs=0.001) == 1.0

    def test_compute_iou_no_overlap(self, tracker):
        """Test IoU of non-overlapping boxes."""
        box1 = [0, 0, 10, 10]
        box2 = [100, 100, 10, 10]
        iou = tracker.compute_iou(box1, box2)
        assert iou == 0.0

    def test_compute_iou_partial_overlap(self, tracker):
        """Test IoU of overlapping boxes."""
        box1 = [0, 0, 20, 20]  # Area: 400
        box2 = [10, 10, 20, 20]  # Area: 400
        # Intersection: [10, 10] to [20, 20] = 10x10 = 100
        # Union: 400 + 400 - 100 = 700
        iou = tracker.compute_iou(box1, box2)
        assert pytest.approx(iou, abs=0.01) == 100 / 700

    def test_compute_embedding_similarity_identical(self, tracker):
        """Test similarity of identical embeddings."""
        emb = np.array([1.0, 0.0, 0.5, 0.3])
        sim = tracker.compute_embedding_similarity(emb, emb)
        assert pytest.approx(sim, abs=0.001) == 1.0

    def test_compute_embedding_similarity_orthogonal(self, tracker):
        """Test similarity of orthogonal embeddings."""
        emb1 = np.array([1.0, 0.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0, 0.0])
        sim = tracker.compute_embedding_similarity(emb1, emb2)
        assert pytest.approx(sim, abs=0.001) == 0.0

    def test_compute_embedding_similarity_none(self, tracker):
        """Test similarity with None embedding."""
        emb = np.array([1.0, 0.0, 0.5, 0.3])
        assert tracker.compute_embedding_similarity(emb, None) == 0.0
        assert tracker.compute_embedding_similarity(None, emb) == 0.0
        assert tracker.compute_embedding_similarity(None, None) == 0.0


class TestReIDTrackerUpdate:
    """Test ReIDTracker update functionality."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker instance."""
        return ReIDTracker(
            iou_threshold=0.3,
            embedding_threshold=0.6,
            max_gap_frames=5,
            max_lost_age=10,
        )

    def test_update_creates_new_track(self, tracker):
        """Test that update creates new tracks for unmatched detections."""
        detections = [
            {
                "object_id": "obj_001",
                "bbox_x": 10, "bbox_y": 10, "bbox_w": 50, "bbox_h": 50,
                "category_name": "person",
            }
        ]

        assignments = tracker.update(0, detections)

        assert len(assignments) == 1
        track_id, obj_id = assignments[0]
        assert track_id == 1
        assert obj_id == "obj_001"
        assert len(tracker.tracks) == 1

    def test_update_matches_existing_track(self, tracker):
        """Test that update matches detection to existing track."""
        # Frame 0: Create track
        det1 = [
            {
                "object_id": "obj_001",
                "bbox_x": 10, "bbox_y": 10, "bbox_w": 50, "bbox_h": 50,
                "category_name": "person",
            }
        ]
        tracker.update(0, det1)

        # Frame 1: Same position, should match
        det2 = [
            {
                "object_id": "obj_002",
                "bbox_x": 12, "bbox_y": 12, "bbox_w": 50, "bbox_h": 50,
                "category_name": "person",
            }
        ]
        assignments = tracker.update(1, det2)

        assert len(assignments) == 1
        track_id, obj_id = assignments[0]
        assert track_id == 1  # Same track
        assert obj_id == "obj_002"

        # Track should have 2 objects
        track = tracker.tracks[1]
        assert len(track["object_ids"]) == 2

    def test_update_with_embedding(self, tracker):
        """Test update with embeddings."""
        emb1 = np.random.randn(768).astype(np.float32)
        emb2 = emb1 + 0.1 * np.random.randn(768).astype(np.float32)  # Similar

        det1 = [
            {
                "object_id": "obj_001",
                "bbox_x": 10, "bbox_y": 10, "bbox_w": 50, "bbox_h": 50,
                "category_name": "person",
            }
        ]
        tracker.update(0, det1, [emb1])

        det2 = [
            {
                "object_id": "obj_002",
                "bbox_x": 100, "bbox_y": 100, "bbox_w": 50, "bbox_h": 50,  # Different position
                "category_name": "person",
            }
        ]
        # Similar embedding should help match even with different position
        tracker.update(1, det2, [emb2])

        # Track should have embedding stored
        track = tracker.tracks[1]
        assert track.get("embedding") is not None

    def test_update_category_mismatch(self, tracker):
        """Test that category mismatch creates new track."""
        det1 = [
            {
                "object_id": "obj_001",
                "bbox_x": 10, "bbox_y": 10, "bbox_w": 50, "bbox_h": 50,
                "category_name": "person",
            }
        ]
        tracker.update(0, det1)

        det2 = [
            {
                "object_id": "obj_002",
                "bbox_x": 12, "bbox_y": 12, "bbox_w": 50, "bbox_h": 50,
                "category_name": "car",  # Different category
            }
        ]
        assignments = tracker.update(1, det2)

        assert len(assignments) == 1
        track_id, obj_id = assignments[0]
        assert track_id == 2  # New track, not matching existing
        assert len(tracker.tracks) == 2

    def test_update_multiple_detections(self, tracker):
        """Test update with multiple detections."""
        detections = [
            {
                "object_id": "obj_001",
                "bbox_x": 10, "bbox_y": 10, "bbox_w": 50, "bbox_h": 50,
                "category_name": "person",
            },
            {
                "object_id": "obj_002",
                "bbox_x": 200, "bbox_y": 200, "bbox_w": 60, "bbox_h": 60,
                "category_name": "car",
            },
        ]

        assignments = tracker.update(0, detections)

        assert len(assignments) == 2
        assert len(tracker.tracks) == 2

    def test_update_empty_detections(self, tracker):
        """Test update with no detections."""
        # First create a track
        det = [
            {
                "object_id": "obj_001",
                "bbox_x": 10, "bbox_y": 10, "bbox_w": 50, "bbox_h": 50,
                "category_name": "person",
            }
        ]
        tracker.update(0, det)

        # Empty update
        assignments = tracker.update(1, [])
        assert len(assignments) == 0


class TestReIDTrackerLostTracks:
    """Test ReIDTracker lost track handling."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker with short gap tolerance."""
        return ReIDTracker(
            iou_threshold=0.3,
            embedding_threshold=0.5,
            max_gap_frames=3,
            max_lost_age=10,
            min_track_hits=2,
        )

    def test_track_becomes_lost(self, tracker):
        """Test that inactive tracks become lost."""
        # Create track with 2 hits
        det1 = [{"object_id": "obj_001", "bbox_x": 10, "bbox_y": 10, "bbox_w": 50, "bbox_h": 50, "category_name": "person"}]
        tracker.update(0, det1)

        det2 = [{"object_id": "obj_002", "bbox_x": 12, "bbox_y": 12, "bbox_w": 50, "bbox_h": 50, "category_name": "person"}]
        tracker.update(1, det2)

        assert len(tracker.tracks) == 1
        assert len(tracker.lost_tracks) == 0

        # Update without matching detection for max_gap_frames
        for i in range(2, 6):
            tracker.update(i, [])

        # Track should be lost now
        assert len(tracker.tracks) == 0
        assert len(tracker.lost_tracks) == 1

    def test_lost_track_reidentified(self, tracker):
        """Test that lost track can be re-identified by embedding."""
        # Create embedding
        emb1 = np.random.randn(768).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)

        # Create track with embedding
        det1 = [{"object_id": "obj_001", "bbox_x": 10, "bbox_y": 10, "bbox_w": 50, "bbox_h": 50, "category_name": "person"}]
        tracker.update(0, det1, [emb1])

        det2 = [{"object_id": "obj_002", "bbox_x": 12, "bbox_y": 12, "bbox_w": 50, "bbox_h": 50, "category_name": "person"}]
        tracker.update(1, det2, [emb1])

        original_track_id = list(tracker.tracks.keys())[0]

        # Make track lost
        for i in range(2, 6):
            tracker.update(i, [])

        assert original_track_id in tracker.lost_tracks

        # Re-identify with similar embedding at different position
        emb_similar = emb1 + 0.05 * np.random.randn(768).astype(np.float32)
        emb_similar = emb_similar / np.linalg.norm(emb_similar)

        det3 = [{"object_id": "obj_010", "bbox_x": 300, "bbox_y": 300, "bbox_w": 50, "bbox_h": 50, "category_name": "person"}]
        assignments = tracker.update(6, det3, [emb_similar])

        # Should reactivate the original track
        assert len(assignments) == 1
        track_id, obj_id = assignments[0]
        assert track_id == original_track_id
        assert original_track_id in tracker.tracks
        assert original_track_id not in tracker.lost_tracks

    def test_lost_track_expires(self, tracker):
        """Test that old lost tracks are removed."""
        # Create track
        det1 = [{"object_id": "obj_001", "bbox_x": 10, "bbox_y": 10, "bbox_w": 50, "bbox_h": 50, "category_name": "person"}]
        tracker.update(0, det1)

        det2 = [{"object_id": "obj_002", "bbox_x": 12, "bbox_y": 12, "bbox_w": 50, "bbox_h": 50, "category_name": "person"}]
        tracker.update(1, det2)

        # Make track lost
        for i in range(2, 6):
            tracker.update(i, [])

        assert len(tracker.lost_tracks) == 1

        # Age beyond max_lost_age
        for i in range(6, 20):
            tracker.update(i, [])

        assert len(tracker.lost_tracks) == 0


class TestReIDTrackerGetTracks:
    """Test ReIDTracker get_tracks functionality."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker instance."""
        return ReIDTracker()

    def test_get_tracks_min_length(self, tracker):
        """Test get_tracks with minimum length filter."""
        # Create track with 1 detection
        det1 = [{"object_id": "obj_001", "bbox_x": 10, "bbox_y": 10, "bbox_w": 50, "bbox_h": 50, "category_name": "person"}]
        tracker.update(0, det1)

        # Create track with 3 detections
        for i in range(1, 4):
            det = [{"object_id": f"obj_00{i+1}", "bbox_x": 10+i, "bbox_y": 10+i, "bbox_w": 50, "bbox_h": 50, "category_name": "person"}]
            tracker.update(i, det)

        # Min length 2 should return the track
        tracks = tracker.get_tracks(min_length=2)
        assert len(tracks) == 1
        assert tracks[0]["length"] >= 2

        # Min length 5 should return nothing
        tracks = tracker.get_tracks(min_length=5)
        assert len(tracks) == 0

    def test_get_tracks_structure(self, tracker):
        """Test that get_tracks returns correct structure."""
        det1 = [{"object_id": "obj_001", "bbox_x": 10, "bbox_y": 10, "bbox_w": 50, "bbox_h": 50, "category_name": "person"}]
        tracker.update(0, det1)

        det2 = [{"object_id": "obj_002", "bbox_x": 12, "bbox_y": 12, "bbox_w": 50, "bbox_h": 50, "category_name": "person"}]
        tracker.update(1, det2)

        tracks = tracker.get_tracks(min_length=1)
        assert len(tracks) == 1

        track = tracks[0]
        assert "track_id" in track
        assert "object_ids" in track
        assert "frame_indices" in track
        assert "category" in track
        assert "length" in track
        assert "is_active" in track

        assert track["object_ids"] == ["obj_001", "obj_002"]
        assert track["frame_indices"] == [0, 1]
        assert track["category"] == "person"

    def test_get_active_tracks(self, tracker):
        """Test get_active_tracks."""
        det = [
            {"object_id": "obj_001", "bbox_x": 10, "bbox_y": 10, "bbox_w": 50, "bbox_h": 50, "category_name": "person"},
            {"object_id": "obj_002", "bbox_x": 100, "bbox_y": 100, "bbox_w": 50, "bbox_h": 50, "category_name": "car"},
        ]
        tracker.update(0, det)

        active = tracker.get_active_tracks()
        assert len(active) == 2

        for track in active:
            assert "track_id" in track
            assert "category" in track
            assert "last_bbox" in track
            assert "last_object_id" in track


class TestReIDTrackerReset:
    """Test ReIDTracker reset functionality."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker instance."""
        return ReIDTracker()

    def test_reset_clears_state(self, tracker):
        """Test that reset clears all state."""
        # Add some tracks
        det = [{"object_id": "obj_001", "bbox_x": 10, "bbox_y": 10, "bbox_w": 50, "bbox_h": 50, "category_name": "person"}]
        tracker.update(0, det)
        tracker.update(1, det)

        assert len(tracker.tracks) == 1
        assert tracker.next_track_id > 1

        # Reset
        tracker.reset()

        assert tracker.tracks == {}
        assert tracker.lost_tracks == {}
        assert tracker.next_track_id == 1
        assert tracker.current_frame == 0


class TestReIDTrackerMotionPrediction:
    """Test ReIDTracker motion prediction."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker instance."""
        return ReIDTracker()

    def test_predict_position_stationary(self, tracker):
        """Test prediction for stationary object."""
        track_state = {
            "last_bbox": [10, 10, 50, 50],
            "bbox_history": [[10, 10, 50, 50], [10, 10, 50, 50]],
            "last_frame": 1,
        }
        tracker.current_frame = 2

        predicted = tracker.predict_position(track_state)

        # Should predict same position for stationary object
        assert predicted[0] == pytest.approx(10, abs=1)
        assert predicted[1] == pytest.approx(10, abs=1)

    def test_predict_position_moving(self, tracker):
        """Test prediction for moving object."""
        track_state = {
            "last_bbox": [20, 20, 50, 50],
            "bbox_history": [[10, 10, 50, 50], [20, 20, 50, 50]],  # Moving +10 per frame
            "last_frame": 1,
        }
        tracker.current_frame = 2

        predicted = tracker.predict_position(track_state)

        # Should predict position at 30, 30 (one frame forward at velocity 10)
        assert predicted[0] == pytest.approx(30, abs=1)
        assert predicted[1] == pytest.approx(30, abs=1)

    def test_predict_position_insufficient_history(self, tracker):
        """Test prediction with insufficient history."""
        track_state = {
            "last_bbox": [10, 10, 50, 50],
            "bbox_history": [[10, 10, 50, 50]],  # Only one position
            "last_frame": 0,
        }
        tracker.current_frame = 1

        predicted = tracker.predict_position(track_state)

        # Should return last known position
        assert predicted == [10, 10, 50, 50]
