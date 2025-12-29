"""Unit tests for Object Registry."""
import os
import sys
import importlib.util
import pytest

# Dynamic import to avoid conflicts with other 'app' packages
_registry_path = os.path.join(
    os.path.dirname(__file__), "../../services/object-registry/app/registry.py"
)
_spec = importlib.util.spec_from_file_location("registry", _registry_path)
_registry_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_registry_module)
ObjectRegistry = _registry_module.ObjectRegistry


class TestObjectRegistry:
    """Test ObjectRegistry class."""

    @pytest.fixture
    def registry(self, temp_data_dir):
        """Create a registry instance with temp directory."""
        return ObjectRegistry(data_dir=temp_data_dir)

    def test_init_creates_directories(self, registry, temp_data_dir):
        """Test that initialization creates necessary directories."""
        assert os.path.exists(temp_data_dir)
        assert os.path.exists(os.path.join(temp_data_dir, "masks"))
        assert os.path.exists(os.path.join(temp_data_dir, "registry.db"))

    def test_register_source_image(self, registry):
        """Test registering an image source."""
        source_id = registry.register_source(
            source_type="image",
            file_path="/path/to/image.jpg",
            width=640,
            height=480,
        )

        assert source_id is not None
        assert source_id.startswith("src_")

        # Verify source was stored
        source = registry.get_source(source_id)
        assert source is not None
        assert source["source_type"] == "image"
        assert source["width"] == 640
        assert source["height"] == 480

    def test_register_source_video(self, registry):
        """Test registering a video source."""
        source_id = registry.register_source(
            source_type="video",
            file_path="/path/to/video.mp4",
            width=1920,
            height=1080,
            frame_count=300,
            fps=30.0,
        )

        source = registry.get_source(source_id)
        assert source["source_type"] == "video"
        assert source["frame_count"] == 300
        assert source["fps"] == 30.0

    def test_get_or_create_category(self, registry):
        """Test category creation and retrieval."""
        # First call creates
        cat_id1 = registry.get_or_create_category("person")
        assert cat_id1 > 0

        # Second call retrieves existing
        cat_id2 = registry.get_or_create_category("person")
        assert cat_id1 == cat_id2

        # Different category gets different ID
        cat_id3 = registry.get_or_create_category("car")
        assert cat_id3 != cat_id1

    def test_list_categories(self, registry):
        """Test listing all categories."""
        registry.get_or_create_category("person")
        registry.get_or_create_category("car")
        registry.get_or_create_category("bicycle")

        categories = registry.list_categories()
        names = [c["name"] for c in categories]

        assert "person" in names
        assert "car" in names
        assert "bicycle" in names

    def test_register_object(self, registry, sample_bbox):
        """Test registering a single object."""
        # First create a source
        source_id = registry.register_source(
            source_type="image",
            file_path="/test/image.jpg",
            width=640,
            height=480,
        )

        # Register object
        object_id = registry.register_object(
            source_id=source_id,
            category_name="person",
            bbox=sample_bbox,
            confidence=0.95,
            detection_model="florence2",
        )

        assert object_id is not None
        assert object_id.startswith("obj_")

        # Verify object was stored
        obj = registry.get_object(object_id)
        assert obj is not None
        assert obj["category_name"] == "person"
        assert obj["confidence"] == 0.95
        assert obj["bbox"] == sample_bbox

    def test_register_objects_batch(self, registry, sample_objects_data):
        """Test batch registration of objects."""
        source_id = registry.register_source(
            source_type="image",
            file_path="/test/image.jpg",
            width=640,
            height=480,
        )

        object_ids = registry.register_objects_batch(
            source_id=source_id,
            objects_data=sample_objects_data,
            project_id="test_project",
        )

        assert len(object_ids) == 3
        assert all(oid.startswith("obj_") for oid in object_ids)

    def test_search_objects_by_category(self, registry, sample_objects_data):
        """Test searching objects by category."""
        source_id = registry.register_source(
            source_type="image",
            file_path="/test/image.jpg",
            width=640,
            height=480,
        )

        registry.register_objects_batch(
            source_id=source_id,
            objects_data=sample_objects_data,
        )

        # Search for persons
        persons = registry.search_objects(category="person")
        assert len(persons) == 2

        # Search for cars
        cars = registry.search_objects(category="car")
        assert len(cars) == 1

    def test_search_objects_by_confidence(self, registry, sample_objects_data):
        """Test searching objects by minimum confidence."""
        source_id = registry.register_source(
            source_type="image",
            file_path="/test/image.jpg",
            width=640,
            height=480,
        )

        registry.register_objects_batch(
            source_id=source_id,
            objects_data=sample_objects_data,
        )

        # Search with high confidence threshold
        high_conf = registry.search_objects(min_confidence=0.9)
        assert len(high_conf) == 2  # 0.95 and 0.92

        # Search with very high threshold
        very_high = registry.search_objects(min_confidence=0.94)
        assert len(very_high) == 1  # Only 0.95

    def test_update_object(self, registry, sample_bbox):
        """Test updating object fields."""
        source_id = registry.register_source(
            source_type="image",
            file_path="/test/image.jpg",
            width=640,
            height=480,
        )

        object_id = registry.register_object(
            source_id=source_id,
            category_name="person",
            bbox=sample_bbox,
        )

        # Update validation status
        success = registry.update_object(object_id, {
            "is_validated": True,
            "validated_by": "test_reviewer",
            "quality_score": 0.95,
        })

        assert success

        obj = registry.get_object(object_id)
        assert obj["is_validated"] == 1  # SQLite stores as 1
        assert obj["validated_by"] == "test_reviewer"
        assert obj["quality_score"] == 0.95

    def test_delete_object(self, registry, sample_bbox):
        """Test deleting an object."""
        source_id = registry.register_source(
            source_type="image",
            file_path="/test/image.jpg",
            width=640,
            height=480,
        )

        object_id = registry.register_object(
            source_id=source_id,
            category_name="person",
            bbox=sample_bbox,
        )

        # Delete
        success = registry.delete_object(object_id)
        assert success

        # Verify deleted
        obj = registry.get_object(object_id)
        assert obj is None

    def test_count_objects(self, registry, sample_objects_data):
        """Test counting objects."""
        source_id = registry.register_source(
            source_type="image",
            file_path="/test/image.jpg",
            width=640,
            height=480,
        )

        registry.register_objects_batch(
            source_id=source_id,
            objects_data=sample_objects_data,
        )

        total = registry.count_objects()
        assert total == 3

        persons = registry.count_objects(category="person")
        assert persons == 2

    def test_get_stats(self, registry, sample_objects_data):
        """Test getting registry statistics."""
        source_id = registry.register_source(
            source_type="image",
            file_path="/test/image.jpg",
            width=640,
            height=480,
        )

        registry.register_objects_batch(
            source_id=source_id,
            objects_data=sample_objects_data,
        )

        stats = registry.get_stats()

        assert stats["sources"] == 1
        assert stats["objects"] == 3
        assert stats["categories"] >= 2  # person, car
        assert stats["validated_objects"] == 0
        assert "objects_per_category" in stats

    def test_create_track(self, registry):
        """Test creating a track from objects."""
        source_id = registry.register_source(
            source_type="video",
            file_path="/test/video.mp4",
            width=1920,
            height=1080,
            frame_count=100,
            fps=30.0,
        )

        # Create objects for track
        object_ids = []
        for i in range(5):
            obj_id = registry.register_object(
                source_id=source_id,
                category_name="person",
                bbox=[100 + i * 10, 100, 50, 80],
                confidence=0.9,
            )
            object_ids.append(obj_id)

        # Create track
        track_id = registry.create_track(
            source_id=source_id,
            object_ids=object_ids,
            category_name="person",
        )

        assert track_id is not None
        assert track_id.startswith("trk_")

        # Verify track
        track = registry.get_track(track_id)
        assert track is not None
        assert len(track["objects"]) == 5

    def test_create_dataset(self, registry, sample_objects_data):
        """Test creating a dataset configuration."""
        source_id = registry.register_source(
            source_type="image",
            file_path="/test/image.jpg",
            width=640,
            height=480,
        )

        registry.register_objects_batch(
            source_id=source_id,
            objects_data=sample_objects_data,
        )

        dataset_id = registry.create_dataset(
            name="test_dataset",
            format="yolo",
            filter_config={"categories": ["person", "car"]},
            split_config={"train": 0.8, "val": 0.1, "test": 0.1},
        )

        assert dataset_id is not None
        assert dataset_id.startswith("ds_")

    def test_build_dataset(self, registry, sample_objects_data):
        """Test building a dataset."""
        source_id = registry.register_source(
            source_type="image",
            file_path="/test/image.jpg",
            width=640,
            height=480,
        )

        registry.register_objects_batch(
            source_id=source_id,
            objects_data=sample_objects_data,
        )

        dataset_id = registry.create_dataset(
            name="test_dataset",
            format="yolo",
        )

        result = registry.build_dataset(dataset_id)

        assert result["dataset_id"] == dataset_id
        assert result["object_count"] == 3
        assert "splits" in result

    # ==================== Track Management Tests ====================

    def test_list_tracks(self, registry):
        """Test listing tracks with filters."""
        source_id = registry.register_source(
            source_type="video",
            file_path="/test/video.mp4",
            width=1920,
            height=1080,
            frame_count=100,
            fps=30.0,
        )

        # Create objects and tracks
        object_ids = []
        for i in range(6):
            obj_id = registry.register_object(
                source_id=source_id,
                category_name="person" if i < 3 else "car",
                bbox=[100 + i * 10, 100, 50, 80],
                confidence=0.9,
            )
            object_ids.append(obj_id)

        # Create two tracks
        track1_id = registry.create_track(source_id, object_ids[:3], "person")
        track2_id = registry.create_track(source_id, object_ids[3:], "car")

        # List all tracks
        all_tracks = registry.list_tracks()
        assert len(all_tracks) == 2

        # List by category
        person_tracks = registry.list_tracks(category="person")
        assert len(person_tracks) == 1
        assert person_tracks[0]["track_id"] == track1_id

        car_tracks = registry.list_tracks(category="car")
        assert len(car_tracks) == 1
        assert car_tracks[0]["track_id"] == track2_id

    def test_update_track(self, registry):
        """Test updating track fields."""
        source_id = registry.register_source(
            source_type="video",
            file_path="/test/video.mp4",
            width=1920,
            height=1080,
            frame_count=100,
            fps=30.0,
        )

        object_ids = []
        for i in range(3):
            obj_id = registry.register_object(
                source_id=source_id,
                category_name="person",
                bbox=[100 + i * 10, 100, 50, 80],
            )
            object_ids.append(obj_id)

        track_id = registry.create_track(source_id, object_ids, "person")

        # Update category
        success = registry.update_track(track_id, category_name="pedestrian")
        assert success

        track = registry.get_track(track_id)
        assert track["category_name"] == "pedestrian"

        # Update validation
        success = registry.update_track(
            track_id,
            is_validated=True,
            validated_by="reviewer1"
        )
        assert success

        track = registry.get_track(track_id)
        assert track["is_validated"] == 1
        assert track["validated_by"] == "reviewer1"

    def test_delete_track(self, registry):
        """Test deleting a track."""
        source_id = registry.register_source(
            source_type="video",
            file_path="/test/video.mp4",
            width=1920,
            height=1080,
        )

        object_ids = []
        for i in range(3):
            obj_id = registry.register_object(
                source_id=source_id,
                category_name="person",
                bbox=[100 + i * 10, 100, 50, 80],
            )
            object_ids.append(obj_id)

        track_id = registry.create_track(source_id, object_ids, "person")

        # Delete track
        success = registry.delete_track(track_id)
        assert success

        # Verify deleted
        track = registry.get_track(track_id)
        assert track is None

        # Objects should still exist
        for obj_id in object_ids:
            obj = registry.get_object(obj_id)
            assert obj is not None

    def test_merge_tracks(self, registry):
        """Test merging multiple tracks."""
        source_id = registry.register_source(
            source_type="video",
            file_path="/test/video.mp4",
            width=1920,
            height=1080,
        )

        # Create objects for two tracks
        objects_track1 = []
        objects_track2 = []
        for i in range(3):
            obj_id = registry.register_object(
                source_id=source_id,
                category_name="person",
                bbox=[100 + i * 10, 100, 50, 80],
            )
            objects_track1.append(obj_id)

        for i in range(2):
            obj_id = registry.register_object(
                source_id=source_id,
                category_name="person",
                bbox=[200 + i * 10, 100, 50, 80],
            )
            objects_track2.append(obj_id)

        track1_id = registry.create_track(source_id, objects_track1, "person")
        track2_id = registry.create_track(source_id, objects_track2, "person")

        # Merge tracks
        merged_id = registry.merge_tracks([track1_id, track2_id])
        assert merged_id is not None
        assert merged_id.startswith("trk_")

        # Verify merged track
        merged = registry.get_track(merged_id)
        assert len(merged["objects"]) == 5

        # Original tracks should be deleted
        assert registry.get_track(track1_id) is None
        assert registry.get_track(track2_id) is None

    def test_merge_tracks_with_new_category(self, registry):
        """Test merging tracks with a new category name."""
        source_id = registry.register_source(
            source_type="video",
            file_path="/test/video.mp4",
            width=1920,
            height=1080,
        )

        objects = []
        for i in range(4):
            obj_id = registry.register_object(
                source_id=source_id,
                category_name="person",
                bbox=[100 + i * 10, 100, 50, 80],
            )
            objects.append(obj_id)

        track1_id = registry.create_track(source_id, objects[:2], "person")
        track2_id = registry.create_track(source_id, objects[2:], "person")

        # Merge with new category
        merged_id = registry.merge_tracks(
            [track1_id, track2_id],
            new_category_name="pedestrian"
        )

        merged = registry.get_track(merged_id)
        assert merged["category_name"] == "pedestrian"

    def test_merge_tracks_single_track_fails(self, registry):
        """Test that merging a single track fails."""
        source_id = registry.register_source(
            source_type="video",
            file_path="/test/video.mp4",
            width=1920,
            height=1080,
        )

        obj_id = registry.register_object(
            source_id=source_id,
            category_name="person",
            bbox=[100, 100, 50, 80],
        )

        track_id = registry.create_track(source_id, [obj_id], "person")

        # Try to merge single track
        result = registry.merge_tracks([track_id])
        assert result is None

    def test_split_track(self, registry):
        """Test splitting a track."""
        source_id = registry.register_source(
            source_type="video",
            file_path="/test/video.mp4",
            width=1920,
            height=1080,
        )

        object_ids = []
        for i in range(5):
            obj_id = registry.register_object(
                source_id=source_id,
                category_name="person",
                bbox=[100 + i * 10, 100, 50, 80],
            )
            object_ids.append(obj_id)

        track_id = registry.create_track(source_id, object_ids, "person")

        # Split at index 2
        result = registry.split_track(track_id, split_index=2)
        assert result is not None

        first_id, second_id = result

        # Verify first track
        first = registry.get_track(first_id)
        assert len(first["objects"]) == 2

        # Verify second track
        second = registry.get_track(second_id)
        assert len(second["objects"]) == 3

        # Original track should be deleted
        assert registry.get_track(track_id) is None

    def test_split_track_invalid_index(self, registry):
        """Test that splitting with invalid index fails."""
        source_id = registry.register_source(
            source_type="video",
            file_path="/test/video.mp4",
            width=1920,
            height=1080,
        )

        object_ids = []
        for i in range(3):
            obj_id = registry.register_object(
                source_id=source_id,
                category_name="person",
                bbox=[100 + i * 10, 100, 50, 80],
            )
            object_ids.append(obj_id)

        track_id = registry.create_track(source_id, object_ids, "person")

        # Split at index 0 should fail
        result = registry.split_track(track_id, split_index=0)
        assert result is None

        # Split at index >= len should fail
        result = registry.split_track(track_id, split_index=3)
        assert result is None

    def test_add_objects_to_track(self, registry):
        """Test adding objects to a track."""
        source_id = registry.register_source(
            source_type="video",
            file_path="/test/video.mp4",
            width=1920,
            height=1080,
        )

        initial_objects = []
        for i in range(3):
            obj_id = registry.register_object(
                source_id=source_id,
                category_name="person",
                bbox=[100 + i * 10, 100, 50, 80],
            )
            initial_objects.append(obj_id)

        track_id = registry.create_track(source_id, initial_objects, "person")

        # Add more objects
        new_objects = []
        for i in range(2):
            obj_id = registry.register_object(
                source_id=source_id,
                category_name="person",
                bbox=[200 + i * 10, 100, 50, 80],
            )
            new_objects.append(obj_id)

        success = registry.add_objects_to_track(track_id, new_objects)
        assert success

        track = registry.get_track(track_id)
        assert len(track["objects"]) == 5

    def test_add_objects_at_position(self, registry):
        """Test inserting objects at specific position."""
        source_id = registry.register_source(
            source_type="video",
            file_path="/test/video.mp4",
            width=1920,
            height=1080,
        )

        initial_objects = []
        for i in range(3):
            obj_id = registry.register_object(
                source_id=source_id,
                category_name="person",
                bbox=[100 + i * 10, 100, 50, 80],
            )
            initial_objects.append(obj_id)

        track_id = registry.create_track(source_id, initial_objects, "person")

        # Add object at position 1
        new_obj = registry.register_object(
            source_id=source_id,
            category_name="person",
            bbox=[50, 100, 50, 80],
        )

        success = registry.add_objects_to_track(track_id, [new_obj], insert_at=1)
        assert success

        track = registry.get_track(track_id)
        assert len(track["objects"]) == 4
        # New object should be at position 1
        assert track["objects"][1]["object_id"] == new_obj

    def test_remove_objects_from_track(self, registry):
        """Test removing objects from a track."""
        source_id = registry.register_source(
            source_type="video",
            file_path="/test/video.mp4",
            width=1920,
            height=1080,
        )

        object_ids = []
        for i in range(5):
            obj_id = registry.register_object(
                source_id=source_id,
                category_name="person",
                bbox=[100 + i * 10, 100, 50, 80],
            )
            object_ids.append(obj_id)

        track_id = registry.create_track(source_id, object_ids, "person")

        # Remove middle objects
        success = registry.remove_objects_from_track(
            track_id,
            [object_ids[1], object_ids[3]]
        )
        assert success

        track = registry.get_track(track_id)
        assert len(track["objects"]) == 3

        # Check sequence is correct (0, 1, 2)
        for i, obj in enumerate(track["objects"]):
            assert obj["sequence_idx"] == i
