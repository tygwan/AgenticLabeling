"""Label Studio Lite - Object Validation UI."""
import os
from io import BytesIO

import httpx
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Configuration
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8010")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")

# Color palette for track visualization
TRACK_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8B500", "#00CED1", "#FF69B4", "#32CD32", "#FF4500",
]

# Page config
st.set_page_config(
    page_title="Label Studio Lite",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    .valid-btn>button {
        background-color: #28a745;
        color: white;
    }
    .reject-btn>button {
        background-color: #dc3545;
        color: white;
    }
    .bbox-info {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==================== API Functions ====================

@st.cache_data(ttl=30)
def get_stats():
    """Get registry statistics."""
    try:
        resp = httpx.get(f"{REGISTRY_URL}/stats", timeout=10.0)
        data = resp.json()
        return data.get("data", {}) if data.get("success") else {}
    except Exception as e:
        st.error(f"Failed to get stats: {e}")
        return {}


@st.cache_data(ttl=10)
def get_categories():
    """Get all categories."""
    try:
        resp = httpx.get(f"{REGISTRY_URL}/categories", timeout=10.0)
        data = resp.json()
        return data.get("data", []) if data.get("success") else []
    except Exception:
        return []


def get_objects(
    is_validated: bool = None,
    category: str = None,
    project_id: str = None,
    limit: int = 100,
):
    """Get objects with filters."""
    params = {"limit": limit}
    if is_validated is not None:
        params["is_validated"] = is_validated
    if category:
        params["category"] = category
    if project_id:
        params["project_id"] = project_id

    try:
        resp = httpx.get(f"{REGISTRY_URL}/objects", params=params, timeout=10.0)
        data = resp.json()
        return data.get("data", []) if data.get("success") else []
    except Exception as e:
        st.error(f"Failed to get objects: {e}")
        return []


def get_object_detail(object_id: str):
    """Get single object details."""
    try:
        resp = httpx.get(f"{REGISTRY_URL}/objects/{object_id}", timeout=10.0)
        data = resp.json()
        return data.get("data") if data.get("success") else None
    except Exception:
        return None


def get_source(source_id: str):
    """Get source details."""
    try:
        resp = httpx.get(f"{REGISTRY_URL}/sources/{source_id}", timeout=10.0)
        data = resp.json()
        return data.get("data") if data.get("success") else None
    except Exception:
        return None


def validate_object(object_id: str, validated_by: str, quality_score: float):
    """Mark object as validated."""
    try:
        resp = httpx.patch(
            f"{REGISTRY_URL}/objects/{object_id}",
            json={
                "is_validated": True,
                "validated_by": validated_by,
                "quality_score": quality_score,
            },
            timeout=10.0,
        )
        return resp.json().get("success", False)
    except Exception:
        return False


def reject_object(object_id: str):
    """Delete/reject an object."""
    try:
        resp = httpx.delete(f"{REGISTRY_URL}/objects/{object_id}", timeout=10.0)
        return resp.json().get("success", False)
    except Exception:
        return False


def update_object_bbox(object_id: str, bbox: dict):
    """Update object bounding box (placeholder - not yet implemented in registry)."""
    # TODO: Implement bbox update in registry
    return False


@st.cache_data(ttl=30)
def get_sources():
    """Get all video sources for track visualization."""
    try:
        resp = httpx.get(f"{REGISTRY_URL}/stats", timeout=10.0)
        data = resp.json()
        # Get sources with tracks (videos)
        return data.get("data", {}) if data.get("success") else {}
    except Exception:
        return {}


def get_track(track_id: str):
    """Get track details with objects."""
    try:
        resp = httpx.get(f"{REGISTRY_URL}/tracks/{track_id}", timeout=10.0)
        data = resp.json()
        return data.get("data") if data.get("success") else None
    except Exception:
        return None


def get_tracks_for_source(source_id: str):
    """Get all tracks for a specific source."""
    try:
        # Search objects for this source and group by tracks
        resp = httpx.get(
            f"{REGISTRY_URL}/objects",
            params={"source_id": source_id, "limit": 1000},
            timeout=30.0,
        )
        data = resp.json()
        objects = data.get("data", []) if data.get("success") else []

        # Group by track_id
        tracks = {}
        for obj in objects:
            track_id = obj.get("track_id")
            if track_id:
                if track_id not in tracks:
                    tracks[track_id] = {
                        "track_id": track_id,
                        "category": obj.get("category_name"),
                        "objects": [],
                    }
                tracks[track_id]["objects"].append(obj)

        # Sort objects by frame_id within each track
        for track in tracks.values():
            track["objects"].sort(key=lambda x: x.get("frame_id", ""))
            track["length"] = len(track["objects"])

        return list(tracks.values())
    except Exception as e:
        st.error(f"Failed to get tracks: {e}")
        return []


def search_objects_by_source(source_id: str, limit: int = 500):
    """Get all objects for a source."""
    try:
        resp = httpx.get(
            f"{REGISTRY_URL}/objects",
            params={"source_id": source_id, "limit": limit},
            timeout=30.0,
        )
        data = resp.json()
        return data.get("data", []) if data.get("success") else []
    except Exception:
        return []


# ==================== Drawing Functions ====================

def draw_bbox_on_image(
    image: Image.Image,
    bbox: tuple,
    label: str,
    color: str = "#00FF00",
    thickness: int = 3,
) -> Image.Image:
    """Draw bounding box on image."""
    draw = ImageDraw.Draw(image)
    x, y, w, h = bbox

    # Draw rectangle
    draw.rectangle([x, y, x + w, y + h], outline=color, width=thickness)

    # Draw label background
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((x, y - 20), label, font=font)
    draw.rectangle(text_bbox, fill=color)
    draw.text((x, y - 20), label, fill="white", font=font)

    return image


def create_placeholder_image(width: int, height: int, text: str = "No Image") -> Image.Image:
    """Create a placeholder image."""
    img = Image.new("RGB", (width, height), color="#f0f0f0")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), text, fill="#999999", font=font)

    return img


def draw_track_trajectory(
    image: Image.Image,
    track_objects: list,
    color: str,
    show_bbox: bool = True,
    line_width: int = 2,
) -> Image.Image:
    """Draw track trajectory with bounding boxes on image."""
    draw = ImageDraw.Draw(image)

    # Get center points for trajectory line
    centers = []
    for obj in track_objects:
        x = obj.get("bbox_x", 0)
        y = obj.get("bbox_y", 0)
        w = obj.get("bbox_w", 0)
        h = obj.get("bbox_h", 0)
        center_x = x + w / 2
        center_y = y + h / 2
        centers.append((center_x, center_y))

    # Draw trajectory line
    if len(centers) > 1:
        for i in range(len(centers) - 1):
            draw.line([centers[i], centers[i + 1]], fill=color, width=line_width)

    # Draw center points
    for cx, cy in centers:
        r = 4
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    # Draw bounding boxes
    if show_bbox:
        for i, obj in enumerate(track_objects):
            x = obj.get("bbox_x", 0)
            y = obj.get("bbox_y", 0)
            w = obj.get("bbox_w", 0)
            h = obj.get("bbox_h", 0)
            # Use lighter color for older boxes
            alpha = 0.3 + 0.7 * (i / max(len(track_objects) - 1, 1))
            draw.rectangle(
                [x, y, x + w, y + h],
                outline=color,
                width=max(1, int(line_width * alpha)),
            )

    return image


def create_track_visualization(
    width: int,
    height: int,
    tracks: list,
    selected_track_ids: list = None,
    show_all_tracks: bool = True,
) -> Image.Image:
    """Create visualization showing multiple tracks."""
    img = Image.new("RGB", (width, height), color="#2C3E50")
    draw = ImageDraw.Draw(img)

    # Draw grid
    grid_color = "#3D566E"
    for x in range(0, width, 50):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, 50):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)

    # Draw tracks
    for idx, track in enumerate(tracks):
        track_id = track.get("track_id")

        # Filter by selection if provided
        if selected_track_ids and track_id not in selected_track_ids:
            if not show_all_tracks:
                continue
            # Draw unselected tracks with reduced opacity (gray)
            color = "#666666"
            line_width = 1
        else:
            color = TRACK_COLORS[idx % len(TRACK_COLORS)]
            line_width = 3

        objects = track.get("objects", [])
        if objects:
            img = draw_track_trajectory(
                img,
                objects,
                color,
                show_bbox=(track_id in (selected_track_ids or [])) or not selected_track_ids,
                line_width=line_width,
            )

    return img


def create_track_timeline(
    tracks: list,
    total_frames: int,
    width: int = 800,
    height: int = 200,
) -> Image.Image:
    """Create a timeline visualization of tracks."""
    img = Image.new("RGB", (width, height), color="#1E1E1E")
    draw = ImageDraw.Draw(img)

    if not tracks or total_frames == 0:
        return img

    # Calculate track row height
    row_height = max(20, min(40, (height - 40) // max(len(tracks), 1)))
    padding = 10

    # Draw frame markers
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()

    for i in range(0, total_frames + 1, max(1, total_frames // 10)):
        x = padding + (width - 2 * padding) * i / total_frames
        draw.line([(x, 0), (x, height)], fill="#333333", width=1)
        draw.text((x, height - 15), str(i), fill="#888888", font=font)

    # Draw each track
    y = padding
    for idx, track in enumerate(tracks):
        color = TRACK_COLORS[idx % len(TRACK_COLORS)]
        objects = track.get("objects", [])

        if objects:
            # Get frame range
            frame_ids = []
            for obj in objects:
                frame_id = obj.get("frame_id", "")
                # Extract frame number from frame_id (e.g., "frame_0001" -> 1)
                try:
                    if frame_id.startswith("frame_"):
                        frame_num = int(frame_id.split("_")[1])
                    else:
                        frame_num = int(frame_id)
                    frame_ids.append(frame_num)
                except (ValueError, IndexError):
                    pass

            if frame_ids:
                min_frame = min(frame_ids)
                max_frame = max(frame_ids)

                x1 = padding + (width - 2 * padding) * min_frame / total_frames
                x2 = padding + (width - 2 * padding) * max_frame / total_frames

                # Draw track bar
                draw.rectangle(
                    [x1, y + 2, x2, y + row_height - 2],
                    fill=color,
                    outline=None,
                )

                # Draw track label
                label = f"{track.get('category', 'obj')} ({len(objects)})"
                draw.text((x1 + 2, y + 4), label, fill="white", font=font)

        y += row_height

    return img


# ==================== UI Components ====================

def render_sidebar():
    """Render sidebar with filters and stats."""
    st.sidebar.title("🏷️ Label Studio Lite")

    # Stats
    stats = get_stats()
    if stats:
        st.sidebar.subheader("📊 Statistics")
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Objects", stats.get("objects", 0))
        col2.metric("Validated", stats.get("validated_objects", 0))

        col3, col4 = st.sidebar.columns(2)
        col3.metric("Categories", stats.get("categories", 0))
        col4.metric("Sources", stats.get("sources", 0))

    st.sidebar.divider()

    # Filters
    st.sidebar.subheader("🔍 Filters")

    # Validation status filter
    validation_filter = st.sidebar.radio(
        "Validation Status",
        ["All", "Pending", "Validated"],
        index=1,  # Default to Pending
    )

    is_validated = None
    if validation_filter == "Pending":
        is_validated = False
    elif validation_filter == "Validated":
        is_validated = True

    # Category filter
    categories = get_categories()
    category_names = ["All"] + [c["name"] for c in categories]
    selected_category = st.sidebar.selectbox("Category", category_names)
    category = None if selected_category == "All" else selected_category

    # Reviewer name
    reviewer_name = st.sidebar.text_input("Reviewer Name", value="reviewer_001")

    # Quality score default
    default_quality = st.sidebar.slider("Default Quality Score", 0.0, 1.0, 0.9, 0.05)

    return {
        "is_validated": is_validated,
        "category": category,
        "reviewer_name": reviewer_name,
        "default_quality": default_quality,
    }


def render_object_card(obj: dict, filters: dict, idx: int):
    """Render a single object card with image and controls."""
    object_id = obj.get("object_id")
    category_name = obj.get("category_name", "unknown")
    confidence = obj.get("confidence")
    is_validated = obj.get("is_validated")
    source_id = obj.get("source_id")

    # Bounding box
    bbox = (
        obj.get("bbox_x", 0),
        obj.get("bbox_y", 0),
        obj.get("bbox_w", 100),
        obj.get("bbox_h", 100),
    )

    # Get source info
    source = get_source(source_id) if source_id else None
    img_width = source.get("width", 640) if source else 640
    img_height = source.get("height", 480) if source else 480

    # Create image with bbox
    # In production, load actual image from file_path
    img = create_placeholder_image(img_width, img_height, f"Source: {source_id[:12]}...")

    # Draw bbox
    label = f"{category_name}"
    if confidence:
        label += f" ({confidence:.2f})"
    color = "#00FF00" if is_validated else "#FF6B6B"
    img = draw_bbox_on_image(img, bbox, label, color)

    # Display
    with st.container():
        st.image(img, use_container_width=True)

        # Info
        st.markdown(f"""
        <div class="bbox-info">
            <strong>ID:</strong> {object_id}<br>
            <strong>Category:</strong> {category_name}<br>
            <strong>Confidence:</strong> {confidence:.2f if confidence else 'N/A'}<br>
            <strong>BBox:</strong> x={bbox[0]:.0f}, y={bbox[1]:.0f}, w={bbox[2]:.0f}, h={bbox[3]:.0f}<br>
            <strong>Status:</strong> {'✅ Validated' if is_validated else '⏳ Pending'}
        </div>
        """, unsafe_allow_html=True)

        # Action buttons
        if not is_validated:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Validate", key=f"validate_{idx}", type="primary"):
                    if validate_object(
                        object_id,
                        filters["reviewer_name"],
                        filters["default_quality"],
                    ):
                        st.success("Validated!")
                        st.rerun()
                    else:
                        st.error("Failed to validate")

            with col2:
                if st.button("❌ Reject", key=f"reject_{idx}", type="secondary"):
                    if reject_object(object_id):
                        st.warning("Rejected and deleted")
                        st.rerun()
                    else:
                        st.error("Failed to reject")


def render_main_content(filters: dict):
    """Render main content area."""
    st.title("Object Validation")

    # Get objects
    objects = get_objects(
        is_validated=filters["is_validated"],
        category=filters["category"],
        limit=50,
    )

    if not objects:
        st.info("No objects found matching the filters.")
        return

    st.write(f"Showing {len(objects)} objects")

    # Display in grid
    cols = st.columns(3)
    for idx, obj in enumerate(objects):
        with cols[idx % 3]:
            render_object_card(obj, filters, idx)


def render_export_tab():
    """Render dataset export tab."""
    st.title("📦 Dataset Export")

    col1, col2 = st.columns(2)

    with col1:
        dataset_name = st.text_input("Dataset Name", value="my_dataset")
        export_format = st.selectbox("Format", ["YOLO", "COCO"])

        # Split ratios
        st.subheader("Split Ratios")
        train_ratio = st.slider("Train", 0.0, 1.0, 0.8, 0.05)
        val_ratio = st.slider("Validation", 0.0, 1.0, 0.1, 0.05)
        test_ratio = 1.0 - train_ratio - val_ratio
        st.write(f"Test: {test_ratio:.2f}")

    with col2:
        # Filters
        st.subheader("Filters")
        only_validated = st.checkbox("Only Validated Objects", value=True)

        categories = get_categories()
        selected_cats = st.multiselect(
            "Categories",
            [c["name"] for c in categories],
            default=[c["name"] for c in categories],
        )

        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)

    if st.button("🚀 Export Dataset", type="primary"):
        with st.spinner("Exporting..."):
            try:
                filter_config = {
                    "categories": selected_cats,
                    "min_confidence": min_confidence,
                }
                if only_validated:
                    filter_config["is_validated"] = True

                resp = httpx.post(
                    f"{GATEWAY_URL}/export",
                    data={
                        "dataset_name": dataset_name,
                        "format": export_format.lower(),
                        "categories": ",".join(selected_cats),
                        "min_confidence": min_confidence,
                        "is_validated": only_validated,
                        "train_ratio": train_ratio,
                        "val_ratio": val_ratio,
                        "test_ratio": test_ratio,
                    },
                    timeout=300.0,
                )
                result = resp.json()

                if result.get("success"):
                    data = result.get("data", {}).get("data", result.get("data", {}))
                    st.success(f"Export completed!")
                    st.json({
                        "dataset_name": data.get("dataset_name"),
                        "format": export_format,
                        "object_count": data.get("object_count"),
                        "image_count": data.get("image_count"),
                        "splits": data.get("splits"),
                    })
                else:
                    st.error(f"Export failed: {result.get('error')}")
            except Exception as e:
                st.error(f"Export error: {e}")


def render_stats_tab():
    """Render statistics dashboard."""
    st.title("📊 Statistics Dashboard")

    stats = get_stats()

    if not stats:
        st.warning("Unable to load statistics")
        return

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Objects", stats.get("objects", 0))
    col2.metric("Validated", stats.get("validated_objects", 0))
    col3.metric("Categories", stats.get("categories", 0))
    col4.metric("Sources", stats.get("sources", 0))

    # Validation progress
    total = stats.get("objects", 0)
    validated = stats.get("validated_objects", 0)
    if total > 0:
        progress = validated / total
        st.progress(progress, text=f"Validation Progress: {validated}/{total} ({progress*100:.1f}%)")

    # Category distribution
    st.subheader("Objects per Category")
    category_counts = stats.get("objects_per_category", {})
    if category_counts:
        df = pd.DataFrame([
            {"Category": k, "Count": v}
            for k, v in category_counts.items()
        ])
        st.bar_chart(df.set_index("Category"))


def render_tracks_tab():
    """Render track visualization tab."""
    st.title("🎯 Track Visualization")

    # Get available sources
    stats = get_stats()

    # Source selector
    st.subheader("Select Source")
    source_id = st.text_input(
        "Source ID",
        placeholder="Enter source ID (e.g., src_abc123)",
        help="Enter the source ID for which you want to visualize tracks",
    )

    if not source_id:
        st.info("Enter a source ID to view tracks. Sources with video content typically have tracks.")

        # Show some instructions
        st.markdown("""
        ### How to use Track Visualization

        1. **Enter Source ID**: Enter the source ID of a video that has been processed with tracking
        2. **View Trajectories**: See object trajectories across frames
        3. **Timeline View**: Understand when each track appears in the video
        4. **Track Details**: Inspect individual track statistics

        ### Track Colors
        Each track is assigned a unique color for easy identification.
        The trajectory shows the center point movement of each tracked object.
        """)
        return

    # Get source info
    source = get_source(source_id)
    if not source:
        st.error(f"Source not found: {source_id}")
        return

    # Display source info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Type", source.get("source_type", "unknown"))
    col2.metric("Width", source.get("width", 0))
    col3.metric("Height", source.get("height", 0))
    col4.metric("Frames", source.get("frame_count", 0))

    st.divider()

    # Get tracks for this source
    with st.spinner("Loading tracks..."):
        tracks = get_tracks_for_source(source_id)

    if not tracks:
        st.warning("No tracks found for this source. Make sure tracking has been run on this video.")

        # Show objects without tracks
        objects = search_objects_by_source(source_id, limit=50)
        if objects:
            st.info(f"Found {len(objects)} objects without track assignments.")
        return

    # Track overview
    st.subheader(f"Found {len(tracks)} Tracks")

    # Track selection
    track_options = [
        f"{t['track_id'][:16]}... - {t.get('category', 'unknown')} ({t.get('length', 0)} objects)"
        for t in tracks
    ]
    selected_tracks = st.multiselect(
        "Select tracks to highlight",
        options=range(len(tracks)),
        format_func=lambda x: track_options[x],
        default=list(range(min(5, len(tracks)))),  # Select first 5 by default
    )

    selected_track_ids = [tracks[i]["track_id"] for i in selected_tracks] if selected_tracks else None

    # Visualization options
    col1, col2 = st.columns(2)
    with col1:
        show_all = st.checkbox("Show all tracks (unselected in gray)", value=True)
    with col2:
        show_timeline = st.checkbox("Show timeline view", value=True)

    # Create main visualization
    st.subheader("Trajectory View")
    width = source.get("width", 1280)
    height = source.get("height", 720)

    # Scale down for display if too large
    max_display_width = 1200
    scale = min(1.0, max_display_width / width)
    display_width = int(width * scale)
    display_height = int(height * scale)

    # Create scaled tracks for visualization
    scaled_tracks = []
    for track in tracks:
        scaled_track = {
            "track_id": track["track_id"],
            "category": track.get("category"),
            "objects": [],
        }
        for obj in track.get("objects", []):
            scaled_obj = {
                "bbox_x": obj.get("bbox_x", 0) * scale,
                "bbox_y": obj.get("bbox_y", 0) * scale,
                "bbox_w": obj.get("bbox_w", 0) * scale,
                "bbox_h": obj.get("bbox_h", 0) * scale,
                "frame_id": obj.get("frame_id"),
            }
            scaled_track["objects"].append(scaled_obj)
        scaled_tracks.append(scaled_track)

    track_img = create_track_visualization(
        display_width,
        display_height,
        scaled_tracks,
        selected_track_ids=selected_track_ids,
        show_all_tracks=show_all,
    )
    st.image(track_img, use_container_width=True)

    # Timeline view
    if show_timeline:
        st.subheader("Timeline View")
        total_frames = source.get("frame_count", 100)
        timeline_height = min(400, max(150, len(tracks) * 30 + 40))
        timeline_img = create_track_timeline(tracks, total_frames, width=1000, height=timeline_height)
        st.image(timeline_img, use_container_width=True)

    # Track details table
    st.subheader("Track Details")
    track_data = []
    for idx, track in enumerate(tracks):
        objects = track.get("objects", [])
        confidences = [obj.get("confidence", 0) for obj in objects if obj.get("confidence")]

        # Get frame range
        frame_ids = []
        for obj in objects:
            frame_id = obj.get("frame_id", "")
            try:
                if frame_id.startswith("frame_"):
                    frame_num = int(frame_id.split("_")[1])
                else:
                    frame_num = int(frame_id) if frame_id else 0
                frame_ids.append(frame_num)
            except (ValueError, IndexError):
                pass

        track_data.append({
            "Color": TRACK_COLORS[idx % len(TRACK_COLORS)],
            "Track ID": track["track_id"][:20] + "...",
            "Category": track.get("category", "unknown"),
            "Objects": len(objects),
            "Start Frame": min(frame_ids) if frame_ids else "-",
            "End Frame": max(frame_ids) if frame_ids else "-",
            "Duration": max(frame_ids) - min(frame_ids) + 1 if frame_ids else 0,
            "Avg Confidence": f"{sum(confidences)/len(confidences):.2f}" if confidences else "-",
        })

    df = pd.DataFrame(track_data)

    # Style the dataframe with track colors
    def color_row(row):
        color = row["Color"]
        return [f"background-color: {color}20" for _ in row]

    styled_df = df.style.apply(color_row, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Individual track inspection
    st.subheader("Inspect Track")
    if selected_tracks:
        selected_idx = st.selectbox(
            "Select a track to inspect",
            options=selected_tracks,
            format_func=lambda x: track_options[x],
        )

        if selected_idx is not None:
            track = tracks[selected_idx]
            objects = track.get("objects", [])

            st.markdown(f"**Track ID:** `{track['track_id']}`")
            st.markdown(f"**Category:** {track.get('category', 'unknown')}")
            st.markdown(f"**Number of detections:** {len(objects)}")

            # Show object details
            if st.checkbox("Show object details", value=False):
                obj_data = []
                for obj in objects:
                    obj_data.append({
                        "Object ID": obj.get("object_id", "")[:16] + "...",
                        "Frame": obj.get("frame_id", ""),
                        "X": f"{obj.get('bbox_x', 0):.1f}",
                        "Y": f"{obj.get('bbox_y', 0):.1f}",
                        "W": f"{obj.get('bbox_w', 0):.1f}",
                        "H": f"{obj.get('bbox_h', 0):.1f}",
                        "Confidence": f"{obj.get('confidence', 0):.2f}" if obj.get('confidence') else "-",
                    })
                st.dataframe(pd.DataFrame(obj_data), use_container_width=True, hide_index=True)


# ==================== Main App ====================

def main():
    """Main application."""
    # Sidebar
    filters = render_sidebar()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏷️ Validation",
        "🎯 Tracks",
        "📦 Export",
        "📊 Statistics",
    ])

    with tab1:
        render_main_content(filters)

    with tab2:
        render_tracks_tab()

    with tab3:
        render_export_tab()

    with tab4:
        render_stats_tab()


if __name__ == "__main__":
    main()
