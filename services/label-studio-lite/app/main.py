"""Label Studio Lite - Object Validation UI.

X-AnyLabeling inspired design with professional annotation workflow.
Claude/Anthropic warm color palette.
"""
import os
from io import BytesIO
from datetime import datetime
from typing import Optional, List, Dict, Any

import httpx
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Configuration
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://object-registry:8010")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8000")
DATA_DIR = os.getenv("DATA_DIR", "/data")

# Anthropic/Claude Color Palette
COLORS = {
    "bg_primary": "#FAF9F6",
    "bg_secondary": "#F5F1EB",
    "bg_dark": "#1A1A1A",
    "bg_card": "#FFFFFF",
    "accent": "#DA7756",
    "accent_hover": "#C66A4A",
    "accent_light": "#F5E6E0",
    "text_primary": "#1A1A1A",
    "text_secondary": "#6B6B6B",
    "text_muted": "#9CA3AF",
    "success": "#10B981",
    "error": "#EF4444",
    "warning": "#F59E0B",
    "info": "#3B82F6",
    "border": "#E5E5E5",
    "border_focus": "#DA7756",
}

# Track colors for visualization
TRACK_COLORS = [
    "#DA7756", "#4ECDC4", "#45B7D1", "#96CEB4", "#F7DC6F",
    "#BB8FCE", "#85C1E9", "#F8B500", "#00CED1", "#32CD32",
]

# Page config
st.set_page_config(
    page_title="AgenticLabeling",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state initialization
if "selected_object_idx" not in st.session_state:
    st.session_state.selected_object_idx = None
if "current_image_idx" not in st.session_state:
    st.session_state.current_image_idx = 0
if "zoom_level" not in st.session_state:
    st.session_state.zoom_level = 1.0
if "show_labels" not in st.session_state:
    st.session_state.show_labels = True
if "show_bboxes" not in st.session_state:
    st.session_state.show_bboxes = True


# X-AnyLabeling inspired CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {{
        background-color: {COLORS["bg_primary"]};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Top Toolbar */
    .toolbar {{
        background: {COLORS["bg_dark"]};
        padding: 8px 16px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 16px;
    }}

    .toolbar-brand {{
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .toolbar-divider {{
        width: 1px;
        height: 24px;
        background: #444;
    }}

    .toolbar-item {{
        color: #ccc;
        font-size: 0.85rem;
        padding: 6px 12px;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.15s;
    }}

    .toolbar-item:hover {{
        background: #333;
        color: white;
    }}

    .toolbar-progress {{
        color: {COLORS["accent"]};
        font-weight: 600;
        margin-left: auto;
    }}

    /* Left Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLORS["bg_secondary"]};
        border-right: 1px solid {COLORS["border"]};
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: {COLORS["text_primary"]};
    }}

    /* Panel Headers */
    .panel-header {{
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {COLORS["text_secondary"]};
        padding: 12px 0 8px 0;
        border-bottom: 1px solid {COLORS["border"]};
        margin-bottom: 12px;
    }}

    /* Tool Buttons */
    .tool-btn {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 6px;
        background: transparent;
        border: 1px solid transparent;
        cursor: pointer;
        transition: all 0.15s;
        color: {COLORS["text_primary"]};
        font-size: 0.875rem;
    }}

    .tool-btn:hover {{
        background: {COLORS["accent_light"]};
        border-color: {COLORS["accent"]};
    }}

    .tool-btn.active {{
        background: {COLORS["accent"]};
        color: white;
    }}

    .tool-icon {{
        width: 20px;
        text-align: center;
    }}

    /* Object List Item */
    .object-item {{
        display: flex;
        align-items: center;
        padding: 10px 12px;
        border-radius: 6px;
        margin-bottom: 4px;
        cursor: pointer;
        transition: all 0.15s;
        border: 1px solid transparent;
    }}

    .object-item:hover {{
        background: {COLORS["bg_secondary"]};
    }}

    .object-item.selected {{
        background: {COLORS["accent_light"]};
        border-color: {COLORS["accent"]};
    }}

    .object-item.validated {{
        border-left: 3px solid {COLORS["success"]};
    }}

    .object-checkbox {{
        margin-right: 10px;
    }}

    .object-label {{
        flex: 1;
        font-size: 0.875rem;
        font-weight: 500;
    }}

    .object-confidence {{
        font-size: 0.75rem;
        color: {COLORS["text_muted"]};
        background: {COLORS["bg_secondary"]};
        padding: 2px 8px;
        border-radius: 12px;
    }}

    /* Canvas Container */
    .canvas-container {{
        background: {COLORS["bg_card"]};
        border-radius: 8px;
        border: 1px solid {COLORS["border"]};
        padding: 16px;
        min-height: 500px;
    }}

    /* Status Bar */
    .status-bar {{
        background: {COLORS["bg_dark"]};
        color: #ccc;
        padding: 8px 16px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        gap: 24px;
        font-size: 0.8rem;
        margin-top: 16px;
    }}

    .status-item {{
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    .status-icon {{
        opacity: 0.7;
    }}

    /* Right Panel - Label Editor */
    .label-editor {{
        background: {COLORS["bg_card"]};
        border-radius: 8px;
        border: 1px solid {COLORS["border"]};
        padding: 16px;
    }}

    .label-field {{
        margin-bottom: 16px;
    }}

    .label-field-label {{
        font-size: 0.75rem;
        font-weight: 500;
        color: {COLORS["text_secondary"]};
        margin-bottom: 6px;
    }}

    /* Action Buttons */
    .action-btn {{
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.15s;
        border: none;
        width: 100%;
        margin-bottom: 8px;
    }}

    .action-btn-primary {{
        background: {COLORS["accent"]};
        color: white;
    }}

    .action-btn-primary:hover {{
        background: {COLORS["accent_hover"]};
    }}

    .action-btn-success {{
        background: {COLORS["success"]};
        color: white;
    }}

    .action-btn-danger {{
        background: {COLORS["error"]};
        color: white;
    }}

    .action-btn-secondary {{
        background: transparent;
        border: 1px solid {COLORS["border"]};
        color: {COLORS["text_primary"]};
    }}

    /* Stats Cards */
    .stat-card {{
        background: {COLORS["bg_card"]};
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid {COLORS["border"]};
    }}

    .stat-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {COLORS["accent"]};
    }}

    .stat-label {{
        font-size: 0.7rem;
        color: {COLORS["text_muted"]};
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Badge */
    .badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
    }}

    .badge-success {{
        background: #D1FAE5;
        color: #065F46;
    }}

    .badge-warning {{
        background: #FEF3C7;
        color: #92400E;
    }}

    .badge-error {{
        background: #FEE2E2;
        color: #991B1B;
    }}

    /* Keyboard Shortcut */
    .kbd {{
        display: inline-block;
        padding: 2px 6px;
        font-size: 0.7rem;
        font-family: monospace;
        background: {COLORS["bg_secondary"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 4px;
        color: {COLORS["text_secondary"]};
    }}

    /* Streamlit overrides */
    .stButton > button {{
        background-color: {COLORS["accent"]};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }}

    .stButton > button:hover {{
        background-color: {COLORS["accent_hover"]};
    }}

    hr {{
        border: none;
        height: 1px;
        background: {COLORS["border"]};
        margin: 16px 0;
    }}

    /* Empty State */
    .empty-state {{
        text-align: center;
        padding: 48px 24px;
        color: {COLORS["text_muted"]};
    }}

    .empty-icon {{
        font-size: 3rem;
        margin-bottom: 16px;
    }}

    /* Error Container */
    .error-container {{
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 8px;
        padding: 16px;
        color: #991B1B;
    }}

    /* Info Box */
    .info-box {{
        background: {COLORS["accent_light"]};
        border: 1px solid {COLORS["accent"]};
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.875rem;
        color: {COLORS["text_primary"]};
    }}
</style>
""", unsafe_allow_html=True)


# ==================== API Functions ====================

def api_request(method: str, url: str, **kwargs) -> tuple:
    """Make API request with error handling."""
    try:
        kwargs.setdefault("timeout", 10.0)
        if method == "GET":
            resp = httpx.get(url, **kwargs)
        elif method == "POST":
            resp = httpx.post(url, **kwargs)
        elif method == "PATCH":
            resp = httpx.patch(url, **kwargs)
        elif method == "DELETE":
            resp = httpx.delete(url, **kwargs)
        else:
            return None, f"Unknown method: {method}"

        data = resp.json()
        if data.get("success"):
            return data.get("data"), None
        return None, data.get("error", "Unknown error")
    except httpx.ConnectError:
        return None, "서비스에 연결할 수 없습니다"
    except httpx.TimeoutException:
        return None, "요청 시간 초과"
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=30)
def get_stats() -> dict:
    """Get registry statistics."""
    data, error = api_request("GET", f"{REGISTRY_URL}/stats")
    return data if data else {}


@st.cache_data(ttl=10)
def get_categories() -> list:
    """Get all categories."""
    data, error = api_request("GET", f"{REGISTRY_URL}/categories")
    return data if data else []


def get_objects(is_validated=None, category=None, limit=100) -> tuple:
    """Get objects with filters."""
    params = {"limit": limit}
    if is_validated is not None:
        params["is_validated"] = is_validated
    if category:
        params["category"] = category
    data, error = api_request("GET", f"{REGISTRY_URL}/objects", params=params)
    return (data if data else [], error)


def get_sources(limit=50) -> list:
    """Get source list."""
    data, error = api_request("GET", f"{REGISTRY_URL}/sources", params={"limit": limit})
    return data if data else []


def get_source(source_id: str) -> Optional[dict]:
    """Get source details."""
    data, error = api_request("GET", f"{REGISTRY_URL}/sources/{source_id}")
    return data


def get_objects_by_source(source_id: str) -> list:
    """Get objects for a specific source."""
    data, error = api_request("GET", f"{REGISTRY_URL}/objects", params={"source_id": source_id, "limit": 200})
    return data if data else []


def validate_object(object_id: str, reviewer: str, quality: float) -> bool:
    """Validate an object."""
    data, error = api_request(
        "PATCH",
        f"{REGISTRY_URL}/objects/{object_id}",
        json={"is_validated": True, "validated_by": reviewer, "quality_score": quality}
    )
    return error is None


def reject_object(object_id: str) -> bool:
    """Reject/delete an object."""
    data, error = api_request("DELETE", f"{REGISTRY_URL}/objects/{object_id}")
    return error is None


def get_tracks(source_id: str) -> list:
    """Get tracks for a source."""
    data, error = api_request("GET", f"{REGISTRY_URL}/tracks", params={"source_id": source_id})
    return data if data else []


# ==================== Image Functions ====================

def load_source_image(source_id: str) -> Optional[Image.Image]:
    """Load source image from file system."""
    source = get_source(source_id)
    if not source:
        return None

    file_path = source.get("file_path")
    if not file_path:
        return None

    paths_to_try = [
        file_path,
        os.path.join(DATA_DIR, file_path),
        os.path.join(DATA_DIR, "uploads", os.path.basename(file_path)),
    ]

    for path in paths_to_try:
        if path and os.path.exists(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                continue
    return None


def create_placeholder_image(width: int, height: int, text: str = "") -> Image.Image:
    """Create a placeholder image."""
    img = Image.new("RGB", (width, height), color="#F5F1EB")
    draw = ImageDraw.Draw(img)

    # Grid pattern
    for i in range(0, width, 20):
        draw.line([(i, 0), (i, height)], fill="#E5E5E5", width=1)
    for i in range(0, height, 20):
        draw.line([(0, i), (width, i)], fill="#E5E5E5", width=1)

    if text:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        x = (width - (bbox[2] - bbox[0])) // 2
        y = (height - (bbox[3] - bbox[1])) // 2
        draw.text((x, y), text, fill="#9CA3AF", font=font)

    return img


def draw_annotations(image: Image.Image, objects: list, selected_idx: Optional[int] = None,
                     show_labels: bool = True, show_bboxes: bool = True) -> Image.Image:
    """Draw bounding boxes and labels on image."""
    if not show_bboxes:
        return image

    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    for idx, obj in enumerate(objects):
        x = obj.get("bbox_x", 0)
        y = obj.get("bbox_y", 0)
        w = obj.get("bbox_w", 0)
        h = obj.get("bbox_h", 0)

        # Color based on selection and validation
        is_selected = idx == selected_idx
        is_validated = obj.get("is_validated", False)

        if is_selected:
            color = COLORS["accent"]
            width = 3
        elif is_validated:
            color = COLORS["success"]
            width = 2
        else:
            color = TRACK_COLORS[idx % len(TRACK_COLORS)]
            width = 2

        # Draw bbox
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)

        # Draw label
        if show_labels:
            label = obj.get("category_name", "unknown")
            conf = obj.get("confidence")
            if conf:
                label += f" {conf:.0%}"

            text_bbox = draw.textbbox((x, y - 18), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x, y - 18), label, fill="white", font=font)

    return img


# ==================== UI Components ====================

def render_toolbar(stats: dict):
    """Render top toolbar."""
    total = stats.get("objects", 0) or 0
    validated = stats.get("validated_objects", 0) or 0
    progress_pct = (validated / total * 100) if total > 0 else 0

    st.markdown(f"""
    <div class="toolbar">
        <div class="toolbar-brand">🏷️ AgenticLabeling</div>
        <div class="toolbar-divider"></div>
        <div class="toolbar-item">📁 파일</div>
        <div class="toolbar-item">👁️ 보기</div>
        <div class="toolbar-item">🔧 도구</div>
        <div class="toolbar-item">❓ 도움말</div>
        <div class="toolbar-progress">
            진행률: {validated}/{total} ({progress_pct:.0f}% 완료)
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_status_bar(source: Optional[dict], objects: list, current_idx: int):
    """Render bottom status bar."""
    filename = source.get("file_path", "없음").split("/")[-1] if source else "선택 없음"
    total = len(objects)
    validated = sum(1 for o in objects if o.get("is_validated"))
    width = source.get("width", 0) if source else 0
    height = source.get("height", 0) if source else 0

    st.markdown(f"""
    <div class="status-bar">
        <div class="status-item">
            <span class="status-icon">📁</span>
            <span>{filename}</span>
        </div>
        <div class="status-item">
            <span class="status-icon">🖼️</span>
            <span>{width} × {height}</span>
        </div>
        <div class="status-item">
            <span class="status-icon">📦</span>
            <span>객체 {total}개 (검증됨 {validated})</span>
        </div>
        <div class="status-item">
            <span class="status-icon">⏱️</span>
            <span>{datetime.now().strftime('%H:%M:%S')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_left_sidebar(stats: dict, categories: list) -> dict:
    """Render left sidebar with tools and filters."""
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="padding: 16px 0; border-bottom: 1px solid #E5E5E5; margin-bottom: 16px;">
            <div style="font-size: 1.25rem; font-weight: 700;">🏷️ AgenticLabeling</div>
            <div style="font-size: 0.75rem; color: #9CA3AF;">AI-Powered Labeling</div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Stats
        st.markdown('<div class="panel-header">📊 통계</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{stats.get('objects', 0)}</div>
                <div class="stat-label">전체</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{stats.get('validated_objects', 0)}</div>
                <div class="stat-label">검증됨</div>
            </div>
            """, unsafe_allow_html=True)

        # Progress
        total = stats.get("objects", 0)
        validated = stats.get("validated_objects", 0)
        if total > 0:
            st.progress(validated / total)
            st.caption(f"{validated/total*100:.1f}% 완료")

        st.markdown("---")

        # View Options
        st.markdown('<div class="panel-header">👁️ 보기 설정</div>', unsafe_allow_html=True)
        show_labels = st.checkbox("라벨 표시", value=True, key="show_labels_cb")
        show_bboxes = st.checkbox("바운딩박스 표시", value=True, key="show_bboxes_cb")

        st.markdown("---")

        # Filters
        st.markdown('<div class="panel-header">🔍 필터</div>', unsafe_allow_html=True)

        validation_filter = st.radio(
            "검증 상태",
            ["전체", "미검증", "검증완료"],
            index=1,
            horizontal=True,
            label_visibility="collapsed"
        )

        is_validated = None
        if validation_filter == "미검증":
            is_validated = False
        elif validation_filter == "검증완료":
            is_validated = True

        category_names = ["전체"] + [c.get("name", "") for c in categories if c.get("name")]
        selected_category = st.selectbox("카테고리", category_names, label_visibility="collapsed")
        category = None if selected_category == "전체" else selected_category

        st.markdown("---")

        # Settings
        st.markdown('<div class="panel-header">⚙️ 설정</div>', unsafe_allow_html=True)
        reviewer_name = st.text_input("검토자", value="reviewer", label_visibility="collapsed",
                                       placeholder="검토자 이름")
        default_quality = st.slider("기본 품질 점수", 0.0, 1.0, 0.9, 0.05)

        st.markdown("---")

        # Keyboard Shortcuts
        st.markdown('<div class="panel-header">⌨️ 단축키</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 0.8rem; color: #6B6B6B;">
            <div style="margin-bottom: 4px;"><span class="kbd">D</span> 다음 이미지</div>
            <div style="margin-bottom: 4px;"><span class="kbd">A</span> 이전 이미지</div>
            <div style="margin-bottom: 4px;"><span class="kbd">V</span> 검증</div>
            <div style="margin-bottom: 4px;"><span class="kbd">X</span> 거부</div>
        </div>
        """, unsafe_allow_html=True)

        return {
            "is_validated": is_validated,
            "category": category,
            "reviewer_name": reviewer_name,
            "default_quality": default_quality,
            "show_labels": show_labels,
            "show_bboxes": show_bboxes,
        }


def render_object_list(objects: list, selected_idx: Optional[int]) -> Optional[int]:
    """Render right panel object list."""
    st.markdown('<div class="panel-header">📋 객체 목록</div>', unsafe_allow_html=True)

    if not objects:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📭</div>
            <p>객체가 없습니다</p>
        </div>
        """, unsafe_allow_html=True)
        return None

    new_selected = selected_idx

    for idx, obj in enumerate(objects):
        category = obj.get("category_name", "unknown")
        confidence = obj.get("confidence", 0) or 0
        is_validated = obj.get("is_validated", False)
        object_id = obj.get("object_id", "")[:8]

        selected_class = "selected" if idx == selected_idx else ""
        validated_class = "validated" if is_validated else ""

        col1, col2, col3 = st.columns([0.5, 3, 1])

        with col1:
            if st.checkbox("", value=(idx == selected_idx), key=f"obj_select_{idx}",
                          label_visibility="collapsed"):
                new_selected = idx

        with col2:
            status = "✓" if is_validated else "○"
            st.markdown(f"**{status} {category}**")
            st.caption(f"ID: {object_id}...")

        with col3:
            st.markdown(f"""
            <div class="object-confidence">{confidence:.0%}</div>
            """, unsafe_allow_html=True)

    return new_selected


def render_label_editor(obj: Optional[dict], filters: dict) -> tuple:
    """Render label editor panel."""
    st.markdown('<div class="panel-header">🏷️ 라벨 편집</div>', unsafe_allow_html=True)

    if not obj:
        st.markdown("""
        <div class="info-box">
            왼쪽 목록에서 객체를 선택하세요
        </div>
        """, unsafe_allow_html=True)
        return None, None

    object_id = obj.get("object_id", "")
    category = obj.get("category_name", "unknown")
    confidence = obj.get("confidence", 0) or 0
    is_validated = obj.get("is_validated", False)
    bbox = (obj.get("bbox_x", 0), obj.get("bbox_y", 0),
            obj.get("bbox_w", 0), obj.get("bbox_h", 0))

    # Object Info
    st.markdown(f"**ID:** `{object_id[:16]}...`")

    st.markdown('<div class="label-field-label">카테고리</div>', unsafe_allow_html=True)
    st.text_input("", value=category, disabled=True, key="cat_display", label_visibility="collapsed")

    st.markdown('<div class="label-field-label">신뢰도</div>', unsafe_allow_html=True)
    st.progress(confidence)
    st.caption(f"{confidence:.1%}")

    st.markdown('<div class="label-field-label">바운딩박스</div>', unsafe_allow_html=True)
    st.code(f"x:{bbox[0]:.0f} y:{bbox[1]:.0f} w:{bbox[2]:.0f} h:{bbox[3]:.0f}")

    st.markdown('<div class="label-field-label">상태</div>', unsafe_allow_html=True)
    if is_validated:
        st.markdown('<span class="badge badge-success">✓ 검증됨</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-warning">⏳ 대기중</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Action Buttons
    validate_clicked = False
    reject_clicked = False

    if not is_validated:
        if st.button("✓ 검증하기", key="validate_btn", use_container_width=True, type="primary"):
            validate_clicked = True

        if st.button("✗ 거부하기", key="reject_btn", use_container_width=True):
            reject_clicked = True
    else:
        st.success("이 객체는 검증되었습니다")

    return validate_clicked, reject_clicked


def render_canvas(source: Optional[dict], objects: list, selected_idx: Optional[int],
                  show_labels: bool, show_bboxes: bool):
    """Render main canvas area."""

    if not source:
        st.markdown("""
        <div class="canvas-container">
            <div class="empty-state">
                <div class="empty-icon">🖼️</div>
                <p>이미지를 선택하세요</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Try to load actual image
    source_id = source.get("source_id", "")
    image = load_source_image(source_id)

    if not image:
        width = source.get("width", 800)
        height = source.get("height", 600)
        image = create_placeholder_image(width, height, "이미지를 로드할 수 없습니다")

    # Draw annotations
    annotated_image = draw_annotations(image, objects, selected_idx, show_labels, show_bboxes)

    # Display
    st.image(annotated_image, use_container_width=True)


# ==================== Main Views ====================

def annotation_view():
    """Main annotation view with X-AnyLabeling layout."""
    stats = get_stats()
    categories = get_categories()

    # Top toolbar
    render_toolbar(stats)

    # Left sidebar (filters and settings)
    filters = render_left_sidebar(stats, categories)

    # Get sources for navigation
    sources = get_sources(limit=100)

    # Main content area
    if not sources:
        st.markdown("""
        <div class="canvas-container">
            <div class="empty-state">
                <div class="empty-icon">📁</div>
                <p>등록된 소스가 없습니다</p>
                <p style="font-size: 0.875rem; margin-top: 8px;">
                    Gateway API를 통해 이미지를 업로드하세요
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Source selector
    source_options = {
        f"{s.get('file_path', '').split('/')[-1]} ({s.get('source_type', 'image')})": s
        for s in sources
    }

    col_nav1, col_nav2, col_nav3 = st.columns([1, 6, 1])
    with col_nav1:
        if st.button("◀ 이전", use_container_width=True):
            if st.session_state.current_image_idx > 0:
                st.session_state.current_image_idx -= 1
                st.session_state.selected_object_idx = None
                st.rerun()

    with col_nav2:
        selected_source_name = st.selectbox(
            "소스 선택",
            list(source_options.keys()),
            index=min(st.session_state.current_image_idx, len(source_options) - 1),
            label_visibility="collapsed"
        )
        current_source = source_options.get(selected_source_name)

    with col_nav3:
        if st.button("다음 ▶", use_container_width=True):
            if st.session_state.current_image_idx < len(sources) - 1:
                st.session_state.current_image_idx += 1
                st.session_state.selected_object_idx = None
                st.rerun()

    # Get objects for current source
    if current_source:
        source_id = current_source.get("source_id", "")
        objects = get_objects_by_source(source_id)
    else:
        objects = []

    # Three-column layout: Object List | Canvas | Label Editor
    col_left, col_center, col_right = st.columns([2, 5, 2])

    with col_left:
        selected_idx = render_object_list(objects, st.session_state.selected_object_idx)
        if selected_idx != st.session_state.selected_object_idx:
            st.session_state.selected_object_idx = selected_idx
            st.rerun()

    with col_center:
        render_canvas(
            current_source,
            objects,
            st.session_state.selected_object_idx,
            filters["show_labels"],
            filters["show_bboxes"]
        )

    with col_right:
        selected_obj = objects[st.session_state.selected_object_idx] if (
            st.session_state.selected_object_idx is not None and
            st.session_state.selected_object_idx < len(objects)
        ) else None

        validate_clicked, reject_clicked = render_label_editor(selected_obj, filters)

        if validate_clicked and selected_obj:
            object_id = selected_obj.get("object_id", "")
            if validate_object(object_id, filters["reviewer_name"], filters["default_quality"]):
                st.success("검증 완료!")
                st.cache_data.clear()
                st.rerun()

        if reject_clicked and selected_obj:
            object_id = selected_obj.get("object_id", "")
            if reject_object(object_id):
                st.warning("객체가 삭제되었습니다")
                st.session_state.selected_object_idx = None
                st.cache_data.clear()
                st.rerun()

    # Status bar
    render_status_bar(current_source, objects, st.session_state.current_image_idx)


def export_view():
    """Dataset export view."""
    st.markdown("## 📦 데이터셋 내보내기")
    st.caption("검증된 데이터를 YOLO 또는 COCO 형식으로 내보냅니다")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="label-editor">', unsafe_allow_html=True)
        st.markdown("### 설정")

        dataset_name = st.text_input("데이터셋 이름", value="my_dataset")
        export_format = st.selectbox("형식", ["YOLO", "COCO"])

        st.markdown("#### 분할 비율")
        train_ratio = st.slider("Train", 0.0, 1.0, 0.8, 0.05)
        val_ratio = st.slider("Validation", 0.0, 1.0, 0.1, 0.05)
        test_ratio = 1.0 - train_ratio - val_ratio
        st.metric("Test", f"{test_ratio:.0%}")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="label-editor">', unsafe_allow_html=True)
        st.markdown("### 필터")

        only_validated = st.checkbox("검증된 객체만", value=True)
        min_confidence = st.slider("최소 신뢰도", 0.0, 1.0, 0.5, 0.05)

        categories = get_categories()
        cat_names = [c.get("name", "") for c in categories if c.get("name")]
        selected_cats = st.multiselect("카테고리", cat_names, default=cat_names)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🚀 내보내기 시작", use_container_width=True, type="primary"):
        with st.spinner("내보내는 중..."):
            try:
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
                    st.success("✓ 내보내기 완료!")
                    st.json({
                        "dataset_name": data.get("dataset_name"),
                        "format": export_format,
                        "object_count": data.get("object_count"),
                        "image_count": data.get("image_count"),
                        "splits": data.get("splits"),
                    })
                else:
                    st.error(f"내보내기 실패: {result.get('error')}")
            except Exception as e:
                st.error(f"오류: {e}")


def stats_view():
    """Statistics dashboard view."""
    st.markdown("## 📊 통계 대시보드")

    stats = get_stats()

    if not stats:
        st.markdown("""
        <div class="error-container">
            <strong>⚠️ 연결 오류</strong><br>
            Registry 서비스에 연결할 수 없습니다
        </div>
        """, unsafe_allow_html=True)
        return

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">📦 {stats.get('objects', 0)}</div>
            <div class="stat-label">전체 객체</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">✓ {stats.get('validated_objects', 0)}</div>
            <div class="stat-label">검증 완료</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">🏷️ {stats.get('categories', 0)}</div>
            <div class="stat-label">카테고리</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">📁 {stats.get('sources', 0)}</div>
            <div class="stat-label">소스</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Progress
    total = stats.get("objects", 0)
    validated = stats.get("validated_objects", 0)
    if total > 0:
        st.markdown("### 검증 진행률")
        st.progress(validated / total)
        st.caption(f"{validated}/{total} 객체 검증됨 ({validated/total*100:.1f}%)")

    st.markdown("---")

    # Category distribution
    st.markdown("### 카테고리별 객체 수")
    category_counts = stats.get("objects_per_category", {})
    if category_counts:
        df = pd.DataFrame([
            {"카테고리": k, "객체 수": v}
            for k, v in category_counts.items()
        ])
        st.bar_chart(df.set_index("카테고리"))
    else:
        st.info("카테고리 데이터가 없습니다")


# ==================== Main App ====================

def main():
    """Main application with tabbed navigation."""

    # Tab navigation
    tab1, tab2, tab3 = st.tabs([
        "🏷️ 주석 편집",
        "📦 내보내기",
        "📊 통계",
    ])

    with tab1:
        annotation_view()

    with tab2:
        export_view()

    with tab3:
        stats_view()


if __name__ == "__main__":
    main()
