"""Agentic Labeling Studio - Claude Style Light Theme UI.

Clean, minimal design inspired by Claude's interface.
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
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8010")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
DATA_DIR = os.getenv("DATA_DIR", "/home/coffin/dev/AgenticLabeling")

# Claude Style Light Color Palette
COLORS = {
    "bg_primary": "#FFFFFF",
    "bg_secondary": "#F9FAFB",
    "bg_card": "#FFFFFF",
    "bg_hover": "#F3F4F6",
    "accent": "#DA7756",  # Claude coral/orange
    "accent_light": "#FCEEE8",
    "accent_dark": "#C4624A",
    "text_primary": "#1F2937",
    "text_secondary": "#6B7280",
    "text_muted": "#9CA3AF",
    "success": "#059669",
    "success_light": "#D1FAE5",
    "error": "#DC2626",
    "error_light": "#FEE2E2",
    "warning": "#D97706",
    "warning_light": "#FEF3C7",
    "info": "#2563EB",
    "info_light": "#DBEAFE",
    "border": "#E5E7EB",
    "border_dark": "#D1D5DB",
}

# Object colors for visualization (visible on light background)
TRACK_COLORS = [
    "#DA7756", "#2563EB", "#059669", "#DC2626", "#7C3AED",
    "#D97706", "#0891B2", "#DB2777", "#4F46E5", "#15803D",
]

# Page config
st.set_page_config(
    page_title="Agentic Labeling Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
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


# Claude Style Light Theme CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Light Theme */
    .stApp {{
        background-color: {COLORS["bg_secondary"]};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: {COLORS["text_primary"]};
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Custom Scrollbar - Light */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {COLORS["bg_secondary"]};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {COLORS["border_dark"]};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS["text_muted"]};
    }}

    /* Main Content Area */
    .main .block-container {{
        padding-top: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }}

    /* Sidebar Light Theme */
    [data-testid="stSidebar"] {{
        background-color: {COLORS["bg_primary"]};
        border-right: 1px solid {COLORS["border"]};
    }}

    /* Headers */
    h1, h2, h3, h4 {{
        color: {COLORS["text_primary"]} !important;
        font-weight: 600 !important;
    }}

    /* Primary Buttons - Claude Orange */
    .stButton > button[kind="primary"],
    .stButton > button {{
        background-color: {COLORS["accent"]} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
        box-shadow: none !important;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button:hover {{
        background-color: {COLORS["accent_dark"]} !important;
        box-shadow: 0 2px 8px rgba(218, 119, 86, 0.3) !important;
    }}

    /* Secondary Buttons */
    .stButton > button[kind="secondary"] {{
        background-color: {COLORS["bg_primary"]} !important;
        color: {COLORS["text_primary"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        box-shadow: none !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        background-color: {COLORS["bg_hover"]} !important;
        border-color: {COLORS["border_dark"]} !important;
    }}

    /* Input Fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {{
        background-color: {COLORS["bg_primary"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 8px !important;
        color: {COLORS["text_primary"]} !important;
    }}
    .stTextInput > div > div > input:focus {{
        border-color: {COLORS["accent"]} !important;
        box-shadow: 0 0 0 3px {COLORS["accent_light"]} !important;
    }}

    /* Checkbox */
    .stCheckbox label {{
        color: {COLORS["text_primary"]} !important;
    }}

    /* Progress Bar */
    .stProgress > div > div > div {{
        background-color: {COLORS["accent"]} !important;
    }}

    /* Metric */
    [data-testid="stMetricValue"] {{
        color: {COLORS["accent"]} !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLORS["text_secondary"]} !important;
    }}

    /* Success/Info/Warning/Error alerts */
    .stSuccess {{
        background-color: {COLORS["success_light"]} !important;
        color: {COLORS["success"]} !important;
        border: none !important;
    }}
    .stInfo {{
        background-color: {COLORS["info_light"]} !important;
        color: {COLORS["info"]} !important;
        border: none !important;
    }}
    .stWarning {{
        background-color: {COLORS["warning_light"]} !important;
        color: {COLORS["warning"]} !important;
        border: none !important;
    }}
    .stError {{
        background-color: {COLORS["error_light"]} !important;
        color: {COLORS["error"]} !important;
        border: none !important;
    }}

    /* Container with height (scrollable panels) */
    [data-testid="stVerticalBlock"] > [style*="height"] {{
        background-color: {COLORS["bg_primary"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 12px !important;
    }}

    /* Divider */
    hr {{
        border-color: {COLORS["border"]} !important;
    }}

    /* Caption */
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {COLORS["text_muted"]} !important;
    }}

    /* Markdown text */
    .stMarkdown {{
        color: {COLORS["text_primary"]};
    }}

    /* Column gaps */
    [data-testid="column"] {{
        padding: 0 0.5rem;
    }}

    /* Select box dropdown */
    [data-baseweb="select"] {{
        background-color: {COLORS["bg_primary"]} !important;
    }}
    [data-baseweb="menu"] {{
        background-color: {COLORS["bg_primary"]} !important;
        border: 1px solid {COLORS["border"]} !important;
    }}

    /* Image container */
    [data-testid="stImage"] {{
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid {COLORS["border"]};
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


@st.cache_data(ttl=5)
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


@st.cache_data(ttl=5)
def get_sources(limit=50) -> list:
    """Get source list."""
    data, error = api_request("GET", f"{REGISTRY_URL}/sources", params={"limit": limit})
    return data if data else []


@st.cache_data(ttl=5)
def get_source(source_id: str) -> Optional[dict]:
    """Get source details."""
    data, error = api_request("GET", f"{REGISTRY_URL}/sources/{source_id}")
    return data


@st.cache_data(ttl=3)
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
    """Render top toolbar with functional menus."""
    total = stats.get("objects", 0) or 0
    validated = stats.get("validated_objects", 0) or 0
    progress_pct = (validated / total * 100) if total > 0 else 0

    # Toolbar with Streamlit components
    cols = st.columns([2, 1, 1, 1, 1, 4])

    with cols[0]:
        st.markdown("### 🏷️ AgenticLabeling")

    with cols[1]:
        with st.popover("📁 파일"):
            if st.button("🔄 새로고침", key="tb_refresh", width="stretch"):
                st.cache_data.clear()
                st.rerun()
            if st.button("📤 내보내기", key="tb_export", width="stretch"):
                st.session_state.active_tab = 1
                st.rerun()

    with cols[2]:
        with st.popover("👁️ 보기"):
            st.session_state.show_labels = st.checkbox("라벨 표시", value=st.session_state.get("show_labels", True), key="tb_labels")
            st.session_state.show_bboxes = st.checkbox("박스 표시", value=st.session_state.get("show_bboxes", True), key="tb_bboxes")

    with cols[3]:
        with st.popover("🔧 도구"):
            st.markdown("**검증 도구**")
            st.caption("V: 현재 객체 검증")
            st.caption("X: 현재 객체 삭제")
            st.caption("←/→: 이전/다음 이미지")

    with cols[4]:
        with st.popover("❓ 도움말"):
            st.markdown("**AgenticLabeling v0.1**")
            st.markdown("AI 기반 자동 라벨링 도구")
            st.markdown("---")
            st.markdown("[문서](https://github.com) | [이슈](https://github.com)")

    with cols[5]:
        st.markdown(f"<div style='text-align: right; padding-top: 8px; color: #DA7756; font-weight: 600;'>진행률: {validated}/{total} ({progress_pct:.0f}%)</div>", unsafe_allow_html=True)


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


def render_image_list(sources: list, current_idx: int) -> int:
    """Render scrollable image list panel."""
    st.markdown("#### 🖼️ 이미지 목록")
    st.caption(f"총 {len(sources)}개")

    new_idx = current_idx

    # Scrollable container with fixed height
    with st.container(height=400):
        for idx, source in enumerate(sources):
            filename = source.get("file_path", "").split("/")[-1] or f"Source {idx}"
            source_type = source.get("source_type", "image")
            is_selected = idx == current_idx

            # Icon based on type
            icon = "🖼️" if source_type == "image" else "🎬" if source_type == "video" else "📄"

            # Button for each image
            label = f"{icon} {filename[:20]}{'...' if len(filename) > 20 else ''}"
            btn_type = "primary" if is_selected else "secondary"

            if st.button(label, key=f"img_btn_{idx}", width="stretch", type=btn_type):
                new_idx = idx

    return new_idx


def render_object_list(objects: list, selected_idx: Optional[int]) -> Optional[int]:
    """Render object list with click-to-select buttons."""
    if not objects:
        st.caption("객체 없음")
        return None

    new_selected = selected_idx

    # Scrollable container for objects
    for idx, obj in enumerate(objects):
        category = obj.get("category_name", "unknown")
        confidence = obj.get("confidence", 0) or 0
        is_validated = obj.get("is_validated", False)
        is_selected = idx == selected_idx

        # Status icon
        if is_validated:
            status = "✅"
        elif is_selected:
            status = "👉"
        else:
            status = "⬜"

        # Button style based on selection
        btn_type = "primary" if is_selected else "secondary"

        # Object button
        label = f"{status} {category} ({confidence:.0%})"
        if st.button(label, key=f"obj_btn_{idx}", width="stretch",
                    type=btn_type if is_selected else "secondary"):
            new_selected = idx

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
        if st.button("✓ 검증하기", key="validate_btn", width="stretch", type="primary"):
            validate_clicked = True

        if st.button("✗ 거부하기", key="reject_btn", width="stretch"):
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
    st.image(annotated_image, width="stretch")


# ==================== Main Views ====================

def render_workflow_steps(current_step: int, stats: dict):
    """Render workflow progress indicator using native Streamlit components."""
    steps = [
        ("📤", "업로드", stats.get("sources", 0)),
        ("🔍", "탐지", stats.get("objects", 0)),
        ("✓", "검증", stats.get("validated_objects", 0)),
        ("📦", "내보내기", stats.get("datasets", 0)),
    ]

    cols = st.columns(4)
    for i, (icon, label, count) in enumerate(steps):
        with cols[i]:
            is_current = i == current_step
            is_active = i <= current_step

            if is_current:
                st.markdown(f"**{icon} {label}**")
                st.caption(f"{count}개 • 진행중")
            elif is_active:
                st.markdown(f"{icon} {label}")
                st.caption(f"{count}개 ✓")
            else:
                st.markdown(f"~~{icon} {label}~~")
                st.caption(f"{count}개")


def unified_workspace_view():
    """Unified workspace showing entire pipeline in one page."""
    stats = get_stats()
    categories = get_categories()
    sources = get_sources(limit=100)

    # Calculate current workflow step
    total_objects = stats.get("objects", 0)
    validated = stats.get("validated_objects", 0)
    if not sources:
        current_step = 0
    elif total_objects == 0:
        current_step = 0
    elif validated == 0:
        current_step = 1
    elif validated < total_objects:
        current_step = 2
    else:
        current_step = 3

    # Workflow Steps Indicator
    render_workflow_steps(current_step, stats)

    # Get current source data early
    if sources:
        if st.session_state.current_image_idx >= len(sources):
            st.session_state.current_image_idx = 0
        current_source = sources[st.session_state.current_image_idx]
        objects = get_objects_by_source(current_source.get("source_id", ""))
    else:
        current_source = None
        objects = []

    # Three-Panel Layout: Gallery | Preview | Actions
    col_gallery, col_preview, col_actions = st.columns([2, 5, 3])

    # ============ LEFT PANEL: Source Gallery ============
    with col_gallery:
        st.markdown("#### 📁 소스")

        if not sources:
            st.info("📤 API로 이미지를 업로드하세요")
        else:
            with st.container(height=450):
                for idx, source in enumerate(sources):
                    fname = source.get("file_path", "").split("/")[-1]
                    is_selected = idx == st.session_state.current_image_idx

                    btn_label = f"{'▶ ' if is_selected else ''}{fname[:18]}{'...' if len(fname) > 18 else ''}"
                    if st.button(
                        btn_label,
                        key=f"src_{idx}",
                        width="stretch",
                        type="primary" if is_selected else "secondary"
                    ):
                        st.session_state.current_image_idx = idx
                        st.session_state.selected_object_idx = None
                        st.rerun()

    # ============ CENTER PANEL: Preview & Detection ============
    with col_preview:
        if current_source:
            # Header with navigation
            nav1, nav2, nav3 = st.columns([1, 8, 1])
            with nav1:
                if st.button("◀", key="nav_prev", width="stretch", disabled=st.session_state.current_image_idx <= 0):
                    st.session_state.current_image_idx -= 1
                    st.session_state.selected_object_idx = None
                    st.rerun()
            with nav2:
                filename = current_source.get("file_path", "").split("/")[-1]
                st.markdown(f"**{filename}**")
            with nav3:
                if st.button("▶", key="nav_next", width="stretch", disabled=st.session_state.current_image_idx >= len(sources) - 1):
                    st.session_state.current_image_idx += 1
                    st.session_state.selected_object_idx = None
                    st.rerun()

            # Detection status
            if objects:
                validated_cnt = sum(1 for o in objects if o.get('is_validated'))
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.success(f"✓ {len(objects)}개 탐지")
                with col_s2:
                    st.info(f"검증: {validated_cnt}/{len(objects)}")

            # Canvas
            render_canvas(
                current_source,
                objects,
                st.session_state.selected_object_idx,
                st.session_state.get("show_labels", True),
                st.session_state.get("show_bboxes", True)
            )

            # View controls
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.show_labels = st.checkbox("라벨", value=True, key="chk_labels")
            with c2:
                st.session_state.show_bboxes = st.checkbox("박스", value=True, key="chk_bboxes")
            with c3:
                if st.button("🔄", key="refresh_canvas", help="새로고침"):
                    st.cache_data.clear()
                    st.rerun()
        else:
            st.markdown("""
            <div style="text-align: center; padding: 100px 40px; background: rgba(255,255,255,0.02); border-radius: 16px; border: 1px dashed rgba(255,255,255,0.1);">
                <div style="font-size: 4rem; margin-bottom: 20px;">🖼️</div>
                <h3 style="color: #fff; margin-bottom: 8px;">이미지를 선택하세요</h3>
                <p style="color: #666;">왼쪽 갤러리에서 이미지를 선택하거나 API로 업로드하세요</p>
            </div>
            """, unsafe_allow_html=True)

    # ============ RIGHT PANEL: Validation & Export ============
    with col_actions:
        # Object List Section
        st.markdown("#### 🎯 탐지된 객체")

        if objects:
            with st.container(height=200):
                for idx, obj in enumerate(objects):
                    category = obj.get("category_name", "unknown")
                    confidence = obj.get("confidence", 0) or 0
                    is_validated = obj.get("is_validated", False)
                    is_selected = idx == st.session_state.selected_object_idx

                    icon = "✅" if is_validated else "⬜"
                    btn_label = f"{icon} {category} ({confidence:.0%})"

                    if st.button(btn_label, key=f"obj_{idx}", width="stretch",
                                type="primary" if is_selected else "secondary"):
                        st.session_state.selected_object_idx = idx
                        st.rerun()
        else:
            st.caption("탐지된 객체 없음")

        st.divider()

        # Validation Section
        st.markdown("#### ✓ 검증")

        selected_obj = objects[st.session_state.selected_object_idx] if (
            st.session_state.selected_object_idx is not None and
            objects and st.session_state.selected_object_idx < len(objects)
        ) else None

        if selected_obj:
            obj_id = selected_obj.get("object_id", "")
            is_validated = selected_obj.get("is_validated", False)

            st.caption(f"{selected_obj.get('category_name')} • {obj_id[:8]}...")

            if is_validated:
                st.success("✓ 검증됨")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✓ 승인", key="approve_btn", width="stretch", type="primary"):
                        validate_object(obj_id, "reviewer", 0.9)
                        st.cache_data.clear()
                        st.rerun()
                with c2:
                    if st.button("✗ 삭제", key="delete_btn", width="stretch"):
                        reject_object(obj_id)
                        st.session_state.selected_object_idx = None
                        st.cache_data.clear()
                        st.rerun()

            # Bulk validation
            unvalidated = [o for o in objects if not o.get("is_validated")]
            if unvalidated:
                if st.button(f"⚡ 모두 검증 ({len(unvalidated)}개)", key="validate_all", width="stretch"):
                    for obj in unvalidated:
                        validate_object(obj.get("object_id", ""), "reviewer", 0.9)
                    st.cache_data.clear()
                    st.rerun()
        else:
            st.caption("객체를 선택하세요")

        st.divider()

        # Quick Export Section
        st.markdown("#### 📦 내보내기")

        validated_count = stats.get("validated_objects", 0)
        st.metric("검증된 객체", f"{validated_count}개")

        export_format = st.selectbox("형식", ["YOLO", "COCO"], key="quick_export_fmt", label_visibility="collapsed")

        if st.button("🚀 내보내기", key="quick_export_btn", width="stretch", type="primary",
                    disabled=validated_count == 0):
            st.success(f"✓ {export_format} 형식으로 내보내기 준비됨")


def export_view():
    """Dataset export view."""
    st.markdown("## 📦 데이터셋 내보내기")
    st.caption("검증된 데이터를 YOLO 또는 COCO 형식으로 내보냅니다")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 설정")

        dataset_name = st.text_input("데이터셋 이름", value="my_dataset")
        export_format = st.selectbox("형식", ["YOLO", "COCO"])

        st.markdown("#### 분할 비율")
        train_ratio = st.slider("Train", 0.0, 1.0, 0.8, 0.05)
        val_ratio = st.slider("Validation", 0.0, 1.0, 0.1, 0.05)
        test_ratio = 1.0 - train_ratio - val_ratio
        st.metric("Test", f"{test_ratio:.0%}")

    with col2:
        st.markdown("### 필터")

        only_validated = st.checkbox("검증된 객체만", value=True)
        min_confidence = st.slider("최소 신뢰도", 0.0, 1.0, 0.5, 0.05)

        categories = get_categories()
        cat_names = [c.get("name", "") for c in categories if c.get("name")]
        selected_cats = st.multiselect("카테고리", cat_names, default=cat_names)

    st.markdown("---")

    if st.button("🚀 내보내기 시작", width="stretch", type="primary"):
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

def render_header():
    """Render compact header with branding."""
    col1, col2 = st.columns([8, 2])
    with col1:
        st.markdown("### ⚡ Agentic**Label**")
        st.caption("AI Vision Labeling Studio")
    with col2:
        st.success("● 연결됨")


def main():
    """Main application - single page workflow."""

    # Check connection
    try:
        import httpx
        r = httpx.get(f"{REGISTRY_URL}/stats", timeout=5)
        stats = r.json().get('data', {})
        connected = True
    except Exception as e:
        connected = False
        stats = {}

    if not connected:
        st.error(f"❌ Registry 연결 실패: {REGISTRY_URL}")
        st.info("Object Registry 서비스가 실행 중인지 확인하세요.")
        return

    # Compact Header
    render_header()

    # Single Page Unified Workspace
    unified_workspace_view()


if __name__ == "__main__":
    main()
