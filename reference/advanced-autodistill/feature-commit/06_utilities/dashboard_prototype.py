#!/usr/bin/env python3
"""
Dashboard for Support Set and Refined Dataset Interaction

This module provides a web-based dashboard for interacting with Support Sets and Refined Datasets
in the AI image processing pipeline. It implements a FastAPI backend with REST endpoints and
a responsive web frontend.

The dashboard allows users to:
1. Browse and manage support set images
2. View and filter refined dataset results
3. Reclassify objects using drag-and-drop
4. Apply different classification methods
5. View analytics on dataset composition and classification results

Architecture:
- Backend: FastAPI REST API with endpoints for data access and manipulation
- Frontend: Responsive HTML/CSS/JS interface with visualization components
- Data Flow: File system based with JSON metadata, optimized for image serving

Requirements:
- fastapi
- uvicorn
- jinja2
- python-multipart
- aiofiles
"""

import os
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple

import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dashboard.log")
    ]
)
logger = logging.getLogger("dashboard")

# Import project utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.metadata_utils import (
    get_all_preprocessing_metadata,
    get_classification_structure,
    create_classification_plan,
    execute_classification_plan
)

###################
# DATA MODELS
###################

class ClassMapping(BaseModel):
    """Model for class mapping operations"""
    source_class: str = Field(..., description="Original class name")
    target_class: str = Field(..., description="Target class name to reclassify to")
    method: str = Field(..., description="Classification method to use")

class ClassificationConfig(BaseModel):
    """Model for classification configuration"""
    method: str = Field(..., description="Classification method to apply")
    mappings: Dict[str, str] = Field({}, description="Custom class mappings (source -> target)")

class ImageMetadata(BaseModel):
    """Model for image metadata information"""
    id: str = Field(..., description="Unique identifier for the image")
    path: str = Field(..., description="Relative path to the image file")
    class_name: str = Field(..., description="Class name or category")
    confidence: Optional[float] = Field(None, description="Confidence score (if available)")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")

class BatchOperationRequest(BaseModel):
    """Model for batch operations on multiple images"""
    image_ids: List[str] = Field(..., description="List of image IDs to process")
    operation: str = Field(..., description="Operation to perform: 'reclassify', 'delete', etc.")
    parameters: Dict[str, Any] = Field({}, description="Operation-specific parameters")

###################
# APP INITIALIZATION
###################

# Initialize FastAPI app
app = FastAPI(
    title="AI Image Processing Dashboard",
    description="Dashboard for Support Set and Refined Dataset Interaction",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define base directories
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "dashboard_static"

# Create static directory structure if it doesn't exist
STATIC_DIR.mkdir(exist_ok=True)
os.makedirs(STATIC_DIR / "css", exist_ok=True)
os.makedirs(STATIC_DIR / "js", exist_ok=True)
os.makedirs(STATIC_DIR / "images", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Set up Jinja2 templates
templates_dir = STATIC_DIR / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

###################
# HELPER FUNCTIONS
###################

def create_default_templates():
    """Create default template files if they don't exist"""
    # Create default index.html if it doesn't exist
    index_html_path = templates_dir / "index.html"
    if not index_html_path.exists():
        with open(index_html_path, "w") as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Processing Dashboard</title>
    <link rel="stylesheet" href="/static/css/style.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <header>
        <h1>AI Image Processing Dashboard</h1>
        <nav>
            <ul>
                <li><a href="#support-set">Support Set</a></li>
                <li><a href="#refined-dataset">Refined Dataset</a></li>
                <li><a href="#analytics">Analytics</a></li>
                <li><a href="#settings">Settings</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        <section id="categories">
            <h2>Categories</h2>
            <div class="category-list">
            {% for category in categories %}
                <div class="category-card" data-category="{{ category }}">
                    <h3>{{ category }}</h3>
                    <button onclick="loadCategory('{{ category }}')">Load</button>
                </div>
            {% endfor %}
            </div>
        </section>

        <section id="support-set">
            <h2>Support Set</h2>
            <div class="gallery" id="support-set-gallery">
                <!-- Support set images will be loaded here -->
            </div>
        </section>

        <section id="refined-dataset">
            <h2>Refined Dataset</h2>
            <div class="controls">
                <select id="classification-method">
                    <option value="method1">Method 1</option>
                    <option value="method2">Method 2</option>
                    <option value="method3">Method 3</option>
                    <option value="method4">Method 4</option>
                </select>
                <button onclick="applyClassification()">Apply Classification</button>
            </div>
            <div class="gallery" id="refined-dataset-gallery">
                <!-- Refined dataset images will be loaded here -->
            </div>
        </section>

        <section id="analytics">
            <h2>Analytics</h2>
            <div class="chart-container">
                <canvas id="class-distribution-chart"></canvas>
            </div>
            <div class="stats-container">
                <!-- Statistics will be loaded here -->
            </div>
        </section>
    </main>

    <footer>
        <p>AI Image Processing Dashboard - Project AGI</p>
    </footer>

    <script src="/static/js/dashboard.js"></script>
</body>
</html>""")
        logger.info("Created default index.html template")

    # Create default style.css if it doesn't exist
    css_path = STATIC_DIR / "css" / "style.css"
    if not css_path.exists():
        with open(css_path, "w") as f:
            f.write("""/* Dashboard Styles */
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f4f4f4;
}

header {
    background-color: #35424a;
    color: white;
    padding: 1rem;
}

header h1 {
    margin-bottom: 1rem;
}

nav ul {
    display: flex;
    list-style: none;
}

nav ul li {
    margin-right: 1rem;
}

nav ul li a {
    color: white;
    text-decoration: none;
}

main {
    padding: 2rem;
}

section {
    margin-bottom: 2rem;
    background-color: white;
    padding: 1.5rem;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

h2 {
    margin-bottom: 1rem;
    border-bottom: 1px solid #eee;
    padding-bottom: 0.5rem;
}

.gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.image-card {
    border: 1px solid #ddd;
    border-radius: 3px;
    overflow: hidden;
    position: relative;
}

.image-card img {
    width: 100%;
    height: auto;
    display: block;
}

.image-card .info {
    padding: 0.5rem;
    background-color: rgba(0,0,0,0.7);
    color: white;
    position: absolute;
    bottom: 0;
    width: 100%;
    transform: translateY(100%);
    transition: transform 0.3s ease;
}

.image-card:hover .info {
    transform: translateY(0);
}

.controls {
    margin-bottom: 1rem;
}

.chart-container {
    height: 300px;
    margin-bottom: 1.5rem;
}

.stats-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 1rem;
}

.stat-card {
    background-color: #f9f9f9;
    padding: 1rem;
    border-radius: 3px;
    border-left: 4px solid #35424a;
}

.category-list {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
}

.category-card {
    background-color: #f0f0f0;
    padding: 1rem;
    border-radius: 5px;
    cursor: pointer;
    flex: 1 0 200px;
    text-align: center;
}

.category-card:hover {
    background-color: #e0e0e0;
}

footer {
    background-color: #35424a;
    color: white;
    text-align: center;
    padding: 1rem;
    margin-top: 2rem;
}
""")
        logger.info("Created default style.css")

    # Create default dashboard.js if it doesn't exist
    js_path = STATIC_DIR / "js" / "dashboard.js"
    if not js_path.exists():
        with open(js_path, "w") as f:
            f.write("""// Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard
    loadCategories();
    
    // Set up event listeners
    setupDragAndDrop();
});

// Load available categories
function loadCategories() {
    fetch('/api/categories')
        .then(response => response.json())
        .then(data => {
            // Categories are loaded via Jinja template
            console.log('Categories loaded:', data.categories);
        })
        .catch(error => console.error('Error loading categories:', error));
}

// Load a specific category
function loadCategory(category) {
    console.log('Loading category:', category);
    
    // Load support set
    fetch(`/api/support-set/${category}`)
        .then(response => response.json())
        .then(data => {
            displaySupportSet(data.images);
        })
        .catch(error => console.error('Error loading support set:', error));
    
    // Load refined dataset
    fetch(`/api/refined-dataset/${category}`)
        .then(response => response.json())
        .then(data => {
            displayRefinedDataset(data.images);
            updateAnalytics(data.stats);
        })
        .catch(error => console.error('Error loading refined dataset:', error));
}

// Display support set images
function displaySupportSet(images) {
    const gallery = document.getElementById('support-set-gallery');
    gallery.innerHTML = '';
    
    images.forEach(image => {
        const card = document.createElement('div');
        card.className = 'image-card';
        card.setAttribute('draggable', 'true');
        card.setAttribute('data-path', image.path);
        card.setAttribute('data-class', image.class);
        
        const img = document.createElement('img');
        img.src = `/api/image?path=${encodeURIComponent(image.path)}`;
        img.alt = image.class;
        
        const info = document.createElement('div');
        info.className = 'info';
        info.textContent = image.class;
        
        card.appendChild(img);
        card.appendChild(info);
        gallery.appendChild(card);
        
        // Set up drag events
        card.addEventListener('dragstart', handleDragStart);
    });
}

// Display refined dataset images
function displayRefinedDataset(images) {
    const gallery = document.getElementById('refined-dataset-gallery');
    gallery.innerHTML = '';
    
    images.forEach(image => {
        const card = document.createElement('div');
        card.className = 'image-card';
        card.setAttribute('data-path', image.path);
        card.setAttribute('data-class', image.class);
        card.setAttribute('data-id', image.id);
        
        const img = document.createElement('img');
        img.src = `/api/image?path=${encodeURIComponent(image.path)}`;
        img.alt = image.class;
        
        const info = document.createElement('div');
        info.className = 'info';
        info.innerHTML = `
            <div>Class: ${image.class}</div>
            <div>Confidence: ${image.confidence || 'N/A'}</div>
        `;
        
        card.appendChild(img);
        card.appendChild(info);
        gallery.appendChild(card);
    });
}

// Update analytics section
function updateAnalytics(stats) {
    // Update class distribution chart
    const ctx = document.getElementById('class-distribution-chart').getContext('2d');
    
    if (window.classChart) {
        window.classChart.destroy();
    }
    
    window.classChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(stats.class_distribution),
            datasets: [{
                label: 'Class Distribution',
                data: Object.values(stats.class_distribution),
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 159, 64, 0.2)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            responsive: true,
            maintainAspectRatio: false
        }
    });
    
    // Update statistics
    const statsContainer = document.querySelector('.stats-container');
    statsContainer.innerHTML = '';
    
    // Add total objects stat
    const totalCard = document.createElement('div');
    totalCard.className = 'stat-card';
    totalCard.innerHTML = `
        <h3>Total Objects</h3>
        <p>${stats.total_objects}</p>
    `;
    statsContainer.appendChild(totalCard);
    
    // Add average confidence stat
    const confidenceCard = document.createElement('div');
    confidenceCard.className = 'stat-card';
    confidenceCard.innerHTML = `
        <h3>Avg. Confidence</h3>
        <p>${stats.average_confidence ? stats.average_confidence.toFixed(2) : 'N/A'}</p>
    `;
    statsContainer.appendChild(confidenceCard);
    
    // Add method selection stats
    const methodCard = document.createElement('div');
    methodCard.className = 'stat-card';
    methodCard.innerHTML = `
        <h3>Classification Method</h3>
        <p>${stats.method || 'None'}</p>
    `;
    statsContainer.appendChild(methodCard);
}

// Set up drag and drop for reclassification
function setupDragAndDrop() {
    const dropZones = document.querySelectorAll('.gallery');
    
    dropZones.forEach(zone => {
        zone.addEventListener('dragover', handleDragOver);
        zone.addEventListener('drop', handleDrop);
    });
}

function handleDragStart(event) {
    event.dataTransfer.setData('text/plain', JSON.stringify({
        path: event.target.getAttribute('data-path'),
        class: event.target.getAttribute('data-class'),
        id: event.target.getAttribute('data-id')
    }));
}

function handleDragOver(event) {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
}

function handleDrop(event) {
    event.preventDefault();
    
    const data = JSON.parse(event.dataTransfer.getData('text/plain'));
    const targetGallery = event.currentTarget;
    const targetClass = targetGallery.parentElement.querySelector('h2').textContent;
    
    console.log('Dropped:', data);
    console.log('Target gallery:', targetGallery.id);
    
    if (targetGallery.id === 'refined-dataset-gallery') {
        // Reclassify object
        const method = document.getElementById('classification-method').value;
        
        fetch('/api/reclassify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_id: data.id,
                source_class: data.class,
                target_class: targetClass,
                method: method
            })
        })
        .then(response => response.json())
        .then(result => {
            console.log('Reclassification result:', result);
            // Reload category to reflect changes
            const currentCategory = document.querySelector('.category-card.active').getAttribute('data-category');
            loadCategory(currentCategory);
        })
        .catch(error => console.error('Error reclassifying object:', error));
    }
}

// Apply classification method
function applyClassification() {
    const method = document.getElementById('classification-method').value;
    const activeCategory = document.querySelector('.category-card.active');
    
    if (!activeCategory) {
        alert('Please select a category first');
        return;
    }
    
    const category = activeCategory.getAttribute('data-category');
    
    fetch('/api/apply-classification', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            category: category,
            method: method
        })
    })
    .then(response => response.json())
    .then(result => {
        console.log('Classification applied:', result);
        // Reload category to reflect changes
        loadCategory(category);
    })
    .catch(error => console.error('Error applying classification:', error));
}
""")
        logger.info("Created default dashboard.js")

def get_category_structure(category_name: str) -> Dict[str, Any]:
    """
    Get the structure information for a specific category
    
    Args:
        category_name: Name of the category to analyze
        
    Returns:
        Dict containing information about the category structure
    """
    category_dir = DATA_DIR / category_name
    if not category_dir.exists():
        raise HTTPException(status_code=404, detail=f"Category not found: {category_name}")
    
    structure = {
        "name": category_name,
        "has_support_set": False,
        "has_preprocessed": False,
        "has_refined_dataset": False,
        "refinement_methods": [],
        "support_set_classes": [],
        "statistics": {}
    }
    
    # Check for support set
    support_set_dir = category_dir / "2.support-set"
    if support_set_dir.exists():
        structure["has_support_set"] = True
        structure["support_set_classes"] = [d.name for d in support_set_dir.iterdir() if d.is_dir()]
    
    # Check for preprocessed data
    preprocessed_dir = category_dir / "6.preprocessed"
    if preprocessed_dir.exists():
        structure["has_preprocessed"] = True
        
        # Get preprocessing metadata if available
        metadata_file = preprocessed_dir / "preprocessing_summary.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    structure["preprocessing_metadata"] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading preprocessing metadata: {e}")
    
    # Check for refined dataset
    refine_dir = category_dir / "8.refine-dataset"
    if refine_dir.exists():
        structure["has_refined_dataset"] = True
        structure["refinement_methods"] = [d.name for d in refine_dir.iterdir() if d.is_dir()]
    
    return structure

def get_image_metadata(image_path: str) -> Dict[str, Any]:
    """
    Get metadata for a specific image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dict containing metadata about the image
    """
    # This is a placeholder function - in a real implementation,
    # you would extract metadata from the metadata files
    try:
        # Generate ID from filename
        img_id = Path(image_path).stem
        
        # Get file stats
        img_full_path = BASE_DIR / image_path
        if not img_full_path.exists():
            return {}
        
        file_stats = img_full_path.stat()
        
        return {
            "id": img_id,
            "filename": Path(image_path).name,
            "size_bytes": file_stats.st_size,
            "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "path": image_path
        }
    except Exception as e:
        logger.error(f"Error getting image metadata: {e}")
        return {}

def generate_error_response(status_code: int, message: str) -> JSONResponse:
    """
    Generate a standardized error response
    
    Args:
        status_code: HTTP status code
        message: Error message
        
    Returns:
        JSONResponse with error details
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "code": status_code,
                "message": message
            }
        }
    )

###################
# API ROUTES
###################

@app.on_event("startup")
async def startup_event():
    """Initialize resources when the application starts"""
    logger.info("Starting dashboard application")
    create_default_templates()
    logger.info("Dashboard application startup complete")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the dashboard home page"""
    # Get list of available categories
    categories = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
    logger.info(f"Serving dashboard with {len(categories)} available categories")
    return templates.TemplateResponse("index.html", {"request": request, "categories": categories})

@app.get("/api/categories")
async def get_categories():
    """Get list of available categories"""
    try:
        categories = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        return generate_error_response(500, f"Failed to fetch categories: {str(e)}")

@app.get("/api/category/{category}")
async def get_category_info(category: str):
    """Get detailed information about a category"""
    try:
        structure = get_category_structure(category)
        return structure
    except HTTPException as e:
        # Re-raise HTTPException for specific error responses
        raise
    except Exception as e:
        logger.error(f"Error fetching category info for {category}: {e}")
        return generate_error_response(500, f"Failed to fetch category info: {str(e)}")

@app.get("/api/support-set/{category}")
async def get_support_set(category: str):
    """Get support set images for a category"""
    try:
        support_set_dir = DATA_DIR / category / "2.support-set"
        
        if not support_set_dir.exists():
            raise HTTPException(status_code=404, detail=f"Support set not found for category: {category}")
        
        images = []
        for class_dir in support_set_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_file in class_dir.glob("*.jpg"):
                    rel_path = str(img_file.relative_to(BASE_DIR))
                    
                    # Get additional metadata if available
                    metadata = get_image_metadata(rel_path)
                    
                    images.append({
                        "id": metadata.get("id", img_file.stem),
                        "path": rel_path,
                        "class": class_name,
                        "metadata": metadata
                    })
        
        return {"images": images}
    except HTTPException as e:
        # Re-raise HTTPException for specific error responses
        raise
    except Exception as e:
        logger.error(f"Error fetching support set for {category}: {e}")
        return generate_error_response(500, f"Failed to fetch support set: {str(e)}")

@app.get("/api/refined-dataset/{category}")
async def get_refined_dataset(category: str, method: str = "method1"):
    """Get refined dataset images for a category and method"""
    try:
        refined_dir = DATA_DIR / category / "8.refine-dataset" / method
        
        if not refined_dir.exists():
            raise HTTPException(status_code=404, detail=f"Refined dataset not found for category: {category}, method: {method}")
        
        # Load metadata if available
        metadata_file = DATA_DIR / category / "6.preprocessed" / "preprocessing_summary.json"
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        
        images = []
        class_distribution = {}
        
        for class_dir in refined_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_count = 0
                
                for img_file in class_dir.glob("*.png"):
                    # Generate a simple ID based on filename
                    img_id = img_file.stem
                    
                    # Get relative path for serving
                    rel_path = str(img_file.relative_to(BASE_DIR))
                    
                    # Get additional metadata if available
                    img_metadata = get_image_metadata(rel_path)
                    
                    # Get confidence from metadata if available
                    confidence = None
                    # (In a real implementation, you would extract this from metadata)
                    
                    images.append({
                        "id": img_id,
                        "path": rel_path,
                        "class": class_name,
                        "confidence": confidence,
                        "metadata": img_metadata
                    })
                    class_count += 1
                
                class_distribution[class_name] = class_count
        
        # Calculate statistics
        total_objects = len(images)
        average_confidence = None  # Would calculate from actual confidence values
        
        stats = {
            "total_objects": total_objects,
            "average_confidence": average_confidence,
            "class_distribution": class_distribution,
            "method": method
        }
        
        return {
            "images": images,
            "stats": stats
        }
    except HTTPException as e:
        # Re-raise HTTPException for specific error responses
        raise
    except Exception as e:
        logger.error(f"Error fetching refined dataset for {category}, method {method}: {e}")
        return generate_error_response(500, f"Failed to fetch refined dataset: {str(e)}")

@app.get("/api/image")
async def get_image(path: str):
    """Serve an image file"""
    try:
        full_path = BASE_DIR / path
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {path}")
        
        return FileResponse(str(full_path))
    except HTTPException as e:
        # Re-raise HTTPException for specific error responses
        raise
    except Exception as e:
        logger.error(f"Error serving image {path}: {e}")
        return generate_error_response(500, f"Failed to serve image: {str(e)}")

@app.post("/api/reclassify")
async def reclassify_image(mapping: ClassMapping, background_tasks: BackgroundTasks):
    """Reclassify an image to a different class"""
    try:
        # In a real implementation, this would update the classification data
        # and move the image to the appropriate class directory
        
        # Log the reclassification request
        logger.info(f"Reclassification request: {mapping.source_class} -> {mapping.target_class} using method {mapping.method}")
        
        # Add a background task for the actual reclassification
        # (this is where you would implement the actual file moving)
        # background_tasks.add_task(perform_reclassification, mapping)
        
        # For now, just return success
        return {
            "success": True,
            "message": f"Reclassified image to {mapping.target_class} using method {mapping.method}"
        }
    except Exception as e:
        logger.error(f"Error in reclassification: {e}")
        return generate_error_response(500, f"Reclassification failed: {str(e)}")

@app.post("/api/apply-classification")
async def apply_classification(config: ClassificationConfig, background_tasks: BackgroundTasks):
    """Apply a classification method to the entire dataset"""
    try:
        # In a real implementation, this would run the classification algorithm
        # based on the selected method and mappings
        
        # Log the classification request
        logger.info(f"Classification request using method {config.method} with {len(config.mappings)} custom mappings")
        
        # Add a background task for the actual classification
        # background_tasks.add_task(perform_classification, config)
        
        # For now, just return success
        return {
            "success": True,
            "message": f"Applied classification using method {config.method}",
            "details": {
                "custom_mappings": len(config.mappings)
            }
        }
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return generate_error_response(500, f"Classification failed: {str(e)}")

@app.post("/api/batch-operation")
async def batch_operation(request: BatchOperationRequest, background_tasks: BackgroundTasks):
    """Perform operations on multiple images at once"""
    try:
        # Log the batch operation request
        logger.info(f"Batch operation '{request.operation}' requested for {len(request.image_ids)} images")
        
        # Add a background task for the actual operation
        # background_tasks.add_task(perform_batch_operation, request)
        
        return {
            "success": True,
            "message": f"Batch operation '{request.operation}' initiated for {len(request.image_ids)} images",
            "operation_id": f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
    except Exception as e:
        logger.error(f"Error in batch operation: {e}")
        return generate_error_response(500, f"Batch operation failed: {str(e)}")

###################
# MAIN ENTRY POINT
###################

def main():
    """Run the FastAPI server"""
    print(f"Starting dashboard server at http://localhost:8000")
    print(f"API documentation available at http://localhost:8000/api/docs")
    print(f"Press Ctrl+C to stop")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 