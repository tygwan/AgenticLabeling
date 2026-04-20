# AI Image Processing Dashboard

## Overview

The AI Image Processing Dashboard provides a web-based interface for interacting with support sets and refined datasets in the AI image processing pipeline. It enables users to:

- Browse and manage support set images
- View and filter refined dataset results
- Reclassify objects using drag-and-drop
- Apply different classification methods
- View analytics on dataset composition and classification results

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages:
  - fastapi
  - uvicorn
  - jinja2
  - python-multipart
  - requests (for testing)

### Installation

1. Clone this repository:
   ```bash
   git clone https://your-repository-url.git
   cd project-agi
   ```

2. Install required packages:
   ```bash
   pip install fastapi uvicorn jinja2 python-multipart aiofiles requests
   ```

3. Set up the dashboard environment:
   ```bash
   ./scripts/run_dashboard.sh setup
   ```

## Usage

### Starting the Dashboard

To start the dashboard in the foreground:
```bash
./scripts/run_dashboard.sh start
```

To start the dashboard in the background:
```bash
./scripts/run_dashboard.sh start-bg
```

Once started, the dashboard will be available at:
- Web interface: http://localhost:8000
- API documentation: http://localhost:8000/api/docs

### Stopping the Dashboard

If running in the background, stop the dashboard with:
```bash
./scripts/run_dashboard.sh stop
```

### Checking Dashboard Status

```bash
./scripts/run_dashboard.sh status
```

### Running Tests

Test the dashboard API functionality:
```bash
./scripts/run_dashboard.sh test
```

Test a specific category:
```bash
./scripts/run_dashboard.sh test test_category
```

## Dashboard Features

### 1. Home Screen

The home screen provides an overview of all available categories and recent activity. It displays:
- Category cards with basic statistics
- Recent activity log
- System status information
- Classification performance metrics

### 2. Support Set Browser

The support set browser allows users to:
- View examples organized by class
- Upload new examples
- Assign examples to classes
- Filter and sort examples
- View example details and metadata

### 3. Refined Dataset Viewer

The refined dataset viewer enables users to:
- Browse classified objects
- Filter by class, confidence, and other attributes
- Sort results
- View result details and metadata
- Perform batch operations on selected objects

### 4. Comparison View

The comparison view provides:
- Side-by-side display of support set examples and refined dataset results
- Visual validation of classification accuracy
- Ability to approve or reclassify results
- Detailed information about selected items

### 5. Analytics Dashboard

The analytics dashboard presents:
- Class distribution charts
- Confidence score visualizations
- Method comparison metrics
- Processing history timeline
- Export options for data and charts

## API Documentation

The API documentation is available at http://localhost:8000/api/docs when the dashboard is running. The main endpoints include:

- `GET /api/categories`: List available categories
- `GET /api/category/{category}`: Get category details
- `GET /api/support-set/{category}`: Get support set images
- `GET /api/refined-dataset/{category}`: Get refined dataset images
- `GET /api/image?path={path}`: Serve an image
- `POST /api/reclassify`: Reclassify an image
- `POST /api/apply-classification`: Apply classification method
- `POST /api/batch-operation`: Perform operations on multiple images

## Architecture

The dashboard implements a layered architecture:

1. **Frontend Layer**:
   - Web-based UI with responsive design
   - Component-based structure for reusability
   - State management for UI interactions

2. **API Layer**:
   - RESTful endpoints for data access and manipulation
   - WebSocket connections for real-time updates
   - Image serving with optimization

3. **Service Layer**:
   - Classification services for organizing images
   - Metadata services for extracting and managing information
   - File operations for organization and persistence

4. **Data Layer**:
   - File system storage for images and metadata
   - JSON-based metadata storage
   - Efficient querying and indexing

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on contributing to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 