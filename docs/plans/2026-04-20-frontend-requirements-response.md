# AgenticLabeling Frontend Requirements Response

This document answers the UI/UX discovery questions using the current MVP implementation and the planned next-step workstation direction.

## 1. Product Definition

### Primary users

The current product should be designed first for:

- ML engineers building detection/segmentation datasets
- Data labelers or internal annotation operators reviewing model output
- Researchers validating model-assisted labeling quality

It is not currently aimed at general end users.

### Core usage scenarios

1. Morning triage
- Open the review workspace
- Check recent sources, object counts, and pending review workload
- Resume from the next unvalidated source/object

2. Repeated labeling workflow
- Upload an image
- Run Florence-2 + SAM2
- Inspect bbox and mask overlays
- Approve or delete objects
- Move to the next pending source

3. Dataset curation and export
- Filter validated outputs
- Confirm export format and dataset name
- Generate a YOLO or COCO package for downstream training

### Reference product tone

The closest references are:

- Label Studio for review and annotation workflow
- Roboflow for dataset-oriented computer vision operations
- Weights & Biases only for the "professional tooling" tone, not for workflow shape

The product should not feel like LangSmith or a prompt-debug console. It is a computer vision review workstation.

### Product format

Target format:

- Single-page web application with workstation-style panels

Not target format:

- Dashboard-first analytics product
- CLI-first interface with a thin admin UI

## 2. Pipeline Structure

### Pipeline stages

The current MVP pipeline is:

1. Image upload
- Input: image file, project_id, class prompt list
- Output: stored source asset

2. Detection
- Backend: Florence-2 grounding
- Input: image bytes, requested classes
- Output: boxes, labels, scores

3. Segmentation
- Backend: SAM2
- Input: image bytes, raw detection boxes
- Output: per-object masks

4. Registry write
- Input: source metadata, object metadata, optional mask paths
- Output: source row, object rows, category rows

5. Human review
- Input: source + objects + overlays
- Output: approved or deleted objects

6. Dataset export
- Input: dataset name, export format, validation filter
- Output: zipped YOLO or COCO dataset

### Input types

Current MVP input:

- Image

Deferred but structurally relevant:

- Video
- Screenshot
- Multimodal image + prompt tasking

Not in current MVP:

- PDF

### Output types

Current MVP outputs:

- Bounding boxes
- Segmentation masks
- Reviewable object rows
- JSON API responses
- YOLO / COCO export packages

Not current MVP outputs:

- Captions
- Long-form text explanations
- LLM-style structured text responses

### Model topology

Current MVP:

- Florence-2 for detection
- SAM2 for segmentation

Planned UX implication:

- The first design version can assume a single active pipeline
- Future versions should leave space for side-by-side model comparison

### Processing mode

Current MVP:

- Request/response style processing for single images

Near-term requirement:

- Support both direct single-image runs and asynchronous batch queues

### Processing latency

Current practical expectation:

- Single image inference is in the seconds range, not sub-second

UX implication:

- The UI should use explicit in-progress state, not instant-feedback assumptions
- Progress and backend status indicators matter

## 3. Major Screens

### Keep

1. Dashboard home
- Purpose: recent activity, runtime health, quick entry points

2. Upload / input screen
- Drag-and-drop upload
- Project ID
- Class prompt input
- Run pipeline action

3. Inference result viewer
- Original image
- BBox overlay
- Segmentation overlay
- Object list and status

4. Evaluation / labeling view
- Approve/delete flow
- Object inspector
- Overlay controls

5. Batch work list
- Needed next
- Show pending / validated / failed work

6. Settings
- Model/runtime settings
- Path and backend visibility

### Add next

- Object inspector panel
- Overlay toggles and opacity controls
- Batch approve/delete
- Source-level filtering

### Defer

- Langflow-style pipeline graph
- Full model-vs-model comparison view
- Token or prompt-debug console

These are not wrong, but they are not the next value-bearing screens for this product.

## 4. Data Samples

### Sample domain images

Current available local reference:

- [data/images/test_street.jpg](/home/coffin/dev/AgenticLabeling/data/images/test_street.jpg)

Recommended next sample set for design quality:

- Street / traffic scene
- Indoor warehouse or manufacturing scene
- Low-light or cluttered scene
- Edge-case image with small or overlapping objects

### Sample request

Example request:

```http
POST /api/pipeline/auto-label
```

Form fields:

- `image`
- `project_id=default-project`
- `classes=person,car,dog`

### Sample response

```json
{
  "success": true,
  "source_id": "src_1234567890ab",
  "object_ids": ["obj_1234567890ab"],
  "detections": 1,
  "file_name": "sample.jpg",
  "segmentation_backend": "sam2"
}
```

### Sample review object schema

Actual object rows are sourced from the registry and include fields such as:

- `object_id`
- `source_id`
- `category_name`
- `bbox_x`
- `bbox_y`
- `bbox_w`
- `bbox_h`
- `confidence`
- `mask_path`
- `is_validated`
- `validated_by`
- `quality_score`
- `created_at`
- `updated_at`

Reference implementation:

- [mvp_app/registry.py](/home/coffin/dev/AgenticLabeling/mvp_app/registry.py)

## 5. Brand And Visual Direction

Recommended tone:

- Technical
- Professional
- Calm
- High signal

Visual direction:

- Light neutral background
- Dense workstation layout
- Strong state colors only where workflow status matters
- Viewer-first composition

Reference tone:

- Closer to Label Studio and Roboflow than to consumer SaaS dashboards

## 6. Project Scope

### MVP screen count

Recommended MVP structure:

1. Home / ingest
2. Review workspace
3. Export / curation summary
4. Settings

### Design output expectation

Recommended deliverables:

- First: wireframes for 3 to 5 layout directions
- Second: one selected high-fidelity UI direction
- Third: implementation-ready page structure and component map

### Immediate design decision

The design should be driven primarily by:

1. Pipeline stage sequence
2. Image + bbox + mask output structure
3. Review operator workflow

Those three elements define most of the page architecture.
