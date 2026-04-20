#!/usr/bin/env python3
"""Test the full auto-labeling pipeline."""
import base64
import json
import httpx
import asyncio
from pathlib import Path

# Service URLs
DETECTION_URL = "http://localhost:8001"
SEGMENTATION_URL = "http://localhost:8002"
REGISTRY_URL = "http://localhost:8010"

async def run_pipeline(image_path: str, classes: list[str], project_id: str = "test_project"):
    """Run the full detection -> segmentation -> registry pipeline."""

    async with httpx.AsyncClient(timeout=120.0) as client:
        print(f"\n{'='*60}")
        print(f"Testing Pipeline with: {image_path}")
        print(f"Classes: {', '.join(classes)}")
        print(f"{'='*60}\n")

        # Step 1: Detection
        print("[1/4] Running object detection...")
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        detect_resp = await client.post(
            f"{DETECTION_URL}/detect",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
            data={"classes": ",".join(classes), "confidence": "0.3"}
        )
        detect_result = detect_resp.json()

        if not detect_result.get("success"):
            print(f"Detection failed: {detect_result}")
            return

        boxes = detect_result["data"]["boxes"]
        labels = detect_result["data"]["labels"]
        confidence = detect_result["data"]["confidence"]
        image_size = detect_result["data"]["image_size"]

        print(f"   Found {len(boxes)} objects:")
        for i, (box, label, conf) in enumerate(zip(boxes, labels, confidence)):
            print(f"   - {label}: confidence={conf:.2f}, bbox={[int(b) for b in box]}")

        if not boxes:
            print("No objects detected!")
            return

        # Step 2: Segmentation
        print(f"\n[2/4] Running segmentation for {len(boxes)} objects...")
        segment_resp = await client.post(
            f"{SEGMENTATION_URL}/segment",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
            data={"boxes": json.dumps(boxes)}
        )
        segment_result = segment_resp.json()

        masks_b64 = []
        if segment_result.get("success"):
            masks = segment_result["data"]["masks"]
            print(f"   Generated {len(masks)} masks")
            for i, mask in enumerate(masks):
                masks_b64.append(mask.get("mask", ""))
                print(f"   - Mask {i+1}: area={mask.get('area', 'N/A')}")
        else:
            print(f"   Segmentation failed: {segment_result}")

        # Step 3: Register Source
        print(f"\n[3/4] Registering source in Object Registry...")
        source_data = {
            "source_type": "image",
            "file_path": str(image_path),
            "width": image_size.get("width") if isinstance(image_size, dict) else image_size[0],
            "height": image_size.get("height") if isinstance(image_size, dict) else image_size[1],
            "metadata": {"project_id": project_id}
        }

        source_resp = await client.post(
            f"{REGISTRY_URL}/sources",
            json=source_data
        )
        source_result = source_resp.json()

        if not source_result.get("success"):
            print(f"   Source registration failed: {source_result}")
            return

        source_id = source_result["source_id"]
        print(f"   Source registered: {source_id}")

        # Step 4: Register Objects
        print(f"\n[4/4] Registering {len(boxes)} objects...")
        objects_data = []
        for i, (box, label) in enumerate(zip(boxes, labels)):
            obj = {
                "category": label,
                "bbox": box,
                "confidence": confidence[i] if i < len(confidence) else 0.5,
                "detection_model": "florence2",
            }
            if i < len(masks_b64) and masks_b64[i]:
                obj["mask_base64"] = masks_b64[i]
            objects_data.append(obj)

        batch_data = {
            "source_id": source_id,
            "project_id": project_id,
            "objects": objects_data
        }

        batch_resp = await client.post(
            f"{REGISTRY_URL}/objects/batch",
            json=batch_data
        )
        batch_result = batch_resp.json()

        if batch_result.get("success"):
            object_ids = batch_result.get("object_ids", [])
            print(f"   Registered {len(object_ids)} objects:")
            for oid in object_ids:
                print(f"   - {oid}")
        else:
            print(f"   Batch registration failed: {batch_result}")
            return

        # Summary
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"Source ID: {source_id}")
        print(f"Objects created: {len(object_ids)}")
        print(f"Categories: {', '.join(set(labels))}")

        # Get stats
        stats_resp = await client.get(f"{REGISTRY_URL}/stats")
        stats = stats_resp.json()
        print(f"\nRegistry Stats:")
        print(f"  - Total objects: {stats.get('objects', 'N/A')}")
        print(f"  - Total sources: {stats.get('sources', 'N/A')}")
        print(f"  - Categories: {stats.get('categories', 'N/A')}")

        return {
            "source_id": source_id,
            "object_ids": object_ids,
            "labels": labels
        }


if __name__ == "__main__":
    # Test with the sample image
    result = asyncio.run(run_pipeline(
        image_path="data/images/test_street.jpg",
        classes=["person", "car", "mountain", "tree", "sky", "building"],
        project_id="test_project"
    ))
