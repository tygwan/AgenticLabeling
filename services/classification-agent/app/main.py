"""Classification Agent Service - Few-shot classification with CLIP/DINOv2/ResNet."""
import io
import json
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .classifier import FeatureExtractor, CosineSimilarityClassifier
from .schemas import ClassificationResponse, SupportSetResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage classifier lifecycle."""
    app.state.classifiers = {}  # model_name -> classifier
    yield
    # Cleanup
    for classifier in app.state.classifiers.values():
        classifier.unload()


app = FastAPI(
    title="Classification Agent",
    description="Few-shot image classification service using CLIP/DINOv2/ResNet",
    version="0.1.0",
    lifespan=lifespan,
)


def get_classifier(model_name: str = "dino") -> CosineSimilarityClassifier:
    """Get or create classifier for the specified model."""
    if model_name not in app.state.classifiers:
        app.state.classifiers[model_name] = CosineSimilarityClassifier(model_name=model_name)
    return app.state.classifiers[model_name]


@app.get("/health")
async def health():
    """Health check endpoint."""
    loaded_models = list(app.state.classifiers.keys())
    return {"status": "healthy", "loaded_models": loaded_models}


@app.post("/support_set/load", response_model=SupportSetResponse)
async def load_support_set(
    images: List[UploadFile] = File(...),
    class_labels: str = Form(...),
    model: str = Form("dino"),
):
    """Load support set images for few-shot classification.

    Args:
        images: List of support set images
        class_labels: JSON string mapping image index to class label
        model: Feature extraction model (clip, dino, resnet)
    """
    try:
        labels = json.loads(class_labels)
        classifier = get_classifier(model)

        # Process images by class
        support_data = {}
        for idx, img_file in enumerate(images):
            class_name = labels.get(str(idx), labels.get(idx, f"class_{idx}"))
            if class_name not in support_data:
                support_data[class_name] = []

            image_bytes = await img_file.read()
            support_data[class_name].append(image_bytes)

        # Extract features for support set
        classifier.load_support_set_from_bytes(support_data)

        return SupportSetResponse(
            success=True,
            message=f"Loaded {len(images)} images for {len(support_data)} classes",
            classes=list(support_data.keys()),
            images_per_class={k: len(v) for k, v in support_data.items()},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/classify", response_model=ClassificationResponse)
async def classify(
    image: UploadFile = File(...),
    model: str = Form("dino"),
    threshold: float = Form(0.75),
    top_k: int = Form(3),
):
    """Classify an image using few-shot learning.

    Args:
        image: Image file to classify
        model: Feature extraction model (clip, dino, resnet)
        threshold: Minimum similarity threshold
        top_k: Number of top predictions to return
    """
    try:
        classifier = get_classifier(model)

        if not classifier.has_support_set():
            raise HTTPException(
                status_code=400,
                detail="No support set loaded. Call /support_set/load first.",
            )

        image_bytes = await image.read()
        result = classifier.classify(image_bytes, threshold=threshold, top_k=top_k)

        return ClassificationResponse(
            success=True,
            data=result,
            message="Classification completed",
        )
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/classify_batch")
async def classify_batch(
    images: List[UploadFile] = File(...),
    model: str = Form("dino"),
    threshold: float = Form(0.75),
):
    """Classify multiple images in batch."""
    try:
        classifier = get_classifier(model)

        if not classifier.has_support_set():
            raise HTTPException(
                status_code=400,
                detail="No support set loaded. Call /support_set/load first.",
            )

        results = []
        for img_file in images:
            image_bytes = await img_file.read()
            result = classifier.classify(image_bytes, threshold=threshold)
            result["filename"] = img_file.filename
            results.append(result)

        return {
            "success": True,
            "results": results,
            "total": len(results),
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/unload")
async def unload_model(model: str = Form(None)):
    """Unload model(s) to free GPU memory."""
    if model:
        if model in app.state.classifiers:
            app.state.classifiers[model].unload()
            del app.state.classifiers[model]
            return {"status": "unloaded", "model": model}
        return {"status": "not_found", "model": model}
    else:
        for clf in app.state.classifiers.values():
            clf.unload()
        app.state.classifiers.clear()
        return {"status": "all_unloaded"}
