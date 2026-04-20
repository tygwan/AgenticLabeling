"""First observation tool: trace Florence-2 → SAM3 pipeline on a single image.

Follows:
- docs/standards/research-observation-protocol.md
- docs/standards/model-inspection-conventions.md

Outputs:
- research/observations/<model>/<YYYY-MM-DD>-<slug>/summary.json
- research/observations/<model>/<YYYY-MM-DD>-<slug>/outputs/raw/*.pt|*.npz (gitignored)
- research/observations/<model>/<YYYY-MM-DD>-<slug>/plan.md (template; user completes)
- research/observations/<model>/<YYYY-MM-DD>-<slug>/report.md (template; user completes)
- research/observations/<model>/<YYYY-MM-DD>-<slug>/analysis.ipynb (template; user populates)

Usage:
    python scripts/inspect_pipeline.py \
        --image data/images/test_street.jpg \
        --classes "car,person,road" \
        --slug initial-pipeline-trace

The script intentionally performs depth-limited observation:
- pipeline / model / output-layer tensor metadata is captured for every run
- forward-hook-based tensor capture is opt-in via --hook-points (deferred to
  follow-up observations since Florence-2 / SAM3 internal module paths differ
  across versions and should be pinned per observation in plan.md)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make repo root importable when running as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.tooling import io as obs_io  # noqa: E402
from research.tooling.stats import describe, describe_many  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--classes",
        required=True,
        help="Comma-separated class names to prompt Florence-2 with",
    )
    parser.add_argument(
        "--slug",
        required=True,
        help="Observation slug, e.g. 'initial-pipeline-trace'",
    )
    parser.add_argument(
        "--fake-models",
        action="store_true",
        help="Use fake models for smoke testing the plumbing",
    )
    parser.add_argument(
        "--model-slug",
        default="pipeline",
        help="Subfolder under research/observations/ (default: 'pipeline')",
    )
    parser.add_argument(
        "--dump-raw/--no-dump-raw",
        dest="dump_raw",
        default=True,
        help="Write raw tensor dumps to outputs/raw/ (gitignored)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16", "half"],
        help="Florence-2 compute dtype. Overrides FLORENCE_DTYPE env var.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup generate() calls before measured run (GPU kernel compile)",
    )
    parser.add_argument(
        "--measure-runs",
        type=int,
        default=3,
        help="Measured generate() calls; median of these is reported",
    )
    return parser.parse_args()


def observe_florence(
    detector,
    image_bytes: bytes,
    classes: list[str],
    warmup_runs: int = 1,
    measure_runs: int = 3,
) -> dict:
    """Run Florence-2 detect and capture step-by-step intermediates.

    Returns a dict with:
        - 'points': list[dict] — observation-point descriptions
        - 'raw': dict[str, tensor/array] — tensors eligible for optional raw dump
        - 'result': dict — detector.detect() API output (pipeline boundary)
        - 'profile': dict — timing & memory measurements (GPU only)
    """
    import statistics
    import time as _time
    from mvp_app.config import get_settings  # noqa

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    points: list[dict] = []
    raw: dict = {}
    profile: dict = {}

    points.append(describe(image_bytes, name="florence-2/processor/inputs.image_bytes"))
    points.append(describe(classes, name="florence-2/processor/inputs.classes"))
    points.append(
        {
            "name": "florence-2/processor/inputs.PIL_image",
            "type": "PIL.Image.Image",
            "mode": image.mode,
            "size": [image.width, image.height],
        }
    )

    from mvp_app.config import get_settings as _get_settings
    if detector is None or _get_settings().fake_models:
        if detector is not None:
            result = detector.detect(image_bytes, classes)
        else:
            result = {
                "boxes": [[image.width * 0.2, image.height * 0.2, image.width * 0.8, image.height * 0.8]],
                "labels": [classes[0] if classes else "object"],
                "scores": [0.99],
                "image_size": {"width": image.width, "height": image.height},
            }
        result["_fake"] = True
        points.append(describe(result, name="florence-2/output/result"))
        return {"points": points, "raw": raw, "result": result, "profile": profile}

    detector._ensure_loaded()  # type: ignore[attr-defined]
    profile["dtype"] = _get_settings().florence_dtype
    profile["device"] = str(detector.device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model_vram_bytes = torch.cuda.memory_allocated()
        profile["model_vram_bytes_after_load"] = int(model_vram_bytes)
    prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    text_input = "A photo of " + ", and ".join(classes) + "."
    processor_inputs = detector._processor(  # type: ignore[attr-defined]
        text=prompt + text_input, images=image, return_tensors="pt",
    )
    model_dtype = next(detector._model.parameters()).dtype  # type: ignore[attr-defined]
    moved = {}
    for k, v in processor_inputs.items():
        if hasattr(v, "to"):
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                moved[k] = v.to(device=detector.device, dtype=model_dtype)
            else:
                moved[k] = v.to(detector.device)
        else:
            moved[k] = v
    processor_inputs = moved
    profile["model_param_dtype"] = str(model_dtype).replace("torch.", "")
    points.extend(describe_many(processor_inputs, prefix="florence-2/processor/inputs"))
    for k, v in processor_inputs.items():
        raw[f"florence-2/processor/inputs.{k}"] = v

    def _one_generate():
        with torch.inference_mode():
            return detector._model.generate(  # type: ignore[attr-defined]
                input_ids=processor_inputs["input_ids"],
                pixel_values=processor_inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=1,
                use_cache=False,
            )

    for _ in range(max(0, warmup_runs)):
        _ = _one_generate()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    measured_ms: list[float] = []
    generated_ids = None
    for _ in range(max(1, measure_runs)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = _time.perf_counter()
        generated_ids = _one_generate()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = _time.perf_counter()
        measured_ms.append((t1 - t0) * 1000.0)

    profile["generate_ms_runs"] = measured_ms
    profile["generate_ms_median"] = statistics.median(measured_ms)
    profile["generate_ms_min"] = min(measured_ms)
    profile["generate_ms_max"] = max(measured_ms)
    if torch.cuda.is_available():
        profile["peak_vram_bytes_during_generate"] = int(torch.cuda.max_memory_allocated())

    points.append(describe(generated_ids, name="florence-2/decoder/generate.output_ids"))
    raw["florence-2/decoder/generate.output_ids"] = generated_ids

    generated_text = detector._processor.batch_decode(  # type: ignore[attr-defined]
        generated_ids, skip_special_tokens=False,
    )[0]
    points.append(
        {
            "name": "florence-2/decoder/generate.output_text",
            "type": "str",
            "length": len(generated_text),
            "sample_head": generated_text[:120],
            "sample_tail": generated_text[-120:],
        }
    )

    post = detector._processor.post_process_generation(  # type: ignore[attr-defined]
        generated_text, task=prompt, image_size=(image.width, image.height),
    )
    points.append(describe(post, name="florence-2/post_process/result"))
    parsed = post.get(prompt, {})
    points.append(describe(parsed.get("bboxes", []), name="florence-2/post_process/bboxes"))
    points.append(describe(parsed.get("labels", []), name="florence-2/post_process/labels"))

    result = detector.detect(image_bytes, classes)
    points.append(describe(result, name="florence-2/output/result"))
    return {"points": points, "raw": raw, "result": result, "profile": profile}


def observe_sam3(segmenter, image_bytes: bytes, boxes: list[list[float]]) -> dict:
    """Run SAM3 segment and capture step-by-step intermediates."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    points: list[dict] = []
    raw: dict = {}

    points.append(describe(image_np, name="sam3/processor/inputs.image_np"))
    points.append(describe(boxes, name="sam3/processor/inputs.boxes_xyxy_px"))

    from mvp_app.config import get_settings as _get_settings
    if segmenter is None or _get_settings().fake_models:
        if segmenter is not None:
            out = segmenter.segment(image_bytes, boxes)
        else:
            out = {"masks": [], "image_size": {"width": image.width, "height": image.height}}
        out["_fake"] = True
        out_view = {
            **out,
            "masks": [
                {"score": m["score"], "mask_base64_length": len(m["mask"])}
                for m in out["masks"]
            ],
        }
        points.append(describe(out_view, name="sam3/output/result (fake)"))
        return {"points": points, "raw": raw, "result": out}

    segmenter._ensure_loaded()  # type: ignore[attr-defined]
    backend = segmenter.backend_name
    points.append(
        {"name": "sam3/processor/backend_name", "type": "str", "value": backend}
    )

    if segmenter._processor is False:  # type: ignore[attr-defined]
        out = segmenter.segment(image_bytes, boxes)
        points.append(describe(out, name="sam3/output/result (fallback)"))
        return {"points": points, "raw": raw, "result": out}

    state = segmenter._processor.set_image(image)  # type: ignore[attr-defined]
    points.append(describe(state, name="sam3/processor/state_after_set_image"))

    for idx, box in enumerate(boxes):
        segmenter._processor.reset_all_prompts(state)  # type: ignore[attr-defined]
        box_norm = segmenter._xyxy_to_cxcywh_norm(box, image.width, image.height)  # type: ignore[attr-defined]
        points.append(
            {
                "name": f"sam3/processor/box_norm[{idx}]",
                "type": "list[float]",
                "value": list(box_norm),
                "source_box_xyxy_px": list(box),
            }
        )
        state = segmenter._processor.add_geometric_prompt(  # type: ignore[attr-defined]
            box=box_norm, label=True, state=state,
        )
        points.append(describe(state, name=f"sam3/mask_head/state[{idx}]_after_prompt"))
        if "masks" in state:
            points.append(describe(state["masks"], name=f"sam3/mask_head/masks_logits[{idx}]"))
            raw[f"sam3/mask_head/masks_logits[{idx}]"] = state["masks"]
        if "scores" in state:
            points.append(describe(state["scores"], name=f"sam3/mask_head/scores[{idx}]"))
            raw[f"sam3/mask_head/scores[{idx}]"] = state["scores"]

    out = segmenter.segment(image_bytes, boxes)
    # Don't dump base64 mask strings into the summary's raw block
    out_view = {
        **out,
        "masks": [
            {"score": m["score"], "mask_base64_length": len(m["mask"])}
            for m in out["masks"]
        ],
    }
    points.append(describe(out_view, name="sam3/output/result (summary)"))
    return {"points": points, "raw": raw, "result": out}


def write_templates(base: Path, inputs_meta: list[dict], classes: list[str]) -> None:
    """Write plan.md, report.md, and a minimal analysis.ipynb template."""
    plan = f"""# Observation Plan

- **Date**: {base.name.split('-', maxsplit=3)[0:3]}
- **Slug**: {base.name}
- **Inputs**:
{chr(10).join(f"  - `{m['path']}` sha256=`{m['sha256']}`" for m in inputs_meta)}
- **Classes prompt**: {classes}

## Observation layer

- [ ] pipeline
- [ ] model
- [ ] module
- [x] tensor (default per protocol)

## Planned observation points

(Filled in after initial run. For forward-hook-based tensor-depth points, pin the
exact `torch.nn.Module` attribute paths here so future runs are reproducible.)

## Hypotheses / questions

- ...
"""
    (base / "plan.md").write_text(plan, encoding="utf-8")

    report = """# Observation Report

## Summary

_(one-paragraph summary after analysis)_

## Setup

- Input image(s): see plan.md
- Model versions: see summary.json `key_packages` / `model_id`
- Commit: see summary.json `commit`

## Observations

_(Facts only — shapes, dtypes, value ranges, structural notes.
Cross-reference summary.json observation points by name.)_

## Interpretation

_(What did I notice, what was unexpected, what design decisions did this inform.
Keep strictly separate from the Observations section above.)_

## Follow-ups

- _(next observation points to pin)_
- _(hypotheses to test)_
"""
    (base / "report.md").write_text(report, encoding="utf-8")

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {base.name}\n",
                    "\n",
                    "연구 관찰 노트북. outputs를 **지우지 말고** 커밋할 것.\n",
                    "프로토콜: `docs/standards/research-observation-protocol.md`\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "from pathlib import Path\n",
                    "summary = json.loads(Path('summary.json').read_text(encoding='utf-8'))\n",
                    "len(summary['observation_points'])\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Example: load a raw dump and visualize\n",
                    "# import torch, matplotlib.pyplot as plt\n",
                    "# t = torch.load('outputs/raw/florence-2__processor__inputs.pixel_values.pt')\n",
                    "# plt.imshow(t[0].permute(1,2,0).clamp(0,1))\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (base / "analysis.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 2

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        print("ERROR: --classes must contain at least one non-empty entry", file=sys.stderr)
        return 2

    if args.fake_models:
        import os
        os.environ["FAKE_MODELS"] = "1"
    if args.dtype:
        import os
        os.environ["FLORENCE_DTYPE"] = args.dtype

    from mvp_app.detector import DetectionService
    from mvp_app.segmenter import SegmentationService
    detector = DetectionService()
    segmenter = SegmentationService()

    image_bytes = image_path.read_bytes()

    base = obs_io.observation_dir(model_slug=args.model_slug, slug=args.slug)
    layout = obs_io.ensure_layout(base)
    print(f"observation folder: {base}")

    inputs_meta = [{"path": str(image_path), "sha256": obs_io.sha256_of_file(image_path)}]

    florence = observe_florence(
        detector, image_bytes, classes,
        warmup_runs=args.warmup_runs, measure_runs=args.measure_runs,
    )
    sam = observe_sam3(segmenter, image_bytes, florence["result"]["boxes"])

    points = florence["points"] + sam["points"]

    if args.dump_raw:
        for source in (florence["raw"], sam["raw"]):
            for name, tensor in source.items():
                obs_io.dump_tensor(tensor, layout["raw"], name)

    import platform
    key_packages = {}
    for mod_name in ("torch", "transformers", "numpy", "PIL"):
        try:
            mod = __import__(mod_name)
            key_packages[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            key_packages[mod_name] = "not-installed"

    from mvp_app.config import get_settings
    settings = get_settings()
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        }

    summary = {
        "model_slug": args.model_slug,
        "slug": args.slug,
        "commit": obs_io.current_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "key_packages": key_packages,
        "gpu": gpu_info,
        "model_ids": {
            "florence": settings.florence_model_id,
            "sam3_checkpoint": settings.sam3_checkpoint,
        },
        "florence_dtype": settings.florence_dtype,
        "fake_models": bool(settings.fake_models),
        "inputs": inputs_meta,
        "classes": classes,
        "florence_result": florence["result"],
        "florence_profile": florence.get("profile", {}),
        "sam_result_shape_only": {
            "masks_count": len(sam["result"].get("masks", [])),
            "image_size": sam["result"].get("image_size"),
        },
        "observation_points": points,
    }
    obs_io.write_summary(summary, base)
    write_templates(base, inputs_meta, classes)

    print(f"wrote {len(points)} observation points")
    print(f"summary: {base / 'summary.json'}")
    print(f"plan / report / notebook templates created. fill them in to complete the observation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
