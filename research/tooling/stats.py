"""Tensor / array description helpers for observation reports."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def describe(value: Any, name: str = "") -> dict:
    """Return a protocol-compliant description of a tensor / array / container.

    The returned dict follows the field set required by
    docs/standards/model-inspection-conventions.md §"텐서 기록 표준 필드".
    """
    desc: dict = {"name": name, "type": type(value).__name__}

    if isinstance(value, torch.Tensor):
        t = value.detach()
        desc.update(
            {
                "type": "torch.Tensor",
                "shape": list(t.shape),
                "dtype": str(t.dtype).replace("torch.", ""),
                "device": str(t.device),
            }
        )
        if t.is_floating_point() or t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            t_cpu = t.float().cpu()
            desc["stats"] = {
                "min": float(t_cpu.min()),
                "max": float(t_cpu.max()),
                "mean": float(t_cpu.mean()),
                "std": float(t_cpu.std()),
            }
            flat = t_cpu.flatten()
            if flat.numel() <= 8:
                desc["sample"] = flat.tolist()
            else:
                desc["sample_head"] = flat[:4].tolist()
                desc["sample_tail"] = flat[-4:].tolist()
        return desc

    if isinstance(value, np.ndarray):
        desc.update(
            {
                "type": "np.ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        )
        if value.size > 0 and np.issubdtype(value.dtype, np.number):
            desc["stats"] = {
                "min": float(value.min()),
                "max": float(value.max()),
                "mean": float(value.mean()),
                "std": float(value.std()),
            }
            flat = value.flatten()
            if flat.size <= 8:
                desc["sample"] = flat.tolist()
            else:
                desc["sample_head"] = flat[:4].tolist()
                desc["sample_tail"] = flat[-4:].tolist()
        return desc

    if isinstance(value, (list, tuple)):
        desc.update({"type": type(value).__name__, "length": len(value)})
        if value:
            desc["element_type"] = type(value[0]).__name__
            if all(isinstance(v, (int, float)) for v in value) and len(value) <= 16:
                desc["sample"] = list(value)
        return desc

    if isinstance(value, dict):
        desc.update({"type": "dict", "keys": list(value.keys())})
        return desc

    if isinstance(value, (int, float, bool, str)) or value is None:
        desc["value"] = value
        return desc

    desc["repr"] = repr(value)[:200]
    return desc


def describe_many(mapping: dict, prefix: str = "") -> list[dict]:
    """Describe each entry of a mapping, preserving insertion order."""
    out = []
    for key, value in mapping.items():
        name = f"{prefix}/{key}" if prefix else key
        out.append(describe(value, name=name))
    return out
