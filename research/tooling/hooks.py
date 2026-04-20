"""Forward-hook registration helpers for external model observation.

Usage:

    captures = {}
    registry = HookRegistry(captures)
    registry.add(model.vision_tower.layers[4].self_attn,
                 "florence-2/encoder/vision_tower.layer[4].self_attn")
    try:
        run_inference(model, ...)
    finally:
        registry.remove_all()

    # captures now holds observed outputs keyed by observation-point name.
"""

from __future__ import annotations

from typing import Any, Callable


class HookRegistry:
    """Groups registered forward hooks so they can all be removed at once."""

    def __init__(self, store: dict[str, Any]):
        self._handles: list = []
        self._store = store

    def _make_hook(self, name: str) -> Callable:
        def hook(module, inputs, output):
            self._store[name] = output
        return hook

    def add(self, module, name: str) -> None:
        handle = module.register_forward_hook(self._make_hook(name))
        self._handles.append(handle)

    def remove_all(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
