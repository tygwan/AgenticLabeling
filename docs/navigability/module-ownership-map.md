# AgenticLabeling Module Ownership Map

This map is for edit scoping, not team assignment. It exists to answer: "where should a change start?"

| Scope | Owns | Does not own |
|---|---|---|
| `mvp_app/main.py` | HTTP routes, HTML responses, overlay endpoints, review actions | model inference internals, sqlite schema details |
| `mvp_app/detector.py` | Florence-2 loading, prompting, detection normalization | HTTP routing, review UI |
| `mvp_app/segmenter.py` | SAM3 loading, mask generation, fallback segmentation | routing, export packaging |
| `mvp_app/registry.py` | source/object/category persistence and review-state updates | image rendering, model loading |
| `mvp_app/storage.py` | asset paths, mask/image file storage, export filesystem handling | detection logic, review rules |
| `project_memory/` | engineering memory schema, recording, querying | product runtime inference |
| `scripts/` | process startup helpers | product logic |
| `tests/` | regression contracts for the repo | production behavior |
| `services/` | historical microservice implementations | default runtime path |
| `docs/navigability/` | repo discovery index | executable behavior |

## Practical edit starting points

If the task is:

- route or page behavior: start with `mvp_app/main.py`
- detection mismatch or prompt normalization: start with `mvp_app/detector.py`
- segmentation backend or mask behavior: start with `mvp_app/segmenter.py`
- source/object persistence: start with `mvp_app/registry.py`
- asset file paths or exports: start with `mvp_app/storage.py`
- engineering memory or repo discovery: start with `project_memory/` and `docs/navigability/`

