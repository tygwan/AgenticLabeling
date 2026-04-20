# AgenticLabeling Failure Memory Index

This index summarizes recurring failures that should not need to be rediscovered.

## F-001 Florence-2 generation cache failure

- Scope: `mvp_app/detector.py`
- Symptom: upload failed during Florence-2 generation with `past_key_values[0][0].shape` access on `None`
- Resolution: force the generation path to avoid the incompatible cache path
- Why it matters: detector regressions can look like general upload failures

## F-002 SAM3 missing module failure

- Scope: `mvp_app/segmenter.py`
- Symptom: `ModuleNotFoundError: No module named 'sam3'`
- Resolution: provide a fallback segmentation path and vendor SAM3 at `vendor/sam3/`
- Why it matters: segmentation backend availability depends on environment and vendored code path

## F-003 Review page stale process confusion

- Scope: local runtime workflow
- Symptom: backend changes existed in code but did not appear in the browser
- Resolution: restart the local server and verify through a route that only the new code can serve
- Why it matters: UI debugging is misleading if an older process still owns the port

## F-004 Docker image bloat from duplicated runtime stacks

- Scope: old microservice architecture
- Symptom: multiple heavy model services produced a very large and expensive container graph
- Resolution: collapse the default path into a single MVP service and keep legacy compose separate
- Why it matters: deployment cost and build latency were architectural symptoms, not just Dockerfile issues

