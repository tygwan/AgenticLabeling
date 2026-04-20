# AgenticLabeling Implementation Memory Summary

This document summarizes the major implementation decisions, failures, and techniques that shaped the current repository state.

It is a human-readable companion to the structured entries stored in `project_memory`.

## Why this exists

The repository has already gone through multiple architectural turns:

- microservices to MVP monolith
- unstable model/runtime edges to guarded fallback paths
- generic UI direction to review-workstation direction
- external standards reference to repository-local standards application

If these changes are not summarized, later work risks repeating the same exploration cost.

## Key implementation records

### 1. Make the MVP monolith the default runtime

- Memory ID: `mem_9b378eded136`
- Type: `decision`
- What changed:
  - the default product boundary moved to `mvp_app/`
  - the microservice stack became legacy rather than primary
- Why:
  - one runtime made upload, detection, segmentation, review, and export easier to stabilize
- Impact:
  - reduced deployment complexity
  - narrowed the default edit surface
  - made future UI work target one active runtime

### 2. Florence-2 generation cache path crashed uploads

- Memory ID: `mem_b16207bd9b03`
- Type: `failure`
- What failed:
  - uploads broke because detector inference failed inside Florence-2 generation
- Why it mattered:
  - the failure appeared at the route level but originated in the model cache path
- Impact:
  - restored upload stability after fix
  - justified detector regression coverage
  - reduced false debugging in unrelated upload code

### 3. Keep segmentation alive with a box-based fallback

- Memory ID: `mem_d2a7b47b30e6`
- Type: `technique`
- What changed:
  - segmentation degrades to a box-derived mask when SAM2 is unavailable
- Why:
  - partial environments should not take down the whole review/export flow
- Impact:
  - preserved end-to-end product behavior
  - separated segmentation availability from total product availability
  - made backend degradation explicit through status surfaces

### 4. Separate original, bbox, and segmentation overlays in review

- Memory ID: `mem_b5096e2d0d35`
- Type: `decision`
- What changed:
  - the review UI exposes distinct visual layers instead of one mixed rendering
- Why:
  - a single composite view hid whether errors came from detection, segmentation, or rendering
- Impact:
  - improved review clarity
  - reduced debugging ambiguity
  - made stage-by-stage visual comparison possible

### 5. Treat the review UI as a workstation, not a dashboard

- Memory ID: `mem_da1b5e556b10`
- Type: `decision`
- What changed:
  - UI direction was fixed around operator workflow, not generic dashboard aesthetics
- Why:
  - the product is fundamentally a review and curation environment
- Impact:
  - stabilized UI prioritization
  - kept design work tied to throughput and validation quality
  - prevented drift toward decorative but low-utility layout choices

### 6. Collapse Docker default to one MVP service

- Memory ID: `mem_bfaf8139e093`
- Type: `decision`
- What changed:
  - the default Docker path became one MVP service
  - the old service graph moved to explicit legacy status
- Why:
  - duplicated heavy runtime layers created avoidable operational cost
- Impact:
  - reduced operational complexity
  - simplified startup and packaging
  - reframed image size as an optimization issue instead of an architecture blocker

### 7. Vendor SAM2 and local Python dependencies

- Memory ID: `mem_b0a737b2a0aa`
- Type: `technique`
- What changed:
  - the active segmentation path moved toward repository-local integration assets
- Why:
  - hidden environment assumptions caused avoidable backend drift
- Impact:
  - improved reproducibility
  - reduced machine-specific setup failures
  - kept the active runtime path more self-contained

### 8. Apply dev-standards through project-local mechanisms

- Memory ID: `mem_1c3acfab602d`
- Type: `decision`
- What changed:
  - standards adoption was bound to local artifacts, not left as an external reference
- Why:
  - repository behavior changes only if standards are tied to docs, tests, and workflow gates
- Impact:
  - made standards actionable inside the repository
  - created a concrete gate for upcoming UI work
  - reduced reliance on implicit process memory

## Existing standards-application records

These records were created earlier and are part of the same standards adoption sequence:

### 9. Adopt dev-standards through repository-local artifacts

- Memory ID: `mem_edb6bf8d3ba3`
- Type: `decision`
- Scope: `docs`

### 10. UI phase must pass the MVP runtime gate

- Memory ID: `mem_c17e398431e6`
- Type: `decision`
- Scope: `mvp_app`

## How to use this summary

Before a major change:

1. read the relevant entry here
2. query `project_memory` for the owning scope
3. check `docs/navigability/` for edit location and interface impact
4. update both the structured memory and this summary if a new non-obvious decision is made

## Query examples

```bash
python3 -m project_memory.memory_store query --module-scope mvp_app --limit 20
```

```bash
python3 -m project_memory.memory_store query --category decision --text docker
```

