# System Map

## Top-Level Components

| Component | Responsibility | Inputs | Outputs | Notes |
|---|---|---|---|---|
| `src/ui` | user-facing workflows | user actions | API requests | do not place domain rules here |
| `src/api` | request and orchestration boundary | UI or client requests | service calls, responses | owns public contracts |
| `src/services` | business logic | validated requests | persistent mutations or results | primary implementation layer |
| `src/storage` | persistence and retrieval | service requests | stored records | isolated persistence logic |
| `project-memory` | structured project memory | summarized events | searchable memory | supports AI-assisted retrieval |

## Where To Read First

- UI task: `src/ui` + interface catalog
- API task: `src/api` + interface catalog
- storage task: `src/storage` + failure memory index
- architectural task: decision records + system map

