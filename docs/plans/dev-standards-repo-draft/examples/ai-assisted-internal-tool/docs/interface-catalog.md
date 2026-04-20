# Interface Catalog

| Name | Type | Location | Consumers | Compatibility Notes |
|---|---|---|---|---|
| public API | HTTP | `src/api` | UI and automation clients | breaking changes require explicit review |
| internal service calls | code contract | `src/services` | API layer | internal, but still discoverable |
| project memory query | SQLite/query layer | `project-memory` | AI retrieval tooling | implementation-specific, not public API |

## Validation And Errors

- public API validates inputs before entering services
- storage errors should not leak raw internal details upward
- memory entries should include source references when possible

