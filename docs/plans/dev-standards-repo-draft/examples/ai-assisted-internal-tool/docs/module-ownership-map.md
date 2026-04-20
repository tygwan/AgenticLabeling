# Module Ownership Map

| Module / Path | Responsibility | Allowed Changes | Common Risks | Notes |
|---|---|---|---|---|
| `src/ui` | rendering and interaction | view logic, local state | putting business logic in UI | keep thin |
| `src/api` | contract and request handling | validation, orchestration | leaking storage details upward | keep contracts stable |
| `src/services` | core behavior | business rules | duplicated workflows | prefer reuse |
| `src/storage` | persistence details | SQL, filesystem, indexes | coupling business logic to storage | isolate queries |
| `project-memory` | AI retrieval support | summaries, links, tags | storing raw noise | prefer structured entries |

