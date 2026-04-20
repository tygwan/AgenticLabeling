# Retrieval Notes

## Goal

Retrieve the highest-value project memory for a task without replaying full transcript history.

## Preferred Retrieval Order

1. same module scope
2. same category and tags
3. recent failures
4. related decisions
5. related techniques

## Preferred Memory Sources

- verified edits
- decision records
- failure records
- resolved techniques

## Lower-Value Sources

- passive code reads with no change
- unfiltered transcript blocks
- memory without source links

## Query Example

For a task scoped to `src/api/auth`:

- fetch `module_scope = 'src/api/auth'`
- rank `category IN ('failure', 'decision', 'technique')`
- boost entries with matching tags
- return summary first, content second

