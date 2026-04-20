-- Fetch high-value memory for a module-scoped task.
SELECT memory_id, category, title, summary, source_ref
FROM memory_entries
WHERE module_scope = :module_scope
  AND category IN ('failure', 'decision', 'technique')
ORDER BY updated_at_utc DESC
LIMIT 20;

-- Fetch related memory for one entry.
SELECT ml.link_type, me.memory_id, me.category, me.title, me.summary
FROM memory_links ml
JOIN memory_entries me ON me.memory_id = ml.to_memory_id
WHERE ml.from_memory_id = :memory_id
ORDER BY me.updated_at_utc DESC;
