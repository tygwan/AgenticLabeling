# Ingestion Notes

## Goal

Turn high-value project events into durable, searchable memory without storing raw noise.

## Preferred Ingestion Flow

1. capture source event
2. decide whether it is worth storing
3. classify it into a memory category
4. write summary and detailed content
5. attach tags, scope, and source link
6. store related-memory links if relevant

## Good Source Events

- accepted decision
- diagnosed failure
- verified fix
- reusable implementation technique
- confirmed project constraint

## Avoid Ingesting

- passive code reads
- raw transcript dumps
- unverified speculation
- duplicate restatements of existing memory

## Quality Rule

If a memory entry does not reduce future repeated reasoning, it probably should not be stored.

