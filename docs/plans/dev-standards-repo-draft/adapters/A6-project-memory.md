# A6 — Project Memory

## Use when

Activate when a project maintains a searchable memory layer outside raw source files and raw chat history.

## Operationalizes

- documentation and discoverability
- problem tracking
- decision records
- AI-assisted development profile

## Adds

- how structured project memory is stored
- how memory categories are defined
- how retrieval prefers high-value artifacts over raw transcript replay
- how searchable memory links back to authoritative project artifacts

## Operating Model

Project memory should be treated as a derived knowledge layer, not as a dump of raw activity.

The preferred flow is:

1. capture high-value source events
2. classify them into stable categories
3. summarize them for retrieval
4. link them back to authoritative project artifacts
5. retrieve them by task scope

## Preferred Source Events

High-value sources usually include:

- accepted decisions
- resolved failures
- reusable techniques
- verified edits
- important constraints

Lower-value sources usually include:

- passive reads
- raw transcript blocks
- unverified speculation
- noisy intermediate steps with no durable result

## Ingestion Guidance

The adapter should prefer ingesting:

- events with a clear outcome
- events with a stable source reference
- events that reduce future repeated reasoning

The adapter should avoid ingesting:

- everything by default
- duplicate summaries of the same event
- memory entries that cannot be traced back to a source artifact

## Retrieval Guidance

Retrieval should prefer:

1. same module or subsystem scope
2. same category and matching tags
3. recent failures relevant to the task
4. related decisions
5. related techniques

The preferred response shape is:

- summary first
- detailed content second
- source link always available

## Link-Back Requirement

Every high-value memory entry should link back to an authoritative source when possible, such as:

- a decision record
- a problem record
- a commit
- a code path
- a runbook or interface document

Without link-back, the memory system becomes a second undocumented truth source.

## Suggested memory categories

- decision
- failure
- technique
- constraint
- open-question

## Suggested storage features

- summary
- detailed content
- tags
- metadata
- source link
- module scope
- change kind
- timestamps
- related memory links

## Suggested write triggers

Good triggers for creating or updating memory entries:

- decision accepted
- failure diagnosed
- fix verified
- technique reused successfully
- constraint discovered and confirmed

## Does not define

- one storage engine
- one embedding model
- one vector database
- one transcript parser implementation
- one ranking algorithm
