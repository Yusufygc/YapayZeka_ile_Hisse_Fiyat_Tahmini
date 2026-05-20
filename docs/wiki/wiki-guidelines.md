---
title: Wiki Guidelines
type: operating-model
status: active
last_updated: 2026-05-09
owner: llm
---

# Wiki Guidelines

This project uses an LLM Wiki pattern: the LLM incrementally compiles knowledge
from source files and conversations into a persistent markdown knowledge base.
The wiki is not a scratchpad. It is a maintained artifact.

## Does This Structure Match The LLM Wiki Pattern?

Yes, after the 2026-05-09 expansion it matches the pattern.

| LLM Wiki Layer | Project Implementation |
|---|---|
| Raw sources | Code, tests, README, existing docs, data schemas, and user-provided decisions |
| Wiki | `docs/wiki/*.md`, generated and maintained by the LLM |
| Schema | `AGENTS.md`, which tells agents how to read, update, and lint the wiki |
| Rules | `RULES.md`, which defines change, wiki-update, and Turkish commit-message discipline |

The initial skeleton was only partially compliant. It had the schema and wiki
directory, but it did not yet define source boundaries, ingest/query/lint
operations, cross-linking conventions, or index/log discipline. Those are now
documented here and in `AGENTS.md`.

## Ownership

- The user owns project direction, source selection, product decisions, and final judgment.
- The LLM owns wiki maintenance: summaries, cross-references, entity pages, logs, and consistency.
- Code and tests remain the primary source of truth for actual behavior.
- The wiki records understanding and decisions; it must be updated when understanding changes.

## Raw Source Rules

Raw sources are inputs to the wiki. They include:

- Source code under `src/`
- Tests under `tests/`
- Project entrypoints such as `python -m src.cli.interactive`, `python -m src.cli.batch`, and `python -m src.cli.forecast`
- Existing documentation such as `README.md`, `docs/glossary.md`, and `docs/MyDocs/`
- Data metadata such as `data/bist_universe.csv` and `data/meta/bist_calendar.csv`
- User-provided architecture decisions and bug findings from conversation
- Repository rules such as `RULES.md`

When ingesting raw sources for documentation purposes, do not rewrite the source
unless the user explicitly asks for a code/doc change. Instead, compile the
knowledge into the wiki.

## Wiki Page Conventions

Each wiki page should use YAML frontmatter:

```yaml
---
title: Page Title
type: concept
status: active
last_updated: YYYY-MM-DD
owner: llm
source_count: 3
---
```

Recommended `type` values:

- `index`
- `operating-model`
- `source-map`
- `concept`
- `entity`
- `workflow`
- `decision`
- `log`

Use relative markdown links between wiki pages. Keep summaries concrete and
source-grounded. Avoid duplicating long code blocks unless they define an
important contract.

## Index Rules

`index.md` is content-oriented. It must list all important wiki pages with a
one-line purpose. Update it when:

- A page is created, renamed, or deleted.
- A page's role changes.
- A new subsystem becomes important enough to navigate directly.

Agents must read `index.md` before touching code or answering architecture
questions.

## Log Rules

`log.md` is chronological and append-first. Add entries at the top using:

```text
## [YYYY-MM-DD] Action | Topic
```

Use actions such as:

- `Ingest`
- `Query`
- `Decision`
- `Bugfix`
- `Feature Plan`
- `Wiki Update`
- `Lint`

Every meaningful architecture change, wiki update, bug solution, or feature plan
must be logged.

## Change Management Rules

`RULES.md` defines the repository-wide change discipline. Any meaningful system
change must update the relevant wiki page. If no suitable page exists, create a
new Markdown file under `docs/wiki/`, link it from `index.md`, and record the
change in `log.md`.

Meaningful changes include:

- File additions, deletions, and renames
- Behavior changes
- Architecture decisions
- Bug solutions
- Feature plans
- Durable maintenance rules

When the user asks for a commit, the commit message must be written in clear
Turkish with correct Turkish characters.

## Ingest Workflow

Use this when the user provides a source, asks to process documentation, or asks
to make the wiki more complete.

1. Read `docs/wiki/index.md`.
2. Identify the source files and read only what is needed.
3. Extract stable facts, decisions, constraints, and contradictions.
4. Update or create the relevant wiki pages.
5. Add cross-links from `index.md` and related pages.
6. Append a top entry to `log.md`.
7. If requested, commit the completed changes with a clear Turkish message.
8. Report what changed and what remains uncertain.

## Query Workflow

Use this when answering questions about the project.

1. Read `docs/wiki/index.md`.
2. Read the most relevant wiki pages.
3. If behavior must be exact, inspect the source code and tests.
4. Answer with file references when useful.
5. If the answer produces reusable analysis or a decision, file it back into the wiki and log it.

## Lint Workflow

Use this periodically or when the user asks for wiki health checks.

Check for:

- Orphan pages not linked from `index.md`
- Pages with stale `last_updated`
- Contradictions between wiki and current source code
- Missing pages for repeatedly mentioned concepts
- Broken relative links
- Decisions mentioned in chat but not logged
- Claims that should cite source files or tests

Any lint pass should append a `Lint` entry to `log.md`, even if no changes are needed.

## When Not To Update The Wiki

Do not update the wiki for:

- Purely mechanical formatting with no knowledge change
- Temporary command output that will not matter later
- Failed exploratory attempts unless they reveal a lasting constraint
- User requests that explicitly say not to update documentation
