# Neo4j Notes

## Main flow
- Shared Neo4j logic lives in `src/llm/eval_voter.py`.
- `src/llm/eval_qa.py` imports `fetch_law_texts(...)` from there, so changes affect both QA and voter prompts.
- Prompt templates only consume `{law_text}`. Rendering logic is not in the templates.

## Data expected from Neo4j
- Legal nodes are fetched by `n.uid`.
- Expected node fields:
  - `uid`
  - `title`
  - `content`
  - `doc_identity`
  - label in `{Article, Clause, Point}`
- Document name is fetched from `(:Document)` by `doc_identity` as `doc_name`.

## UID format
- Article: `{doc_identity}::article::{N}`
- Clause: `{doc_identity}::article::{N}::clause::{M}`
- Point: `{doc_identity}::article::{N}::clause::{M}::point::{letter}`

## Current rendering rule
- Article UID:
  - render `doc_name`
  - render `Điều N. {title}`
  - render full article `content`
- Clause UID:
  - keep old behavior
  - render article title + clause content
  - fallback may include article content if clause content is missing
- Point UID:
  - keep old behavior
  - render article title + clause content + point content
  - skip clause content if it is identical to article content

## Important constraint
- Only article-only UIDs were changed to include full article content.
- Clause and point behavior should stay unchanged unless explicitly requested.

## Best place to modify
- If future Neo4j prompt formatting changes are needed, edit `_build_law_text(...)` in `src/llm/eval_voter.py`.
