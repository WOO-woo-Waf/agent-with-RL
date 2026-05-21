---
id: novel_chapter_analysis
version: 1
task: novel_chapter_analysis
output_contract: json_object
---

# Task

Merge chunk analyses from one chapter into stable chapter-level story state for
planning, retrieval, validation, and continuation.

# Rules

- Do not invent events absent from the chunk analyses.
- Produce a chapter synopsis, event chain, scene sequence, character state
  updates, relationship updates, world-rule confirmations, style signals, open
  questions, and continuation hooks.
- Keep low-confidence facts as candidates or open questions.
- Keep chunk ids and retrieval keywords useful for later evidence lookup.
- Return exactly one JSON object matching the user-provided schema.
