---
id: novel_chunk_analysis
version: 1
task: novel_chunk_analysis
output_contract: json_object
---

# Task

Analyze one novel source chunk for a continuation agent. Return structured state,
not literary commentary.

# Rules

- Use only the provided chunk and task context.
- Separate confirmed source facts, character viewpoint, uncertainty, and possible foreshadowing.
- Treat chunk analysis as evidence for later chapter/global synthesis.
- Preserve source quotes and retrieval keywords that can help future RAG.
- Do not promote auxiliary reference material into primary-story canon unless the input explicitly says so.
- Return exactly one JSON object matching the user-provided schema.
