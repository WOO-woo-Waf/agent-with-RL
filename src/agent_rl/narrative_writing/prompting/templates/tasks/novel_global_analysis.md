---
id: novel_global_analysis
version: 1
task: novel_global_analysis
output_contract: json_object
---

# Task

Synthesize chapter analyses into a global story state for a novel-continuation
agent.

# Rules

- Build reusable state, not a generic summary.
- Cover characters, relationships, plot threads, timeline, world rules,
  settings, objects, organizations, foreshadowing, style bible, continuation
  constraints, and retrieval index suggestions when evidence is available.
- Keep source authority clear. Primary-story material may become canon
  candidates; style/reference material should stay reference-only unless the
  author later promotes it.
- Keep uncertainty visible through open questions and completeness gaps.
- Return exactly one JSON object matching the user-provided schema.
