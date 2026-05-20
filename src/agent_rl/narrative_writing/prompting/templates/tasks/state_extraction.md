---
id: state_extraction
version: 1
task: state_extraction
output_contract: json_object
---

Extract candidate canonical state changes from a generated draft. Only propose changes supported by the draft text. Separate events, plot progress, character state, world facts, relationship changes, and style notes. Return JSON proposals with confidence and source rationale.
