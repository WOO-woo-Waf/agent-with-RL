# Core Package Layering Design

## Background

The project started as a compact learning repo, so early files were intentionally simple. After adding the narrative-writing scenario, the root package began mixing several responsibilities:

- generic Agent/RL concepts and runtime
- framework adapters
- narrative domain models
- narrative-writing application policies
- examples and public package exports

That is no longer a good boundary. The package now needs explicit layers so new scenarios can reuse the core without importing narrative-specific code.

## Goal

Establish a layered package structure that keeps OOAD responsibilities separate:

```text
src/agent_rl/
  core/                 # reusable Agent/RL abstractions and runtime
  domains/              # domain models, no runtime dependency
  narrative_writing/    # narrative scenario application service and policies
  examples/             # runnable demos only
  integrations.py       # optional external framework adapters
```

## Applied Standards

The project follows these engineering standards:

- **Layered architecture**: core abstractions must not depend on scenario/domain/application code.
- **Dependency inversion**: application code depends on ports/protocols, not concrete infrastructure.
- **Single responsibility**: DTOs, ports, policies, scenario adapter, runtime agent, and examples are separate files.
- **Clean initial API**: no root compatibility shim modules are kept because the project is still pre-release.
- **Open extension**: LLM, memory, retrieval, LangGraph, OpenAI Agents SDK, and mem0 integrations should replace ports, not rewrite the core.
- **Repository hygiene**: environment, requirements, ignore rules, text attributes, tests, docs, and examples are explicit.

## Current Layers

### Core Layer

Location: `src/agent_rl/core/`

Responsibilities:

- `concepts.py`: `Goal`, `Observation`, `Action`, `Reward`, `Transition`, `Trajectory`, `Policy`, `Environment`, `Tool`, `Guardrail`.
- `runtime.py`: generic observe-decide-act-record loop.
- `policies.py`: generic reusable policies.
- `architectures.py`: ReAct, Plan-and-Execute, Multi-Agent architecture helpers.
- `memory.py`: generic memory store implementation.

Rules:

- May import only Python stdlib and other `agent_rl.core` modules.
- Must not import narrative domain or scenario code.
- Must remain testable without external services.

### Domain Layer

Location: `src/agent_rl/domains/`

Responsibilities:

- Domain vocabulary and state models.
- No concrete runtime, no model calls, no persistence.

Current domain:

- `domains/narrative.py`: source, world, character, plot, scene, style, author intent, memory, retrieval, generation, evaluation, and task-state objects.

### Scenario/Application Layer

Location: `src/agent_rl/narrative_writing/`

Responsibilities:

- Author-facing request/result DTOs.
- Ports for interchangeable policies.
- Default local policies.
- Scenario adapter.
- Application service that runs the narrative Agent use case.

Structure:

```text
narrative_writing/
  requests.py
  ports.py
  bootstrap.py
  utils.py
  policies/
  scenario.py
  agent.py
```

Rules:

- May depend on `core` and `domains`.
- Must depend on policy ports where behavior should be replaceable.
- Must not require LLM keys, databases, or vector stores for local tests.

### Integration Layer

Location: `src/agent_rl/integrations.py`

Responsibilities:

- Thin optional adapters for external implementations.
- Keep framework-specific imports lazy.

Current adapters:

- Gymnasium environment adapter.
- PettingZoo AEC adapter.
- OpenAI Agents SDK tool adapter.

Future direction:

- If adapters grow, split into `integrations/` package in a dedicated refactor.

## Dependency Direction

Allowed direction:

```text
examples -> scenario/application -> domains -> core
examples -> core
integrations -> core
```

Forbidden direction:

```text
core -> domains
core -> narrative_writing
domains -> narrative_writing
domains -> integrations
```

## Verification

Required checks after structural changes:

```powershell
python -m pytest
$env:PYTHONPATH="src"; python -m agent_rl.examples.narrative_demo
```

## Open Questions

- Whether `integrations.py` should become an `integrations/` package once more adapters are added.
