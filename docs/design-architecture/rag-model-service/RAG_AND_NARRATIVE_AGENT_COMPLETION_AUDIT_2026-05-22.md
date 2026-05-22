# RAG And Narrative Agent Completion Audit

Date: 2026-05-22

## Scope

Objective audited:

- Finish the design-document implementation for the long-running narrative agent.
- Make the RAG local model service runnable from this project.
- Provide a conda environment for RAG work.
- Adapt the remote server RAG pattern from the old `narrative-state-engine`.
- Keep the implementation aligned with OOAD/design-pattern boundaries.

## Requirement Audit

| Requirement | Evidence | Status |
|---|---|---|
| Local RAG model can run with `qwen3-embedding:4b` | `python -m agent_rl.rag.cli start-local ...` succeeded; `python -m agent_rl.rag.cli warm` returned dimension `2560`; `python -m agent_rl.rag.cli embed --text ...` returned dimension `2560` | Complete |
| RAG conda environment exists | `environment-rag.yml`; `conda env create -f environment-rag.yml --dry-run` passed | Complete |
| RAG env contract is reusable and cross-platform | `.env.example` contains `RAG_*` settings; `agent_rl.rag.config.RAGServiceConfig`; `rag_env_snapshot()` redacts sensitive values | Complete |
| RAG is a reusable core package, not narrative-only code | `src/agent_rl/rag/` contains provider, reranker, vector store, remote manager, CLI, and service facade | Complete |
| Local vector storage exists | `SQLiteVectorStore`; tests cover index, count, and cosine search | Complete |
| Local RAG CLI exists | `agent-rag` script in `pyproject.toml`; `python -m agent_rl.rag.cli --help` passed | Complete |
| Remote server RAG is adapted from old project pattern | `OpenAICompatibleEmbeddingProvider`, `HTTPReranker`, `RemoteRAGServiceManager`; old `NOVEL_AGENT_*` aliases supported | Complete |
| Remote health command works | `python -m agent_rl.rag.cli remote-health --base-url http://127.0.0.1:18080` and `--base-url http://172.18.36.87:18080` completed and returned `healthy=false` | Complete for adapter; live remote service unavailable in current network |
| Narrative state can be indexed into RAG | `NarrativeRAGIndexingService`; `narrative_state_documents`; tests cover conversion and indexing | Complete |
| Narrative retrieval can use vector RAG | `RAGVectorNarrativeRetrievalPolicy`; tests cover vector evidence merge into `EvidencePack` and trace | Complete |
| Successful commits can enter RAG index automatically | `NarrativeWritingSession(auto_rag_index=True, rag_service=...)`; `RAG_AUTO_INDEX_ON_COMMIT`; `--auto-rag-index`; tests cover auto indexing after commit | Complete |
| Explicit workbench command exists for RAG indexing | `narrative-agent index-rag`; CLI help shows command | Complete |
| Background job exists for RAG indexing | `NarrativeJob` type `rag_index`; `NarrativeJobRunner` handles it | Complete |
| Long-running narrative session and workbench are implemented | `NarrativeWritingSession`, `save/resume`, author conversation, blueprint revision, repair, branches, rollback/export/show commands; tests cover main flows | Complete |
| Memory governance is implemented | SQLite memory repository, memory decay, invalidation API/CLI/job, compression policy, commit upsert; tests cover repository and invalidation | Complete |
| Design patterns are explicit | Strategy/Adapter/Facade/Repository/Composite mapping recorded in `RAG_MODEL_SERVICE_IMPLEMENTATION_STATUS_2026-05-22.md` | Complete |
| Full test suite passes | `74 passed` | Complete |
| Whitespace/patch hygiene passes | `git diff --check` passed | Complete |

## Non-Blocking Notes

- The old direct remote endpoint `http://172.18.36.87:18080` is not healthy/reachable from the current machine or network at audit time. The adapter, env compatibility, CLI, and health checks are implemented.
- The old project did not contain a concrete remote vector-store HTTP implementation beyond embedding/rerank clients and a `RemoteVectorStore` protocol. This project therefore implements local SQLite vector storage plus remote embedding/rerank adapters. A remote vector-store adapter can be added later if that service exposes index/search endpoints.
- A real always-on worker service is still future infrastructure. Current jobs are local file-backed background-style units runnable by CLI/API.

## Verification Commands

```powershell
rtk python -m pytest
rtk git diff --check
rtk powershell -NoProfile -Command "`$env:PYTHONPATH='src'; python -m agent_rl.rag.cli --help"
rtk powershell -NoProfile -Command "`$env:PYTHONPATH='src'; python -m agent_rl.narrative_writing.cli --help"
rtk conda env create -f environment-rag.yml --dry-run
```

Live local RAG smoke already run:

```powershell
python -m agent_rl.rag.cli start-local --ollama-executable "C:\Users\98289\AppData\Local\Programs\Ollama\ollama.exe" --ollama-models "D:\models\ollama"
python -m agent_rl.rag.cli warm
python -m agent_rl.rag.cli embed --text "测试本地 RAG embedding"
```

Observed live result:

```text
warm -> dimension 2560
embed -> dimension 2560
```
