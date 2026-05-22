# RAG Model Service Implementation Status

Date: 2026-05-22

## Goal

Make RAG a reusable project subsystem that can run with the local Ollama `qwen3-embedding:4b` model, adapt to the old remote embedding/rerank service pattern, and plug into the narrative-writing agent without hard-coding vector infrastructure into domain logic.

## Implemented

- `agent_rl.rag` package:
  - `RAGServiceConfig`
  - `OllamaEmbeddingProvider`
  - `OpenAICompatibleEmbeddingProvider`
  - `HTTPReranker`
  - `SQLiteVectorStore`
  - `RAGModelService`
  - `RemoteRAGServiceManager`
- CLI entrypoint:
  - `agent-rag = agent_rl.rag.cli:main`
  - `python -m agent_rl.rag.cli env`
  - `start-local`
  - `warm`
  - `embed`
  - `index-jsonl`
  - `search`
  - `remote-health`
  - `remote-start`
  - `remote-stop`
- Environment contract:
  - `.env.example` includes `RAG_*` settings.
  - `RAG_*` supports old `NOVEL_AGENT_*` aliases for remote embedding/rerank service migration.
- Conda:
  - `environment-rag.yml` defines a RAG-capable conda environment.
- Narrative integration:
  - `NarrativeRAGIndexingService`
  - `narrative_state_documents`
  - `NarrativeWritingSession.index_rag`
  - optional `NarrativeWritingSession` auto-index on successful commit
  - `RAGVectorNarrativeRetrievalPolicy`
  - `narrative-agent index-rag`
  - `NarrativeJob` type `rag_index`
- Public API:
  - Core package exports RAG service, providers, vector store, and narrative RAG policy.

## Design Pattern Mapping

- Strategy: embedding provider, reranker, retrieval policy, vector store.
- Adapter: Ollama HTTP adapter, OpenAI-compatible embedding adapter, remote SSH service manager.
- Facade: `RAGModelService` coordinates embedding, vector storage, search, and rerank.
- Repository/Port: `SQLiteVectorStore` implements the replaceable `VectorStore` port.
- Composite retrieval: `RAGVectorNarrativeRetrievalPolicy` merges local structural evidence and vector evidence into one `EvidencePack`.
- Graceful degradation: vector retrieval failure records a trace and falls back to local structural retrieval.

## Verification

Current automated verification:

```text
74 passed
git diff --check passed
python -m agent_rl.rag.cli --help passed
python -m agent_rl.narrative_writing.cli --help passed
python -m agent_rl.rag.cli start-local ... passed
python -m agent_rl.rag.cli warm -> dimension 2560 passed
python -m agent_rl.rag.cli embed --text ... -> dimension 2560 passed
python -m agent_rl.rag.cli remote-health --base-url http://127.0.0.1:18080 passed; service not running locally
python -m agent_rl.rag.cli remote-health --base-url http://172.18.36.87:18080 passed; service not reachable/healthy in current network
conda env create -f environment-rag.yml --dry-run passed
```

Covered behavior:

- RAG env config and old-system aliases.
- Ollama `/api/embed` request shape.
- SQLite vector index and cosine search.
- `RAGModelService` batching and search.
- Narrative state to RAG documents.
- Narrative vector retrieval evidence merge and trace.
- Optional narrative commit auto-indexing into RAG vector storage.

## External / Future Infrastructure Notes

- Remote live smoke depends on the remote endpoint or SSH tunnel being available. Current `remote-health` commands work, but both `127.0.0.1:18080` and old direct endpoint `http://172.18.36.87:18080` report `healthy=false` from this machine/network.
- Add a stronger remote vector-store adapter only if the remote service exposes search/index endpoints beyond embeddings/rerank. The old codebase only defined a `RemoteVectorStore` protocol and concrete HTTP embedding/rerank clients.
