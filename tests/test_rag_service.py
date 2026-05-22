import json
from io import BytesIO

from agent_rl.domains.narrative import MemoryAtom, NarrativeQuery, NarrativeTaskState, SourceChunk
from agent_rl.narrative_writing import AuthorRequest, NarrativeWritingSession, ReferenceMaterial
from agent_rl.narrative_writing.policies import RAGVectorNarrativeRetrievalPolicy
from agent_rl.narrative_writing.rag_index import NarrativeRAGIndexingService, narrative_state_documents
from agent_rl.narrative_writing.utils import new_id
from agent_rl.rag import RAGDocument, RAGModelService, RAGServiceConfig, SQLiteVectorStore
from agent_rl.rag.embeddings import OllamaEmbeddingProvider


class _FakeEmbeddingProvider:
    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "river" in lowered or "bridge" in lowered else 0.0,
                    1.0 if "warehouse" in lowered else 0.0,
                    1.0 if "style" in lowered else 0.0,
                ]
            )
        return vectors


class _FakeRAGService:
    def __init__(self) -> None:
        self.indexed_batches: list[list[RAGDocument]] = []

    def index_documents(self, documents: list[RAGDocument], *, collection_id: str = "default", batch_size: int | None = None) -> int:
        self.indexed_batches.append(list(documents))
        return len(documents)


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_rag_config_reads_new_and_legacy_env(monkeypatch) -> None:
    monkeypatch.setenv("RAG_PROVIDER", "openai-compatible")
    monkeypatch.setenv("NOVEL_AGENT_VECTOR_STORE_URL", "http://remote:18080")
    monkeypatch.setenv("NOVEL_AGENT_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setenv("NOVEL_AGENT_EMBEDDING_DIMENSION", "2560")

    config = RAGServiceConfig.from_env()

    assert config.provider == "openai-compatible"
    assert config.embedding_base_url == "http://remote:18080"
    assert config.embedding_model == "Qwen/Qwen3-Embedding-4B"
    assert config.embedding_dimension == 2560


def test_ollama_embedding_provider_calls_embed_api(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        assert request.full_url == "http://127.0.0.1:11434/api/embed"
        assert timeout == 120.0
        body = json.loads(request.data.decode("utf-8"))
        assert body["model"] == "qwen3-embedding:4b"
        assert body["input"] == ["hello"]
        return _Response({"embeddings": [[0.1, 0.2, 0.3]]})

    monkeypatch.setattr("agent_rl.rag.embeddings.urllib.request.urlopen", fake_urlopen)

    vectors = OllamaEmbeddingProvider().embed_texts(["hello"])

    assert vectors == [[0.1, 0.2, 0.3]]


def test_sqlite_vector_store_indexes_and_searches(tmp_path) -> None:
    store = SQLiteVectorStore(tmp_path / "vectors.sqlite3")
    documents = [
        RAGDocument(document_id="d1", text="river bridge clue", story_id="story", evidence_type="event"),
        RAGDocument(document_id="d2", text="quiet style sentence", story_id="story", evidence_type="style_snippet"),
    ]
    embeddings = _FakeEmbeddingProvider().embed_texts([document.text for document in documents])

    assert store.upsert_documents(documents, embeddings, collection_id="narrative") == 2
    results = store.search(embedding=_FakeEmbeddingProvider().embed_query("river"), story_id="story", collection_id="narrative")

    assert results[0].evidence_id == "d1"
    assert store.count(collection_id="narrative", story_id="story") == 2


def test_rag_model_service_indexes_batches_and_searches(tmp_path) -> None:
    service = RAGModelService(
        config=RAGServiceConfig(vector_db_path=str(tmp_path / "vectors.sqlite3"), batch_size=1),
        embedding_provider=_FakeEmbeddingProvider(),
        vector_store=SQLiteVectorStore(tmp_path / "vectors.sqlite3"),
        reranker=None,
    )
    documents = [
        RAGDocument(document_id="d1", text="warehouse clue", story_id="story", evidence_type="event"),
        RAGDocument(document_id="d2", text="river bridge", story_id="story", evidence_type="event"),
    ]

    assert service.index_documents(documents, collection_id="narrative") == 2
    assert service.search("river", story_id="story", collection_id="narrative", limit=1, rerank=False)[0].evidence_id == "d2"


def test_narrative_state_can_be_indexed_for_rag(tmp_path) -> None:
    state = NarrativeTaskState(
        task_id="task",
        story_id="story",
        goal="continue",
        source_chunks=[SourceChunk(chunk_id="chunk-1", document_id="doc", source_type="target_continuation", text="river bridge clue")],
        memory_atoms=[MemoryAtom(memory_id="memory-1", memory_type="plot_progress", text="warehouse clue", canonical=True)],
    )
    service = RAGModelService(
        config=RAGServiceConfig(vector_db_path=str(tmp_path / "vectors.sqlite3")),
        embedding_provider=_FakeEmbeddingProvider(),
        vector_store=SQLiteVectorStore(tmp_path / "vectors.sqlite3"),
        reranker=None,
    )

    assert [document.document_id for document in narrative_state_documents(state)] == ["chunk-1", "memory-1"]
    assert NarrativeRAGIndexingService(service).index_state(state) == 2


def test_rag_vector_narrative_retrieval_merges_vector_evidence(tmp_path) -> None:
    state = NarrativeTaskState(
        task_id="task",
        story_id="story",
        goal="continue",
        memory_atoms=[MemoryAtom(memory_id="memory-1", memory_type="event", text="river bridge clue", canonical=True)],
    )
    service = RAGModelService(
        config=RAGServiceConfig(vector_db_path=str(tmp_path / "vectors.sqlite3")),
        embedding_provider=_FakeEmbeddingProvider(),
        vector_store=SQLiteVectorStore(tmp_path / "vectors.sqlite3"),
        reranker=None,
    )
    NarrativeRAGIndexingService(service).index_state(state)

    pack = RAGVectorNarrativeRetrievalPolicy(service).retrieve(
        state,
        NarrativeQuery(query_id=new_id("query"), query_text="river bridge", query_type="chapter_continuation"),
    )

    assert any(item.source == "memory_atoms" and item.score_vector > 0 for item in pack.plot_evidence)
    assert pack.retrieval_trace[-1]["status"] == "succeeded"


def test_session_auto_indexes_rag_after_commit() -> None:
    rag_service = _FakeRAGService()
    session = NarrativeWritingSession(
        AuthorRequest(
            request="continue",
            session_id="rag-auto-session",
            story_id="rag-auto-story",
            task_id="rag-auto-task",
            references=(ReferenceMaterial(title="ref", text="warehouse letter river bridge"),),
            writing_direction="continue toward the river bridge",
            confirm_plan=True,
        ),
        rag_service=rag_service,  # type: ignore[arg-type]
        auto_rag_index=True,
        rag_collection_id="narrative",
    )

    result = session.run_until_pause()

    assert result.committed is True
    assert rag_service.indexed_batches
    assert session.state.metadata["rag_index"]["mode"] == "auto_on_commit"
    assert session.state.metadata["rag_index"]["status"] == "succeeded"
