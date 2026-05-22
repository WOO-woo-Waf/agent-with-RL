"""High-level RAG model service facade."""

from __future__ import annotations

from agent_rl.rag.config import RAGServiceConfig
from agent_rl.rag.embeddings import EmbeddingProvider, OllamaEmbeddingProvider, OpenAICompatibleEmbeddingProvider
from agent_rl.rag.rerankers import HTTPReranker, Reranker
from agent_rl.rag.types import RAGDocument, VectorSearchResult
from agent_rl.rag.vector_store import SQLiteVectorStore, VectorStore


class RAGModelService:
    """Coordinates embedding providers, optional rerankers, and vector storage."""

    def __init__(
        self,
        *,
        config: RAGServiceConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config or RAGServiceConfig.from_env()
        self.embedding_provider = embedding_provider or _embedding_provider(self.config)
        self.vector_store = vector_store or SQLiteVectorStore(self.config.vector_db_path)
        self.reranker = reranker or _reranker(self.config)

    @classmethod
    def from_env(cls) -> "RAGModelService":
        return cls(config=RAGServiceConfig.from_env())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_provider.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedding_provider.embed_query(text)

    def warm(self) -> int:
        return len(self.embedding_provider.embed_query("warmup"))

    def index_documents(self, documents: list[RAGDocument], *, collection_id: str = "default", batch_size: int | None = None) -> int:
        if not documents:
            return 0
        size = max(1, batch_size or self.config.batch_size)
        total = 0
        for start in range(0, len(documents), size):
            batch = documents[start : start + size]
            embeddings = self.embedding_provider.embed_texts([document.text for document in batch])
            total += self.vector_store.upsert_documents(batch, embeddings, collection_id=collection_id)
        return total

    def search(
        self,
        query: str,
        *,
        story_id: str = "",
        evidence_types: list[str] | None = None,
        collection_id: str = "default",
        limit: int = 20,
        rerank: bool = True,
    ) -> list[VectorSearchResult]:
        embedding = self.embedding_provider.embed_query(query)
        candidates = self.vector_store.search(
            embedding=embedding,
            story_id=story_id,
            evidence_types=evidence_types,
            collection_id=collection_id,
            limit=max(limit, self.config.rerank_top_n if rerank else limit),
        )
        if rerank and self.reranker is not None and candidates:
            return _rerank_results(self.reranker, query, candidates, limit=limit, top_n=self.config.rerank_top_n)
        return candidates[:limit]


def _embedding_provider(config: RAGServiceConfig) -> EmbeddingProvider:
    if config.provider == "ollama":
        return OllamaEmbeddingProvider(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            timeout_s=config.request_timeout_s,
        )
    return OpenAICompatibleEmbeddingProvider(
        base_url=config.embedding_base_url,
        api_key=config.embedding_api_key,
        model=config.embedding_model,
        timeout_s=config.request_timeout_s,
    )


def _reranker(config: RAGServiceConfig) -> Reranker | None:
    if not config.rerank_base_url:
        return None
    return HTTPReranker(
        base_url=config.rerank_base_url,
        api_key=config.rerank_api_key,
        model=config.rerank_model,
        timeout_s=config.request_timeout_s,
    )


def _rerank_results(
    reranker: Reranker,
    query: str,
    candidates: list[VectorSearchResult],
    *,
    limit: int,
    top_n: int,
) -> list[VectorSearchResult]:
    pool = candidates[: max(1, top_n)]
    ranked = reranker.rerank(query=query, documents=[item.text for item in pool], top_n=len(pool))
    by_index = {item.index: item for item in ranked}
    reranked: list[VectorSearchResult] = []
    for index, candidate in enumerate(pool):
        rank = by_index.get(index)
        if rank is None:
            reranked.append(candidate)
            continue
        reranked.append(
            VectorSearchResult(
                evidence_id=candidate.evidence_id,
                evidence_type=candidate.evidence_type,
                source=candidate.source,
                text=candidate.text,
                score=max(candidate.score, rank.score),
                related_entities=list(candidate.related_entities),
                related_plot_threads=list(candidate.related_plot_threads),
                chapter_index=candidate.chapter_index,
                metadata={**dict(candidate.metadata), "rerank_score": rank.score},
            )
        )
    reranked.sort(key=lambda item: item.score, reverse=True)
    tail_ids = {item.evidence_id for item in reranked}
    return [*reranked, *[item for item in candidates if item.evidence_id not in tail_ids]][:limit]
