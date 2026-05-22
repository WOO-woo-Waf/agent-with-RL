"""Reusable RAG model service adapters and local vector storage."""

from agent_rl.rag.config import RAGServiceConfig, rag_env_snapshot
from agent_rl.rag.embeddings import EmbeddingProvider, OllamaEmbeddingProvider, OpenAICompatibleEmbeddingProvider
from agent_rl.rag.remote_service import RemoteRAGServiceConfig, RemoteRAGServiceManager
from agent_rl.rag.rerankers import HTTPReranker, Reranker
from agent_rl.rag.service import RAGModelService
from agent_rl.rag.types import RAGDocument, RerankResult, VectorSearchResult
from agent_rl.rag.vector_store import SQLiteVectorStore, VectorStore

__all__ = [
    "EmbeddingProvider",
    "HTTPReranker",
    "OllamaEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
    "RAGDocument",
    "RAGModelService",
    "RAGServiceConfig",
    "RemoteRAGServiceConfig",
    "RemoteRAGServiceManager",
    "RerankResult",
    "Reranker",
    "SQLiteVectorStore",
    "VectorSearchResult",
    "VectorStore",
    "rag_env_snapshot",
]
