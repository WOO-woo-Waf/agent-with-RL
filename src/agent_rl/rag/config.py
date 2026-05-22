"""RAG service configuration loaded from cross-platform environment variables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_rl.config import env_snapshot, get_env, get_env_bool, get_env_float, get_env_int, load_project_env


RAG_ENV_KEYS = (
    "RAG_PROVIDER",
    "RAG_OLLAMA_BASE_URL",
    "RAG_OLLAMA_MODEL",
    "RAG_EMBEDDING_BASE_URL",
    "RAG_EMBEDDING_API_KEY",
    "RAG_EMBEDDING_MODEL",
    "RAG_EMBEDDING_DIMENSION",
    "RAG_VECTOR_DB_PATH",
    "RAG_RERANK_BASE_URL",
    "RAG_RERANK_API_KEY",
    "RAG_RERANK_MODEL",
    "RAG_RERANK_TOP_N",
    "RAG_REMOTE_ON_DEMAND",
    "RAG_REMOTE_STOP_AFTER_USE",
    "RAG_AUTO_INDEX_ON_COMMIT",
    "RAG_REMOTE_SSH_HOST",
    "RAG_REMOTE_SERVICE_DIR",
    "RAG_REMOTE_CUDA_DEVICES",
)


@dataclass(frozen=True)
class RAGServiceConfig:
    """Configuration for local Ollama or remote OpenAI-compatible RAG services."""

    provider: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3-embedding:4b"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "qwen3-embedding:4b"
    embedding_dimension: int = 2560
    vector_db_path: str = "artifacts/rag/vector_store.sqlite3"
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = "Qwen/Qwen3-Reranker-4B"
    rerank_top_n: int = 30
    request_timeout_s: float = 120.0
    batch_size: int = 32
    remote_on_demand: bool = False
    remote_stop_after_use: bool = False
    auto_index_on_commit: bool = False

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> "RAGServiceConfig":
        load_project_env(env_path, start=Path.cwd())
        provider = get_env("RAG_PROVIDER", "ollama").strip().lower()
        ollama_base_url = get_env("RAG_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        ollama_model = get_env("RAG_OLLAMA_MODEL", "qwen3-embedding:4b")
        embedding_base_url = get_env(
            "RAG_EMBEDDING_BASE_URL",
            "",
            aliases=("NOVEL_AGENT_VECTOR_STORE_URL",),
        ).rstrip("/")
        embedding_model = get_env(
            "RAG_EMBEDDING_MODEL",
            ollama_model if provider == "ollama" else "Qwen/Qwen3-Embedding-4B",
            aliases=("NOVEL_AGENT_EMBEDDING_MODEL",),
        )
        rerank_base_url = get_env(
            "RAG_RERANK_BASE_URL",
            embedding_base_url,
            aliases=("NOVEL_AGENT_VECTOR_STORE_URL",),
        ).rstrip("/")
        return cls(
            provider=provider,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=get_env("RAG_EMBEDDING_API_KEY", "", aliases=("NOVEL_AGENT_VECTOR_STORE_API_KEY",)),
            embedding_model=embedding_model,
            embedding_dimension=get_env_int(
                "RAG_EMBEDDING_DIMENSION",
                2560,
                aliases=("NOVEL_AGENT_EMBEDDING_DIMENSION",),
            ),
            vector_db_path=get_env("RAG_VECTOR_DB_PATH", "artifacts/rag/vector_store.sqlite3"),
            rerank_base_url=rerank_base_url,
            rerank_api_key=get_env("RAG_RERANK_API_KEY", "", aliases=("NOVEL_AGENT_VECTOR_STORE_API_KEY",)),
            rerank_model=get_env("RAG_RERANK_MODEL", "Qwen/Qwen3-Reranker-4B", aliases=("NOVEL_AGENT_RERANK_MODEL",)),
            rerank_top_n=get_env_int("RAG_RERANK_TOP_N", 30, aliases=("NOVEL_AGENT_RERANK_TOP_N",)),
            request_timeout_s=get_env_float("RAG_REQUEST_TIMEOUT_S", 120.0),
            batch_size=get_env_int("RAG_EMBED_BATCH_SIZE", 32, aliases=("NOVEL_AGENT_GENERATED_EMBED_BATCH_SIZE",)),
            remote_on_demand=get_env_bool("RAG_REMOTE_ON_DEMAND", False, aliases=("NOVEL_AGENT_REMOTE_EMBEDDING_ON_DEMAND",)),
            remote_stop_after_use=get_env_bool(
                "RAG_REMOTE_STOP_AFTER_USE",
                False,
                aliases=("NOVEL_AGENT_REMOTE_EMBEDDING_STOP_AFTER_USE",),
            ),
            auto_index_on_commit=get_env_bool("RAG_AUTO_INDEX_ON_COMMIT", False),
        )

    @property
    def effective_embedding_base_url(self) -> str:
        return self.ollama_base_url if self.provider == "ollama" else self.embedding_base_url

    @property
    def effective_embedding_model(self) -> str:
        return self.ollama_model if self.provider == "ollama" else self.embedding_model


def rag_env_snapshot() -> dict[str, str]:
    """Return a redacted RAG env snapshot for diagnostics."""

    return env_snapshot(RAG_ENV_KEYS)
