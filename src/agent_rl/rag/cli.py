"""Command-line utilities for local and remote RAG model services."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_rl.rag.config import RAGServiceConfig, rag_env_snapshot
from agent_rl.rag.ollama_manager import OllamaServiceConfig, OllamaServiceManager
from agent_rl.rag.remote_service import RemoteRAGServiceConfig, RemoteRAGServiceManager
from agent_rl.rag.service import RAGModelService
from agent_rl.rag.types import RAGDocument


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = RAGServiceConfig.from_env(args.env)

    if args.command == "env":
        print(json.dumps(rag_env_snapshot(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "start-local":
        manager = _ollama_manager(config, args)
        manager.ensure_running()
        print(json.dumps({"status": "running", "base_url": config.ollama_base_url, "model": config.ollama_model}, indent=2))
        return 0

    if args.command == "warm":
        dim = RAGModelService(config=config).warm()
        print(json.dumps({"status": "warmed", "dimension": dim, "model": config.effective_embedding_model}, indent=2))
        return 0

    if args.command == "embed":
        vectors = RAGModelService(config=config).embed_texts(args.text)
        print(json.dumps({"count": len(vectors), "dimensions": [len(vector) for vector in vectors]}, indent=2))
        return 0

    if args.command == "index-jsonl":
        service = RAGModelService(config=config)
        documents = _load_jsonl_documents(Path(args.path))
        count = service.index_documents(documents, collection_id=args.collection_id, batch_size=args.batch_size)
        print(json.dumps({"indexed": count, "collection_id": args.collection_id}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "search":
        service = RAGModelService(config=config)
        results = service.search(
            args.query,
            story_id=args.story_id,
            evidence_types=args.evidence_type,
            collection_id=args.collection_id,
            limit=args.limit,
            rerank=not args.no_rerank,
        )
        print(json.dumps({"results": [result.__dict__ for result in results]}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "remote-health":
        manager = RemoteRAGServiceManager(RemoteRAGServiceConfig.from_env(args.env, base_url=args.base_url or None))
        print(json.dumps({"healthy": manager.is_healthy(), "base_url": manager.config.base_url}, indent=2))
        return 0

    if args.command == "remote-start":
        manager = RemoteRAGServiceManager(RemoteRAGServiceConfig.from_env(args.env, base_url=args.base_url or None))
        manager.ensure_running()
        print(json.dumps({"status": "running", "base_url": manager.config.base_url}, indent=2))
        return 0

    if args.command == "remote-stop":
        manager = RemoteRAGServiceManager(RemoteRAGServiceConfig.from_env(args.env, base_url=args.base_url or None))
        manager.stop()
        print(json.dumps({"status": "stop_requested", "base_url": manager.config.base_url}, indent=2))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG model service utilities")
    parser.add_argument("--env", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("env")

    start = subparsers.add_parser("start-local")
    start.add_argument("--ollama-executable", default="ollama")
    start.add_argument("--ollama-models", default="")

    subparsers.add_parser("warm")

    embed = subparsers.add_parser("embed")
    embed.add_argument("--text", action="append", required=True)

    index = subparsers.add_parser("index-jsonl")
    index.add_argument("--path", required=True)
    index.add_argument("--collection-id", default="default")
    index.add_argument("--batch-size", type=int, default=None)

    search = subparsers.add_parser("search")
    search.add_argument("--query", required=True)
    search.add_argument("--story-id", default="")
    search.add_argument("--evidence-type", action="append", default=[])
    search.add_argument("--collection-id", default="default")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--no-rerank", action="store_true")

    for name in ("remote-health", "remote-start", "remote-stop"):
        remote = subparsers.add_parser(name)
        remote.add_argument("--base-url", default="")
    return parser


def _ollama_manager(config: RAGServiceConfig, args: Any) -> OllamaServiceManager:
    return OllamaServiceManager(
        OllamaServiceConfig(
            executable=args.ollama_executable,
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            models_dir=args.ollama_models,
        )
    )


def _load_jsonl_documents(path: Path) -> list[RAGDocument]:
    documents: list[RAGDocument] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        documents.append(
            RAGDocument(
                document_id=str(payload.get("document_id") or payload.get("id")),
                text=str(payload["text"]),
                story_id=str(payload.get("story_id") or ""),
                evidence_type=str(payload.get("evidence_type") or "memory"),
                source=str(payload.get("source") or "jsonl"),
                related_entities=list(payload.get("related_entities") or []),
                related_plot_threads=list(payload.get("related_plot_threads") or []),
                chapter_index=payload.get("chapter_index"),
                metadata=dict(payload.get("metadata") or {}),
            )
        )
    return documents


if __name__ == "__main__":
    raise SystemExit(main())
