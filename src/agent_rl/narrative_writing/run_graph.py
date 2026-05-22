"""Small run-graph helpers for candidate-only parallel work."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class NarrativeTaskNode:
    """One independent node in a candidate run graph."""

    node_id: str
    task_type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeTaskResult:
    """Output of a candidate task node."""

    node_id: str
    success: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class NarrativeRunGraph:
    """A simple graph container for independent candidate tasks."""

    graph_id: str
    nodes: list[NarrativeTaskNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ParallelToolExecutor:
    """Executes independent candidate tasks without mutating canonical state."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max(1, max_workers)

    def run(
        self,
        nodes: Iterable[NarrativeTaskNode],
        handler: Callable[[NarrativeTaskNode], dict[str, Any]],
    ) -> list[NarrativeTaskResult]:
        items = list(nodes)
        if not items:
            return []
        results: list[NarrativeTaskResult] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as executor:
            futures = {executor.submit(handler, node): node for node in items}
            for future in as_completed(futures):
                node = futures[future]
                try:
                    results.append(NarrativeTaskResult(node_id=node.node_id, success=True, payload=future.result()))
                except Exception as exc:  # noqa: BLE001 - candidate task failures must be collected.
                    results.append(NarrativeTaskResult(node_id=node.node_id, success=False, error=str(exc)))
        return sorted(results, key=lambda item: item.node_id)


__all__ = ["NarrativeRunGraph", "NarrativeTaskNode", "NarrativeTaskResult", "ParallelToolExecutor"]
