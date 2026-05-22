"""File-backed evaluation report persistence for narrative writing."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from agent_rl.domains.narrative import EvaluationReport
from agent_rl.narrative_writing.serialization import to_jsonable


class FileNarrativeEvaluationRepository:
    """Stores evaluation reports as audit artifacts."""

    def __init__(self, root: str | Path = Path("artifacts") / "narrative-evaluations") -> None:
        self.root = Path(root)

    def save_reports(self, story_id: str, reports: Sequence[EvaluationReport], *, run_id: str = "") -> list[Path]:
        paths: list[Path] = []
        base_dir = self.root / _safe_path_part(story_id or "story")
        for report in reports:
            name = f"{_safe_path_part(report.report_id)}{_suffix(run_id)}.json"
            path = base_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "story_id": story_id,
                "run_id": run_id,
                "report": to_jsonable(report),
            }
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            temp.replace(path)
            paths.append(path)
        return paths


def _safe_path_part(value: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())
    return clean[:120] or "item"


def _suffix(run_id: str) -> str:
    return f"-{_safe_path_part(run_id)}" if run_id else ""


__all__ = ["FileNarrativeEvaluationRepository"]
