"""File-backed author conversation repository."""

from __future__ import annotations

import json
import re
from pathlib import Path

from agent_rl.domains.narrative import AuthorConversation
from agent_rl.narrative_writing.serialization import from_jsonable, to_jsonable


class FileNarrativeConversationRepository:
    """Stores long-running author conversations as JSON snapshots."""

    def __init__(self, root: str | Path = Path("artifacts") / "narrative-conversation") -> None:
        self.root = Path(root)

    def save_conversation(self, conversation: AuthorConversation) -> Path:
        path = self._story_dir(conversation.story_id) / f"{_safe_path_part(conversation.session_id)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_jsonable(conversation), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load_conversation(self, session_id: str, *, story_id: str = "") -> AuthorConversation | None:
        if story_id:
            path = self._story_dir(story_id) / f"{_safe_path_part(session_id)}.json"
            if not path.exists():
                return None
        else:
            candidates = sorted(self.root.glob(f"*/{_safe_path_part(session_id)}.json"))
            if not candidates:
                return None
            path = candidates[-1]
        return from_jsonable(AuthorConversation, json.loads(path.read_text(encoding="utf-8")))

    def _story_dir(self, story_id: str) -> Path:
        return self.root / _safe_path_part(story_id or "story")


def _safe_path_part(value: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())
    return clean[:120] or "item"
