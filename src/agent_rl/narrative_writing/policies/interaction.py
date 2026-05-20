"""Author interaction policy."""

from __future__ import annotations

from agent_rl.domains.narrative import NarrativeTaskState
from agent_rl.narrative_writing.requests import AuthorQuestion, AuthorRequest


class BasicAuthorInteractionPolicy:
    """Asks for reference material and writing direction before execution."""

    def missing_questions(self, request: AuthorRequest, state: NarrativeTaskState | None) -> list[AuthorQuestion]:
        questions: list[AuthorQuestion] = []
        if not request.references and state is None:
            questions.append(
                AuthorQuestion(
                    question_id="reference_material",
                    prompt="请提供参考小说/原文片段，或说明这是一个全新原创任务。",
                    reason="小说 Agent 需要 canon、风格或世界观材料作为初始观测。",
                )
            )
        if not request.writing_direction.strip():
            questions.append(
                AuthorQuestion(
                    question_id="writing_direction",
                    prompt="请说明这次写作方向：要推进什么剧情、角色关系或章节目标？",
                    reason="作者意图必须结构化成 ChapterBlueprint 和 AuthorConstraint。",
                )
            )
        return questions
