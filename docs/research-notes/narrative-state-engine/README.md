# Narrative State Engine Notes

这个目录整理 `D:\buff\narrative-state-engine\docs` 中和小说场景、领域建模、RAG、记忆、状态机、任务闭环相关的内容。

当前整合结论见：

- `docs/design-architecture/narrative-agent-system/NARRATIVE_AGENT_DOMAIN_MODEL_2026-05-20.md`

整合原则：

- 不迁移前端设计。
- 保留小说场景、任务、状态、记忆、检索、评估和作者规划相关概念。
- 用当前项目的 Agent/RL 视角重新定义：小说写作环境是一个 stateful scenario adapter，RL/RAG/记忆用于提升可控性、可评估性和持续改进能力。
