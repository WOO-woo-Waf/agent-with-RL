# Agent With RL

这个项目用于沉淀一套面向学习和工程复用的 Agent/RL 建模方式。

核心定位：

- 用 OOAD 定义强化学习与 LLM Agent 的共同概念：状态、观测、动作、策略、环境、奖励、轨迹、记忆、约束和评估。
- 用设计模式组织不同 Agent 架构：Strategy、Adapter、Composite、Template Method。
- 核心实现保持轻量，不封闭训练或编排能力；需要训练时对接 Gymnasium、PettingZoo、Stable-Baselines3、RLlib，需要生产 Agent 编排时对接 OpenAI Agents SDK、LangGraph、AutoGen 等。

快速验证：

```powershell
python -m pytest
```

示例运行：

```powershell
$env:PYTHONPATH="src"; python -m agent_rl.examples.gridworld
```

设计文档：

- `docs/design-architecture/agent-rl-ooad/AGENT_RL_OOAD_DESIGN_2026-05-20.md`
- `docs/design-architecture/narrative-agent-system/NARRATIVE_AGENT_DOMAIN_MODEL_2026-05-20.md`

研究笔记：

- `docs/research-notes/agent-rl/`
- `docs/research-notes/narrative-state-engine/`
