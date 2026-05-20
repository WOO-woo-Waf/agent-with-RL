# Agent/RL OOAD 建模设计

## Background

当前项目已有两份核心材料：

- `docs/research-notes/agent-rl/Agent与强化学习关系 (2026-03-18).md`：把 LLM Agent 形式化为带记忆、工具 options、约束和分层控制的 POMDP/SMDP。
- `docs/agent-rl-system-guide.md`：明确项目定位是“大模型 API 型 Agent runtime + 强化学习思想”，不是一开始就训练完整策略网络。

本次实现把这些概念落为 Python 代码骨架，用于持续学习、比较框架、记录轨迹和后续接入训练。

## Goal

1. 用 OOAD 定义共同领域对象：`Goal`、`Observation`、`Action`、`Reward`、`Transition`、`Trajectory`、`Policy`、`Environment`、`MemoryStore`、`Guardrail`。
2. 用设计模式表达 Agent 架构：
   - Strategy：`Policy` / `SequencePolicy` / `GreedyActionPolicy`
   - Template Method：`AgentRuntime.run()`
   - Adapter：`GymnasiumEnvAdapter`、`PettingZooAECAdapter`、`openai_function_tool`
   - Composite：`MultiAgentCoordinator`
3. 保持实现开放：核心抽象轻量自有，训练和生产编排优先借助权威开源/官方框架。

## Non-goals

- 不重写 PPO、DQN、SAC、RLHF、RLAIF 等训练算法。
- 不复刻 OpenAI Agents SDK、LangGraph、AutoGen、CrewAI 的完整 runtime。
- 不把当前最小 GridWorld 示例当成最终 RL 环境标准；它只是测试领域模型闭环。

## Authoritative References

### RL 基础接口

- Gymnasium 是单 Agent RL 环境 API 的事实标准。本项目的 `Environment` 参考其 `reset()` / `step()` agent-environment loop，但返回项目自己的领域对象。  
  Source: https://gymnasium.farama.org/introduction/basic_usage/
- PettingZoo 面向多 Agent RL。本项目的 `PettingZooAECAdapter` 参考其 AEC API：`agent_iter()`、`last()`、`step(action)`。  
  Source: https://pettingzoo.farama.org/content/basic_usage/
- Stable-Baselines3 是 PyTorch 上的 RL 算法实现集合，可作为后续 PPO/DQN/SAC 等训练入口。  
  Source: https://stable-baselines3.readthedocs.io/en/master/
- Ray RLlib 是可扩展 RL 库，适合后续多 Agent、离线数据、分布式训练和生产工作负载。  
  Source: https://docs.ray.io/en/latest/rllib/index.html
- CleanRL 提供单文件 RL 算法实现，适合学习算法细节和做最小可读 baseline。  
  Source: https://docs.cleanrl.dev/

### Agent 编排接口

- OpenAI Agents SDK 的核心 primitives 是 agents、tools、handoffs、guardrails、sessions、tracing。本项目只做薄适配，不复制其生产 runtime。  
  Source: https://openai.github.io/openai-agents-python/
- LangGraph 是低层 orchestration runtime，适合 long-running、stateful、human-in-the-loop、durable execution 的 Agent。  
  Source: https://docs.langchain.com/oss/python/langgraph/overview
- Microsoft AutoGen 提供 AgentChat、Core、Extensions；可作为事件驱动 multi-agent 系统参考。  
  Source: https://microsoft.github.io/autogen/stable/index.html
- CrewAI 面向 collaborative agents、crews、flows，也可作为多 Agent 角色分工与流程编排参考。  
  Source: https://docs.crewai.com/

## Recommended Design

### Core Domain Layer

核心层位于 `src/agent_rl/concepts.py`，只定义领域对象和 Protocol，不绑定任何框架：

- `Environment` 对齐 Gymnasium 的 `reset()` / `step()` 思路，但返回项目自己的 `Observation` 和 `Transition`。
- `Policy` 是策略接口，可以由规则、LLM、训练模型、planner 或 router 实现。
- `Trajectory` 是最重要的数据资产，用于 replay、评估、偏好标注、离线学习和策略对比。
- `MemoryStore` 是 belief state 的工程化近似，当前有 `InMemoryStore`，后续可换 SQL/vector store。

### Runtime Layer

`src/agent_rl/runtime.py` 的 `AgentRuntime` 是最小闭环：

1. reset environment
2. 构造 `AgentState`
3. policy 选择 `Decision`
4. guardrails 校验
5. environment step
6. 写入 trajectory 和 memory
7. 直到 terminated/truncated/stopped/max_steps

这个 runtime 是教学和统一轨迹记录用的，不替代 LangGraph/OpenAI Agents SDK 的生产 runtime。

### Architecture Layer

`src/agent_rl/architectures.py` 显式表达几种 Agent 架构：

- `ReActAgent`：单层在线闭环 controller，每一步重新观察和决策。
- `PlanAndExecuteAgent`：高层 planner 先产出 plan，低层 executor 顺序执行。
- `MultiAgentCoordinator`：router 选择子 policy，体现分层/组合式 multi-agent。

### Integration Layer

`src/agent_rl/integrations.py` 只做 Adapter：

- `GymnasiumEnvAdapter`：接 Gymnasium 风格环境。
- `PettingZooAECAdapter`：接 PettingZoo AEC 多 Agent 环境。
- `openai_function_tool()`：把本项目 `Tool` 包装成 OpenAI Agents SDK function tool。

训练时优先把环境交给 SB3/RLlib/CleanRL；生产编排时优先把 agent loop 交给 OpenAI Agents SDK、LangGraph、AutoGen 或 CrewAI。

## Official Comparison

| 领域 | 本项目实现 | 权威/开源实现 | 取舍 |
|---|---|---|---|
| RL 环境接口 | `Environment` Protocol + `GridWorldEnv` | Gymnasium | 本项目保持概念清晰；真实训练接 Gymnasium |
| 多 Agent RL | `MultiAgentCoordinator` + `PettingZooAECAdapter` | PettingZoo / RLlib | 本项目记录和协调；复杂 MARL 交给外部框架 |
| RL 算法 | 暂不实现 | SB3 / RLlib / CleanRL | 避免低质量重复造轮子 |
| Agent runtime | `AgentRuntime` 最小闭环 | OpenAI Agents SDK / LangGraph / AutoGen / CrewAI | 本项目用于学习、抽象、轨迹统一；生产使用成熟 runtime |
| Tools/Handoff | `Action(kind="tool")` / `agent_id` / adapter | OpenAI Agents SDK tools/handoffs | 概念对齐，执行细节交给 SDK |
| Guardrails | `Guardrail` Protocol | OpenAI Agents SDK Guardrails / CrewAI Guardrails | 先定义约束接口，后续接入官方 guardrail |
| Tracing/Eval | `Trajectory` | LangSmith / OpenAI tracing / 自建 eval | 先保留本地可回放格式，再接平台 |

## Change List

- 新增 Python 项目配置：`pyproject.toml`
- 新增工程基础文件：`.gitignore`、`.editorconfig`、`README.md`
- 新增核心包：`src/agent_rl/`
- 新增测试：`tests/`
- 新增本设计文档

## Risks And Constraints

- 可选依赖包名和 API 会随版本变化，适配器必须保持薄层，不要让核心领域模型依赖外部框架。
- OpenAI Agents SDK、LangGraph 等 runtime 的抽象粒度不同，不应强行映射成同一种 class hierarchy。
- 轨迹结构一旦用于训练数据集，字段变更需要版本化。
- 当前 `GridWorldEnv` 只覆盖离散动作和单 Agent，不能代表真实 LLM Agent 的异步工具调用和长时任务。

## Verification Plan

- 运行 `python -m pytest` 验证核心闭环和架构样例。
- 运行 `PYTHONPATH=src python -m agent_rl.examples.gridworld` 验证最小 demo。
- 后续接入 Gymnasium 时，增加 contract test：reset/step/action_space/terminated/truncated。
- 后续接入 OpenAI Agents SDK 时，增加 fake tool test，避免 CI 依赖真实 API key。

## Next Steps

1. 增加 `Trajectory` JSONL 导出与版本号，作为后续 offline RL/eval 数据基础。
2. 增加 Gymnasium 自定义环境 wrapper，让 `GridWorldEnv` 可直接被 SB3 训练。
3. 增加 OpenAI Agents SDK 示例：一个本地 `Tool` -> `function_tool` -> handoff 的最小样例。
4. 增加 LangGraph 示例：把 `AgentState` 映射为 graph state，把 `TrajectoryStep` 写入 tracing/eval。
