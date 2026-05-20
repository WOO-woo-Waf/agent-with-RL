# 面向大模型 API 的 Agent 系统与强化学习指导

本文档用于明确本项目的核心目标：构建一个更好的 agent 系统。这里的 agent 主要指“围绕大模型 API、工具调用、记忆、规划、执行和评估组成的应用系统”，而不是狭义上在强化学习中被直接训练的策略网络。强化学习仍然重要，但它在这个项目中更适合被看作一种“评估、优化和经验利用的方法论”，而不是系统的唯一出发点。

## 1. 两种 Agent 的区别

### 1.1 大模型 API 型 Agent

大模型 API 型 agent 是一个应用运行时。它本身通常不训练模型参数，而是调用已有的大模型完成理解、规划、推理、工具选择和文本生成。

典型组成：

- 大模型 API：负责推理、规划、生成和工具调用决策。
- Prompt / System Instruction：定义角色、边界、任务风格和输出格式。
- Tool Layer：搜索、读写文件、运行代码、访问数据库、调用业务 API。
- Memory：短期上下文、长期经验、用户偏好、任务历史。
- Planner：将目标拆成步骤，决定先做什么、后做什么。
- Executor：执行工具调用、处理失败、重试、合并结果。
- Evaluator：检查结果是否完成目标，例如测试是否通过、答案是否可靠、输出是否符合约束。
- Orchestrator：管理单 agent、多 agent、子任务、上下文传递和终止条件。

这类 agent 的优化重点不是“训练策略网络”，而是：

- 更好的任务分解。
- 更可靠的工具调用。
- 更少的无效 token 消耗。
- 更稳定的上下文管理。
- 更强的错误恢复能力。
- 更可观察、可回放、可评估的执行轨迹。
- 更安全的权限、边界和人工确认机制。

### 1.2 强化学习中的 Agent

强化学习中的 agent 是一个学习主体。它和环境交互，根据状态选择动作，从奖励中学习策略。

典型定义：

- 状态空间 `S`：环境或任务当前所处的状态。
- 动作空间 `A`：agent 可以执行的动作集合。
- 转移函数 `P(s' | s, a)`：执行动作后状态如何变化。
- 奖励函数 `R(s, a, s')`：动作产生的收益或惩罚。
- 策略 `pi(a | s)`：在状态下选择动作的规则。
- 价值函数 `V(s)` / `Q(s, a)`：衡量状态或动作的长期收益。
- 轨迹 `trajectory`：一次交互过程中的状态、动作、奖励序列。

强化学习里的 agent 更像一个“可训练决策策略”。它可能是神经网络、表格策略、树搜索策略，也可能是由大模型参与的复合策略。

### 1.3 最关键的差异

| 维度 | 大模型 API 型 Agent | 强化学习 Agent |
|---|---|---|
| 核心能力来源 | 预训练大模型、工具、上下文工程 | 通过环境反馈学习策略 |
| 是否必须训练 | 不必须 | 通常需要训练或策略优化 |
| 主要问题 | 如何规划、调用工具、管理上下文、可靠执行 | 如何定义状态、动作、奖励并学习最优策略 |
| 典型优化方式 | Prompt、工具设计、记忆、评估器、工作流 | DQN、PPO、SAC、RLHF、离线 RL |
| 输出形态 | 一个可运行的软件系统 | 一个决策策略或训练流程 |
| 适合本项目的位置 | 主体系统 | 优化层、评估层、经验学习层 |

本项目不应该把两者混成一个概念。更好的理解是：

> LLM API agent 是产品和运行时，强化学习是改进这个运行时决策质量的方法之一。

## 2. 如何把强化学习思想引入 LLM Agent 系统

虽然本项目的 agent 不是传统 RL agent，但依然可以借用 RL 的抽象来设计系统。

### 2.1 映射关系

| RL 概念 | LLM Agent 系统中的对应物 |
|---|---|
| 环境 `Env` | 用户、文件系统、浏览器、工具、数据库、代码仓库、其他 agent |
| 观察 `Observation` | 用户请求、工具返回、报错、当前文件、检索结果 |
| 状态 `State` | 目标、计划、上下文摘要、记忆、约束、已完成步骤 |
| 动作 `Action` | 回复、搜索、读文件、写文件、运行测试、调用 API、委派子 agent、请求用户输入 |
| 奖励 `Reward` | 任务完成、测试通过、用户确认、成本降低、错误减少、安全约束满足 |
| 策略 `Policy` | LLM 决策、规则系统、工具选择器、计划器、路由器 |
| 轨迹 `Trajectory` | 一次任务从开始到完成的完整执行日志 |
| 价值评估 `Value` | 某一步动作是否值得继续、是否应该回滚、是否需要问用户 |

这个映射不是为了立即训练模型，而是为了让系统从第一天开始就具备“可度量、可复盘、可改进”的结构。

### 2.2 对 LLM Agent 最有用的 RL 思想

1. 明确定义状态

不要把完整聊天记录直接等同于状态。更好的状态应该包含：

- 当前目标。
- 已知约束。
- 已完成步骤。
- 失败尝试。
- 可用工具。
- 当前风险。
- 需要用户确认的事项。

2. 明确定义动作空间

动作空间不能无限散开。建议先定义一组结构化动作：

```text
answer
ask_user
search
read_file
write_file
run_command
run_test
call_api
delegate_agent
reflect
finish
```

动作越清晰，日志越好分析，后续越容易训练工具选择器或工作流策略。

3. 明确定义奖励

奖励不一定是一个单独分数，可以是多维评价：

```text
task_success
test_pass_rate
user_satisfaction
cost
latency
tool_error_count
context_length
safety_violation
manual_intervention_count
```

更实际的总分可以这样设计：

```text
score =
  5.0 * task_success
+ 2.0 * test_pass_rate
+ 1.0 * user_satisfaction
- 1.0 * safety_violation
- 0.5 * tool_error_count
- 0.2 * cost_normalized
- 0.2 * latency_normalized
```

4. 记录轨迹

没有轨迹，就没有强化学习，也没有系统改进。每一步都应该记录：

```json
{
  "task_id": "task_001",
  "step": 3,
  "observation": "...",
  "state_summary": "...",
  "candidate_actions": ["read_file", "run_test", "ask_user"],
  "chosen_action": "run_test",
  "action_args": {},
  "result": "...",
  "reward_signals": {
    "test_passed": false,
    "tool_error": true,
    "cost": 0.02
  },
  "reflection": "测试失败，下一步应读取失败用例。"
}
```

5. 先做评估，再做训练

对于 LLM agent，优先级应该是：

```text
可运行闭环
→ 可记录轨迹
→ 可自动评估
→ 可对比不同策略
→ 可做离线分析
→ 再考虑训练或强化学习优化
```

## 3. 推荐系统架构

建议把系统分成以下层次：

```text
User / Task
   ↓
Task Parser
   ↓
Agent Runtime
   ├─ State Manager
   ├─ Memory Manager
   ├─ Planner
   ├─ Policy / Router
   ├─ Tool Executor
   ├─ Reflection Module
   └─ Safety Guard
   ↓
Environment / Tools
   ↓
Observation / Result
   ↓
Evaluator
   ↓
Trajectory Store
   ↓
Optimizer / Trainer
```

### 3.1 Task Parser

负责把用户请求转成结构化任务：

```json
{
  "goal": "生成项目文档",
  "constraints": ["中文", "放在 docs 文件夹", "初始化 git"],
  "expected_outputs": ["markdown 文档", "git repository"],
  "risk_level": "low"
}
```

### 3.2 State Manager

负责维护当前状态，不只是堆上下文。

推荐状态结构：

```json
{
  "goal": "...",
  "current_plan": [],
  "completed_steps": [],
  "open_questions": [],
  "known_constraints": [],
  "recent_observations": [],
  "available_tools": [],
  "failure_history": []
}
```

### 3.3 Planner

Planner 不应该每一步都重写全部计划。推荐分两层：

- High-level plan：任务级计划，变化较慢。
- Step-level policy：下一步动作，变化较快。

例如：

```text
High-level plan:
1. 理解用户目标
2. 检查项目现状
3. 创建文档
4. 初始化 git
5. 验证结果

Step-level action:
read_file -> write_file -> git_init -> git_status
```

### 3.4 Tool Executor

工具层要做到：

- 每个工具有明确 schema。
- 每次调用有输入、输出、错误、耗时。
- 高风险工具需要权限或策略约束。
- 工具失败后返回结构化错误，而不是只把异常塞回上下文。

### 3.5 Memory Manager

记忆可以分为三类：

- 工作记忆：当前任务上下文。
- 项目记忆：项目结构、规范、常用命令、历史决策。
- 经验记忆：过去任务中有效或失败的策略。

记忆不是越多越好。推荐记录“可复用经验”：

```json
{
  "pattern": "Windows PowerShell 中文环境",
  "lesson": "读取中文文件时显式设置 UTF-8 输出，避免误判编码。",
  "confidence": 0.9,
  "source_task": "task_2026_05_18_001"
}
```

### 3.6 Evaluator

Evaluator 是系统变好的关键。它可以由多种方式组成：

- 自动测试：代码任务最可靠。
- 静态检查：lint、类型检查、格式检查。
- 规则检查：文件是否存在、字段是否完整。
- LLM judge：适合评估主观质量，但不能作为唯一标准。
- 用户反馈：最高价值，但成本高，适合作为稀疏奖励。

## 4. 强化学习在本项目中的合理位置

不要一开始就训练大模型。更合理的路线是逐层引入学习能力。

### 4.1 第一阶段：无训练 Agent

目标：做出稳定可用的 agent runtime。

方法：

- 固定 prompt。
- 固定工具 schema。
- 固定工作流。
- 人工设计规则。
- 完整记录轨迹。

这一阶段的成功标准：

- 能稳定完成一类任务。
- 每次执行都能回放。
- 失败原因可以定位。

### 4.2 第二阶段：经验驱动 Agent

目标：让 agent 从历史任务中变聪明。

方法：

- 检索相似任务。
- 复用成功计划。
- 避免历史失败动作。
- 建立任务模板和技能库。

这不一定需要 RL，但已经具备“经验改进”的味道。

### 4.3 第三阶段：策略选择优化

目标：训练或优化“小策略”，而不是训练整个大模型。

可优化对象：

- 选择哪个工具。
- 是否需要先搜索。
- 是否需要问用户。
- 是否委派子 agent。
- 下一步是读文件、写文件还是运行测试。
- 失败后重试、反思还是换策略。

适合方法：

- 规则 + 打分。
- 多臂老虎机。
- 上下文 bandit。
- 离线学习。
- 偏好优化。

### 4.4 第四阶段：RL 优化 Agent 行为

当你已经有足够轨迹和可靠奖励后，再考虑：

- PPO / GRPO：优化可训练策略模型。
- RLHF / RLAIF：用人类或 AI 偏好训练奖励模型。
- Offline RL：从历史轨迹学习。
- Decision Transformer：把任务轨迹作为序列建模。

这时 RL 的作用是优化 agent 的决策过程，而不是替代整个 agent 系统。

## 5. 多 Agent 系统设计

多 agent 不应该只是“多开几个聊天窗口”。它需要明确分工、状态同步和结果合并。

### 5.1 推荐角色

- Planner Agent：拆解任务、制定路线、定义验收标准。
- Worker Agent：执行具体子任务，例如写代码、查资料、生成文档。
- Reviewer Agent：检查结果、找漏洞、指出风险。
- Evaluator Agent：运行测试、打分、生成反馈。
- Coordinator Agent：分派任务、处理冲突、合并输出。

### 5.2 多 Agent 与强化学习的关系

多 agent 系统可以借鉴多智能体强化学习中的思想：

- 每个 agent 有局部观察。
- 全局任务有共享状态。
- 团队有共同奖励。
- 个体也可以有贡献奖励。
- 通信本身也是一种动作。

但在 LLM API 型 agent 系统中，优先落地的是工程机制：

- 任务边界清晰。
- 上下文隔离。
- 输出格式统一。
- 合并策略可靠。
- 失败时能回收和重试。

## 6. 项目落地路线

### 6.1 MVP

先做一个单 agent 闭环：

```text
input task
→ parse task
→ build state
→ plan
→ choose action
→ execute tool
→ evaluate
→ log trajectory
→ finish
```

最小动作空间：

```text
answer
ask_user
read_file
write_file
run_command
run_test
search
finish
```

最小数据表：

- `tasks`
- `steps`
- `tool_calls`
- `evaluations`
- `memories`

### 6.2 第二版

加入：

- 长期记忆检索。
- 任务模板。
- 失败恢复策略。
- 自动评价器。
- 成本统计。
- 策略对比实验。

### 6.3 第三版

加入：

- 多 agent 协作。
- Planner / Worker / Reviewer / Evaluator 分工。
- 轨迹数据集导出。
- 离线分析 dashboard。
- 对工具选择策略做 bandit 或离线学习。

### 6.4 第四版

加入：

- 奖励模型。
- 偏好数据收集。
- 小策略模型训练。
- RL 或类 RL 优化。
- 自动课程生成。

## 7. 评价指标

一个更好的 agent 系统，不应该只看“回答像不像人”。建议使用以下指标：

| 指标 | 含义 |
|---|---|
| Task Success Rate | 任务完成率 |
| First-pass Success | 第一次完成就正确的比例 |
| Tool Error Rate | 工具调用失败率 |
| Recovery Rate | 出错后成功恢复的比例 |
| Average Steps | 平均完成步数 |
| Cost per Task | 单任务成本 |
| Latency | 完成耗时 |
| Human Intervention Rate | 需要用户介入的比例 |
| Regression Rate | 修改后引入新问题的比例 |
| User Acceptance Rate | 用户接受结果的比例 |

对代码型 agent，优先指标应该是：

```text
测试通过率 > 任务完成率 > diff 可控性 > 成本 > 速度
```

## 8. 推荐阅读与论文

### 8.1 强化学习基础

- Sutton, R. S. and Barto, A. G. Reinforcement Learning: An Introduction.  
  https://incompleteideas.net/book/the-book-2nd.html

- Mnih et al. Human-level control through deep reinforcement learning. Nature, 2015.  
  https://www.nature.com/articles/nature14236

- Schulman et al. Proximal Policy Optimization Algorithms, 2017.  
  https://arxiv.org/abs/1707.06347

- Haarnoja et al. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, 2018.  
  https://arxiv.org/abs/1801.01290

- Chen et al. Decision Transformer: Reinforcement Learning via Sequence Modeling, 2021.  
  https://arxiv.org/abs/2106.01345

### 8.2 LLM Agent 与工具调用

- Yao et al. ReAct: Synergizing Reasoning and Acting in Language Models, 2022.  
  https://arxiv.org/abs/2210.03629

- Shinn et al. Reflexion: Language Agents with Verbal Reinforcement Learning, 2023.  
  https://arxiv.org/abs/2303.11366

- Schick et al. Toolformer: Language Models Can Teach Themselves to Use Tools, 2023.  
  https://arxiv.org/abs/2302.04761

- Park et al. Generative Agents: Interactive Simulacra of Human Behavior, 2023.  
  https://arxiv.org/abs/2304.03442

- Wang et al. Voyager: An Open-Ended Embodied Agent with Large Language Models, 2023.  
  https://arxiv.org/abs/2305.16291

### 8.3 Agent 与强化学习结合

- Agent Lightning: Train ANY AI Agents with Reinforcement Learning.  
  https://arxiv.org/abs/2508.03680

- The Landscape of Agentic Reinforcement Learning for LLMs: A Survey.  
  https://arxiv.org/abs/2509.02547

- AGILE: A Novel Reinforcement Learning Framework of LLM Agents.  
  https://arxiv.org/abs/2405.14751

### 8.4 工程框架参考

- OpenAI Agents SDK：agent、tools、handoffs、guardrails 的工程抽象。  
  https://openai.github.io/openai-agents-python/

- LangChain Agents：工具调用、agent runtime、工作流编排参考。  
  https://python.langchain.com/docs/concepts/agents/

- AutoGen：多 agent 协作框架参考。  
  https://microsoft.github.io/autogen/

- Model Context Protocol：工具、资源和上下文协议参考。  
  https://modelcontextprotocol.io/

- Gymnasium：强化学习环境接口参考。  
  https://gymnasium.farama.org/

- PettingZoo：多智能体强化学习环境接口参考。  
  https://pettingzoo.farama.org/

## 9. 本项目的建议定位

本项目最适合定位为：

> 一个面向大模型 API 的 agent runtime，通过强化学习思想组织状态、动作、奖励、轨迹和评估，并逐步引入经验学习与策略优化，最终做出更可靠、更可复盘、更能持续进化的 agent 系统。

短期目标不是训练一个大模型，而是先做出：

- 结构化任务表示。
- 清晰动作空间。
- 可靠工具执行。
- 完整轨迹记录。
- 自动评价体系。
- 可复用记忆和技能。
- 多 agent 协作机制。

中长期目标才是：

- 从轨迹中学习。
- 优化工具选择。
- 优化规划策略。
- 建立奖励模型。
- 使用 RL 或偏好优化提升 agent 决策质量。

这条路线能避免一开始陷入“训练什么模型”的泥潭，而是先把 agent 系统本身做成一个可以观察、可以评估、可以进化的工程平台。

## 10. 当前先进 Agent 能力学习路线

本项目后续学习和实现不要只盯着 RL。RL 提供“状态、动作、奖励、轨迹、策略优化”的底层语言，但更好的 Agent 系统还需要同时吸收 RAG、长期记忆、记忆压缩、多 Agent 编排、评估、安全约束和可观测性。

### 10.1 Agent Runtime

需要重点学习：

- OpenAI Agents SDK：agents、tools、handoffs、guardrails、sessions、tracing。
- LangGraph：stateful graph、durable execution、checkpoint、human-in-the-loop。
- AutoGen / CrewAI：多 Agent 角色协作、任务分派、汇总与冲突处理。

本项目代码中对应：

- `AgentRuntime`
- `ReActAgent`
- `PlanAndExecuteAgent`
- `MultiAgentCoordinator`

### 10.2 RAG 与知识检索

RAG 在 Agent 系统中不是简单“问向量库”，而是 observation builder：它决定 Agent 每一步看到哪些外部事实。

后续应补充：

- `Retriever`
- `Reranker`
- `ContextBuilder`
- `Citation`
- retrieval eval

重点能力：

- query rewriting
- multi-hop retrieval
- hybrid search
- reranking
- grounded answer checking
- agentic RAG：由 Agent 决定何时检索、检索什么、如何验证检索结果。

### 10.3 长期记忆与记忆压缩

Agent 的记忆至少分为：

- 工作记忆：当前任务上下文。
- 情节记忆：过去任务轨迹。
- 语义记忆：稳定事实、用户偏好、项目知识。
- 程序性记忆：可复用技能、流程和策略。

需要重点研究：

- mem0：面向 assistant/agent 的持久化上下文记忆。
- Letta / MemGPT：agent 自主管理长期记忆与上下文窗口。
- LangGraph / LangChain memory：thread state、store、checkpoint。

后续代码建议：

- `MemoryCompressor`
- `MemoryWritePolicy`
- `MemoryRetrievalPolicy`
- `MemoryValidator`
- `MemoryDecayPolicy`

记忆不是越多越好。好的记忆系统要能写入、检索、压缩、纠错、遗忘。

### 10.4 评估、奖励与策略优化

先把 Agent 的行为变成数据，再考虑强化学习：

- 记录 `Trajectory`。
- 记录 action、observation、reward、cost、latency、error、human feedback。
- 用 evaluator 判断任务成功、风险、成本和质量。
- 从轨迹中学习小策略，而不是一开始训练整个 LLM。

优先优化的小策略：

- 工具选择策略。
- RAG 检索策略。
- 停止策略。
- 反思/重试策略。
- 子 Agent 路由策略。
- 上下文预算分配策略。

### 10.5 项目原则

本项目要坚持：

- 核心抽象自己掌握。
- 训练算法借助 SB3、RLlib、CleanRL。
- RL 环境接口对齐 Gymnasium、PettingZoo。
- 生产 Agent 编排参考 OpenAI Agents SDK、LangGraph、AutoGen、CrewAI。
- RAG 和记忆系统优先薄适配成熟框架，再抽象自己的统一接口。
- 所有实验都要产出可回放轨迹和可比较指标。

## 11. 小说创作作为第一个复杂 Agent 场景

`D:\buff\narrative-state-engine\docs` 中的小说续写项目应作为本项目的第一个复杂场景来吸收，而不是简单照搬。

核心判断：

- 保留小说领域：角色、关系、世界规则、剧情线、伏笔、场景、风格、作者计划、证据、记忆、校验。
- 不保留前端复杂度。
- 不把 `NovelAgentState` 做成封闭大对象，而是抽象为 `NarrativeTaskState` + `NarrativeScenarioAdapter`。
- 用当前项目的 Agent/RL 抽象重新组织：状态、观测、动作、策略、轨迹、奖励/评估、记忆、RAG、guardrail。

当前已整合：

- `src/agent_rl/core/`：通用 Agent/RL 核心抽象、runtime、通用 policy 和架构 helper。
- `src/agent_rl/domains/narrative.py`：小说场景领域模型。
- `src/agent_rl/narrative_writing/`：可运行的小说写作 Agent，实现作者交互、检索、规划、生成、抽取、评估、提交和记忆压缩闭环，并按 OOAD 分成 DTO、端口、策略、场景适配器和应用服务。
- `docs/design-architecture/narrative-agent-system/NARRATIVE_AGENT_DOMAIN_MODEL_2026-05-20.md`：小说 Agent 场景建模与改进设计。
- `docs/design-architecture/core-package-layering/AGENT_RL_CORE_LAYERING_DESIGN_2026-05-20.md`：核心包分层规范。

小说场景的目标不是“让模型多写字”，而是建立一个可控创作环境：

```text
作者输入
-> 意图/约束抽取
-> 记忆与证据检索
-> 章节/场景规划
-> 草稿生成
-> 状态变化抽取
-> 角色/剧情/风格/世界/作者意图评估
-> 修复或人工确认
-> canonical state 提交
-> 记忆压缩与索引更新
```

这个场景非常适合后续研究：

- Agent runtime 如何承载长任务。
- RAG 如何从普通问答升级为任务感知检索。
- mem0/Letta 式长期记忆如何和状态机结合。
- RL 如何优化检索、修复、停止、分支选择、记忆写入等小策略。
