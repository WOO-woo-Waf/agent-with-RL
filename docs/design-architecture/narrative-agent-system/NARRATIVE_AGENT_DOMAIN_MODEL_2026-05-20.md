# 小说 Agent 场景建模与改进设计

## Background

来源项目：`D:\buff\narrative-state-engine\docs`

该项目的核心判断是正确的：小说续写不是单纯文本生成，而是一个 `stateful agent architecture` 问题。它已经提出：

- `NovelAgentState` 作为统一运行态。
- `Story State`、`Chapter State`、`Style State`、`Validation State`、`MemoryBundle`。
- `read state -> retrieve memory -> plan -> draft -> extract -> validate -> commit/rollback`。
- 分层长期记忆、RAG、作者剧情规划、角色卡、风格画像、证据包、状态提交。

当前问题不是“没有概念”，而是概念太多但尚未被统一为一个清晰的 Agent 场景协议。更好的方向是：保留小说场景和任务部分，把它重新收敛成当前项目里的一个 `NarrativeScenarioAdapter`。

## Core Position

小说写作系统应该被定义为：

```text
Narrative Agent = Agent Runtime + Narrative Scenario Adapter + Canonical State + Memory/RAG + Evaluators + Human Confirmation
```

强化学习在这里不是先训练大模型，而是提供决策系统语言：

- 状态：故事 canon、角色动态、作者约束、风格画像、压缩记忆。
- 观测：本轮用户输入、检索证据、当前章节状态、运行信号。
- 动作：分析、检索、规划、生成、抽取、校验、修复、提交、回滚、请求作者确认。
- 奖励/评价：作者满意度、角色一致性、剧情推进、风格匹配、成本、风险。
- 轨迹：一次从作者输入到状态提交的完整运行记录。

## Frontier References

本设计借鉴但不复制这些框架：

- OpenAI Agents SDK：agents、tools、handoffs、guardrails、sessions、tracing。适合作为工具调用、handoff、guardrail 和 trace 的参考。  
  https://openai.github.io/openai-agents-python/
- LangGraph：long-running stateful agents、durable execution、human-in-the-loop、memory、streaming、tracing。适合作为长任务状态图和 checkpoint 参考。  
  https://docs.langchain.com/oss/python/langgraph/overview
- mem0：AI agent 长期记忆、graph memory、hybrid search。适合作为写入/检索长期记忆的参考。  
  https://docs.mem0.ai/open-source
- Letta / MemGPT：上下文窗口与外部记忆分层，适合作为长篇小说记忆换入/换出的参考。  
  https://docs.letta.com/concepts/memory-management
- Gymnasium / PettingZoo：RL 环境接口参考。小说环境不一定直接训练，但应该能记录状态、动作、奖励和轨迹。

## Scenario Boundary

不用管前端。小说场景只提供以下能力：

```python
class NarrativeScenarioAdapter:
    def build_observation(task_state) -> Observation: ...
    def list_actions(task_state) -> list[Action]: ...
    def retrieve_context(task_state, query) -> EvidencePack: ...
    def propose_plan(task_state, author_input) -> ChapterBlueprint: ...
    def generate_draft(task_state, plan, evidence) -> DraftCandidate: ...
    def extract_changes(task_state, draft) -> list[StateChangeProposal]: ...
    def evaluate(task_state, draft, changes) -> list[EvaluationReport]: ...
    def commit_or_rollback(task_state, changes, reports) -> NarrativeTaskState: ...
```

Agent Runtime 只负责循环、工具调度、轨迹、guardrail、人工确认和可观测性。小说 Adapter 负责领域状态、检索、生成、校验和落库语义。

## Domain Model

当前代码化模型见 `src/agent_rl/domains/narrative.py`。它不是替代 `narrative-state-engine` 的全部代码，而是把可迁移的领域概念先统一下来。

### L0 Source Layer

对象：

- `SourceDocument`
- `SourceSpan`

职责：

- 记录原文、参考文本、作者 notes、生成文本的来源。
- 所有证据、抽取概念和校验报告都应能追溯到 source span。

改进点：

- 不要让 chunk 成为长期业务概念。chunk 是索引和分析实现细节；业务层主要使用章节、场景、事件、证据。

### L1 Story World Layer

对象：

- `WorldRule`
- `LocationState`
- `ObjectState`

职责：

- 维护世界规则、地点、物品、组织、能力体系、时代约束。
- 为生成和校验提供硬约束。

改进点：

- 世界规则要带 `stability`：candidate、confirmed、contested、deprecated。
- 规则要拆分 `forbidden_implications` 和 `required_implications`，让校验器能执行。

### L2 Character Layer

对象：

- `CharacterCard`
- `CharacterDynamicState`
- `RelationshipState`

职责：

- `CharacterCard` 保存稳定人格、口吻、价值观、禁行行为、知识边界。
- `CharacterDynamicState` 保存当前情绪、身体状态、地点、目标、已知事实和弧线阶段。
- `RelationshipState` 保存双向关系、信任、张力、未解冲突。

改进点：

- 不要只做静态角色卡。续写真正需要的是“当前角色动态状态 + 稳定角色卡 + 关系状态”。
- 人物一致性校验必须检查知识边界：角色不能知道未揭示事实。

### L3 Plot Layer

对象：

- `NarrativeEvent`
- `PlotThreadState`
- `Beat`
- `ForeshadowingState`

职责：

- 事件维护因果、参与者、状态变化、揭示事实。
- 剧情线维护开放问题、预期节拍、阻塞节拍、解决条件。
- 节拍是场景内部最小叙事动作。
- 伏笔独立建模，避免提前揭示或忘记回收。

改进点：

- 原项目的 plot/event 概念应强化为“事件图 + 剧情线状态 + 伏笔状态”。
- 续写不是只写下一段，而是选择哪些 plot thread 和 beat 被推进。

### L4 Scene Layer

对象：

- `SceneState`

职责：

- 场景是生成的主工作单位。
- 记录 location、POV、进入状态、退出状态、冲突、节拍、情绪曲线和风格要求。

改进点：

- 长篇续写不应直接以“章节全文”作为唯一生成单位。更稳的做法是：章节蓝图 -> 场景计划 -> 场景生成 -> 场景提交 -> 章节汇总。

### L5 Style Layer

对象：

- `StyleProfile`
- `StyleSnippet`

职责：

- 风格画像结构化：叙事距离、句长分布、对话比例、描写比例、修辞标记、词汇指纹、禁用模式。
- 风格片段可检索、可引用、可评估。

改进点：

- “像原文”不能只靠 prompt。需要 style evidence + style metrics + style repair plan。

### L6 Author Intent Layer

对象：

- `AuthorConstraint`
- `ChapterBlueprint`

职责：

- 作者输入先成为候选意图，再经确认进入约束或蓝图。
- 作者确认约束优先级高于模型推断。

改进点：

- 作者说的话不要直接污染 canon。它应进入 `AuthorIntent -> Constraint Proposal -> Confirmation -> AuthorConstraint/ChapterBlueprint`。

### L7 Memory Layer

对象：

- `MemoryAtom`
- `CompressedMemoryBlock`

职责：

- `MemoryAtom` 是可检索、可晋升、可压缩的最小记忆。
- `CompressedMemoryBlock` 是概念感知的压缩结果，记录保留/丢弃哪些记忆。

改进点：

- 记忆压缩不是摘要文本，而是按小说概念压缩：事件、角色、伏笔、风格、作者意图。
- 未 commit 的草稿不能进入长期记忆。

### L8 Retrieval Layer

对象：

- `NarrativeQuery`
- `NarrativeEvidence`
- `EvidencePack`

职责：

- 查询从任务状态派生，不只是用户输入。
- 证据分区：style、character、plot、world、author plan、scene case。
- retrieval trace 必须可审计。

改进点：

- 不要把 top-k 检索结果直接塞进 prompt。应该组装成写作材料包：

```text
作者硬约束
章节蓝图
角色卡与动态状态
近期剧情压缩记忆
伏笔与未解决问题
世界规则/地点/物品
原文风格片段
相似场景案例
禁止事项与修复提示
```

### L9 Generation Layer

对象：

- `ChapterPlan`
- `DraftCandidate`
- `StateChangeProposal`

职责：

- `ChapterPlan` 是本次执行计划。
- `DraftCandidate` 是未提交草稿。
- `StateChangeProposal` 是从草稿或作者输入中抽出的候选状态变化。

改进点：

- 正文不是最终真相。最终真相是 proposal 通过校验后提交到 canonical state。

### L10 Evaluation Layer

对象：

- `EvaluationIssue`
- `EvaluationReport`

职责：

- 统一角色、剧情、风格、世界、检索、作者意图的评估报告。
- `blocker` issue 阻止 commit。

改进点：

- 评估不是最后打分，而是 Agent 决策闭环的一部分：失败后应该触发 repair、clarify、rollback 或 human review。

## Task Scenarios

### 1. 全文分析

目标：把长篇原文变成结构化状态，而不是一次性摘要。

推荐流程：

```text
source ingest
-> chapter/chunk analysis
-> entity/event/style extraction
-> merge aliases and duplicate events
-> build baseline NarrativeTaskState
-> index source/evidence/memory
```

需要增强：

- chunk 级分析要最终汇总到 chapter/story/scene/event 状态。
- 每个抽取结果要有 source span。
- 分析结果应该产生 trajectory，后续能比较不同抽取策略。

### 2. 作者剧情规划

目标：把作者自然语言意图变成可执行、可校验的约束和蓝图。

推荐流程：

```text
author input
-> intent extraction
-> constraint proposal
-> clarification question if uncertain
-> author confirmation
-> AuthorConstraint / ChapterBlueprint
-> retrieval hints
```

需要增强：

- confirmed author constraints 必须进入检索和校验。
- 禁止发展、必经情节、伏笔安排、节奏要求要结构化。

### 3. 章节/场景续写

目标：生成可提交的小说内容，并更新状态。

推荐流程：

```text
load canonical state
-> retrieve author plan / memory / evidence
-> build chapter plan
-> build scene plan
-> generate draft
-> extract state changes
-> evaluate character/plot/style/world/retrieval
-> repair loop or human review
-> commit accepted changes
-> compress memory
-> update retrieval index
```

需要增强：

- 优先场景级生成，章节级汇总。
- 每轮生成后的状态变化要回写。
- 失败原因要变成可学习信号。

### 4. 审稿与修订

目标：不是简单重写，而是基于报告做定向修复。

推荐流程：

```text
draft
-> reports
-> revision plan
-> rewrite with must-preserve/must-remove
-> re-evaluate
-> commit or rollback
```

需要增强：

- 修复计划要结构化。
- 修订不能破坏已通过的剧情/角色/风格约束。

### 5. 状态回流

目标：把新生成内容变成下一轮可用状态。

推荐流程：

```text
accepted draft
-> information extraction
-> StateChangeProposal
-> conflict detection
-> canonical state update
-> compressed memory
-> evidence index update
```

需要增强：

- rollback 不进入长期记忆。
- contested proposal 进入 conflict queue，不进入 canonical 检索池。

## Current Design Assessment

### 值得保留

- state-first 方向。
- proposal -> validate -> commit/rollback。
- 分层记忆与 RAG。
- 作者计划独立于 canon。
- EvidencePack、WorkingMemoryContext、retrieval trace。
- task_id/story_id 隔离。
- 远端 embedding/reranker 与本地 PG 解耦。

### 主要问题

1. `NovelAgentState` 容易成为上帝对象。需要把领域状态与通用 Agent runtime 分开。
2. 节点链像固定工作流，缺少策略层。不同任务应能选择不同 policy：分析、规划、续写、修复、澄清、人工确认。
3. RAG 很强，但还需要变成 `NarrativeRetrievalPolicy`，支持按任务动态路由证据类型。
4. 记忆压缩还需要 write policy、retrieval policy、validation policy、decay policy。
5. 评估报告需要进入奖励/轨迹体系，而不是只作为日志。
6. 多 Agent 不应先追求角色很多，而应先拆清楚 Planner、Retriever、Writer、Critic、Memory Manager、Continuity Guard 的责任。

## Improved Agent Architecture

建议拆为七个 Agent/Policy，而不是一个巨型状态机：

| 角色 | 职责 | 输入 | 输出 |
|---|---|---|---|
| Intent/Task Policy | 判断任务类型和是否需要澄清 | 作者输入、当前状态 | task type、clarification |
| Retrieval Policy | 决定查哪些证据 | NarrativeQuery、状态 | EvidencePack |
| Planning Policy | 生成章节/场景计划 | 作者约束、状态、证据 | ChapterPlan/ScenePlan |
| Writer Policy | 生成草稿 | plan、evidence、style | DraftCandidate |
| Extractor Policy | 抽取状态变化 | draft、source spans | StateChangeProposal[] |
| Evaluator/Critic Policy | 校验和评分 | draft、changes、state | EvaluationReport[] |
| Memory Policy | 写入、压缩、晋升、遗忘 | committed changes、reports | MemoryAtom/CompressedMemoryBlock |

这些 policy 可以先是规则 + LLM，后续可用 bandit/offline RL 优化。

## RL Optimization Points

第一阶段不要训练大模型，先记录轨迹：

```text
task_state
-> selected action/policy
-> evidence pack
-> draft
-> reports
-> commit decision
-> human feedback
```

可优化的小策略：

- retrieval routing：当前任务更需要角色证据、风格证据还是伏笔证据。
- context budget allocation：每类证据占多少 token。
- stop/continue policy：是否继续生成、修复、请求作者确认。
- repair policy：遇到角色问题、风格问题、剧情问题时选哪种修复动作。
- branch selection policy：多分支生成时选哪条进入审稿。
- memory write policy：哪些变化值得进入长期记忆。

奖励建议：

```text
reward =
  author_alignment
  + character_consistency
  + plot_continuity
  + style_match
  + retrieval_coverage
  - cost_penalty
  - latency_penalty
  - human_review_penalty
  - rollback_penalty
```

## Minimal Rebuild Path

如果要把 `narrative-state-engine` 做好，不建议继续堆 UI 或继续扩大状态机。建议按这个顺序重构：

1. 抽出 `NarrativeTaskState` 和领域模型，和 runtime 解耦。
2. 抽出 `NarrativeScenarioAdapter`，让通用 Agent Runtime 调它。
3. 统一 `Trajectory`，每次分析/规划/续写/修复都记录 action、observation、reward/report。
4. 把 RAG 改成 `NarrativeRetrievalPolicy`，所有检索结果都进入 `EvidencePack`。
5. 把记忆改成 `MemoryPolicy`：write、retrieve、compress、promote、decay。
6. 把 evaluator 结果变成 commit gate 和 reward source。
7. 后续再接 LangGraph/OpenAI Agents SDK 作为生产编排，不让领域模型依赖它们。

## Code Integration

本项目已实现：

- `src/agent_rl/domains/narrative.py`
- `src/agent_rl/narrative_writing/`
- `src/agent_rl/examples/narrative_demo.py`

`narrative.py` 把小说场景中的核心对象类化：

- source/world/character/plot/scene/style/author/memory/retrieval/generation/evaluation/task state。
- `SourceChunk` 和 `NarrativeSourceAnalysis` 用于承载“先分析参考小说，再进入续写”的分析资产。

`narrative_writing/` 把这些对象组织成可运行的 Agent，并按 OOAD 分层：

```text
narrative_writing/
  requests.py      # AuthorRequest / ReferenceMaterial / result DTO
  ingestion.py     # 本地参考小说 txt 读取与 AuthorRequest 构造
  ports.py         # 可替换策略端口
  bootstrap.py     # 从作者输入和参考材料构造初始状态
  prompting/       # prompt profile / global prompt / task prompt 管理
  policies/        # interaction/retrieval/planning/writing/extraction/evaluation/memory
  scenario.py      # NarrativeScenarioAdapter
  agent.py         # NarrativeWritingAgent 应用服务
```

运行链路：

```text
AuthorRequest
-> BasicAuthorInteractionPolicy
-> RuleBasedSourceAnalysisPolicy
-> NarrativeScenarioAdapter
-> CompositeNarrativeRetrievalPolicy
-> BasicRetrievalEvaluationPolicy
-> RuleBasedPlanningPolicy
-> BudgetedNarrativeContextPolicy
-> TemplateNarrativeWriterPolicy
-> RuleBasedExtractorPolicy
-> CompositeNarrativeEvaluatorPolicy
-> SimpleNarrativeMemoryPolicy
-> NarrativeRunResult
```

这套实现先用规则和模板，目的是让概念闭环可运行。后续可逐步替换为 LLM planner、mem0/Letta memory、LangGraph runtime、OpenAI Agents SDK tools/guardrails，而不改变核心领域模型和端口。

参考小说摄入已经作为核心包能力存在：`narrative_writing/ingestion.py` 支持读取 UTF-8/GB18030 文本文件或目录，把它们转换为 `ReferenceMaterial`。`RuleBasedSourceAnalysisPolicy` 会把参考文本构造成 `SourceDocument`、`SourceChunk`、`NarrativeEvent`、`StyleSnippet` 和初始 `MemoryAtom`，并把分析覆盖率写入 `NarrativeSourceAnalysis.coverage`。`CompositeNarrativeRetrievalPolicy` 会按作者计划、人物、剧情/记忆、世界、风格、source chunk、scene case 分通道召回，并根据 source_type 权重和配额融合到 `EvidencePack`。这不是最终向量数据库实现，但接口位置已经对齐真实 RAG：后续可把 embedding、reranker 和持久化索引替换进 retrieval policy，而不用改 `NarrativeWritingAgent`。

上下文管理已经进入核心包：`BudgetedNarrativeContextPolicy` 会把作者请求、章节计划、作者约束、剧情证据、人物证据、世界规则、风格参照和状态摘要装配成 `WorkingMemoryContext`。这一步位于检索和写作之间，目的是把原项目重视的上下文预算、顺序、优先级和可审计性前移到领域模型，而不是让 writer 临时拼字符串。

提示词管理已经有轻量文件化实现：`narrative_writing/prompting/` 提供 `PromptRegistry` 和 `PromptComposer`，默认包含 `global/default.md`、`profiles/default.yaml` 和 `draft_generation`、`state_extraction`、`source_analysis`、`author_planning` 任务提示词。当前 template writer 还不调用真实 LLM，但 LLM writer/extractor 可以直接复用这个 prompt 边界。

LLM 接入已经有包级基础设施边界：`agent_rl.llm` 定义 `ChatModelClient`、`JsonBlobParser`、`OpenAICompatibleChatClient`、审计日志和 usage 记录。`LLMNarrativeWriterPolicy` 与 `LLMNarrativeExtractorPolicy` 是小说场景的使用方，通过 prompt composer 和 working context 调用模型，要求 JSON 输出；模型调用、JSON 解析或 schema 校验失败时，会分别回退到 `TemplateNarrativeWriterPolicy` 和 `RuleBasedExtractorPolicy`。这让真实模型能力可以进入写作链路，同时测试和本地 demo 不依赖 API key。

模型调用适配器不是小说场景私有能力，而是包级 `agent_rl.llm` 基础设施。它已经吸收旧项目的审计思路：`OpenAICompatibleChatClient` 从 `LLM_API_BASE`、`LLM_API_KEY`、`LLM_MODEL` 等环境变量读取配置，支持 `LLM_API_BASES` / `LLM_API_KEYS` 多 endpoint 轮换，JSON mode contract、失败重试、请求开始/成功/失败事件、prompt metadata、token usage 都写入 JSONL。默认日志路径在 `artifacts/llm/` 下，不进入 git。小说写作只是通过 `LLMNarrativeWriterPolicy` / `LLMNarrativeExtractorPolicy` 使用这个包级模块。

### 使用方式

第一轮不确认计划，只让 Agent 问问题或给出章节蓝图：

```python
from agent_rl.narrative_writing import AuthorRequest, NarrativeWritingAgent, ReferenceMaterial

agent = NarrativeWritingAgent()
result = agent.run(
    AuthorRequest(
        request="规划并续写下一章",
        references=(ReferenceMaterial(title="参考", text="原文片段..."),),
        writing_direction="下一章必须找到密信线索；不要让主角立刻原谅对方",
        constraints=("不要让主角立刻原谅对方",),
        confirm_plan=False,
    )
)
```

从本地参考小说文件构造请求：

```python
from agent_rl.narrative_writing import NarrativeWritingAgent, build_author_request_from_files

request = build_author_request_from_files(
    request="规划并续写下一章",
    reference_paths=("data/my-novel/chapter-01.txt", "data/my-novel/chapter-02.txt"),
    writing_direction="下一章继续推进密信线索，不要让主角立刻原谅对方",
    constraints=("不要让主角立刻原谅对方",),
    confirm_plan=True,
)
result = NarrativeWritingAgent().run(request)
```

作者确认后重新运行：

```python
result = agent.run(
    AuthorRequest(
        request="规划并续写下一章",
        references=(ReferenceMaterial(title="参考", text="原文片段..."),),
        writing_direction="下一章必须找到密信线索；不要让主角立刻原谅对方",
        constraints=("不要让主角立刻原谅对方",),
        confirm_plan=True,
    )
)
```

命令行 demo：

```powershell
$env:PYTHONPATH="src"; python -m agent_rl.examples.narrative_demo
```

### 当前交互边界

这不是前端实现，但已经支持用户操作语义：

- 缺少参考小说或写作方向时，返回 `AuthorQuestion[]`。
- 可以从本地参考小说 txt 文件或目录读取初始参考材料。
- 初始参考材料会进入 `SourceDocument`、`StyleSnippet`、`MemoryAtom`，并在检索阶段进入 `EvidencePack`。
- 生成章节蓝图后，返回 `requires_confirmation=True`。
- 作者确认后才执行草稿生成和 canonical state 提交。
- 评估报告出现 blocker 时回滚，不写入长期记忆。
- 成功提交后写入 `MemoryAtom` 和 `CompressedMemoryBlock`。

## Verification Plan

- 单元测试保证 `EvidencePack` 和 `EvaluationReport` 的基础行为。
- 后续新增 scenario adapter 时，测试应覆盖：
  - task state -> observation
  - task state -> available actions
  - retrieval policy -> evidence pack
  - evaluator blocker -> commit blocked
  - committed changes -> memory compression

## Key Decision

保留小说场景，重做运行模式：

```text
不要把小说项目继续做成一个前端工作台驱动的复杂状态机。
要把它做成一个可插拔的 Narrative Agent Scenario：
状态可信、证据可追溯、记忆可压缩、动作可评估、提交可回滚、策略可学习。
```

## Follow-up Design

- `NARRATIVE_ENGINE_ABSORPTION_PLAN_2026-05-20.md`：对比原 `narrative-state-engine` 与当前 `agent-with-RL` 的运行链路，并定义分析、RAG、提示词、上下文管理、并行 LLM 调用如何吸收到当前 Agent 场景中。
