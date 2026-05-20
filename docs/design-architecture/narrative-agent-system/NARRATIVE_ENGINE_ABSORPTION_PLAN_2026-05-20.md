# 小说状态引擎能力吸收方案

## 背景

来源项目：`D:\buff\narrative-state-engine`

当前项目：`agent-with-RL`

这份文档回答一个关键问题：原小说系统已经把“分析小说、情节规划、续写、RAG、提示词、上下文管理、并行 LLM 调用”做成较完整的状态机；当前项目是否保留了这些能力，还是改成了新的 Agent 结构。

结论：

```text
当前项目没有复制原系统的重型运行时。
当前项目把原系统重新设计为两层：
1. 包级 Agent/LLM 基础设施：core concepts、agent_rl.llm、integrations。
2. narrative scenario：领域状态 + 策略端口 + 场景适配器 + 轨迹/奖励/评估 + 人类确认。
```

也就是说，当前实现是“包级基础设施 + 场景概念骨架 + 可运行最小闭环”，不是原系统所有工程能力的迁移版。后续应以当前项目为主体，把原系统能力逐步吸收为可替换 policy、service、adapter，而不是把旧状态机整块搬进来。

## 两套系统的定位差异

| 维度 | 原 `narrative-state-engine` | 当前 `agent-with-RL` |
|---|---|---|
| 核心定位 | 面向小说生产的 state-first 写作引擎 | 面向学习和复用的 Agent/RL OOAD 场景实现 |
| 中心对象 | `NovelAgentState` | `NarrativeTaskState` + `Trajectory` |
| 运行方式 | 固定节点链 / LangGraph 风格 state machine | `NarrativeWritingAgent` 调用 scenario adapter 和 policy ports |
| 分析能力 | chunk/chapter/global LLM 分析，原文回流入索引 | 当前只有轻量规则 bootstrap，尚未有 LLM 分析管线 |
| RAG 能力 | keyword + structured + vector + rerank + source_type quota | 当前是本地 keyword/entity overlap + `EvidencePack` |
| 续写能力 | LLM draft generator + fallback + 多轮章节完成策略 | 当前是 template writer，占位但可运行 |
| 提示词系统 | 文件化 prompt registry/profile/binding/hash/audit | 当前尚未实现 prompt registry |
| 上下文管理 | `PromptContextSection`、预算、可见性、优先级、mode | 当前只有 `EvidencePack`，缺少 context manifest/budget |
| 并行模型调用 | run graph、child runs、分析分块并行、续写分支并行 | 当前串行执行，只有 trajectory step |
| 持久化 | PostgreSQL/pgvector、状态版本、分支、retrieval runs | 当前核心包内存对象，无数据库依赖 |
| 交互原则 | 主对话 + 后台状态机工具 + 确认门 | 当前支持缺失信息提问和 plan confirmation |

## 原系统运行链路

原项目偏“生产写作状态机”。典型链路是：

```text
task_id/story_id
-> ingest txt
-> source_chunk / chapter split
-> llm_chunk_analysis
-> llm_chapter_analysis
-> llm_global_analysis
-> evidence_indexing
-> embedding backfill
-> author_session / author_plan
-> author confirmation
-> retrieve_author_plan
-> retrieve_story_state
-> hybrid evidence retrieval
-> working memory context
-> plot_planner
-> llm_draft_generator
-> llm_information_extractor
-> consistency_validator
-> character_consistency_evaluator
-> plot_alignment_evaluator
-> style_evaluator
-> repair_loop
-> commit_or_rollback
-> memory_compressor
-> generated_chapter_indexing
-> generated_chapter_embedding
-> clean chapter export
```

它的强项不是某一个 prompt，而是：

- 原文先分析成结构化资产。
- 检索不是普通问答，而是围绕写作目标取证据。
- 提示词、上下文、LLM 日志、JSON 解析失败都有审计。
- 生成结果不会直接污染 canon，必须先抽取 proposal，再校验，再提交。
- 章节写完后会反向分析，成为下一轮的状态和检索资产。

## 当前系统运行链路

当前项目偏“Agent 概念实现”。现有链路是：

```text
AuthorRequest
-> BasicAuthorInteractionPolicy
-> build_initial_state
-> propose_plan
-> author confirmation gate
-> build_query
-> KeywordNarrativeRetrievalPolicy
-> build_chapter_plan
-> TemplateNarrativeWriterPolicy
-> RuleBasedExtractorPolicy
-> CompositeNarrativeEvaluatorPolicy
-> commit_or_rollback
-> SimpleNarrativeMemoryPolicy
-> Trajectory / Reward
-> NarrativeRunResult
```

当前已经有这些概念位置：

- 分析/初始化：`bootstrap.py`
- 作者交互：`BasicAuthorInteractionPolicy`
- 情节规划：`NarrativePlanningPolicy`
- RAG：`NarrativeRetrievalPolicy` + `EvidencePack`
- 续写：`NarrativeWriterPolicy`
- 信息抽取：`NarrativeExtractorPolicy`
- 校验：`NarrativeEvaluatorPolicy`
- 记忆提交/压缩：`NarrativeMemoryPolicy`
- 环境交互：`NarrativeScenarioAdapter`
- Agent 轨迹：`TrajectoryStep` / `Reward`

但当前实现仍是轻量版：

- 没有 LLM chunk/chapter/global analyzer。
- 没有 prompt registry。
- 没有 context section/budget/token 管理。
- 没有 vector/rerank RAG。
- 没有 run graph 和并行 child runs。
- 没有持久化状态版本和生成章节回流索引。

这不是错误，而是当前阶段的设计选择：先把 Agent 场景概念闭环跑通，再把重能力吸收到对应端口。

## 设计判断

当前项目不应该回到原项目的“大状态机 + 重数据库 + Web 工作台”主体。

更合理的主体是：

```text
Narrative Agent Scenario
  = Agent/RL core concepts
  + OOAD domain model
  + replaceable policies
  + author-in-the-loop
  + retrieval/memory/context as first-class state
  + run trajectory / reward / evaluation
```

原系统能力要被吸收，但吸收方式应是“能力分层”，不是“代码平移”。

## 能力吸收映射

| 原系统能力 | 当前应放入的位置 | 吸收方式 |
|---|---|---|
| `TextChunker` / txt chapter split | `narrative_writing.ingestion` 或新 `analysis/` | 先做纯 Python chunk/chapter splitter，无数据库 |
| `LLMNovelAnalyzer` | `NarrativeAnalysisPolicy` | 新增分析端口，产出角色、事件、世界、风格、伏笔 |
| chunk/chapter/global analysis | `domains.narrative` + `analysis policies` | 映射成 `NarrativeEvent`、`CharacterCard`、`PlotThreadState` 等 |
| `EvidencePackBuilder` | `NarrativeRetrievalPolicy` | 当前 keyword policy 升级为 composite retrieval service |
| hybrid search / vector / rerank | `narrative_writing.retrieval` 子包 | 作为可选 adapter，不让核心依赖 pgvector |
| source_type quota | `EvidencePack` ranking policy | 保留 target/style/world/crossover 配额思想 |
| `WorkingMemoryContext` | 新 `NarrativeContextPolicy` | 从 `EvidencePack` 进一步生成有预算的上下文包 |
| `PromptRegistry` / profiles | 新 `narrative_writing.prompting` | 文件化 prompt 管理，支持 metadata/hash/version |
| `build_draft_messages` | `LLMNarrativeWriterPolicy` + `agent_rl.llm` | writer policy 使用包级 LLM 调用体系 |
| `build_extraction_messages` | `LLMNarrativeExtractorPolicy` + `agent_rl.llm` | extractor policy 使用包级 LLM 调用体系 |
| JSON parser/fallback | `agent_rl.llm` | 提供全项目 LLM adapter、审计、usage、JSON parser |
| run graph / child runs | `core.runtime` 或新 `core.run_graph` | 与 `Trajectory` 并存：trajectory 记录决策，run graph 记录并行执行 |
| parallel chunk analysis | `NarrativeAnalysisPolicy.analyze_sources` | 返回多个子任务结果并合并 |
| multi-branch continuation | `NarrativeBranchingPolicy` | 作为 action selection / branch selection 问题 |
| memory compressor | `NarrativeMemoryPolicy` | 当前简单压缩升级为 rolling/archival/retrieval memory |
| generated chapter indexing | `NarrativeMemoryPolicy` + `RetrievalIndexPolicy` | commit 后把新章节回流为 source/evidence |

## 上下文管理应该如何进入当前项目

原项目重视上下文管理，这是必须保留的。当前 `EvidencePack` 只回答“拿到了哪些证据”，还没有回答“哪些内容应该进入模型上下文、进入哪个位置、占多少预算、作者是否可见”。

建议新增：

```text
NarrativeContextPolicy
PromptContextSection
WorkingMemoryContext
ContextBudget
ContextManifest
```

职责：

- 把 `NarrativeTaskState`、`EvidencePack`、`AuthorConstraint`、`ChapterPlan`、`CompressedMemoryBlock` 组装成模型上下文。
- 每个 section 有 `priority`、`order`、`budget_chars/token_budget`、`visible_to_author`、`visible_to_model`。
- 输出 `ContextManifest`，进入 `DraftCandidate.metadata` 和 `TrajectoryStep`。
- 上下文超预算时按策略裁剪，而不是临时截字符串。

推荐链路：

```text
retrieve_context
-> build_working_memory_context
-> compose_prompt_context
-> writer_policy.generate
```

## RAG 应该如何升级

当前 `KeywordNarrativeRetrievalPolicy` 应保留为 baseline，但新增组合检索：

```text
CompositeNarrativeRetrievalPolicy
  -> AuthorPlanRetriever
  -> StructuredStateRetriever
  -> SourceMemoryRetriever
  -> StyleSnippetRetriever
  -> SceneCaseRetriever
  -> VectorRetriever(optional)
  -> Reranker(optional)
  -> EvidencePackRanker
```

第一阶段不必接 pgvector，也可以先本地实现：

- chunk reference text
- 建立 source memory index
- keyword + source_type quota
- evidence type quota
- retrieval trace
- retrieval evaluation report

第二阶段再接：

- embedding provider
- vector store adapter
- reranker adapter
- persistent retrieval runs

## 分析、规划、续写应如何分开

当前代码已经用 policy 分开，但还缺少“分析源材料”这一层。

建议形成四类 task policy：

```text
NarrativeAnalysisPolicy
  输入：ReferenceMaterial / SourceDocument
  输出：characters, events, plot_threads, world_rules, style_profile, style_snippets, source_memory

NarrativePlanningPolicy
  输入：AuthorRequest + state + evidence
  输出：AuthorPlotPlan / ChapterBlueprint / ChapterPlan

NarrativeWriterPolicy
  输入：state + chapter_plan + working_context
  输出：DraftCandidate

NarrativeExtractorPolicy
  输入：draft + state
  输出：StateChangeProposal[]
```

这比原项目更 Agent 化：每个 policy 都是可替换策略，未来可以做 bandit/RL 选择。例如：

- 什么时候用便宜模型分析？
- 哪些证据进入上下文？
- 生成几条分支？
- 哪条分支值得进入评估？
- 哪些记忆应晋升长期记忆？

## 并行调用应如何进入当前项目

原系统的并行 LLM 调用应被建模成 run graph，而不是直接塞进 `Trajectory`。

两者区别：

```text
Trajectory：Agent 决策序列，回答“系统做了哪些动作，得到什么 reward”。
RunGraph：工程执行图，回答“某个动作内部并行跑了哪些子任务，状态如何”。
```

建议新增：

```text
RunNode
RunGraph
ParallelExecutionPolicy
```

例子：

```text
Action: analyze_sources
RunGraph:
  chunk_analysis_001
  chunk_analysis_002
  chunk_analysis_003
  merge_chapter_analysis
  build_global_story_state
```

```text
Action: generate_branches
RunGraph:
  plan_branch_001
  draft_branch_001
  draft_branch_002
  evaluate_branch_001
  evaluate_branch_002
  select_branch
```

这样当前项目仍保持 Agent/RL 主体：并行只是 action 内部执行形态，不改变 Agent 的概念结构。

## Prompt 系统吸收原则

原项目的 prompt 系统值得吸收，但不能把 prompt 写成业务核心。建议：

```text
narrative_writing/prompting/
  registry.py
  composer.py
  templates/
    global/default.md
    tasks/source_analysis.md
    tasks/author_planning.md
    tasks/draft_generation.md
    tasks/state_extraction.md
```

关键规则：

- prompt 是策略实现的配置资产，不是领域模型。
- prompt 必须带 id/version/hash。
- user text、reference text、retrieved evidence 只能作为 data context，不能提升到 system prompt。
- LLM 输出必须有 schema，解析失败要有 trace 和 fallback。
- prompt metadata 应进入 `DraftCandidate.metadata`、`EvaluationReport.metrics` 或 `TrajectoryStep.metadata`。

## 推荐实施顺序

### Phase 1: 保持当前项目主体，补上下文与 prompt 边界

1. 新增 `NarrativeContextPolicy` 和 `WorkingMemoryContext`。
2. 新增 prompt registry/composer，但不先接真实 LLM。
3. 把 writer/extractor 的模板上下文改为通过 context policy 获取。
4. 给 `EvidencePack` 增加 retrieval evaluation report。

当前落地状态：

- 已新增 `NarrativeAnalysisPolicy` 和 `RuleBasedSourceAnalysisPolicy`，参考小说会先形成 `SourceChunk`、`NarrativeSourceAnalysis`、事件、风格片段和初始记忆。
- 已新增 `CompositeNarrativeRetrievalPolicy`，按作者计划、人物、剧情/记忆、世界、风格、source chunk、scene case 分通道检索，并用 source_type 权重和配额融合。
- 已新增 `PromptContextSection` 和 `WorkingMemoryContext`。
- 已新增 `BudgetedNarrativeContextPolicy`，运行链路变为 `retrieve_context -> build_working_context -> generate_draft`。
- 已新增 `PromptRegistry` / `PromptComposer` 和默认 prompt profile/task prompts。
- 已新增包级 `agent_rl.llm`：`ChatModelClient`、`JsonBlobParser`、`OpenAICompatibleChatClient`、`OpenAICompatibleConfig` 和 JSONL audit/usage recorder，吸收旧项目的模型调用审计、usage 记录、endpoint failover 和 retry 思路。
- 已新增 `LLMNarrativeWriterPolicy`、`LLMNarrativeExtractorPolicy`，小说场景通过包级 LLM 模块接模型，同时保留 template/rule fallback。
- 已新增 `BasicRetrievalEvaluationPolicy`，将检索覆盖率、缺失证据类型和检索评分写入 `state.metadata["retrieval_evaluation_report"]` 与 `retrieval_trace`。

### Phase 2: 吸收原系统分析层

1. 新增 `NarrativeAnalysisPolicy`。
2. 移植/重写 txt chunker 和 chapter splitter。
3. 先做规则分析实现，再接 LLM 分析实现。
4. 分析结果进入 `NarrativeTaskState`，不是进入旧 `NovelAgentState`。

### Phase 3: 吸收 RAG

1. 新增 composite retrieval。
2. 支持 source_type quota。
3. 支持 local in-memory source index。
4. 可选接 embedding/vector/reranker adapter。

### Phase 4: 吸收 LLM 生成和抽取

1. 新增 `LLMNarrativeWriterPolicy`。
2. 新增 `LLMNarrativeExtractorPolicy`。
3. 引入 JSON parser/fallback。
4. 保留 template/rule policy 作为测试和降级路径。

### Phase 5: 吸收并行运行

1. 新增 `RunGraph`。
2. 分析 chunk 并行。
3. 多分支续写并行。
4. branch selection policy。
5. 把 run graph 和 trajectory 关联起来。

## 最终目标链路

当前项目后续理想链路应是：

```text
AuthorRequest / SourceFiles
-> ask missing author context
-> analyze_sources
   -> chunk/chapter/global analysis
   -> source memory / style / character / plot indexing
-> author_plan_dialogue
   -> propose constraints
   -> ask clarifying questions
   -> confirmation gate
-> build_query
-> composite_retrieval
-> retrieval_evaluation
-> build_working_memory_context
-> plan_chapter
-> compose_prompt
-> generate_draft
-> extract_state_changes
-> evaluate character / plot / style / world / retrieval
-> repair or rollback
-> commit canonical state
-> compress memory
-> index generated chapter
-> record trajectory + run graph + reward
```

这条链路吸收了原系统的强项，但表达方式仍属于当前项目：

```text
Agent observes state, chooses actions, calls tools/policies, receives evaluation/reward, updates memory/state.
```

## 当前判断

如果问“现在是否已有原系统那些能力”，答案应分开说：

- 有概念位置：分析、规划、续写、RAG、抽取、校验、记忆、作者确认都已经被拆成独立 policy 或 domain object。
- 有最小可运行闭环：本地参考小说读取、规划、RAG-like 检索、模板续写、规则抽取、评估、提交、记忆压缩、trajectory。
- 还没有原系统强工程能力：并行 LLM、正式 prompt registry、LLM 分析、hybrid RAG、上下文预算、pgvector、reranker、章节生成回流。

因此后续推进方向不是推倒当前实现，而是把原系统的重能力逐个放进当前的 Agent 场景端口里。
