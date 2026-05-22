# Narrative Codex-Like Long-Running Agent Plan

Date: 2026-05-22

## 背景

当前小说 Agent 已经从一次性 pipeline 进化到 ReAct 风格工作流：

```text
Observation -> Action -> Tool/Policy -> Reward -> Trajectory -> State Update
```

已有核心能力：

- `NarrativeReActEnvironment`：把小说写作暴露成可观察、可执行、可暂停的环境。
- `NarrativeWorkflowState`：记录当前工作流阶段、动作历史、蓝图、草稿、产物。
- `NarrativeTaskState`：记录故事领域状态，包括 source、角色、事件、剧情线、世界规则、记忆、上下文、草稿和评估报告。
- `NarrativeWritingSession`：保留同一个环境、状态、轨迹和 memory，可在作者确认后继续运行。
- `LLMDeepNarrativeAnalysisPolicy`：支持 `chunk -> chapter -> global` 的深度分析。
- `FileNarrativeAnalysisRepository` / `FileNarrativeStateRepository`：支持分析产物和运行状态本地持久化。
- `Trajectory` / `Reward`：已经能把执行过程变成可复盘数据。

下一阶段目标不是简单“让模型续写”，而是做一个更像 Codex 的高级 Agent：能持续运行、能暂停询问、能并行处理、能保存上下文、能恢复任务、能生成多个候选、能评估和修复、能沉淀项目级经验。

## Codex-Like 能力抽象

这里说的 Codex-like 不是复制 coding agent，而是吸收它稳定的工程模式：

1. **Durable objective**

   Agent 不只响应一次 prompt，而是围绕一个长期目标持续推进。

   小说领域映射：

   ```text
   “把这部小说按作者方向持续写下去”
   -> 多轮分析
   -> 多轮作者规划
   -> 多章节续写
   -> 状态回流
   -> 记忆压缩
   -> 后续任务继续接上
   ```

2. **Task sandbox / environment**

   Codex 的任务在隔离环境中执行。小说 Agent 也需要“故事工作区”，而不是每次把所有文件塞进 prompt。

   小说领域映射：

   ```text
   story workspace
     source/
     analysis/
     state_snapshots/
     blueprints/
     drafts/
     branches/
     trajectories/
     memory/
   ```

3. **Ask mode / execute mode**

   Codex 区分只分析建议和实际改代码。小说 Agent 也应区分：

   - Ask：只分析、解释、规划、问问题。
   - Plan：生成蓝图、等待作者确认。
   - Execute：生成正文、抽取状态、评估、提交。
   - Review：对已有草稿/分支做审阅和修复建议。

4. **Background and parallel work**

   Codex 可在后台并行处理多个任务。小说 Agent 应允许并行做：

   - chunk 分析
   - 章节摘要
   - 人物卡抽取
   - 世界观术语抽取
   - 多候选续写分支
   - 风格评估
   - 连续性评估

5. **Work log**

   Codex 的工作过程要能复盘。小说 Agent 也必须保留：

   - 每一步 action
   - 每次 LLM 调用目的
   - 使用了哪些证据
   - 生成了什么蓝图
   - 作者确认了什么
   - 哪些状态被提交
   - 哪些状态被回滚

6. **Skills / team standards**

   Codex 可通过项目规则和技能沉淀重复工作经验。小说 Agent 应该有“写作技能”和“项目规范”：

   - 文风技能
   - 类型小说技能
   - 人物对话技能
   - 伏笔管理技能
   - 修仙/科幻/悬疑等设定系统技能
   - 作者偏好规则
   - 禁忌设定和世界观硬约束

7. **Human-in-the-loop permissions**

   Codex 重要变更要让人 review。小说 Agent 里对应：

   - 蓝图确认
   - 世界观新增确认
   - 角色关系重大变化确认
   - 主线分支选择
   - 正文最终提交

8. **Recoverable state**

   Codex 任务可以从日志和环境恢复。小说 Agent 应从以下内容恢复：

   - `NarrativeTaskState`
   - `NarrativeWorkflowState`
   - `Trajectory`
   - `AuthorConversation`
   - `AnalysisRepository`
   - `MemoryRepository`

## 小说领域应该充分利用的定义

当前项目已经定义了很多有价值的领域对象。高级 Agent 不应该把它们降级为 prompt 文本，而应该把它们作为动作、检索、评估和记忆的结构化基础。

### Source / Analysis

已定义：

- `SourceDocument`
- `SourceChunk`
- `NarrativeSourceAnalysis`
- `ChunkAnalysisResult`
- `ChapterAnalysisResult`
- `GlobalStoryAnalysisResult`

应该利用方式：

- chunk 只作为分析和索引单位。
- chapter 是写作和检索主要单位。
- global 是长期状态和故事 bible。
- 分析结果进入 repository，而不是只进入 prompt。
- 每次续写前优先读取已有 analysis cache，避免重复分析。

### Story State

已定义：

- `CharacterCard`
- `CharacterDynamicState`
- `RelationshipState`
- `NarrativeEvent`
- `PlotThreadState`
- `WorldRule`
- `LocationState`
- `ObjectState`
- `ForeshadowingState`
- `SceneState`

应该利用方式：

- 生成前：作为 hard/soft constraints 和 retrieval source。
- 生成中：约束人物知识边界、关系状态、事件因果。
- 生成后：通过 extractor 产生 `StateChangeProposal`。
- 提交前：通过 evaluator 判断是否破坏 canon。
- 提交后：进入 canonical state 和 memory。

### Author State

已定义：

- `AuthorRequest`
- `AuthorQuestion`
- `AuthorConstraint`
- `ChapterBlueprint`
- `ChapterBlueprintSegment`

应该利用方式：

- 作者输入不是普通上下文，而是最高优先级意图。
- 蓝图不是 prompt 片段，而是必须确认的计划对象。
- segment 是长章写作的最小执行单位。
- 作者约束要进入 retrieval、context、evaluation、repair。

### Memory

已定义：

- `MemoryAtom`
- `CompressedMemoryBlock`
- `LongformContextSelector`
- `WorkingMemoryContext`
- `PromptContextSection`

应该利用方式：

- near memory：最近章节、当前场景、最近事件。
- mid memory：剧情线、人物动态、关系变化、章节摘要。
- global memory：人物卡、世界规则、风格 bible、全书梗概。
- compression 不是为了省 token，而是为了保留未来写作会用到的状态。

### Evaluation / RL

已定义：

- `EvaluationReport`
- `EvaluationIssue`
- `Reward`
- `Trajectory`

应该利用方式：

- 每次写作形成可学习轨迹。
- reward 不只是“写出来了”，而是多维：

```text
author_alignment
character_consistency
plot_progress
world_consistency
style_match
retrieval_grounding
memory_update_quality
```

后续可用这些轨迹做：

- prompt 迭代
- policy 对比
- writer branch selection
- 自动回归测试
- 轻量 RL/偏好优化实验

## 目标架构

```text
NarrativeWritingSession
  owns:
    NarrativeReActEnvironment
    NarrativeWorkflowState
    NarrativeTaskState
    MemoryStore
    Trajectory

Environment actions:
  scan_workspace
  load_state_snapshot
  load_analysis
  analyze_source
  ask_author
  propose_blueprint
  revise_blueprint
  confirm_blueprint
  retrieve_context
  build_working_context
  generate_segment
  merge_draft_segments
  evaluate_draft
  repair_draft
  compress_new_draft
  commit_state
  save_artifacts
  schedule_followup
  stop

Repositories:
  NarrativeAnalysisRepository
  NarrativeStateRepository
  NarrativeConversationRepository
  NarrativeBranchRepository
  NarrativeMemoryRepository
  NarrativeEvaluationRepository

Policies:
  AnalysisPolicy
  AuthorInteractionPolicy
  PlanningPolicy
  RetrievalPolicy
  ContextPolicy
  WriterPolicy
  ExtractorPolicy
  EvaluatorPolicy
  RepairPolicy
  MemoryPolicy
  BranchSelectionPolicy
```

## 需要新增的核心能力

### 1. Conversation State

当前已有 `NarrativeWritingSession`，但还没有完整作者会话模型。

建议新增：

```python
AuthorMessage
AuthorConversation
AuthorDecision
AuthorPreferenceProfile
NarrativeConversationRepository
```

用途：

- 记录作者每次方向、修改意见、确认/拒绝。
- 从历史对话中抽取长期偏好。
- 让下一次规划知道作者偏好，而不是只看当前 request。

### 2. Persistent Session Resume

当前 session 是内存中的持续运行。下一步需要落盘恢复：

```text
session_id
story_id
task_id
workflow_snapshot
state_snapshot
trajectory
conversation
last_observation
```

需要 API：

```python
session = NarrativeWritingSession.create(request)
session.save()
session = NarrativeWritingSession.resume(session_id)
```

### 3. Blueprint Revision Loop

现在蓝图只有 propose/confirm。需要增加：

```text
propose_blueprint
-> wait_for_author
-> revise_blueprint
-> wait_for_author
-> confirm_blueprint
```

作者反馈例子：

```text
第二段冲突更强
不要这么早暴露反派
主角要更被动一点
加一个配角视角
```

这些反馈应变成：

- blueprint patch
- author constraints
- retrieval hints
- evaluation checks

### 4. Repair Draft Loop

现在评估失败会 rollback。高级 Agent 应该先 repair：

```text
evaluate_draft
-> if blocker:
     build_repair_plan
     repair_draft
     extract_changes
     evaluate_again
-> commit / rollback
```

需要新增：

```python
NarrativeRepairPolicy
DraftRepairPlan
DraftRevisionCandidate
```

### 5. Branch Generation

Codex 可以准备一个 PR。小说 Agent 应该能准备多个续写分支：

```text
generate_branch_A
generate_branch_B
generate_branch_C
evaluate_branches
rank_branches
ask_author_to_choose
commit_selected_branch
```

分支评价维度：

- 作者方向符合度
- 剧情推进效率
- 人物一致性
- 风格匹配
- 伏笔利用
- 后续可写空间

### 6. Parallel Tool Execution

适合并行的任务：

- chunk analysis
- character extraction
- world-setting extraction
- style extraction
- branch generation
- branch evaluation

建议新增：

```python
NarrativeRunGraph
NarrativeTaskNode
ParallelToolExecutor
```

注意：并行不应破坏 canonical state。并行任务只能写 candidate artifacts，最终由 commit action 合并。

### 7. Retrieval Upgrade

当前 retrieval 是本地 composite 检索。后续升级方向：

```text
keyword retrieval
-> JSONL local index
-> SQLite FTS
-> embedding vector store
-> hybrid retrieval
-> reranker
-> graph retrieval
```

小说场景里 retrieval 应按任务意图路由：

- 写对话：人物 voice、关系、最近情绪、对话禁忌。
- 写战斗：能力体系、场景地形、动作风格、伤害约束。
- 写揭秘：伏笔、未解问题、已知/未知边界。
- 写日常：风格片段、关系微变化、节奏。
- 写转折：剧情线阶段、不能提前泄露的信息。

### 8. Memory Upgrade

需要把 memory 从“写入一些摘要”升级为“可治理的长期记忆”：

```text
memory write policy
memory importance scoring
memory decay
memory invalidation
memory compression
memory retrieval
memory conflict review
```

小说里尤其要处理：

- 新章节改变了角色状态。
- 作者推翻了旧设定。
- 某个伏笔已经回收。
- 某个关系状态从敌对变成合作。
- 某个规则被新剧情补充或限制。

### 9. Workbench Without Frontend

用户暂时不要前端，但可以先做好 CLI/API 级 workbench：

```text
agent status
agent continue
agent confirm-blueprint
agent revise-blueprint
agent show-context
agent show-branches
agent accept-branch
agent rollback
agent export-chapter
```

这些命令应该调用核心包，不复制业务逻辑。

### 10. Scheduled / Background Jobs

类似任务系统，小说 Agent 后续可以支持：

- 每晚自动分析新增章节。
- 生成下一章候选蓝图。
- 定期压缩 memory。
- 检查世界观冲突。
- 为作者生成“待确认问题列表”。

第一版不需要真正后台服务，可以先做 job repository：

```python
NarrativeJob
NarrativeJobRepository
NarrativeJobRunner
```

## 高效率的关键设计

小说写作效率不是靠“更长 prompt”，而是靠结构化状态减少无效上下文。

### 1. 分析一次，多次复用

参考小说原文要先进入：

```text
source -> chunk analysis -> chapter analysis -> global story state -> retrieval index
```

续写时不要重复全文分析。

### 2. 生成前先检索，不直接塞全文

每次写作只取相关证据：

```text
author direction
-> query
-> evidence pack
-> working context
-> draft
```

### 3. 长章按 segment 生成

长章不应一次生成。当前已有 segment，后续要让 segment 更强：

- 每段有 entry/exit state。
- 每段有 required beats。
- 每段生成后可局部评估。
- 段落之间有状态传递。

### 4. 状态回流必须结构化

生成正文后必须抽取：

- 新事件
- 角色状态变化
- 关系变化
- 世界规则变化
- 伏笔种植/回收
- 新记忆

不能只保存正文。

### 5. 作者偏好长期化

作者反馈要变成可检索偏好：

```text
“不要太快揭秘”
“主角要克制”
“对话少解释，多暗示”
“战斗写得短一点”
```

这些应该进入 `AuthorPreferenceProfile`，后续自动影响 planning、writing、evaluation。

## 实施路线

### Phase 1: Session 可恢复

目标：真正像长期 Agent。

改动：

- 新增 `NarrativeConversationRepository`
- 新增 `session_id`
- 保存/恢复 `NarrativeWritingSession`
- CLI 支持 `start/resume/status/step`
- 作者确认后从同一个 session 继续

验收：

- 启动 session，停在蓝图确认。
- 关闭进程。
- 重新 resume。
- 提交作者确认。
- 继续生成并 commit。

### Phase 2: 蓝图修订循环

目标：作者可以和 Agent 反复讨论剧情。

改动：

- `revise_blueprint`
- `AuthorFeedback`
- `BlueprintPatch`
- 蓝图版本历史

验收：

- 作者提出修改意见。
- Agent 生成新蓝图版本。
- 旧蓝图保留。
- 新蓝图确认后进入写作。

### Phase 3: 草稿修复循环

目标：评估失败不直接 rollback。

改动：

- `NarrativeRepairPolicy`
- `repair_draft`
- `DraftRevisionCandidate`
- repair trace

验收：

- evaluator 发现 blocker。
- repair policy 修复。
- 二次评估通过后 commit。

### Phase 4: 分支生成和选择

目标：让作者像 review PR 一样 review 续写分支。

改动：

- `NarrativeBranchRepository`
- `DraftBranch`
- `BranchEvaluationReport`
- `BranchSelectionPolicy`

验收：

- 同一个蓝图生成 2-3 个分支。
- 每个分支有评分和解释。
- 作者选择一个。
- 选中分支提交状态。

### Phase 5: RAG / Memory 持久升级

目标：支撑长篇小说和多章节持续写作。

改动：

- SQLite repository
- FTS retrieval
- optional vector adapter
- memory invalidation
- memory compression policy

验收：

- 多章节状态可快速检索。
- 新章节提交后自动进入索引。
- 作者改设定后相关 memory 失效或降权。

### Phase 6: 任务调度和后台运行

目标：支持主动工作。

改动：

- `NarrativeJob`
- `NarrativeJobRunner`
- `ScheduledAnalysisJob`
- `MemoryCompressionJob`
- `BlueprintProposalJob`

验收：

- 可以提交后台分析任务。
- 任务完成后生成 artifacts 和通知信息。
- 不需要前端也能从 CLI/API 查看状态。

## 最近最该做的三件事

1. **Session 持久化恢复**

   这是“像 Codex 一样持续运行”的基础。

2. **蓝图修订循环**

   这是小说作者真实交互的核心。

3. **草稿修复循环**

   这是从“能写”走向“能稳定写”的关键。

## 风险

1. 状态对象过多，但没有 repository 边界，会导致后续迁移困难。

   对策：所有长期对象都走 repository port。

2. Agent 自主性过强，越过作者。

   对策：蓝图、重大状态变化、分支选择必须 human-in-the-loop。

3. prompt 越来越长。

   对策：坚持 retrieval + context budget，不走全文 prompt。

4. 记忆污染。

   对策：区分 candidate / confirmed / author_locked / deprecated。

5. 并行任务写坏 canonical state。

   对策：并行只写 candidate artifacts，commit action 单点合并。

## 参考资料

- OpenAI Codex Cloud 文档：Codex 可读写执行代码、在云端后台并行处理任务，并使用任务专属环境。  
  https://platform.openai.com/docs/codex
- OpenAI Codex agent internet access 文档：Agent 网络访问需要按环境控制，并应审查工作日志。  
  https://platform.openai.com/docs/codex/agent-network
- OpenAI Codex use cases：包含 durable objective、skills、workflow automation 等用例方向。  
  https://developers.openai.com/codex/explore
- OpenAI Agents 文档：Agent 是为目标设计工作流并连接工具、评估与优化的系统。  
  https://platform.openai.com/docs/guides/agents
