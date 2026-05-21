# 作者对话式 Narrative ReAct 系统设计

## 目标

本设计描述一个作者与小说 Agent 共同创作的交互系统。系统不是一次性 prompt，也不是固定 pipeline，而是一个持续运行的 ReAct loop：

```text
作者输入
-> Agent 观察当前环境和故事状态
-> Agent 根据策略选择下一步动作
-> 调用分析、规划、检索、写作、保存等工具
-> 环境状态转移
-> Agent 形成新的观察
-> 继续对话、请求作者确认或执行下一步
```

核心目标：

- 作者主导创作方向。
- Agent 管理状态、记忆、分析、检索、蓝图、草稿和校验。
- 每个关键动作都形成可回放轨迹。
- 分析机制和续写机制都作为工具纳入动作空间。
- 下一轮写作可以从上轮状态快照继续，而不是重新开始。

长篇小说状态、三层上下文和压缩机制见：

- `NARRATIVE_LONGFORM_STATE_MEMORY_DESIGN_2026-05-22.md`

## 交互总览

一个典型创作会话如下：

```text
作者：我要分析这本小说。
Agent：观察当前 workspace，检查是否已有分析产物、状态快照、配置和目标文件。
Agent -> Tool：scan_workspace
Agent -> Tool：load_existing_state 或 analyze_source
Agent：如果小说过长，选择 chunk analysis -> chapter analysis -> global analysis。
Agent -> Tool：save_analysis
Agent：报告分析完成，但不泄露敏感正文。

作者：我们来规划剧情。
Agent：观察当前 story state、作者目标、开放剧情线、角色状态。
Agent -> Tool：propose_blueprint
Agent：给出章节蓝图、分段、字数、关键节点和禁止发展，等待作者确认。

作者：修改第二段，增加冲突，第三段减少字数。
Agent -> Tool：revise_blueprint
Agent：更新蓝图，再次等待确认。

作者：确认。
Agent -> Tool：confirm_blueprint
Agent -> Tool：persist_blueprint

作者：按蓝图开始写。
Agent：观察蓝图、当前状态、目标字数、已写内容。
Agent -> Tool：retrieve_context
Agent -> Tool：build_working_context
Agent -> Tool：generate_segment 或 generate_chapter
Agent -> Tool：evaluate_draft
Agent -> Tool：save_draft
Agent -> Tool：commit_state
Agent：输出完成摘要和文件路径，等待下一轮。
```

## 角色分工

### 作者

作者不是外部参数提供者，而是创作系统中的 human policy owner。

作者负责：

- 指定目标小说或已有分析目录。
- 指定写作方向。
- 提供后续情节设想。
- 确认或修改剧情蓝图。
- 决定每段字数和关键节点。
- 决定是否接受草稿、修复草稿或回滚。
- 提供偏好反馈。

作者不应该被迫处理：

- 原文切块。
- 上下文拼装。
- 状态抽取。
- 角色一致性校验。
- 记忆压缩。
- 工程路径和中间文件管理。

### Agent

Agent 是创作执行者和状态管理者。

Agent 负责：

- 观察当前 workspace、配置、文件、分析产物和 state snapshot。
- 判断缺少哪些信息。
- 选择下一步 action。
- 调用工具。
- 管理 `NarrativeTaskState` 和 `NarrativeWorkflowState`。
- 把作者自然语言意图转成蓝图、约束和成功标准。
- 在关键节点请求作者确认。
- 记录 trajectory。
- 保存分析、蓝图、草稿、状态快照和运行结果。

### 工具

工具只执行明确动作，不做全局决策。全局决策由 policy 根据 observation 作出。

## 状态模型

### Conversation State

对话状态记录作者和 Agent 的交互意图。

```python
ConversationState:
    conversation_id: str
    author_messages: list[AuthorMessage]
    agent_messages: list[AgentMessage]
    pending_confirmation: ConfirmationRequest | None
    author_preferences: dict[str, Any]
```

用途：

- 保存作者偏好。
- 保存蓝图确认历史。
- 保存修订原因。
- 区分“作者设想”与“canonical story state”。

### NarrativeTaskState

领域状态，保存小说世界本身：

- source documents
- source chunks
- characters
- character dynamic states
- relationships
- events
- plot threads
- world rules
- style profile
- memory atoms
- compressed memory
- evidence pack
- working context
- chapter plan
- draft
- reports

它回答：

```text
这本小说当前是什么状态？
角色知道什么？
剧情线推进到哪里？
哪些规则不能违反？
风格是什么？
```

### NarrativeWorkflowState

工作流状态，保存 Agent 当前执行到哪一步：

```python
NarrativeWorkflowState:
    phase: str
    current_goal: Goal
    current_author_request: AuthorRequest
    active_blueprint_id: str
    current_segment_index: int
    min_target_chars: int
    generated_char_count: int
    pending_questions: list[AuthorQuestion]
    last_observation_summary: str
    last_action_result: dict[str, Any]
```

它回答：

```text
现在该分析、规划、等待确认、写作、校验还是提交？
```

### Artifact State

产物状态保存所有可恢复文件：

- analysis directory
- source_analysis.json
- chapter_blueprint.json
- draft segments
- draft.txt
- state_snapshot.json
- workflow_snapshot.json
- trajectory.json
- run_result.json

它回答：

```text
如果明天继续写，从哪里恢复？
```

## Observation 设计

Observation 是 Agent 决策所需的“可见事实摘要”，不是全文。

### Workspace Observation

来自文件系统和配置：

```json
{
  "has_env": true,
  "llm_configured": true,
  "source_files": ["..."],
  "analysis_artifacts": ["source_analysis.json", "global_analysis.json"],
  "state_snapshots": ["state_snapshot.json"],
  "draft_outputs": ["draft.txt"]
}
```

### Story Observation

来自 `NarrativeTaskState`：

```json
{
  "story_id": "story-default",
  "chapter_count": 5,
  "characters_count": 12,
  "plot_threads_count": 4,
  "open_questions_count": 9,
  "world_rules_count": 18,
  "style_profile_available": true,
  "memory_atoms_count": 120
}
```

### Workflow Observation

来自 `NarrativeWorkflowState`：

```json
{
  "phase": "blueprint_proposed",
  "has_blueprint": true,
  "blueprint_confirmed": false,
  "target_total_chars": 30000,
  "segments_count": 5,
  "current_segment_index": 0,
  "needs_author_confirmation": true,
  "available_actions": ["revise_blueprint", "confirm_blueprint", "ask_author"]
}
```

### Tool Result Observation

工具执行后的结果摘要：

```json
{
  "last_action": "analyze_source",
  "success": true,
  "created_artifacts": ["source_analysis.json", "global_analysis.json"],
  "chunk_count": 42,
  "chapter_count": 5,
  "fallback_used": false
}
```

## Action Space 和工具设计

### Workspace Tools

#### `scan_workspace`

目的：观察当前工程环境。

输入：

```json
{
  "root": "D:/buff/agent-with-RL",
  "external_roots": ["D:/buff/narrative-state-engine"]
}
```

输出：

```json
{
  "env_exists": true,
  "analysis_dirs": [],
  "snapshots": [],
  "candidate_sources": []
}
```

决策用途：

- 如果已有分析，优先 `load_analysis`。
- 如果没有分析但有 source，进入 `analyze_source`。
- 如果缺少文件，进入 `ask_author`。

#### `load_state_snapshot`

目的：从上一轮继续。

输入：

```json
{
  "state_snapshot_path": "...",
  "workflow_snapshot_path": "..."
}
```

输出：

```json
{
  "loaded": true,
  "state_version_no": 6,
  "last_committed_chapter": 6
}
```

### Analysis Tools

#### `analyze_source`

目的：把小说原文变成 `NarrativeSourceAnalysis`。

内部机制：

```text
source text
-> source chunking
-> chunk analysis
-> chapter analysis
-> global analysis
-> merge into NarrativeSourceAnalysis
-> save analysis artifacts
```

输入：

```json
{
  "source_paths": ["..."],
  "chunk_chars": 12000,
  "analysis_repository_root": "...",
  "privacy_mode": true
}
```

输出：

```json
{
  "analysis_id": "...",
  "chunk_count": 42,
  "chapter_count": 5,
  "characters_count": 12,
  "plot_threads_count": 4,
  "style_profile_available": true,
  "artifacts_root": "..."
}
```

注意：

- 工具可以读敏感原文，但 Agent 对用户的摘要不展示正文。
- audit 日志需要支持关闭 prompt preview 或脱敏。
- chunk 大小应根据模型上下文和 JSON 稳定性调整，不应固定过小。

#### `load_analysis`

目的：从已落盘的分析结果恢复状态，不重新读原文。

输入：

```json
{
  "source_analysis_path": "..."
}
```

输出：

```json
{
  "state_built": true,
  "characters_count": 12,
  "events_count": 80,
  "plot_threads_count": 4
}
```

### Planning Tools

#### `propose_blueprint`

目的：根据作者方向和当前 story state 生成可确认蓝图。

输入：

```json
{
  "author_intent": "...",
  "target_total_chars": 30000,
  "hard_constraints": [],
  "story_observation": {},
  "open_plot_threads": [],
  "style_constraints": {}
}
```

输出：

```json
{
  "blueprint_id": "...",
  "chapter_goal": "...",
  "target_total_chars": 30000,
  "segments": [
    {
      "segment_id": "scene-1",
      "goal": "...",
      "target_chars": 6000,
      "required_beats": [],
      "forbidden_beats": [],
      "entry_state": "...",
      "exit_state": "..."
    }
  ],
  "requires_author_confirmation": true
}
```

决策规则：

- 生成蓝图后必须进入 `needs_blueprint_confirmation`。
- 未经作者确认，不能写正文。

#### `revise_blueprint`

目的：根据作者反馈修改蓝图。

输入：

```json
{
  "blueprint_id": "...",
  "author_feedback": "第二段增加冲突，第三段减少字数",
  "previous_blueprint": {}
}
```

输出：

```json
{
  "blueprint_id": "...",
  "revision_no": 2,
  "segments": []
}
```

#### `confirm_blueprint`

目的：把蓝图标记为可执行计划。

输出：

```json
{
  "confirmed": true,
  "confirmed_blueprint_id": "..."
}
```

### Retrieval and Context Tools

#### `retrieve_context`

目的：从故事状态和分析结果中取出当前蓝图相关证据。

输入：

```json
{
  "blueprint_id": "...",
  "segment_id": "optional",
  "query_type": "chapter_continuation"
}
```

输出：

```json
{
  "evidence_pack_id": "...",
  "author_evidence_count": 2,
  "character_evidence_count": 5,
  "plot_evidence_count": 6,
  "world_evidence_count": 4,
  "style_evidence_count": 4
}
```

#### `build_working_context`

目的：把 evidence pack、蓝图、作者约束、状态摘要压成模型可用上下文。

输出：

```json
{
  "context_id": "...",
  "section_count": 8,
  "estimated_tokens": 3000
}
```

### Writing Tools

#### `generate_chapter`

目的：一次性生成整章。

适合：

- 目标字数较短。
- 模型输出窗口足够。
- 作者强烈要求一次生成。

风险：

- 长文容易截断。
- 后半段可能偏离蓝图。
- 难以分段校验。

#### `generate_segment`

目的：按蓝图逐段生成。

输入：

```json
{
  "blueprint_id": "...",
  "segment_id": "scene-2",
  "target_chars": 6000,
  "previous_segment_summary": "...",
  "previous_tail": "...",
  "working_context": "...",
  "hard_constraints": []
}
```

输出：

```json
{
  "segment_id": "scene-2",
  "draft_segment_id": "...",
  "char_count": 6200,
  "continuity_notes": [],
  "needs_followup": false
}
```

建议默认使用 `generate_segment`，因为它更符合 ReAct：

```text
写一段 -> 观察结果 -> 校验 -> 决定继续/修复/询问作者
```

#### `merge_draft_segments`

目的：合并各段为整章草稿。

输出：

```json
{
  "draft_id": "...",
  "total_chars": 31200,
  "segment_count": 5
}
```

### Evaluation Tools

#### `evaluate_draft`

目的：检查草稿是否满足蓝图和状态约束。

检查项：

- 必写节点是否完成。
- 禁止发展是否发生。
- 角色知识边界是否被破坏。
- 角色关系是否无铺垫突变。
- 世界规则是否违反。
- 风格是否明显偏离。
- 字数是否满足。

输出：

```json
{
  "status": "passed",
  "overall_score": 0.86,
  "blocking_issues": [],
  "warnings": []
}
```

#### `repair_draft`

目的：根据评估报告修复草稿。

决策规则：

- blocking 问题默认不自动提交。
- 可自动修复小问题。
- 涉及剧情方向的大修必须问作者。

### Memory and Persistence Tools

#### `extract_state_changes`

目的：从草稿抽取候选状态变化。

输出：

```json
{
  "changes": [
    {
      "update_type": "event",
      "summary": "...",
      "confidence": 0.82
    }
  ]
}
```

#### `commit_state`

目的：把通过校验的变化写入 canonical state。

输出：

```json
{
  "committed": true,
  "state_version_no": 7,
  "memory_atoms_added": 12,
  "compressed_memory_blocks": 2
}
```

#### `save_artifacts`

目的：保存蓝图、草稿、状态、轨迹。

输出文件：

```text
chapter_blueprint.json
draft_segments/*.txt
draft.txt
state_snapshot.json
workflow_snapshot.json
trajectory.json
run_result.json
```

## Policy 设计

### Deterministic Author-Led Policy

第一阶段使用确定性策略，不依赖 LLM 选择动作。

伪代码：

```python
if missing_required_input:
    ask_author
elif no_state and existing_snapshot:
    load_state_snapshot
elif no_analysis:
    analyze_source
elif author_asks_planning and no_blueprint:
    propose_blueprint
elif blueprint_proposed and not confirmed:
    wait_for_author_confirmation
elif author_feedback_on_blueprint:
    revise_blueprint
elif author_confirmed_blueprint:
    confirm_blueprint
elif ready_to_write and no_evidence:
    retrieve_context
elif evidence_ready and no_context:
    build_working_context
elif context_ready and has_unwritten_segment:
    generate_segment
elif segment_ready:
    evaluate_segment
elif all_segments_ready:
    merge_draft_segments
elif draft_ready:
    evaluate_draft
elif evaluation_has_blockers:
    ask_author_or_repair
elif evaluation_passed:
    commit_state
else:
    stop
```

优点：

- 可测试。
- 可解释。
- 不会越过作者确认。
- 后续可以替换成 LLM policy。

### LLM Policy

第二阶段可以让 LLM policy 在动作空间中选择动作，但必须受 guardrail 限制。

LLM policy 输入：

- 当前 observation。
- 可用 action 列表。
- 每个 action 的 precondition。
- 作者确认状态。
- 最近 trajectory。

LLM policy 输出：

```json
{
  "action_name": "retrieve_context",
  "payload": {},
  "rationale": "蓝图已确认，但还没有证据包，必须先检索上下文。"
}
```

Guardrail 规则：

- 未确认蓝图不能写正文。
- 缺少 state 不能检索。
- blocking 校验不能 commit。
- 作者明确禁止的内容不能作为写作目标。

## 人在环路中的作用

人不是每一步都要审批，但必须控制方向。

### 必须问作者

- 缺少写作目标。
- 缺少原文或分析状态。
- 蓝图首次生成。
- 蓝图字数分配或关键节点不明确。
- 校验出现 blocking 问题。
- Agent 需要改变作者已确认的剧情方向。

### 可以自动执行

- 扫描 workspace。
- 加载已有 state。
- 切块分析。
- 检索上下文。
- 装配 working context。
- 按已确认蓝图写当前段。
- 保存草稿和轨迹。
- 非阻塞评价。

### 作者反馈如何进入状态

作者反馈分三类：

1. Intent

   写作目标和剧情方向。

2. Constraint

   禁止或必须发生的内容。

3. Preference

   风格偏好、节奏偏好、角色处理偏好。

它们不应全部混成 prompt，而应结构化为：

- `AuthorRequest`
- `AuthorConstraint`
- `ChapterBlueprint`
- `ConversationState.author_preferences`

## 分析机制纳入 ReAct

原有分析机制保留，但成为工具：

```text
observe source length and existing artifacts
-> decide analyze_source
-> chunk source
-> analyze chunks
-> analyze chapters
-> analyze global story
-> merge source analysis
-> save artifacts
-> observe analysis summary
```

如果上下文足够大，chunk 可以调大。但仍建议保留 chunk/chapter/global 三层：

- chunk 负责局部事实和证据。
- chapter 负责章节结构和状态变化。
- global 负责角色表、剧情线、世界规则、风格画像。

这样续写时可以检索不同粒度的信息，而不是只依赖一个超长摘要。

## 续写机制纳入 ReAct

续写不再只是 `generate_draft` 一步，而是：

```text
confirmed blueprint
-> retrieve_context for chapter
-> for each segment:
     retrieve_context for segment
     build_segment_context
     generate_segment
     evaluate_segment
     maybe repair_segment
     save_segment
-> merge_draft_segments
-> evaluate_draft
-> extract_state_changes
-> commit_state
-> save_snapshot
```

如果作者要求一次性生成，可以使用 `generate_chapter`，但系统应记录风险：

```json
{
  "risk": "long_output_truncation",
  "mitigation": "fallback to segment generation"
}
```

## 状态转移示例

### 分析阶段

```text
phase=initialized
observation: no state, source path provided
decision: analyze_source
action: analyze_source
transition: source_analysis saved
next_observation: analysis_ready
```

### 规划阶段

```text
phase=analysis_ready
observation: story state available, author asks for plot planning
decision: propose_blueprint
action: propose_blueprint
transition: blueprint created
next_observation: blueprint_proposed, needs_author_confirmation=true
```

### 作者确认

```text
phase=blueprint_proposed
observation: blueprint exists but not confirmed
decision: wait_for_author
action: ask_author
transition: paused
next_observation: waiting_confirmation
```

### 写作阶段

```text
phase=blueprint_confirmed
decision: retrieve_context
transition: evidence_ready

phase=evidence_ready
decision: generate_segment
transition: segment_drafted

phase=segment_drafted
decision: evaluate_segment
transition: segment_passed

phase=all_segments_passed
decision: merge_draft_segments
transition: draft_ready
```

### 提交阶段

```text
phase=draft_ready
decision: extract_state_changes
transition: changes_pending

phase=changes_pending
decision: evaluate_draft
transition: evaluated

phase=evaluated
decision: commit_state
transition: committed
```

## 轨迹记录

每个步骤记录：

```json
{
  "index": 4,
  "observation": {"phase": "blueprint_confirmed"},
  "action": {"name": "retrieve_context", "kind": "tool"},
  "rationale": "蓝图已确认，需要为写作检索证据。",
  "reward": {
    "value": 0.1,
    "dimensions": {"evidence_count": 28}
  },
  "next_observation": {"phase": "evidence_ready"},
  "metadata": {
    "tool_result": {},
    "artifacts": []
  }
}
```

轨迹不应保存敏感正文全文，只保存摘要、计数、id、路径、hash 和状态变化。

## 隐私和审计

对于敏感小说原文和作者大纲：

- Agent 不主动打印正文。
- LLM audit 可配置关闭 prompt preview。
- 输出给用户的摘要只包含状态计数和产物路径。
- trajectory 默认不保存长正文。
- draft 正文单独保存到受控 artifact 文件。

## 第一阶段实现范围

当前项目已经有最小 ReAct 环境。下一阶段建议实现：

1. `ConversationState`
2. `NarrativeArtifactRepository`
3. `load_analysis` tool
4. `save_state_snapshot` / `load_state_snapshot`
5. segment 级写作工具：

   - `generate_segment`
   - `evaluate_segment`
   - `merge_draft_segments`

6. CLI：

   - `narrative_chat_session`
   - `narrative_plan`
   - `narrative_write_from_blueprint`
   - `narrative_resume`

## 最终形态

最终作者体验应该是：

```text
作者：分析这本小说。
Agent：我看到没有已有状态，将进行长文分析。是否关闭审计预览？
作者：关闭，开始。
Agent：分析完成，识别出 5 章、12 个角色、4 条剧情线、若干世界规则。

作者：我们规划下一章，3 万字。
Agent：这是蓝图，分 5 段，每段目标字数和关键节点如下。请确认或修改。
作者：第二段加冲突，第四段减少 1000 字。
Agent：已修订。是否确认？
作者：确认。

作者：开始写。
Agent：我将按段写作并保存。第 1 段完成，已通过校验。继续第 2 段。
...
Agent：整章完成，共 31200 字，已保存 draft.txt，状态已提交，可继续下一章。
```

这就是作者主导、Agent 执行、工具可观测、状态可恢复的小说创作系统。
