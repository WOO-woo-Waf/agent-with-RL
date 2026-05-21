# 长篇小说状态、分层上下文与压缩设计

## 目标

长篇小说 Agent 不能依赖单次上下文窗口，也不能把所有历史都塞给模型。它需要一套长期运行的状态维护系统：

```text
真实可续写剧情
-> 章节/角色抽象剧情
-> 全局高度压缩剧情
```

目标：

- 当前章节写作时能保留必要细节。
- 远期历史不会占满上下文，但不会被遗忘。
- 伏笔、角色弧线、关系变化、世界规则和作者风格能持续维护。
- Agent 每一步都能观察工具结果、状态摘要和可用动作。
- 分析、规划、续写三个大目标都能通过动作空间和工具实现。

## 当前旧系统分析状态在哪里

旧系统吸收过来的分析结果现在属于状态空间的一部分，主要有两层保存位置。

### 1. 持久化产物

深度分析会落盘到：

```text
artifacts/narrative/{story_id}/{task_id}/
  source_analysis.json
  source_documents.json
  source_chunks.jsonl
  chunk_analysis.jsonl
  chapter_analysis.jsonl
  global_analysis.json
  manifest.json
```

它们是长期可恢复的 analysis artifacts。

用途：

- `source_chunks.jsonl`：原文或参考材料的检索单元。
- `chunk_analysis.jsonl`：局部细节、事件、角色提及、风格特征。
- `chapter_analysis.jsonl`：章节级摘要、事件链、人物状态变化、伏笔、开放问题。
- `global_analysis.json`：故事级人物表、剧情线、世界规则、风格画像、连续性约束。
- `source_analysis.json`：把以上分析聚合成一个完整快照。

### 2. 运行时状态空间

运行时通过 `NarrativeTaskState` 承载：

```python
NarrativeTaskState:
    source_analyses
    source_documents
    source_chunks
    characters
    character_states
    relationships
    events
    plot_threads
    world_rules
    style_profile
    style_snippets
    memory_atoms
    compressed_memory
    evidence_pack
    working_context
    chapter_plan
    draft
    reports
```

其中 `NarrativeSourceAnalysis` 又包含：

```python
NarrativeSourceAnalysis:
    chunk_analyses
    chapter_analyses
    global_analysis
    characters
    events
    plot_threads
    world_rules
    style_profile
    style_snippets
    memory_atoms
```

所以旧系统分析出来的角色、事件、剧情线、世界规则、风格和章节分析不是外部附属物，而应该被视为小说 Agent 的 canonical state seed。后续要做的是让这些状态可加载、可检索、可压缩、可更新。

## 三层上下文模型

### L1 可续写剧情层

这是模型真正用于当前续写的近场上下文。

内容：

- 最近章节全文或片段。
- 当前段落上一段尾部。
- 当前场景 entry/exit state。
- 当前角色所在位置、情绪、目标、已知事实。
- 当前剧情线最近事件。
- 当前必须回收或强化的伏笔。
- 作者刚确认的蓝图和段落目标。

特点：

- 细节多。
- 直接进入模型 working context。
- 时间范围短，通常覆盖当前章、上一章、关键相邻章。
- 权重最高。

典型对象：

- `SourceChunk`
- `StyleSnippet`
- `NarrativeEvidence`
- `WorkingMemoryContext`
- `ChapterBlueprintSegment`
- `DraftCandidate`

### L2 章节/角色抽象剧情层

这是中层记忆，回答“角色和剧情长期发展到哪里了”。

内容：

- 章节摘要。
- 章节事件链。
- 角色状态更新。
- 关系变化。
- 章节级开放问题。
- 每条剧情线当前阶段。
- 伏笔状态。
- 重要物品、地点、组织的状态。

特点：

- 不保存全文。
- 以章节、角色、关系、剧情线为单位。
- 需要按当前任务检索。
- 可以进入 working context 的摘要区。

典型对象：

- `ChapterAnalysisResult`
- `NarrativeEvent`
- `CharacterDynamicState`
- `RelationshipState`
- `PlotThreadState`
- `ForeshadowingState`
- `MemoryAtom`

### L3 全局高度压缩层

这是远场记忆，回答“这本书整体是什么”。

内容：

- 故事总梗概。
- 角色注册表。
- 世界规则。
- 关系图。
- 全局剧情线。
- 风格 bible。
- 作者长期偏好。
- 全局禁忌和连续性约束。

特点：

- 高度压缩。
- 与当前无关时只提供很少摘要。
- 只在规划、校验、长线伏笔回收时强化进入上下文。
- 用于防止遗忘，不用于替代当前细节。

典型对象：

- `GlobalStoryAnalysisResult`
- `StyleProfile`
- `WorldRule`
- `CompressedMemoryBlock`
- `AuthorConstraint`

## 优先级和重要性

上下文选择不能只按相似度，还要按重要性和任务相关性综合评分。

建议每个可检索对象带这些字段：

```python
importance: float
freshness: float
relevance: float
canonical: bool
recency_rank: int
source_layer: Literal["near", "mid", "global"]
continuity_risk: Literal["low", "medium", "high"]
author_priority: Literal["low", "normal", "high", "must"]
```

### 优先级来源

1. 作者显式指定

   作者说“这一点必须保留”“不要忘记这个伏笔”，优先级最高。

2. Canonical 稳定性

   已确认的 canon 高于候选状态。

3. 连续性风险

   角色知识边界、关系状态、世界规则、伏笔回收属于高风险。

4. 时间近因

   最近章节和当前章节相关性高。

5. 检索相关性

   与当前蓝图、段落目标、角色、地点、剧情线相关。

6. 风格代表性

   风格片段不一定剧情相关，但对续写质量重要。

## 压缩机制

### 压缩不是删除

压缩应该产生新的 state object，同时保留 provenance：

```python
CompressedMemoryBlock:
    block_id
    block_type
    scope
    summary
    key_points
    preserved_ids
    dropped_ids
    compression_ratio
    valid_until_state_version
```

`preserved_ids` 和 `dropped_ids` 很重要：它让 Agent 知道摘要来自哪些源事实，也知道哪些细节被省略了，必要时可以回查。

### 压缩时机

1. 分析完成后

   把 chunk/chapter/global 分析合成初始 memory。

2. 每章提交后

   把新草稿抽取成事件、角色变化、关系变化和压缩记忆。

3. 上下文预算不足时

   对低优先级历史进行二次压缩。

4. 规划长线剧情前

   重新压缩剧情线和伏笔，帮助生成全局蓝图。

5. 作者修改 canon 后

   标记旧压缩块为过期，重新生成相关摘要。

### 压缩层级

#### Chunk Compression

输入：原文 chunk 或 draft segment。

输出：

- 局部事件。
- 角色提及。
- 信息揭示。
- 风格特征。
- 检索关键词。

#### Chapter Compression

输入：本章所有 chunk/segment。

输出：

- 章节梗概。
- 事件链。
- 角色状态变化。
- 关系变化。
- 伏笔变化。
- 下一章入口。

#### Arc Compression

输入：多章或一条剧情线。

输出：

- 剧情线阶段。
- 已解决问题。
- 未解决问题。
- 阻塞发展。
- 未来可推进方向。

#### Global Compression

输入：全书状态。

输出：

- 故事总览。
- 角色表。
- 世界规则。
- 风格 bible。
- 全局连续性约束。

## 上下文装配策略

写作时模型不应该直接看“所有状态”，而应该看一个预算化的 `WorkingMemoryContext`。

推荐结构：

```text
1. 作者当前目标和硬约束
2. 已确认章节蓝图
3. 当前 segment 目标和字数
4. 近场剧情细节
5. 当前角色状态
6. 当前剧情线和伏笔
7. 世界规则和禁忌
8. 风格片段
9. 高度压缩全局摘要
10. 上一段尾部
```

预算分配示例：

```text
作者目标和蓝图：15%
当前 segment：15%
近场剧情：25%
角色/关系：15%
剧情线/伏笔：10%
世界规则：5%
风格：10%
全局摘要：5%
```

这个比例应可配置，后续可以按任务调整：

- 情节规划：提高全局摘要、剧情线、伏笔比例。
- 续写：提高近场剧情、角色状态、风格比例。
- 分析：提高 source chunk 和 chapter context 比例。

## 三大目标和动作空间

当前 Agent 主要支持三个作者目标：

```text
1. 情节分析
2. 剧情规划
3. 小说续写
```

每个目标对应一组动作和辅助工具。

### 目标一：情节分析

目的：把原文或草稿变成可维护状态。

动作空间：

- `scan_workspace`
- `load_source`
- `split_source`
- `analyze_chunk`
- `analyze_chapter`
- `analyze_global_story`
- `extract_characters`
- `extract_plot_threads`
- `extract_world_rules`
- `extract_style_profile`
- `extract_foreshadowing`
- `compress_analysis`
- `save_analysis`
- `build_initial_state`

观察：

- source length
- chunk count
- analysis coverage
- fallback count
- characters count
- chapter count
- open questions count
- artifact paths

状态转移：

```text
no_analysis -> analyzing_chunks -> analyzing_chapters -> analyzing_global -> analysis_ready
```

### 目标二：剧情规划

目的：作者与 Agent 共同形成可执行蓝图。

动作空间：

- `observe_story_state`
- `retrieve_plot_context`
- `retrieve_character_arcs`
- `retrieve_foreshadowing`
- `propose_blueprint`
- `allocate_segment_chars`
- `revise_blueprint`
- `validate_blueprint`
- `ask_author_for_confirmation`
- `confirm_blueprint`
- `save_blueprint`

观察：

- author intent
- current plot threads
- unresolved conflicts
- character arcs
- foreshadowing states
- target total chars
- segment count
- author feedback

状态转移：

```text
analysis_ready -> planning -> blueprint_proposed -> blueprint_revision -> blueprint_confirmed
```

### 目标三：小说续写

目的：按已确认蓝图生成并提交正文。

动作空间：

- `retrieve_chapter_context`
- `retrieve_segment_context`
- `build_working_context`
- `generate_segment`
- `evaluate_segment`
- `repair_segment`
- `merge_draft_segments`
- `evaluate_draft`
- `extract_state_changes`
- `compress_new_draft`
- `commit_state`
- `save_draft`
- `save_state_snapshot`

观察：

- confirmed blueprint
- current segment index
- target chars
- generated chars
- segment evaluation report
- draft evaluation report
- pending state changes
- memory compression result

状态转移：

```text
blueprint_confirmed
-> context_ready
-> writing_segment_n
-> segment_evaluated
-> all_segments_written
-> draft_evaluated
-> state_committed
```

## 工具结果如何进入观察

工具不能只返回给调用方，它们的结果必须变成下一步 observation。

示例：

```json
{
  "last_action": "compress_new_draft",
  "success": true,
  "result": {
    "memory_atoms_added": 12,
    "compressed_blocks_added": 3,
    "high_priority_items": 4,
    "continuity_risks": ["角色A知道秘密B的时间点需要校验"]
  },
  "next_available_actions": [
    "commit_state",
    "repair_segment",
    "ask_author"
  ]
}
```

这样 Agent 才能基于工具结果继续思考：

```text
如果 compression 发现高风险连续性问题 -> evaluate_draft 或 ask_author
如果 segment 字数不足 -> repair_segment
如果 evaluation passed -> commit_state
```

## 长篇状态维护系统

### State Repository

需要一个持久化仓库：

```python
NarrativeStateRepository:
    save_state_snapshot(state)
    load_state_snapshot(story_id, version)
    save_workflow_snapshot(workflow)
    save_trajectory(trajectory)
    save_memory_blocks(blocks)
    load_memory_blocks(query)
```

本地第一版可以是 JSON/JSONL：

```text
artifacts/narrative-state/{story_id}/
  state_snapshots/state-v0007.json
  workflow_snapshots/run-xxxx.json
  trajectories/run-xxxx.json
  memory_atoms.jsonl
  compressed_memory.jsonl
  blueprints/chapter-006.json
  drafts/chapter-006.txt
```

后续可替换为 SQLite/Postgres/Vector store。

### Memory Index

需要按多维度索引：

- character_id
- plot_thread_id
- chapter_index
- location_id
- object_id
- foreshadowing_id
- importance
- freshness
- source_layer
- state_version_no

检索时不是只查关键词，而是：

```text
query = author goal + blueprint + current segment + involved characters + plot threads
```

## 小说状态字段演进

当前 `NarrativeTaskState` 已经有基础字段，但长篇小说还需要逐步增强：

### 伏笔

已有 `ForeshadowingState`，后续建议增强：

```python
payoff_policy
last_reinforced_chapter
risk_if_forgotten
author_priority
```

### 作者风格

已有 `StyleProfile` 和 `StyleSnippet`，后续建议增强：

```python
scene_style_profiles
dialogue_style_by_character
pacing_by_arc
forbidden_style_patterns
positive_style_examples
```

### 角色动态

已有 `CharacterCard` 和 `CharacterDynamicState`，后续建议增强：

```python
knowledge_boundary
emotional_debt
relationship_pressure
current_mask
private_motivation
```

### 世界规则

已有 `WorldRule`，后续建议增强：

```python
violation_examples
required_implications
scope_by_location_or_arc
confidence
last_confirmed_source
```

### 情节线

已有 `PlotThreadState`，后续建议增强：

```python
arc_stage
planned_payoff
blocked_by
must_not_resolve_before
author_priority
```

## ReAct 决策示例

### 作者要求分析

```text
Observation:
  source_path exists
  no source_analysis
  LLM configured

Decision:
  analyze_source

Action:
  analyze_source(chunk_chars=12000)

Transition:
  analysis artifacts saved
  NarrativeTaskState initialized
```

### 作者要求规划

```text
Observation:
  analysis_ready
  author_intent exists
  open plot threads available

Decision:
  retrieve_plot_context
  propose_blueprint

Transition:
  blueprint_proposed
  needs_author_confirmation=true
```

### 作者确认蓝图后续写

```text
Observation:
  blueprint_confirmed
  segment 1 not written

Decision:
  retrieve_segment_context
  build_working_context
  generate_segment

Transition:
  segment_drafted
  generated_char_count updated
```

### 写完一章后压缩

```text
Observation:
  draft_ready
  evaluation_passed

Decision:
  extract_state_changes
  compress_new_draft
  commit_state
  save_state_snapshot

Transition:
  state_version_no += 1
  memory_atoms and compressed_memory updated
```

## 设计约束

1. 长正文不直接进入 trajectory。

   trajectory 保存 id、路径、hash、摘要和计数。

2. 分析状态必须可恢复。

   不允许只存在于一次进程内存中。

3. 压缩结果必须带 provenance。

   任何摘要都要能回查来源。

4. 作者确认是硬门。

   未确认蓝图不能执行写作。

5. 高风险状态优先。

   角色知识边界、世界规则、伏笔和作者硬约束应优先进入上下文。

6. 状态字段可扩展。

   后续新增伏笔、风格、角色、世界规则字段时，不应破坏 core ReAct runtime。

## 落地顺序

1. 当前已经完成：

   - ReAct environment 最小闭环。
   - ChapterBlueprint segment 和总字数。
   - 旧接口兼容。

2. 下一步：

   - 增加 `NarrativeStateRepository`。
   - 增加 state snapshot 保存/恢复。
   - 增加 `load_analysis` tool。
   - 增加三层上下文选择策略。
   - 增加 `compress_new_draft` tool。

3. 再下一步：

   - segment 级生成和评估。
   - 伏笔状态增强。
   - 作者偏好状态。
   - 多轮对话会话管理。
   - SQLite/向量索引适配。

## 总结

长篇小说 Agent 的核心不是“生成更长”，而是：

```text
维护状态
-> 分层压缩
-> 按目标检索
-> 按作者蓝图写作
-> 抽取变化
-> 重新压缩并提交状态
```

这样模型既不会被前文细节淹没，也不会忘记关键 canon、伏笔、角色弧线和作者风格。
