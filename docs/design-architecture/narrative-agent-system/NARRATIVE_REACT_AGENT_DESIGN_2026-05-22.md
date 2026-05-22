# Narrative ReAct Agent 设计

## 背景

当前小说写作链路已经可以跑通：

```text
分析原文 -> 生成章节蓝图 -> 检索上下文 -> 生成草稿 -> 抽取变化 -> 校验 -> 提交状态
```

这个链路证明了领域能力可用，但它仍然更像一个固定 pipeline。项目的核心概念已经在 `agent_rl.core` 中定义：

- `Goal`
- `Observation`
- `Action`
- `Decision`
- `AgentState`
- `Environment`
- `Policy`
- `Trajectory`
- `Reward`

下一步目标不是把小说场景塞进 core，而是让小说场景成为一个真正的 ReAct-style environment：Agent 有目标，观察当前故事和工作流状态，选择动作，调用工具，获得新观察，再决定下一步。

## 设计原则

1. Core 保持通用

   `agent_rl.core.concepts` 不承载小说专用对象。小说中的角色、剧情线、蓝图、草稿、作者确认等概念留在 `agent_rl.domains.narrative` 和 `agent_rl.narrative_writing`。

2. 小说场景实现 Environment

   新增 `NarrativeReActEnvironment`，实现 core 的 `Environment` 协议：

   ```python
   reset() -> tuple[Observation, JsonMap]
   available_actions() -> Sequence[Action]
   step(action: Action) -> Transition
   close() -> None
   ```

3. 作者主导

   Agent 不能越过作者确认写作关键决策。至少这些阶段应显式停下来等待作者：

   - 缺少参考材料或写作方向
   - 蓝图已生成但未确认
   - 蓝图需要修订
   - 草稿校验失败，需要作者选择修复方向

4. 分析和续写都是工具动作

   分析不是启动前的隐式准备，续写也不是唯一主流程。它们都是 action space 中的工具动作：

   - `load_analysis`
   - `analyze_source`
   - `propose_blueprint`
   - `revise_blueprint`
   - `retrieve_context`
   - `build_working_context`
   - `generate_draft`
   - `generate_segment`
   - `evaluate_draft`
   - `repair_draft`
   - `commit_state`
   - `rollback`
   - `ask_author`
   - `stop`

5. 轨迹优先

   每一步都要形成 `TrajectoryStep`，记录：

   - 当前观察
   - 选择的动作
   - 选择理由
   - 动作结果
   - 评价信号
   - 状态变化摘要

   这为后续训练、评估、回放和调试留下统一数据。

## 概念映射

### Goal

小说写作目标由作者主导，描述本轮任务和成功标准。

示例：

```python
Goal(
    description="基于作者后续情节续写第 6 章",
    success_criteria=(
        "正文至少 30000 字",
        "遵守作者蓝图",
        "保持角色和剧情连续性",
        "通过校验后提交状态",
    ),
)
```

### State Space

状态分两层：

1. `NarrativeTaskState`

   领域状态，保存故事和写作材料：

   - source documents
   - source chunks
   - characters
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
   - pending changes
   - reports

2. `NarrativeWorkflowState`

   ReAct 工作流状态，保存执行阶段：

   - 当前 phase
   - 作者请求
   - 是否需要作者输入
   - 是否需要蓝图确认
   - 当前 action 历史
   - 已确认蓝图 id
   - 当前生成段落索引
   - 最小字数要求
   - 输出路径或快照路径

### Long-Running Session

`NarrativeWritingAgent.run(...)` 适合一次性命令行或脚本调用。要做接近 Codex 的持续工作方式，应该使用 `NarrativeWritingSession`：

```python
session = NarrativeWritingSession(request)
result = session.run_until_pause()

if result.requires_confirmation:
    session.apply_author_input(confirm_plan=True)
    result = session.run_until_pause()
```

Session 会保留同一个 `NarrativeReActEnvironment`、`NarrativeWorkflowState`、`NarrativeTaskState`、`MemoryStore` 和 `Trajectory`，因此可以在作者确认、补充方向、后续修订时继续运行，而不是每次重新启动一个完整 pipeline。

当前 Session 能处理的暂停点：

- 缺少参考小说或写作方向：`needs_author_input`
- 蓝图生成后等待作者确认：`needs_confirmation`
- 正常提交：`committed`
- 校验失败回滚：`rolled_back`

后续要继续增强的方向是把 `repair_draft`、多轮作者反馈、分支草稿选择和长期后台任务也接入同一个 Session。

### Observation

Observation 不是全文，也不是完整状态 dump，而是 Agent 当前决策所需的摘要。

建议 payload：

```python
{
    "phase": "blueprint_proposed",
    "story_id": "...",
    "task_id": "...",
    "goal": "...",
    "has_analysis": true,
    "has_blueprint": true,
    "blueprint_confirmed": false,
    "has_evidence": false,
    "has_draft": false,
    "draft_char_count": 0,
    "questions": [...],
    "available_action_names": [...],
}
```

### Action Space

动作分为四类：

1. `control`

   - `ask_author`
   - `wait_for_confirmation`
   - `stop`

2. `tool`

   - `load_analysis`
   - `analyze_source`
   - `retrieve_context`
   - `build_working_context`

3. `planning`

   - `propose_blueprint`
   - `revise_blueprint`
   - `confirm_blueprint`

4. `writing`

   - `generate_draft`
   - `generate_segment`
   - `evaluate_draft`
   - `repair_draft`
   - `commit_state`
   - `rollback`

### Decision

Policy 输出 `Decision`，包含动作和理由。

早期可以先实现确定性策略 `NarrativeAuthorLedPolicy`，按照 phase 选择下一步：

```text
needs_author_input -> ask_author
ready_for_analysis -> analyze_source 或 load_analysis
ready_for_blueprint -> propose_blueprint
blueprint_proposed -> wait_for_confirmation
blueprint_confirmed -> retrieve_context
context_ready -> generate_draft 或 generate_segment
draft_ready -> evaluate_draft
evaluated -> commit_state 或 repair_draft
completed -> stop
```

后续可以替换为 LLM policy 或 multi-agent policy。

### Trajectory

每一步 action 都记录到 trajectory。与当前固定 pipeline 不同，trajectory 是一等产物，可以被保存、回放和评估。

### Reward

奖励先弱化为工程评价信号：

- 是否生成蓝图
- 检索证据数量
- 是否满足最小字数
- 评估报告平均分
- 是否 blocking
- 是否 commit

后续再接作者反馈或偏好学习。

## 作者主导对话状态

建议新增 phase：

```text
initialized
needs_author_input
ready_for_analysis
analysis_ready
ready_for_blueprint
blueprint_proposed
blueprint_confirmed
context_ready
drafting
draft_ready
evaluated
needs_repair_decision
committed
rolled_back
completed
```

作者交互规则：

1. 没有写作方向，必须问作者。
2. 没有 reference 且没有已有 state，必须问作者。
3. 生成蓝图后默认停止，等待确认。
4. 蓝图确认后才能进入续写。
5. 校验失败时默认停止，等待作者选择修复或放弃。

## 章节蓝图升级

当前 `ChapterBlueprint` 只有章级目标和 required beats。长篇写作需要 segment 级蓝图。

建议新增：

```python
@dataclass
class ChapterBlueprintSegment:
    segment_id: str
    title: str = ""
    goal: str = ""
    target_chars: int = 0
    required_beats: list[str] = field(default_factory=list)
    forbidden_beats: list[str] = field(default_factory=list)
    involved_character_ids: list[str] = field(default_factory=list)
    plot_thread_ids: list[str] = field(default_factory=list)
    entry_state: str = ""
    exit_state: str = ""
```

然后在 `ChapterBlueprint` 增加：

```python
target_total_chars: int = 0
segments: list[ChapterBlueprintSegment] = field(default_factory=list)
requires_author_confirmation: bool = True
confirmed: bool = False
```

这样 Agent 可以先问作者：

```text
本章计划 30000 字，分 5 段：
1. 5000 字，目标...
2. 7000 字，目标...
...
是否确认？
```

## 写作策略

长篇续写有两种执行方式：

1. 单次整章生成

   适合模型输出足够长、章节较短、对连贯性要求高的场景。

2. 分段生成

   适合 30000 字以上正文。每次生成一个 segment，但每次都带：

   - 全章蓝图
   - 当前 segment 目标
   - 已写内容摘要
   - 上一段尾部
   - 人物和剧情约束
   - 风格约束

默认应采用分段生成，因为它更稳定，也能在每段后观察状态并修复。

## 快照和续写

要支持“下一次继续写”，必须持久化：

- `state_snapshot.json`
- `workflow_snapshot.json`
- `trajectory.json`
- `chapter_blueprint.json`
- `draft.txt`
- `run_result.json`

下一轮从 snapshot 启动，而不是重新读原文。

## 实现步骤

1. 扩展 narrative domain：

   - `ChapterBlueprintSegment`
   - `ChapterBlueprint.target_total_chars`
   - `ChapterBlueprint.segments`
   - `ChapterBlueprint.confirmed`

2. 新增 `narrative_writing/react.py`：

   - `NarrativeWorkflowState`
   - `NarrativeWorkflowPhase`
   - `NarrativeReActEnvironment`
   - `NarrativeAuthorLedPolicy`

3. 让 `NarrativeWritingAgent` 使用 ReAct runtime，保留旧接口兼容。

4. 把固定 pipeline 中的动作迁移到 environment step。

5. 补测试：

   - 缺信息时 ask_author
   - 未确认蓝图时停止
   - 确认后完整跑到 commit
   - trajectory 中包含 ReAct 动作
   - segment 字数分配进入 blueprint

6. 后续再加：

   - state snapshot repository
   - resume from snapshot
   - LLM policy
   - repair loop

## 近期落地目标

本轮先完成 ReAct 化的最小闭环：

```text
AuthorRequest
-> NarrativeReActEnvironment
-> AgentRuntime
-> NarrativeAuthorLedPolicy
-> ask/propose/retrieve/context/write/evaluate/commit
-> NarrativeRunResult
```

旧的 CLI 和测试应继续可用。新的 ReAct 轨迹应能体现 Agent 的观察、决策、动作和状态转移。
