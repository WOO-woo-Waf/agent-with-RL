# Narrative Long-Running Agent Implementation Status

Date: 2026-05-22

## 目标

本轮实现把小说写作场景从一次性 pipeline 推进为可持续运行的 agent session。核心目标不是只生成文本，而是让 agent 在小说领域内持续维护：

- 作者会话和长期偏好
- 可恢复的任务状态、workflow、trajectory、memory
- 可确认和可修订的章节蓝图
- 可评估、可修复、可提交的草稿
- 可供作者选择的多分支候选
- 可通过 CLI/API 层继续推进的后台式任务

## 已落地能力

### 1. Durable Session

入口：`NarrativeWritingSession`

能力：

- `run_until_pause()`：持续运行直到需要作者输入、蓝图确认、分支选择或提交完成。
- `apply_author_input()`：把作者的新方向、蓝图反馈、确认、分支选择合并回同一个 session。
- `save()` / `resume()`：恢复 `NarrativeTaskState`、`NarrativeWorkflowState`、`Trajectory`、`AuthorConversation`、`Observation` 和 in-memory trajectory。
- 自动 checkpoint：每一步 agent action 后都会保存 session snapshot，进程中断后可继续。

### 2. Author Conversation

领域对象：

- `AuthorMessage`
- `AuthorDecision`
- `AuthorPreferenceProfile`
- `AuthorConversation`

持久化：

- `FileNarrativeConversationRepository`

用途：

- 记录作者初始请求、后续修改、确认、分支选择。
- 把作者反馈沉淀为偏好，后续可用于 planning、writing、evaluation。

### 3. Blueprint Revision Loop

链路：

```text
propose_blueprint
-> wait_for_confirmation
-> apply_author_input(blueprint_feedback=..., writing_direction=...)
-> propose_blueprint
-> wait_for_confirmation
-> confirm_blueprint
```

实现点：

- `ChapterBlueprint` 增加 `revision_no`、`parent_blueprint_id`、`revision_notes`。
- 旧蓝图保存在 `NarrativeTaskState.metadata["blueprint_revision_history"]`。
- 新蓝图继承旧蓝图 id，形成可追踪版本链。

### 4. Draft Repair Loop

链路：

```text
evaluate_draft
-> repair_draft when blocker exists and attempts remain
-> evaluate_draft
-> compress_new_draft
-> commit_state
```

实现点：

- 新增 `DraftRepairPlan`、`DraftRevisionCandidate`。
- 新增 `NarrativeRepairPolicy` port。
- 默认实现：`RuleBasedNarrativeRepairPolicy`。
- `NarrativeReActEnvironment` 在 blocker 出现时优先 repair，不直接 rollback。

### 5. Branch Generation And Selection

链路：

```text
build_working_context
-> generate_branches
-> wait_for_branch_selection
-> apply_author_input(selected_branch_id=...)
-> select_branch
-> evaluate_draft
-> commit_state
```

实现点：

- 新增 `DraftBranch`、`BranchEvaluationReport`。
- `AuthorRequest.branch_count > 1` 时开启多候选分支。
- 选中分支后才进入正式 evaluate/commit。
- 分支可通过 `FileNarrativeStateRepository.save_branches()` 落盘。

### 6. Background Job Baseline

入口：

- `NarrativeJob`
- `FileNarrativeJobRepository`
- `NarrativeJobRunner`

当前 job 类型：

- `continue_session`
- `confirm_blueprint`
- `revise_blueprint`
- `select_branch`

设计约束：

- job runner 只调用 `NarrativeWritingSession`，不复制业务逻辑。
- 后续可以替换为真实 scheduler、worker service 或 API 层。

### 7. Parallel Candidate Executor

入口：

- `NarrativeRunGraph`
- `NarrativeTaskNode`
- `ParallelToolExecutor`

约束：

- 用于 candidate-only work，例如多分支生成、并行分析、候选评估。
- 不直接写 canonical state。
- canonical state 仍由 commit action 单点合并。

### 8. CLI Workbench

模块：

```text
agent_rl.narrative_writing.cli
```

安装后命令：

```text
narrative-agent
```

开发态可直接运行：

```powershell
$env:PYTHONPATH='src'
python -m agent_rl.narrative_writing.cli --help
```

核心命令：

- `start`
- `status`
- `step`
- `continue`
- `confirm-blueprint`
- `revise-blueprint`
- `select-branch`

## 当前可运行链路

### 普通续写

```text
start request
-> scan_workspace
-> analyze_source/load_analysis
-> propose_blueprint
-> wait_for_confirmation
-> confirm-blueprint
-> retrieve_context
-> build_working_context
-> generate_draft
-> evaluate_draft
-> repair_draft if needed
-> compress_new_draft
-> commit_state
```

### 多分支续写

```text
start request with branch_count > 1
-> confirm blueprint
-> retrieve/build context
-> generate_branches
-> select-branch
-> evaluate/repair/commit selected branch
```

## 验证

全量测试：

```text
61 passed
```

覆盖点：

- ReAct 蓝图确认暂停
- 确认后提交
- 长章节 segment 写作
- session pause/resume
- session 自动 checkpoint 与 memory 恢复
- 蓝图修订版本链
- 草稿 repair loop
- 多分支生成和作者选择
- job runner 恢复并确认蓝图
- parallel executor 收集候选结果

## 仍然需要继续升级

- LLM 版 repair policy：当前默认 repair 是本地规则实现。
- 更完整的 RAG 后端：当前仍是本地结构化检索，后续可升级 SQLite FTS、vector store、reranker、graph retrieval。
- 长期 memory 治理：当前已有 memory atom/compression 对象，仍需补 invalidation、importance scoring、decay。
- 真后台服务：当前 job runner 是本地文件队列 baseline，还不是常驻 worker。
