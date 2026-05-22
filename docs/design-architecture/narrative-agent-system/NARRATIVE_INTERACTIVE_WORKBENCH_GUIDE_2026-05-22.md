# Narrative Interactive Workbench Guide

## Usage Scenario

这个入口不是“自然语言转参数表单”。它是一个外层 ReAct 操作员 Agent，并且拥有和 Codex 类似的持久会话概念：

```text
author message
  -> observe current workspace/session state
  -> decide next intent
  -> call the right tool/session function
  -> show result and wait for the next author message
```

因此作者可以像使用 Codex 一样直接发目标：

```text
分析 data/first-novel/1.txt，续写第2章，3000字。延续上一章悬念，不要提前揭示核心真相。
```

系统会直接创建小说写作 session 并执行到第一个作者关口。只有工具调用缺少必要参数时才追问，例如没有参考 txt 路径、导出时没有草稿、分支选择缺少 branch id。

## Operator Session

外层会话是 `NarrativeOperatorSessionState`，默认存储在：

```text
artifacts/narrative-operator-sessions/<operator-session-id>.json
```

它保存：

- 作者和助手消息。
- 当前作者目标。
- 当前活跃的 `NarrativeWritingSession`。
- 当前 story id。
- 参考小说路径。
- 每次 operator intent 和工具调用轨迹。

这使入口更接近 Codex 模式：你不是一次性调用某个“续写命令”，而是在一个长期上下文里持续发目标。Agent 会根据上下文决定下一步是分析、确认规划、修改规划、续写、查看草稿还是导出。

可以指定外层会话 id：

```powershell
python -m agent_rl.narrative_writing.cli --operator-session-id my-novel-operator
```

下次用同一个 id 启动时，会自动恢复其活跃小说 session：

```powershell
python -m agent_rl.narrative_writing.cli --operator-session-id my-novel-operator
```

## Two-Layer Agent Model

当前模式分两层：

```text
Outer operator Agent
  - 接收作者消息
  - 观察是否已有 session、当前 phase、是否有草稿/蓝图/分支
  - 决定 start/resume/continue/confirm/revise/export/show 等 intent
  - 调用核心 session API

Inner narrative Agent
  - 导入小说
  - 深度分析
  - 生成章节蓝图
  - 等待作者确认规划
  - 检索上下文
  - 续写
  - 评估/修复
  - 提交状态和记忆
```

`导入小说 -> 深度分析 -> 作者确认规划 -> 检索上下文 -> 续写 -> 评估/修复 -> 提交记忆` 是内层小说 Agent 的实际运行链路，不是作者需要手工填写的流程。

## Prerequisites

在项目根目录运行：

```powershell
conda activate agent-with-rl
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

不用 RAG 时不要加 `--use-rag-vector` 或 `--auto-rag-index`。

## Start

调用真实模型：

```powershell
python -m agent_rl.narrative_writing.cli --use-llm
```

本地规则 smoke：

```powershell
python -m agent_rl.narrative_writing.cli --no-llm
```

进入后直接说目标：

```text
分析 data/first-novel/1.txt，续写第2章，3000字。下一章要延续上一章的悬念，不要提前揭示核心真相，不要改变主角性格。
```

预期行为：

```text
observe: no active narrative session
decide: start_session (...)
action: start narrative session
[1] initialized -> scan_workspace
[2] workspace_observed -> analyze_source
[3] analysis_ready -> propose_blueprint
[4] blueprint_proposed -> wait_for_confirmation
next: say 'confirm' to write, or describe how to revise the plan.
```

## Natural Messages

有 active session 后，普通消息会被外层操作员解释为 intent：

- `确认` / `confirm`: 如果当前蓝图待确认，调用 `confirm_plan` 并继续续写。
- `继续` / `run`: 继续运行到下一个作者关口。
- `看分析`: 展示分析摘要。
- `看规划`: 展示章节蓝图。
- `看草稿`: 展示草稿。
- `导出 artifacts/narrative/first-novel/chapter-002.txt`: 导出草稿。
- `不要提前揭示幕后黑手`: 作为硬约束加入，并继续运行。
- 其他普通文本: 作为新的作者目标/写作方向，合并到 session 后继续运行。

命令仍然保留，便于精确控制：

```text
/status
/analysis
/plan
/draft
/context
/confirm
/run
/revise <feedback>
/constraint <text>
/select <branch-id>
/export [path]
/resume <session-id> [story-id]
/quit
```

## Code Structure

核心文件：

- `src/agent_rl/narrative_writing/workbench.py`
  - `NarrativeInteractiveWorkbench`: 外层运行循环。
  - `NarrativeOperatorSessionState`: Codex-like 外层会话状态。
  - `FileNarrativeOperatorSessionRepository`: 外层会话持久化。
  - `WorkbenchOperatorPolicy`: 根据当前 session 状态和作者消息选择 intent。
  - `WorkbenchRequestDraft`: 新 session 的可执行参数对象。
  - `_HeuristicWorkbenchRequestParser` / `_LLMWorkbenchRequestParser`: 把目标消息解析成启动参数。
- `src/agent_rl/narrative_writing/session.py`
  - 长会话、checkpoint、resume、作者输入合并。
- `src/agent_rl/narrative_writing/react.py`
  - 内层小说 Agent 的 ReAct 环境和阶段机。
- `src/agent_rl/narrative_writing/scenario.py`
  - 小说领域工具编排。

## Validation

已覆盖的自动测试：

```text
message -> start_session -> analyze/propose_blueprint
message(confirm) -> confirm_plan -> write/evaluate/commit
message(export path) -> export_draft
operator_session_id -> restart -> resume active narrative session
```

运行：

```powershell
python -m pytest tests/test_narrative_workbench.py -q
python -m pytest -q
```
