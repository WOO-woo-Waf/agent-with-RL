# 基于 opencode 魔改小说 Agent 工具的设计方案

Date: 2026-05-22

## 背景

目标是把 `anomalyco/opencode` 这类 Codex-like 工具改造成小说分析、学习、规划、续写和状态回流工具。这里的重点不是“换一套系统提示词让它会写小说”，而是把本项目已经定义的小说领域状态、分析结果、记忆、评估、分支、作者确认和 ReAct 轨迹嵌入到一个成熟的长运行 Agent 宿主里。

本轮已将上游仓库克隆到：

```text
D:\buff\opencode
```

本地检查的上游 commit：

```text
76d9c2cd7
```

## 目标

1. 借用 `opencode` 的交互外壳、session、工具调用、权限、子 Agent、MCP、插件和 TUI/桌面/Web 形态。
2. 把本项目的小说领域对象作为一组可调用、可追踪、可恢复的小说工具接进去。
3. 让作者能在一个类似 Codex 的工作台里完成：
   - 原文导入和深度分析
   - 人物、剧情线、世界规则、伏笔、风格画像抽取
   - 章节/场景蓝图生成和反复修订
   - 多分支续写、评估、选择
   - 角色一致性、剧情连续性、文风、世界观约束检查
   - 草稿修复、提交、回滚
   - 长期记忆压缩、失效、检索

## 非目标

1. 第一阶段不直接重写 `opencode` 的核心 session、消息表、LLM 调用链。
2. 第一阶段不把 Python 小说内核整体翻译成 TypeScript。
3. 第一阶段不做完整前端产品化，只先把能力接入 `opencode` 可运行。
4. 不让模型自由改写 canon。蓝图确认、重大状态变更、分支选择、最终提交必须保留 human-in-the-loop。

## opencode 技术环境判断

`opencode` 是一个 Bun + TypeScript + Effect 的 monorepo。

核心环境：

- 包管理器：`bun@1.3.14`
- 主语言：TypeScript ESM
- Effect runtime：大量服务、Layer、Schema、Effect flow
- LLM 调用：Vercel AI SDK 6
- 服务端：Hono/OpenAPI 风格 HTTP API
- 本地存储：SQLite + Drizzle
- 终端 UI：OpenTUI/Solid
- Web/Desktop：Solid/Vite/Electron
- 扩展：plugin、MCP、custom tools、agent markdown、skill discovery

关键目录：

```text
D:\buff\opencode\packages\opencode\src\agent
D:\buff\opencode\packages\opencode\src\tool
D:\buff\opencode\packages\opencode\src\session
D:\buff\opencode\packages\opencode\src\plugin
D:\buff\opencode\packages\opencode\src\mcp
D:\buff\opencode\packages\opencode\src\permission
D:\buff\opencode\packages\opencode\src\server
D:\buff\opencode\packages\plugin
```

最重要的结论：

- `opencode` 已经有 primary/subagent 模式，适合映射成小说总控、分析员、规划员、写手、审稿、记忆管理员。
- `opencode` 已经有工具注册表，支持内置工具、`.opencode/tools/*.ts` 自定义工具、外部 plugin tool、MCP tool。
- `opencode` 已经有权限系统，可以把“确认蓝图”“提交状态”“导出草稿”“失效记忆”等高风险操作设为 ask。
- `opencode` 已经有 session、child session、fork、message parts、tool calls，适合承载多轮小说任务和分支探索。
- `opencode` 的领域默认是“代码仓库”，所以小说领域不应该硬改到所有核心抽象里，而应该先作为一个“narrative workspace adapter + tool pack”接入。

## 本项目已有小说内核

本项目已经具备一个可嵌入宿主的小说领域内核，核心代码在：

```text
src/agent_rl/domains/narrative.py
src/agent_rl/narrative_writing/
```

已定义的关键对象：

- Source/Analysis：`SourceDocument`、`SourceChunk`、`NarrativeSourceAnalysis`、`ChunkAnalysisResult`、`ChapterAnalysisResult`、`GlobalStoryAnalysisResult`
- Story State：`CharacterCard`、`CharacterDynamicState`、`RelationshipState`、`NarrativeEvent`、`PlotThreadState`、`WorldRule`、`LocationState`、`ObjectState`、`ForeshadowingState`、`SceneState`
- Author State：`AuthorRequest`、`AuthorQuestion`、`AuthorConversation`、`AuthorDecision`、`AuthorPreferenceProfile`、`AuthorConstraint`、`ChapterBlueprint`、`ChapterBlueprintSegment`
- Memory/RAG：`MemoryAtom`、`CompressedMemoryBlock`、`NarrativeQuery`、`NarrativeEvidence`、`EvidencePack`、`WorkingMemoryContext`
- Generation/Evaluation：`ChapterPlan`、`DraftCandidate`、`DraftBranch`、`StateChangeProposal`、`EvaluationReport`、`DraftRepairPlan`
- Runtime：`NarrativeScenarioAdapter`、`NarrativeReActEnvironment`、`NarrativeWritingSession`、`Trajectory`、`Reward`

这说明魔改 `opencode` 时，正确的边界是：

```text
opencode = 长运行 Agent 宿主、界面、权限、session、工具调度
agent-with-RL narrative = 小说领域内核、状态、分析、检索、写作、评估、记忆、提交
```

## 推荐路线

推荐不要一上来深 fork。先走三层路线。

### Route A: 低侵入接入，最快可跑

在小说项目工作区放置 `.opencode` 配置：

```text
novel-workspace/
  .opencode/
    opencode.jsonc
    agents/
      narrative-director.md
      story-analyst.md
      plot-planner.md
      character-guardian.md
      style-critic.md
      continuity-auditor.md
      branch-writer.md
      memory-librarian.md
    tools/
      narrative.ts
  .narrative/
    source/
    analysis/
    state/
    sessions/
    blueprints/
    drafts/
    branches/
    memory/
    evaluation/
    exports/
```

`narrative.ts` 作为 opencode custom tool，调用本项目 Python CLI/API：

```text
opencode tool call
-> .opencode/tools/narrative.ts
-> python -m agent_rl.narrative_writing.cli ...
-> JSON result
-> opencode tool result
-> LLM 继续决策
```

优点：

- 不改 `opencode` 源码。
- 能快速验证交互模式、工具颗粒度、权限和提示词。
- 可以保留 Python 内核作为唯一真实领域实现。

缺点：

- UI 仍是通用 coding agent UI。
- 工具结果展示需要靠文本/JSON。
- 对 session/message schema 的深度定制较少。

这是第一阶段首选。

### Route B: 本地 narrative service + opencode plugin

把小说内核封装成本地服务或 MCP server：

```text
opencode plugin / MCP
-> narrative service
-> NarrativeWritingSession
-> repositories / memory / RAG / artifacts
```

优点：

- 工具调用稳定，不需要每次启动 Python 进程。
- 可以更好地维护长运行 session。
- MCP 可以被其他 Agent 宿主复用。
- plugin hook 可以注入系统提示、拦截 tool definition、记录事件。

适合第二阶段。

### Route C: 深 fork 成小说专用产品

等 Route A/B 跑通后，再 fork `D:\buff\opencode` 成一个小说工具仓库，例如：

```text
D:\buff\novelcode
```

深 fork 内容：

- 修改品牌、默认文案、默认 agent、默认工作区结构。
- 新增 `packages/narrative`，提供 TypeScript SDK/client，连接 Python narrative service。
- 新增 first-class narrative tools，不再只依赖 `.opencode/tools`。
- 在 TUI/Web/Desktop 中增加小说状态面板、蓝图面板、分支对比、记忆检索、评估报告视图。
- 必要时扩展 session part 类型，支持 structured narrative artifacts。

只有当通用 `opencode` 外壳限制明显影响小说工作流时，才进入这一阶段。

## Agent 设计

建议新增这些 agents：

| Agent | 类型 | 职责 |
|---|---|---|
| `narrative-director` | primary | 作者主交互入口，决定分析、规划、写作、修复、提交何时发生 |
| `story-analyst` | subagent | 原文分析、章节/全局结构抽取、证据归档 |
| `plot-planner` | subagent | 章节蓝图、场景计划、伏笔和剧情线推进 |
| `character-guardian` | subagent | 人物卡、关系状态、知识边界、一致性检查 |
| `style-critic` | subagent | 文风画像、风格证据、草稿文风评估 |
| `continuity-auditor` | subagent | 世界规则、事件因果、canon 冲突、提交 gate |
| `branch-writer` | subagent | 多分支草稿生成，不直接提交 canonical state |
| `memory-librarian` | subagent | 记忆写入、压缩、失效、检索、冲突队列 |

`narrative-director` 的权限应允许读取、检索、提问、生成候选，但对以下工具使用 `ask`：

- `narrative_confirm_blueprint`
- `narrative_commit_state`
- `narrative_select_branch`
- `narrative_invalidate_memory`
- `narrative_export_chapter`

## 小说工具清单

第一阶段 custom tool 应该集中成一个 `narrative.ts`，内部按 action 分发，避免一次性创建过多 TS 文件。

建议工具：

| Tool | 输入 | 输出 | 说明 |
|---|---|---|---|
| `narrative_scan_workspace` | cwd/story_id | workspace summary | 检查 source、state、session、memory 是否存在 |
| `narrative_start_session` | request/reference_paths/writing_direction | session_id/result | 创建 `NarrativeWritingSession` |
| `narrative_resume_session` | session_id | status/result | 从 snapshot 恢复 |
| `narrative_analyze_source` | source paths | analysis artifact | 深度分析原文 |
| `narrative_show_state` | session_id/story_id | compact state | 展示人物、剧情线、记忆、草稿状态 |
| `narrative_propose_blueprint` | session_id | blueprint | 生成章节/场景蓝图 |
| `narrative_revise_blueprint` | session_id/feedback | blueprint revision | 根据作者反馈修订蓝图 |
| `narrative_confirm_blueprint` | session_id | status | 作者确认后进入写作 |
| `narrative_retrieve_context` | session_id/query | EvidencePack summary | 检索写作证据 |
| `narrative_generate_branch` | session_id/count | branch list | 生成候选续写分支 |
| `narrative_select_branch` | session_id/branch_id | selected branch | 作者选择分支 |
| `narrative_evaluate_draft` | session_id/draft_id | reports | 角色/剧情/风格/世界观评估 |
| `narrative_repair_draft` | session_id/report ids | revision | 定向修复 |
| `narrative_commit_state` | session_id | committed state | 提交 canon、memory、index |
| `narrative_export_chapter` | session_id/path | export path | 导出正文 |
| `narrative_invalidate_memory` | story_id/query/ids/reason | invalidated ids | 废弃错误记忆 |
| `narrative_index_rag` | story_id/session_id | indexed count | 更新 RAG 索引 |

工具结果必须返回 compact summary，而不是把完整草稿、完整 state、完整分析全塞回上下文。大输出写入 `.narrative/artifacts` 或 `.narrative/drafts`，只返回路径、摘要、评分和下一步建议。

## 工作区与持久化

推荐把小说运行状态和 opencode 自身配置分开：

```text
.opencode/
  opencode.jsonc
  agents/
  tools/

.narrative/
  source/
  analysis/
  state/
  sessions/
  conversations/
  blueprints/
  drafts/
  branches/
  memory/
  evaluation/
  trajectories/
  exports/
```

原因：

- `.opencode` 代表 Agent 宿主配置。
- `.narrative` 代表小说产品数据。
- 后续即使换掉 `opencode`，`.narrative` 仍然是小说状态仓库。

建议跟踪策略：

- 可以跟踪：`source/` 中作者明确纳入项目的参考材料、稳定 blueprints、最终 exports、必要配置。
- 默认忽略：session snapshots、trajectory、大模型调用日志、临时 branches、SQLite memory、evaluation run artifacts。
- 如果某些分析结果要作为长期 canon，应从 artifacts 晋升为 reviewable state snapshot。

## 数据契约

不要让 TypeScript 和 Python 各自定义一套小说领域模型。第一阶段以 Python dataclass 为 canonical model，通过 JSON 作为跨进程契约。

建议新增：

```text
src/agent_rl/narrative_writing/api_contracts.py
```

职责：

- 定义 tool input/output JSON 结构。
- 提供 `to_jsonable` / `from_jsonable` 入口。
- 输出 schema 或示例，供 `.opencode/tools/narrative.ts` 做轻量校验。

TypeScript 侧只保留 wrapper 类型：

```text
type NarrativeToolResult = {
  success: boolean
  summary: string
  artifacts: string[]
  metrics: Record<string, number>
  payload?: unknown
}
```

## 与 opencode 的具体结合点

### 1. Agent markdown

`opencode` 会从 `{agent,agents}/**/*.md` 加载 agent。每个小说 agent 用 markdown frontmatter 定义 mode、description、permission、model、steps，然后正文写领域指令。

### 2. Custom tools

`opencode` 会扫描 `{tool,tools}/*.{js,ts}`，这正适合放 `.opencode/tools/narrative.ts`。该工具通过 Bun/Node 子进程调用 Python CLI，并把结果转成 opencode tool output。

### 3. Plugin hooks

`@opencode-ai/plugin` 支持：

- `tool`
- `chat.message`
- `chat.params`
- `tool.execute.before`
- `tool.execute.after`
- `permission.ask`
- `experimental.chat.system.transform`
- `experimental.session.compacting`

第二阶段可用 plugin 做：

- 自动注入小说项目说明和当前 story summary。
- 自动记录每次工具调用到 `.narrative/trajectories`。
- 修改 compaction，保护作者决策、蓝图和 canon 摘要不被压丢。
- 对高风险 narrative tool 强制 ask。

### 4. MCP

如果希望小说工具也能给其他 Agent 使用，应该把 Python narrative service 包成 MCP server。`opencode` 会把 MCP tools 与内置工具一起提供给 LLM，并可通过权限控制。

### 5. Session

`opencode` session 保存对话、工具调用、token、cost、fork 等通用信息。小说 session 应继续由 `NarrativeWritingSession` 保存到 `.narrative/sessions`。两者用 metadata 绑定：

```json
{
  "opencode_session_id": "...",
  "narrative_session_id": "...",
  "story_id": "...",
  "task_id": "..."
}
```

不要第一阶段修改 `opencode` 的 `SessionTable`，否则会把后续上游同步成本放大。

## 典型工作流

### 全文分析

```text
author asks narrative-director
-> narrative_scan_workspace
-> narrative_analyze_source
-> story-analyst reviews analysis summary
-> narrative_index_rag
-> save analysis/state artifacts
```

### 蓝图规划

```text
author direction
-> narrative_start_session or narrative_resume_session
-> narrative_show_state
-> narrative_retrieve_context
-> narrative_propose_blueprint
-> ask author to confirm/revise
-> narrative_revise_blueprint until accepted
-> narrative_confirm_blueprint
```

### 多分支续写

```text
confirmed blueprint
-> branch-writer generates A/B/C
-> narrative_evaluate_draft for each
-> continuity-auditor checks blockers
-> author selects branch
-> narrative_select_branch
-> optional repair
-> narrative_commit_state
-> narrative_export_chapter
```

## 实施路线

### Phase 0: opencode 本地验证

目标：确认上游可在本机跑起来。

动作：

- 在 `D:\buff\opencode` 安装依赖。
- 运行 `bun --version`、`bun install`。
- 运行 `bun run --cwd packages/opencode typecheck`。
- 尝试 `bun run dev` 或使用已发布 `opencode` 打开测试项目。

验收：

- 能列出 agents/tools。
- 能从项目 `.opencode/agents` 加载自定义 agent。
- 能从 `.opencode/tools` 加载一个 hello tool。

### Phase 1: 无 fork 小说工具包

目标：在任意小说 workspace 内通过 opencode 调用本项目小说内核。

动作：

- 新建 `docs` 中的示例 `.opencode` 配置。
- 新增 `scripts/opencode_narrative_tool.py` 或扩展 `narrative-agent` CLI，支持 JSON action 分发。
- 新增 `.opencode/tools/narrative.ts` 示例 wrapper。
- 新增 `narrative-director.md` 和几个 subagent 示例。
- 把工具输出限制为 compact result + artifacts path。

验收：

- 用 opencode 发起“分析这部小说并生成下一章蓝图”。
- 工具生成 `.narrative/analysis` 和 `.narrative/sessions`。
- 蓝图生成后暂停，等待作者确认。
- 确认后能继续生成草稿、评估、提交或回滚。

### Phase 2: narrative service / MCP

目标：稳定承载长运行 session，减少进程启动和 JSON CLI 复杂度。

动作：

- 封装本地 narrative service。
- 或实现 Python MCP server，暴露上述 narrative tools。
- opencode 配置 `mcp.narrative`。
- 加入 tool permission 默认策略。

验收：

- opencode 能通过 MCP 调用所有 narrative tools。
- 关闭 opencode 后重新打开，仍能 resume narrative session。
- 多分支任务不会污染 canonical state。

### Phase 3: opencode fork 产品化

目标：把通用 coding agent 壳改成小说专用工作台。

动作：

- fork 为 `D:\buff\novelcode`。
- 新增 `packages/narrative`。
- 默认 agent 改成 narrative agents。
- TUI/Web 增加 narrative panes：
  - 状态概览
  - 蓝图确认
  - 分支对比
  - 评估报告
  - 记忆检索
  - 提交/回滚历史
- 修改 compaction 策略，保护 author decisions、canon summary、current blueprint。

验收：

- 打开后第一屏就是小说工作台，而不是代码工作台。
- 作者能不用记 tool name，通过自然语言完成分析、规划、分支、评估、提交。

## 必要重构

本项目侧建议先做小重构：

1. 给 narrative CLI 增加 machine-readable JSON 模式。
2. 把所有长输出落盘，返回 artifact path。
3. 明确 `.narrative` workspace layout。
4. 给 `NarrativeWritingSession` 增加 `opencode_session_id` metadata。
5. 给 tools 增加稳定 action input/output contract。
6. 增加 fake LLM / rule-based smoke tests，避免 opencode 集成测试依赖真实模型。

`opencode` 侧第一阶段不改源码。

## 风险

1. 上下文污染：如果每个工具都返回完整状态，opencode 会很快爆上下文。
   - 对策：工具只返回摘要、评分、artifact path。
2. 双 session 不一致：opencode session 和 narrative session 各自持久化。
   - 对策：显式保存绑定 metadata，所有 narrative action 都带 `narrative_session_id`。
3. 模型越权提交 canon。
   - 对策：提交、失效记忆、选择分支、确认蓝图全部权限 ask。
4. TypeScript/Python 模型分裂。
   - 对策：Python 是 canonical model，TS 只做 wrapper。
5. 深 fork 维护成本高。
   - 对策：A/B 两阶段跑通后再 fork。

## 验证计划

第一轮验证只需要覆盖最小闭环：

```text
opencode narrative-director
-> narrative_scan_workspace
-> narrative_start_session
-> narrative_analyze_source
-> narrative_propose_blueprint
-> pause for confirmation
-> narrative_confirm_blueprint
-> narrative_generate_branch
-> narrative_evaluate_draft
-> narrative_commit_state
-> narrative_export_chapter
```

本项目测试：

- `pytest tests/test_narrative_workbench.py`
- 新增 JSON tool contract tests
- 新增 CLI action 分发 tests

opencode 侧验证：

- 从 `packages/opencode` 运行 `bun typecheck`
- 用示例 `.opencode/tools/narrative.ts` 加载工具
- 用一个小型小说样例跑一次 end-to-end smoke

## 当前决策

本轮建议的工程决策是：

```text
先不要深 fork opencode。
先把 opencode 当作 Codex-like 宿主，用 agent markdown + custom tools / MCP 接入本项目小说内核。
当工具粒度、状态仓库和作者确认链路稳定后，再 fork 做小说专用 UI 和 session 扩展。
```

这样最符合当前项目现实：小说领域模型已经在 Python 里成型，`opencode` 最有价值的是长运行 Agent 宿主和交互环境，而不是它的 coding 领域默认工具。

