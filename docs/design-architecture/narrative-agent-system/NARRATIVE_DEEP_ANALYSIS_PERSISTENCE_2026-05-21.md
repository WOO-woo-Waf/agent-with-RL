# Narrative Deep Analysis Persistence

Date: 2026-05-21

## Goal

把旧版小说系统里有效的 `chunk -> chapter -> global` 深度分析链路吸收到当前项目，但不照搬旧系统的数据库、工作台和重服务。当前项目仍以 Agent/RL OOAD 概念为主体：分析是 `NarrativeAnalysisPolicy`，存储是 `NarrativeAnalysisRepository`，模型调用走包级 `agent_rl.llm`。

## Old System Experience Kept

- `chunk` 是章节内分析实现细节，不是长期业务主对象。
- `chapter` 是对外稳定分析单位，要沉淀章节摘要、事件链、人物状态、关系变化、世界规则、伏笔和下一章入口。
- `global` 是续写 Agent 的全局故事状态，要沉淀人物卡、剧情线、世界规则、风格 bible、检索建议和连续性约束。
- LLM 分析必须有 fallback，不能因为一次 JSON 失败中断整个分析。
- 分析产物必须可复盘、可断点续跑、可被后续 RAG/记忆压缩复用。

## Current OOAD Implementation

领域对象在 `src/agent_rl/domains/narrative.py`：

- `ChunkAnalysisResult`
- `ChapterAnalysisResult`
- `GlobalStoryAnalysisResult`
- `NarrativeSourceAnalysis.chunk_analyses`
- `NarrativeSourceAnalysis.chapter_analyses`
- `NarrativeSourceAnalysis.global_analysis`

端口在 `src/agent_rl/narrative_writing/ports.py`：

- `NarrativeAnalysisRepository`

本地适配器在 `src/agent_rl/narrative_writing/persistence/file_repository.py`：

- `FileNarrativeAnalysisRepository`

LLM policy 在 `src/agent_rl/narrative_writing/policies/deep_analysis.py`：

- `LLMDeepNarrativeAnalysisPolicy`

prompt 资产在：

- `prompting/templates/tasks/novel_chunk_analysis.md`
- `prompting/templates/tasks/novel_chapter_analysis.md`
- `prompting/templates/tasks/novel_global_analysis.md`
- `prompting/templates/profiles/default.yaml`

## Local Storage Layout

第一版使用本地 JSON/JSONL，默认根目录是 ignored `artifacts/narrative`：

```text
artifacts/narrative/{story_id}/{task_id}/
  manifest.json
  source_analysis.json
  source_documents.json
  source_chunks.jsonl
  chunk_analysis.jsonl
  chapter_analysis.jsonl
  global_analysis.json
```

用途：

- `source_chunks.jsonl`：原文 chunk，供后续 RAG/indexing 使用。
- `chunk_analysis.jsonl`：每个 chunk 的 LLM 细粒度分析，天然支持并行与断点续跑。
- `chapter_analysis.jsonl`：章节级稳定状态，供作者规划、检索和续写使用。
- `global_analysis.json`：故事级状态图，供 Agent 的长期上下文和记忆压缩使用。
- `manifest.json`：记录布局、覆盖率、trace，后续可加入模型、prompt hash、source hash。
- `source_analysis.json`：完整快照，方便调试和迁移。

## Runtime Path

启用 LLM 后，factory 默认把深度分析接入 narrative agent：

```text
AuthorRequest
-> LLMDeepNarrativeAnalysisPolicy
   -> RuleBasedSourceAnalysisPolicy builds documents/chunks fallback base
   -> novel_chunk_analysis
   -> novel_chapter_analysis
   -> novel_global_analysis
   -> FileNarrativeAnalysisRepository.save_source_analysis
-> NarrativeTaskState
-> retrieval / context / planning / writing / extraction / evaluation
```

CLI 可控制是否跳过：

```powershell
python -m agent_rl.examples.narrative_continue `
  --reference data/my-novel/chapter-01.txt `
  --direction "下一章方向" `
  --llm `
  --confirm-plan
```

跳过深度分析：

```powershell
python -m agent_rl.examples.narrative_continue `
  --reference data/my-novel/chapter-01.txt `
  --direction "下一章方向" `
  --llm `
  --no-llm-analysis
```

## Upgrade Path

本地 JSON/JSONL 是第一阶段 adapter，不是最终绑定：

```text
FileNarrativeAnalysisRepository
-> SQLiteNarrativeAnalysisRepository
-> PostgresNarrativeAnalysisRepository
-> Vector/RAG indexing adapter
-> memory compression adapter
```

后续增强点：

- 给 `manifest.json` 增加 source hash、prompt hash、model name、temperature、analysis version。
- 支持按 chunk hash 跳过已分析 chunk，实现断点续跑。
- 把 `chunk_analysis.jsonl` 和 `chapter_analysis.jsonl` 回流到 retrieval index。
- 增加 SQLite repository，解决大文件 JSONL 的查询问题。
- 增加 embedding/reranker adapter，但保持 `NarrativeWritingAgent` 不直接依赖向量库。
