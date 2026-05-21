# Agent With RL

这个项目用于沉淀一套面向学习和工程复用的 Agent/RL 建模方式。

核心定位：

- 用 OOAD 定义强化学习与 LLM Agent 的共同概念：状态、观测、动作、策略、环境、奖励、轨迹、记忆、约束和评估。
- 用设计模式组织不同 Agent 架构：Strategy、Adapter、Composite、Template Method。
- 核心实现保持轻量，不封闭训练或编排能力；需要训练时对接 Gymnasium、PettingZoo、Stable-Baselines3、RLlib，需要生产 Agent 编排时对接 OpenAI Agents SDK、LangGraph、AutoGen 等。
- `agent_rl.llm` 是全项目通用模型调用基础设施，提供 OpenAI-compatible 调用、JSON 解析、endpoint failover、JSONL 审计和 token usage 记录；具体 Agent/场景通过端口注入使用它。

快速验证：

```powershell
conda env create -f environment.yml
conda activate agent-with-rl
python -m pytest
```

DeepSeek 环境准备：

```powershell
Copy-Item .env.example .env
# 然后在 .env 里填写 LLM_API_KEY，默认 LLM_API_BASE=https://api.deepseek.com
# 默认模型示例是 LLM_MODEL=deepseek-v4-flash
```

配置模块：

- `agent_rl.config` 提供跨 Windows/Linux 的 `.env` 加载、向上查找、`${VAR}` 展开、typed env 读取和脱敏快照。
- `.env` 是本地私有文件，`.env.example` 是可提交的配置契约。

示例运行：

```powershell
$env:PYTHONPATH="src"; python -m agent_rl.examples.gridworld
$env:PYTHONPATH="src"; python -m agent_rl.examples.narrative_demo
```

真实小说续写入口：

```powershell
# 先只生成章节蓝图，确认方向
$env:PYTHONPATH="src"; python -m agent_rl.examples.narrative_continue `
  --reference data/my-novel/chapter-01.txt `
  --direction "下一章继续推进密信线索，不要让主角立刻原谅对方" `
  --constraint "不要让主角立刻原谅对方" `
  --llm

# 确认蓝图后再生成正文，并导出纯正文
$env:PYTHONPATH="src"; python -m agent_rl.examples.narrative_continue `
  --reference data/my-novel/chapter-01.txt `
  --direction "下一章继续推进密信线索，不要让主角立刻原谅对方" `
  --constraint "不要让主角立刻原谅对方" `
  --confirm-plan `
  --llm `
  --output artifacts/narrative/chapter-continue.txt
```

小说写作 Agent 支持把本地参考小说文本读入为 `ReferenceMaterial`，再进入初始状态、RAG 证据和记忆层：

```python
from agent_rl.narrative_writing import NarrativeWritingAgent, build_author_request_from_files

request = build_author_request_from_files(
    request="规划并续写下一章",
    reference_paths=("data/my-novel/chapter-01.txt",),
    writing_direction="下一章继续推进密信线索，不要让主角立刻原谅对方",
    constraints=("不要让主角立刻原谅对方",),
    confirm_plan=True,
)
result = NarrativeWritingAgent().run(request)
```

小说写作核心包目前还包含：

- `NarrativeAnalysisPolicy`：把参考小说分析为 source chunks、角色、事件、剧情线、世界规则、风格片段和初始记忆。
- `CompositeNarrativeRetrievalPolicy`：按作者计划、人物、剧情/记忆、世界、风格、source chunk、scene case 分通道检索并配额融合。
- `WorkingMemoryContext`：把作者计划、检索证据、状态摘要按预算装配为模型上下文。
- `PromptRegistry` / `PromptComposer`：文件化 prompt profile、global prompt、task prompt 和 hash metadata。
- `LLMNarrativeWriterPolicy` / `LLMNarrativeExtractorPolicy`：通过包级 `agent_rl.llm.ChatModelClient` 接入真实模型，JSON 解析失败时回退到本地策略。
- `retrieval_evaluation_report`：记录 EvidencePack 覆盖率、缺失证据类型和检索质量指标。

设计文档：

- `docs/design-architecture/agent-rl-ooad/AGENT_RL_OOAD_DESIGN_2026-05-20.md`
- `docs/design-architecture/core-package-layering/AGENT_RL_CORE_LAYERING_DESIGN_2026-05-20.md`
- `docs/design-architecture/narrative-agent-system/NARRATIVE_AGENT_DOMAIN_MODEL_2026-05-20.md`
- `docs/design-architecture/narrative-agent-system/NARRATIVE_ENGINE_ABSORPTION_PLAN_2026-05-20.md`
- `docs/design-architecture/narrative-agent-system/NARRATIVE_DEEP_ANALYSIS_PERSISTENCE_2026-05-21.md`

研究笔记：

- `docs/research-notes/agent-rl/`
- `docs/research-notes/narrative-state-engine/`
