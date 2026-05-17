好！下面给你一套 **实操步骤和指令示例**，演示如何在 **OpenAI 官方的 Codex CLI** 中启用并使用“多会话 / 多 agent（Sub‑agents）”的实验性功能——也就是让它能 **并行生成和协作多个子 agent** 去处理复杂任务（例如多人协作的开发规划→执行）。这些是目前官方 CLI 支持的方式（虽然仍是实验/进阶模式）。([aiHola][1])

---

## 🚀 0) 安装 & 准备

如果你还没装过 Codex CLI：

```bash
npm install -g @openai/codex
```

然后配置 API Key：

```bash
export OPENAI_API_KEY="你的 OpenAI Key"
```

或者 Windows PowerShell：

```powershell
$Env:OPENAI_API_KEY="你的 OpenAI Key"
```

这会让 CLI 自动通过你的 Key 调用 OpenAI 模型。([OpenAI Help Center][2])

---

## 🧩 1) 启用实验性多 agent 功能（Sub‑agents）

Codex CLI 的“多 agent / 子 agent（subagents）”支持是默认关闭的，需要先在配置文件里打开：

### 📌 方法 A — 修改 config

编辑或创建：

```bash
nano ~/.codex/config.toml
```

加入：

```toml
[features]
multi_agent = true

[agents]
max_threads = 6      # 最多并行子 agent 个数（默认可改）
max_depth   = 1      # 嵌套层级为 1 就够 （不要太深）
```

保存并退出。([Codex Blog][3])

---

### 📌 方法 B — 在会话里开关

进入 CLI：

```bash
codex
```

然后输入：

```
/experimental
```

选择 **Enable Multi‑agents**（启用多 agent 实验特性）。这会自动写入配置并重启。([aiHola][1])

---

## 🧠 2) 启动 Codex CLI 会话

```bash
codex
```

现在你进入交互式会话。在这里你可以下自然语言命令让 Codex 工作，也可以用多 agent 特性分配任务。

---

## 🛠 3) 使用子 agent 并行执行任务

现在你已启用 multi‑agent，可以用一些内置机制让子 agent 并行工作。我提供两种典型方法：

---

### 📌 方式 A — Codex 自动分配

只需要给大任务 prompt，Codex 会决定是否 spawn 子 agent 并行拆任务（例如审查代码、分模块编写等）：

```
请分析这个大型工程，生成三个子任务：
1) 前端 UI 功能
2) 后端 API 逻辑
3) 单元测试覆盖

然后分别让 Codex 并行处理这些任务，并汇总结果。
```

CLI 会自动尝试 spawn 子 agent 去并行处理。([aiHola][1])

> 这种是“让模型自己决定 spawn 时机”，对大任务最简单。

---

### 📌 方式 B — 明确 spawn 子 agent（更可控）

你还可以明确控制子 agent 生命周期。Codex CLI 有一套内部命令：

```
spawn_agent
send_input
wait_agent
close_agent
```

不过目前文档不完整，但基本思路如下（实验性质强）：

```
spawn_agent --role frontend "请为这个工程创建前端 UI 功能"
spawn_agent --role backend  "请为这个工程实现后端 API"
spawn_agent --role test     "请为这个工程编写测试脚本"

# 等待所有子 agent 完成
wait_agent

# 汇总或合成输出
请将 frontend、backend、test 子 agent 的结果汇总成一个最终方案。
```

Codex 会并行执行这些 agent，每个 agent 是一个独立线程。([Zylos][4])

⚠️ 真正在 CLI 里出现这些命令的语法可能略有不同（因为这是实验功能），你可以先通过试探：

```
/help
```

或看 CLI TUI 内的自动补全提示。

---

## 🧪 4) 子 agent 的角色 &策略

你可以在配置文件里为不同角色设置不同行为，例如：

```toml
[agents.frontend]
model = "gpt-5.5"
description = "UI 前端开发 agent"
sandbox_mode = "danger-full-access"

[agents.backend]
model = "gpt-5.5"
description = "后端 API 开发 agent"
sandbox_mode = "danger-full-access"
```

这样当你 spawn 一个 `frontend` agent 时，它会自动带上设定的模型、角色提示和权限。([Codex Blog][3])

---

## 📊 5) 查看正在运行的 agent 或状态

Codex CLI 提供一些内置 TUI 或命令，可以检查当前多个会话/agent：

```
/agents
```

它会列出子 agent 当前的状态（运行中、等待、完成等）。([Reddit][5])

---

## 🧠 6) 典型协作流程示例

1. 启动 main agent（你在 CLI 里）

   ```bash
   codex
   ```

2. 主 agent 接收工程规划 prompt

   ```
   请规划整个工程并拆成三个并行任务：前端、后端、测试。
   ```

3. 主 agent spawn 子 agent 分配任务

   ```
   spawn_agent --role frontend "根据规划写前端代码"
   spawn_agent --role backend  "根据规划写后端代码"
   spawn_agent --role test     "写自动化测试"
   ```

4. 等待子 agent 完成

   ```
   wait_agent
   ```

5. 主 agent 合并结果

   ```
   请汇总所有子 agent 输出并生成最终项目交付包。
   ```

以上流程是最典型的“并行作业 + 汇总结果”的 agent 协作。([aiHola][1])

---

## 💡 注意事项

🧪 **实验性功能**

* 多 agent / sub agents 是实验功能，需要手动启用。
* 不同 CLI 版本命令细节可能略有不同（例如 spawn/agent 命令）。
* 官方文档可能还不完整，但大方向是这套“并行 agent → 汇总模式”。([aiHola][1])

⚠️ **Token & 成本**
每个并行 agent 都会独立调用 API 和消耗 token。并行越多，消耗越大。

📁 **上下文隔离**
每个 agent 有独立上下文窗口，可以清晰分工，但需要主 agent 合并。([aiHola][1])

---

## 🧠 总结（最关键的命令）

| 场景          | 指令示例                                    |
| ----------- | --------------------------------------- |
| 启动 CLI      | `codex`                                 |
| 启用多 agent   | `/experimental` → *Enable Multi‑agents* |
| 修改配置        | 编辑 `~/.codex/config.toml`               |
| spawn agent | `spawn_agent --role <name> "<prompt>"`  |
| 等待 agent 完成 | `wait_agent`                            |
| 查看 agent    | `/agents`                               |
| 汇总结果        | 普通自然语言 prompt                           |

---

如果你愿意，我还可以给你：

🔹 一份 **完整 demo 脚本**（可以直接复制到终端里运行）
🔹 一个 **Codex CLI workflow 模板**（planner → multiple agents → merge）

要不要我帮你写这两份？😊

[1]: https://aihola.com/article/codex-cli-multi-agent-mode?utm_source=chatgpt.com "Codex CLI Adds Multi-Agent Mode for Parallel Coding"
[2]: https://help.openai.com/en/articles/11096431-openai-codex-ligetting-started?utm_source=chatgpt.com "OpenAI Codex CLI – Getting Started | OpenAI Help Center"
[3]: https://codex.danielvaughan.com/2026/03/26/codex-cli-subagents-toml-parallelism/?utm_source=chatgpt.com "Codex CLI Subagents: TOML Format, Parallelism and spawn_agents_on_csv | Codex Blog"
[4]: https://zylos.ai/research/2026-03-26-openai-codex-cli-architecture-multi-runtime-patterns?utm_source=chatgpt.com "OpenAI Codex CLI Architecture and Multi-Runtime Agent Patterns | Zylos Research"
[5]: https://www.reddit.com/r/codex/comments/1r49q5d/how_do_you_use_subagents_in_codex_cli/?utm_source=chatgpt.com "How do you use sub-agents in Codex CLI?"
