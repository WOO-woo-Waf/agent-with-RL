# RAG Local Model Service Runbook

## Background

This project can use a local embedding model for RAG indexing and retrieval. The current Windows-native setup uses Ollama to serve `qwen3-embedding:4b` on the local machine. WSL is intentionally not required for this runbook.

Current machine baseline:

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU, 8 GB VRAM
- Driver: 581.95
- Ollama: 0.24.0
- Conda environment: `llm-gpu`
- Embedding model: `qwen3-embedding:4b`
- Model storage: `D:\models\ollama`
- Hugging Face cache root: `D:\models\huggingface`

## What Is Installed

### Ollama service

Ollama executable:

```powershell
C:\Users\98289\AppData\Local\Programs\Ollama\ollama.exe
```

The user `PATH` also includes the Ollama install directory, so a new terminal should normally accept:

```powershell
ollama --version
```

### Local model

The RAG embedding model is:

```text
qwen3-embedding:4b
```

Verified output:

- Embedding dimension: `2560`
- Cold start test: about `7.9s` total for 2 short inputs, including about `4.0s` model load
- Warm test: 16 short inputs in about `2.374s`, about `6.74 items/s`
- Loaded VRAM usage observed: about `6.1 GB` total GPU memory used by the system

### Python client

The `llm-gpu` Conda environment includes:

- `torch 2.11.0+cu128`
- CUDA runtime `12.8`
- cuDNN from the PyTorch wheel
- `ollama`
- `requests`
- `transformers`
- `sentence-transformers`
- `accelerate`
- `bitsandbytes`
- `faiss-cpu`

Activate it with:

```powershell
conda activate llm-gpu
```

## Start The Service

For a normal interactive session, start Ollama in a hidden background process:

```powershell
$env:OLLAMA_MODELS="D:\models\ollama"
Start-Process -FilePath "C:\Users\98289\AppData\Local\Programs\Ollama\ollama.exe" `
  -ArgumentList "serve" `
  -WindowStyle Hidden
```

If `ollama` is already available in the terminal `PATH`, this is enough:

```powershell
$env:OLLAMA_MODELS="D:\models\ollama"
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
```

The HTTP API listens on:

```text
http://127.0.0.1:11434
```

## Warm The Model

The first request loads the model and is slower. For RAG workloads, warm it once before indexing a large batch:

```powershell
$body = @{
  model = "qwen3-embedding:4b"
  input = "warmup"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:11434/api/embed" `
  -ContentType "application/json" `
  -Body $body
```

## Use From Python

Install-time setup has already been done in `llm-gpu`. A minimal embedding call:

```python
import ollama

texts = [
    "第一段需要进入向量库的文本",
    "第二段需要进入向量库的文本",
]

response = ollama.embed(model="qwen3-embedding:4b", input=texts)
embeddings = response["embeddings"]

print(len(embeddings))
print(len(embeddings[0]))  # 2560
```

For throughput, batch inputs:

```python
import ollama

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = ollama.embed(model="qwen3-embedding:4b", input=texts)
    return response["embeddings"]

chunks = ["chunk text"] * 32
vectors = embed_texts(chunks)
```

Avoid sending one request per chunk unless latency matters more than throughput.

## Use From This Project

The core package now provides reusable RAG adapters under `agent_rl.rag`.
They are not tied to the novel-writing package:

- `OllamaEmbeddingProvider`: local Ollama `/api/embed`.
- `OpenAICompatibleEmbeddingProvider`: remote `/v1/embeddings`.
- `HTTPReranker`: remote `/v1/rerank`, compatible with the old narrative-state-engine service.
- `SQLiteVectorStore`: local dependency-light vector store for development and small projects.
- `RAGModelService`: facade for embed, index, search, and rerank.
- `RemoteRAGServiceManager`: SSH on-demand remote service manager adapted from the old system.

### Conda Environment

The lightweight project environment is still:

```powershell
conda env create -f environment.yml
conda activate agent-with-rl
```

For local RAG work, use the RAG environment:

```powershell
conda env create -f environment-rag.yml
conda activate agent-with-rl-rag
```

If you want to reuse the already prepared GPU environment from this machine:

```powershell
conda activate llm-gpu
pip install -e .[rag]
```

### Env Contract

Local Ollama defaults:

```env
RAG_PROVIDER=ollama
RAG_OLLAMA_BASE_URL=http://127.0.0.1:11434
RAG_OLLAMA_MODEL=qwen3-embedding:4b
RAG_EMBEDDING_DIMENSION=2560
RAG_VECTOR_DB_PATH=artifacts/rag/vector_store.sqlite3
RAG_EMBED_BATCH_SIZE=32
RAG_AUTO_INDEX_ON_COMMIT=0
```

Remote OpenAI-compatible embedding service:

```env
RAG_PROVIDER=openai-compatible
RAG_EMBEDDING_BASE_URL=http://127.0.0.1:18080
RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
RAG_EMBEDDING_DIMENSION=2560
```

Remote rerank and SSH-managed service:

```env
RAG_RERANK_BASE_URL=http://127.0.0.1:18080
RAG_RERANK_MODEL=Qwen/Qwen3-Reranker-4B
RAG_RERANK_TOP_N=30
RAG_REMOTE_ON_DEMAND=1
RAG_REMOTE_STOP_AFTER_USE=1
RAG_REMOTE_SSH_HOST=zjgGroup-A800
RAG_REMOTE_SERVICE_DIR=/home/data/nas_hdd/jinglong/waf/novel-embedding-service
RAG_REMOTE_CUDA_DEVICES=6
```

The new variables also accept old-system aliases such as:

- `NOVEL_AGENT_VECTOR_STORE_URL`
- `NOVEL_AGENT_VECTOR_STORE_API_KEY`
- `NOVEL_AGENT_EMBEDDING_MODEL`
- `NOVEL_AGENT_EMBEDDING_DIMENSION`
- `NOVEL_AGENT_RERANK_MODEL`
- `NOVEL_AGENT_RERANK_TOP_N`
- `NOVEL_AGENT_REMOTE_EMBEDDING_*`

### CLI Commands

Show redacted RAG env:

```powershell
$env:PYTHONPATH="src"
python -m agent_rl.rag.cli env
```

Start local Ollama through the project CLI:

```powershell
$env:PYTHONPATH="src"
python -m agent_rl.rag.cli start-local `
  --ollama-executable "C:\Users\98289\AppData\Local\Programs\Ollama\ollama.exe" `
  --ollama-models "D:\models\ollama"
```

Warm the embedding model:

```powershell
$env:PYTHONPATH="src"
python -m agent_rl.rag.cli warm
```

Embed test text:

```powershell
$env:PYTHONPATH="src"
python -m agent_rl.rag.cli embed --text "测试本地 RAG embedding"
```

Index a JSONL file:

```powershell
$env:PYTHONPATH="src"
python -m agent_rl.rag.cli index-jsonl `
  --path artifacts/rag/index_input.jsonl `
  --collection-id narrative
```

Each JSONL row should contain:

```json
{"document_id":"doc-1","story_id":"story-1","evidence_type":"source_chunk","source":"manual","text":"需要进入向量索引的文本"}
```

Search the local SQLite vector store:

```powershell
$env:PYTHONPATH="src"
python -m agent_rl.rag.cli search `
  --query "仓库 密信" `
  --story-id story-1 `
  --collection-id narrative `
  --no-rerank
```

Remote service health and lifecycle:

```powershell
$env:PYTHONPATH="src"
python -m agent_rl.rag.cli remote-health --base-url http://127.0.0.1:18080
python -m agent_rl.rag.cli remote-start --base-url http://127.0.0.1:18080
python -m agent_rl.rag.cli remote-stop --base-url http://127.0.0.1:18080
```

### Narrative Agent Integration

The narrative package remains the main scenario implementation, but RAG is now a replaceable adapter.

Index a persisted narrative session into the RAG vector store:

```powershell
$env:PYTHONPATH="src"
python -m agent_rl.narrative_writing.cli `
  --memory-db artifacts/narrative-memory/memory.sqlite3 `
  index-rag `
  --session-id my-session `
  --story-id my-story `
  --collection-id narrative
```

Automatically index after a successful narrative commit:

```powershell
$env:PYTHONPATH="src"
python -m agent_rl.narrative_writing.cli `
  --auto-rag-index `
  --rag-collection-id narrative `
  --rag-index-batch-size 16 `
  start `
  --session-id my-session `
  --story-id my-story `
  --request "续写下一章" `
  --reference data/my-novel/chapter-01.txt `
  --writing-direction "继续推进仓库密信线索" `
  --confirm-plan
```

You can also enable the same behavior through `.env`:

```env
RAG_AUTO_INDEX_ON_COMMIT=1
```

This remains opt-in because starting a local or remote embedding model may consume GPU memory. When enabled, the session records `state.metadata["rag_index"]["mode"] == "auto_on_commit"`.

Queue the same work as a local background job:

```powershell
$env:PYTHONPATH="src"
python -m agent_rl.narrative_writing.cli enqueue-job `
  --session-id my-session `
  --story-id my-story `
  --job-id rag-index-001 `
  --job-type rag_index `
  --payload-json "{\"collection_id\":\"narrative\",\"batch_size\":16}"

python -m agent_rl.narrative_writing.cli run-job --job-id rag-index-001
```

Programmatic assembly:

```python
from agent_rl.narrative_writing import RAGVectorNarrativeRetrievalPolicy
from agent_rl.rag import RAGModelService

rag_service = RAGModelService.from_env()
retrieval_policy = RAGVectorNarrativeRetrievalPolicy(rag_service, collection_id="narrative")
```

The retrieval policy first builds local structural evidence, then merges vector results into the same `EvidencePack` with retrieval trace metadata. If the vector service fails, it degrades to the local structural retriever.

## Use From PowerShell

```powershell
$body = @{
  model = "qwen3-embedding:4b"
  input = @("第一段文本", "第二段文本")
} | ConvertTo-Json -Depth 5

$result = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:11434/api/embed" `
  -ContentType "application/json" `
  -Body $body

$result.embeddings.Count
$result.embeddings[0].Count
```

Expected dimension:

```text
2560
```

## Check Status

Check whether the service is running:

```powershell
Get-Process -Name ollama -ErrorAction SilentlyContinue
```

Check loaded Ollama models:

```powershell
$env:OLLAMA_MODELS="D:\models\ollama"
ollama ps
```

Check installed Ollama models:

```powershell
$env:OLLAMA_MODELS="D:\models\ollama"
ollama list
```

Check GPU usage:

```powershell
nvidia-smi
```

The model is not fully stopped until `nvidia-smi` no longer shows `ollama.exe` as a compute process.

## Stop The Service

First ask Ollama to unload the model:

```powershell
$env:OLLAMA_MODELS="D:\models\ollama"
ollama stop qwen3-embedding:4b
```

Then stop remaining Ollama service processes:

```powershell
Get-Process -Name ollama -ErrorAction SilentlyContinue | Stop-Process -Force
```

Verify:

```powershell
Get-Process -Name ollama -ErrorAction SilentlyContinue
nvidia-smi
```

Expected result:

- No `ollama` process from `Get-Process`
- No `ollama.exe` compute process in `nvidia-smi`

## Restart Cleanly

If the service behaves strangely, do a clean restart:

```powershell
Get-Process -Name ollama -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
$env:OLLAMA_MODELS="D:\models\ollama"
Start-Process -FilePath "C:\Users\98289\AppData\Local\Programs\Ollama\ollama.exe" `
  -ArgumentList "serve" `
  -WindowStyle Hidden
```

Then warm the model again.

## Resource Budget

Measured after setup:

| Item | Path | Size |
|---|---:|---:|
| Conda GPU environment | `C:\Users\98289\.conda\envs\llm-gpu` | `5.01 GB` |
| Ollama program | `C:\Users\98289\AppData\Local\Programs\Ollama` | `6.54 GB` |
| Ollama models | `D:\models\ollama` | `2.33 GB` |
| Hugging Face cache | `D:\models\huggingface` | `0 GB` |
| CUDA llama.cpp | `D:\tools\llama.cpp-b9222-cuda12` | `1.04 GB` |
| Download cache | `D:\tools\downloads` | `0.57 GB` |
| pip cache | `C:\Users\98289\AppData\Local\pip\Cache` | `4.61 GB` |

Core retained footprint:

```text
about 14.92 GB
```

Including download cache:

```text
about 15.49 GB
```

Including pip cache:

```text
about 20.10 GB
```

Observed GPU state:

- Loaded embedding model: system GPU memory around `6.1 GB` used
- After stopping Ollama: system GPU memory around `2.45 GB` used by desktop/browser/VS Code and other GUI processes

The 8 GB GPU can run `qwen3-embedding:4b`, but large batches and long text chunks have limited headroom. Close browser video, games, dynamic wallpaper, and other GPU-heavy apps before large indexing jobs.

## RAG Usage Guidance

Recommended pipeline:

1. Split documents into chunks.
2. Start Ollama.
3. Warm `qwen3-embedding:4b`.
4. Batch embed chunks, for example 16 to 64 chunks per call depending on chunk length.
5. Store vectors in FAISS, sqlite-vss, LanceDB, Chroma, or another vector store.
6. Stop Ollama after indexing or retrieval if GPU memory is needed elsewhere.

Practical chunking defaults:

- Start with 400 to 800 Chinese characters per chunk.
- Use 80 to 150 characters overlap for narrative text.
- Keep metadata: source file, chapter, section, chunk index, and character range.
- Normalize retrieval scores in the RAG layer, not in the embedding service.

## Troubleshooting

### `ollama` command is not recognized

Open a new terminal first. If it still fails, call the full path:

```powershell
& "C:\Users\98289\AppData\Local\Programs\Ollama\ollama.exe" --version
```

### API cannot connect

Start the service:

```powershell
$env:OLLAMA_MODELS="D:\models\ollama"
Start-Process -FilePath "C:\Users\98289\AppData\Local\Programs\Ollama\ollama.exe" `
  -ArgumentList "serve" `
  -WindowStyle Hidden
```

### Model is not found

Check the model directory and list:

```powershell
$env:OLLAMA_MODELS="D:\models\ollama"
ollama list
```

If needed, pull again:

```powershell
$env:OLLAMA_MODELS="D:\models\ollama"
ollama pull qwen3-embedding:4b
```

### GPU memory is too high after a run

Unload the model and stop service:

```powershell
$env:OLLAMA_MODELS="D:\models\ollama"
ollama stop qwen3-embedding:4b
Get-Process -Name ollama -ErrorAction SilentlyContinue | Stop-Process -Force
nvidia-smi
```

### Performance is lower than expected

Check:

- Whether the model is already warm.
- Whether `nvidia-smi` shows `ollama.exe`.
- Whether other GUI apps are using GPU memory.
- Whether batches are too small.
- Whether chunks are too long.

## Notes

Do not enable WSL for this workflow. vLLM and TensorRT-LLM are Linux/WSL-oriented options and are intentionally outside this Windows-native RAG service runbook.
