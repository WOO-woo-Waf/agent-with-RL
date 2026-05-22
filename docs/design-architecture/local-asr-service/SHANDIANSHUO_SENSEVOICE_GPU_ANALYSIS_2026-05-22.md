# Shandianshuo SenseVoice Local GPU Analysis

Date: 2026-05-22

## Conclusion

The current Shandianshuo installation uses its local `SenseVoice Small` ONNX model for ASR and uses a separate LLM provider for text correction. Ollama can be used for the correction/LLM provider path, but it does not serve or accelerate the `SenseVoice Small` ASR model inside Shandianshuo.

The local ASR model cannot be moved completely out of system memory. Even if a GPU backend is active, the desktop app still needs RAM for the process, model/session metadata, audio buffers, tokenizer data, and provider fallback.

Based on local inspection, Shandianshuo loads `directml.dll`, and the binary contains ONNX Runtime provider names including CPU, CUDA, and DirectML. However, the current config does not expose a GPU backend switch, and `nvidia-smi` did not show `shandianshuo.exe` using RTX 3070 GPU memory during inspection. Treat current SenseVoice ASR as not controllably GPU-backed from user configuration.

## Local Evidence

Installed app:

```text
D:\Shandianshuo\shandianshuo.exe
Version observed: 0.6.2
```

User data and model:

```text
C:\Users\98289\AppData\Roaming\Shandianshuo
C:\Users\98289\AppData\Roaming\Shandianshuo\models\sensevoice-small\model.onnx
```

Observed disk usage:

| Item | Path | Size |
| --- | --- | ---: |
| App binary directory | `D:\Shandianshuo` | 52.4 MB |
| Shandianshuo roaming data | `C:\Users\98289\AppData\Roaming\Shandianshuo` | 3.931 GB |
| SenseVoice Small model directory | `C:\Users\98289\AppData\Roaming\Shandianshuo\models\sensevoice-small` | 894.5 MB |
| SenseVoice ONNX model file | `model.onnx` | 937,615,562 bytes |

Observed runtime:

| Process | Memory |
| --- | ---: |
| `shandianshuo.exe` working set | about 1.0 GB |
| `shandianshuo.exe` private memory | about 1.0 GB |

Observed Shandianshuo config shape:

```text
asr.provider = local
asr.aliyun.model = qwen3-asr-flash
asr.volcengine.model = bigmodel_nostream
ai.correction.provider = deepseek
ai.providers.ollama.endpoint = http://localhost:11434
ai.providers.ollama.model = llama3.2
```

Do not print or commit Shandianshuo API keys. The local config contains provider credentials.

Recent logs show fast local ASR already:

```text
45s audio -> 4.88s transcription
13s audio -> 0.89s transcription
5s audio -> 0.27s transcription
10s audio -> 0.93s transcription
```

## Architecture Boundary

Shandianshuo has two separate model paths:

| Path | Current role | Can Ollama replace it? |
| --- | --- | --- |
| ASR/transcription | Local `SenseVoice Small` ONNX model | No |
| AI correction/rewriting | DeepSeek, Ollama, OpenAI-compatible providers, etc. | Yes |

Ollama serves text models and embedding models through HTTP. It does not load Shandianshuo's `model.onnx` file, and it cannot directly accelerate the local SenseVoice ASR path.

## Feasible Options

### Option A: Keep Local SenseVoice ASR, Use Ollama For Local Correction

This is the most practical local-only setup.

Use Shandianshuo for recording and transcription, then configure its AI correction provider to Ollama. Ollama can use the RTX 3070 for the text correction model if the selected model fits GPU memory.

Installed Ollama models currently visible under `D:\models\ollama`:

```text
qwen3:4b-instruct
nomic-embed-text:latest
```

Use a chat/instruct model for correction. `qwen3:4b-instruct` is suitable for correction. Embedding models are not suitable for correction.

Start Ollama:

```powershell
$env:OLLAMA_MODELS="D:\models\ollama"
Start-Process -FilePath "C:\Users\98289\AppData\Local\Programs\Ollama\ollama.exe" `
  -ArgumentList "serve" `
  -WindowStyle Hidden
```

Check models:

```powershell
$env:OLLAMA_MODELS="D:\models\ollama"
& "C:\Users\98289\AppData\Local\Programs\Ollama\ollama.exe" list
```

Recommended Shandianshuo UI settings, if using its provider settings page:

```text
大模型服务商: Ollama
Endpoint: http://localhost:11434
Model: qwen3:4b-instruct
API Key: empty
```

If editing config manually, back up first:

```powershell
$cfg="$env:APPDATA\Shandianshuo\config.json"
Copy-Item -LiteralPath $cfg -Destination "$cfg.bak-2026-05-22"
```

Then change only these logical fields:

```text
ai.correction.provider = ollama
ai.providers.ollama.enabled = true
ai.providers.ollama.endpoint = http://localhost:11434
ai.providers.ollama.model = qwen3:4b-instruct
```

Because the current config has encoding/JSON parsing issues, prefer the Shandianshuo UI over manual JSON editing.

Stop Ollama when done:

```powershell
$env:OLLAMA_MODELS="D:\models\ollama"
& "C:\Users\98289\AppData\Local\Programs\Ollama\ollama.exe" stop qwen3:4b-instruct
Get-Process -Name ollama -ErrorAction SilentlyContinue | Stop-Process -Force
```

### Option B: Use Cloud ASR To Avoid Local ASR Memory

The config contains ASR cloud provider entries:

```text
aliyun: qwen3-asr-flash
volcengine: bigmodel_nostream
```

If Shandianshuo exposes these in the UI and credentials/quota are available, using cloud ASR can reduce local CPU/RAM pressure because transcription is no longer done by the local SenseVoice model. This depends on network latency, provider quota, and privacy requirements.

### Option C: Build A Separate GPU ASR Service

A separate local GPU ASR service can be built with FunASR/SenseVoice through PyTorch CUDA or ONNX Runtime GPU. This can use the RTX 3070 more directly.

The blocker is integration: the current Shandianshuo config does not show a custom ASR HTTP endpoint or plugin setting. Without an official custom ASR provider hook, an external GPU ASR service cannot be cleanly wired into Shandianshuo's recording/input workflow.

Use this option only if one of these becomes true:

- Shandianshuo documents a custom ASR endpoint/plugin.
- The vendor exposes a GPU backend setting.
- The workflow moves outside Shandianshuo into a custom recorder/transcriber.

## GPU Verification

Check whether Ollama is using GPU:

```powershell
nvidia-smi
```

Look for `ollama.exe` in the process list after a correction request.

Check whether Shandianshuo appears on GPU:

```powershell
nvidia-smi
```

If `shandianshuo.exe` does not appear and GPU memory does not change during transcription, there is no practical evidence that local SenseVoice ASR is running on the RTX 3070.

Check Shandianshuo process memory:

```powershell
Get-Process shandianshuo -ErrorAction SilentlyContinue |
  Select-Object Id,ProcessName,Path,WorkingSet64,PrivateMemorySize64
```

Check loaded GPU-related modules:

```powershell
Get-Process shandianshuo -ErrorAction SilentlyContinue |
  ForEach-Object { $_.Modules } |
  Where-Object { $_.ModuleName -match 'onnx|cuda|cudnn|directml|dml|nvcuda|nvml|tensorrt' } |
  Select-Object ModuleName,FileName
```

## Recommended Operating Mode

For now:

1. Keep Shandianshuo local ASR enabled with its bundled `SenseVoice Small` model.
2. Use Ollama only for local AI correction with `qwen3:4b-instruct`.
3. Keep RAG embeddings separate from Shandianshuo correction. `qwen3-embedding:4b` or other embedding models are for vector retrieval, not text correction.
4. Use `nvidia-smi` to verify actual GPU use instead of relying on provider names in the binary.
5. Do not try binary patching or DLL replacement for Shandianshuo unless the vendor documents the GPU backend. The risk of breaking input, recording, or updates is high.

## Open Questions

- Whether Shandianshuo has a hidden or paid GPU acceleration option for local ASR.
- Whether Shandianshuo supports a custom ASR HTTP endpoint in a newer version.
- Whether cloud ASR privacy/latency is acceptable for the user's daily dictation workflow.
