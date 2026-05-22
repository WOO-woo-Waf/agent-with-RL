# Shandianshuo SenseVoice Local GPU Analysis

Date: 2026-05-22

## Conclusion

The main goal is to run Shandianshuo's local `SenseVoice Small` ONNX ASR model on the RTX 3070 instead of relying on CPU execution. Remote ASR/API services are intentionally out of scope and should remain unchanged.

The local ASR model cannot be moved completely out of system memory. ONNX Runtime, DirectML, and CUDA all still require system RAM for the desktop process, model/session metadata, graph initialization, tokenizer data, audio buffers, and often staging/copy buffers. GPU execution can reduce CPU compute and may improve latency, but it cannot make RAM usage zero.

Based on local inspection and Shandianshuo's official help pages, the Windows app depends on DirectML and the process loads `directml.dll`, `d3d12.dll`, and `dxgi.dll`. The binary also contains ONNX Runtime provider names including CPU, CUDA, and DirectML. However, the current config does not expose a runtime backend switch such as `cpu`, `cuda`, `directml`, or `gpu`, and `nvidia-smi` did not list `shandianshuo.exe` among active GPU processes during inspection.

Practical conclusion: within the closed-source Shandianshuo app, there is no supported local setting found that can force the bundled `model.onnx` to run on CUDA or guarantee that all ASR compute runs on the RTX 3070. The supported local levers are keeping DirectML available, replacing the ONNX model file, and using a quantized ONNX model to reduce memory/compute pressure.

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

Observed Shandianshuo ASR config shape:

```text
asr.provider = local
asr.aliyun.model = qwen3-asr-flash
asr.volcengine.model = bigmodel_nostream
```

Do not print or commit Shandianshuo API keys. The local config contains provider credentials.

Recent logs show fast local ASR already:

```text
45s audio -> 4.88s transcription
13s audio -> 0.89s transcription
5s audio -> 0.27s transcription
10s audio -> 0.93s transcription
```

## ASR Boundary

The target model is this local ONNX ASR file:

```text
%AppData%\Shandianshuo\models\sensevoice-small\model.onnx
```

Changing LLM/correction providers does not affect this ASR runtime. Those settings are intentionally left unchanged.

## Feasible Options

### Option A: Use Shandianshuo's Supported Local ONNX Path

This keeps the software workflow unchanged.

Supported controls found in official docs:

- Keep Windows DirectML available. Shandianshuo documents `directml.dll` as a required Windows component.
- Keep the local SenseVoice directory structure intact:

```text
%AppData%\Shandianshuo\models\sensevoice-small\
  model.onnx
  tokens.json
  config.yaml
```

- To reduce memory and compute pressure, replace `model.onnx` with the official/compatible quantized `model_quant.onnx`, renamed to `model.onnx`.

This may lower RAM and CPU/GPU pressure, but it is not the same as forcing CUDA execution.

### Option B: Request Or Wait For A Vendor GPU Backend Switch

The clean solution inside Shandianshuo is a vendor-supported runtime setting, for example:

```text
ASR backend: DirectML / CUDA / CPU
Device: NVIDIA RTX 3070
```

No such exposed field was found in the current local config, backup config, model config, or official help pages checked.

### Option C: Build A Separate GPU ONNX ASR Runtime

A separate local GPU ASR service can be built with FunASR/SenseVoice through PyTorch CUDA or ONNX Runtime GPU. This can use the RTX 3070 more directly.

The blocker is integration: the current Shandianshuo config does not show a custom ASR HTTP endpoint or plugin setting. Without an official custom ASR provider hook, an external GPU ASR service cannot be cleanly wired into Shandianshuo's recording/input workflow.

Use this option only if one of these becomes true:

- Shandianshuo documents a custom ASR endpoint/plugin.
- The vendor exposes a GPU backend setting.
- The workflow moves outside Shandianshuo into a custom recorder/transcriber.

## GPU Verification

Check whether Shandianshuo appears on GPU:

```powershell
nvidia-smi
```

If `shandianshuo.exe` does not appear and GPU memory does not change during transcription, there is no practical evidence that local SenseVoice ASR is running on the RTX 3070.

DirectML workloads may be clearer in Windows Task Manager than in `nvidia-smi`. In Task Manager, add the `GPU engine` column and watch `shandianshuo.exe` while actively transcribing.

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
2. Do not change remote API/correction providers as part of this task.
3. If memory pressure is the main problem, use the official quantized ONNX replacement path and accept a possible accuracy drop.
4. Use Windows Task Manager GPU Engine and `nvidia-smi` to verify actual GPU use instead of relying on provider names in the binary.
5. Do not try binary patching or DLL replacement for Shandianshuo unless the vendor documents the GPU backend. The risk of breaking input, recording, or updates is high.

## Sources Checked

- Shandianshuo official homepage: documents local speech recognition model support.
- Shandianshuo help: `directml.dll` is documented as a required Windows DirectML component.
- Shandianshuo help: local SenseVoice ONNX model installation uses `model.onnx`, `tokens.json`, and `config.yaml`.
- Shandianshuo help: memory pressure guidance recommends replacing the default model with a quantized ONNX model.

## Open Questions

- Whether Shandianshuo has a hidden or paid GPU acceleration option for local ASR.
- Whether Shandianshuo supports a custom ASR HTTP endpoint in a newer version.
