# Hindi ASR 辅助检查

`hi_h200_native.asr_eval` 是原创的小型验收工具，实际识别调用 `openai-whisper==20250625`。默认只列计划；不下载、不 import Torch、不运行识别。先用同域真实 Hindi 验证音频校准 ASR，再比较使用相同文本、prompt 和生成参数的 original / merged WAV。WER/CER 不能独立证明 TTS 可懂度、自然度或音色通过；ASR 自身的错误和幻觉不一定来自 TTS，仍需懂 Hindi 的人审听。

官方 [v20250625 README](https://github.com/openai/whisper/tree/v20250625#license) 明确代码与权重采用 [MIT](https://github.com/openai/whisper/blob/v20250625/LICENSE)。本工具只选择多语种 `small`，不是 `small.en`；包含 Hindi 语言标签不等于 Hindi 可靠性已经验证。固定公开 checkpoint 为 **483,617,219 bytes**，SHA256：

```text
9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794
```

下载地址来自上述官方版本包的 `_MODELS["small"]`，SHA 同时包含在固定 URL 路径中。默认单文件预算 500,000,000 bytes；不读取 Hub 身份、token 或 gated 数据，也不借此接受任何条款。下载后保留已安装 Whisper 包的 MIT 许可。当前 CPU 测试没有下载这些权重，也没有把合成测试音频冒充真实 Hindi 识别。

## 先列计划，再显式下载

所有输入、模型、输出必须位于本项目独立持久目录 `$WORK` 内；不要指定 ComfyUI 模型目录。使用已有且版本吻合的独立 Python 环境，不修改共享服务环境。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m hi_h200_native.asr_eval --project-root "$WORK"

# 可选：确认磁盘余量后，显式下载唯一固定公开模型；目录必须全新。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m hi_h200_native.asr_eval --project-root "$WORK" \
  --model-dir assets-local/whisper-small --download
```

固定 HTTPS 域名、默认 TLS 校验、禁用环境代理、无 Authorization/Cookie、完整字节数与 SHA256 检查。已有目录不覆盖；失败可能留下本次创建的 partial，但不能当作有效权重。网络传输中断时停止，不自动重试或自动改用其他模型。

## 准备最多 12 条检查记录

将 JSONL 放在被 `.gitignore` 排除的 `outputs/acceptance/asr-cases.jsonl`；音频路径相对 `$WORK`。`reference` 必须由操作者确认是真实 held-out 验证音频及其实际转写，不得用生成音频冒充校准。工具核验角色标签，不能独立证明来源、说话人隔离或转写准确性。

```json
{"id":"calibration-01","condition":"reference","audio":"outputs/acceptance/val-01.wav","text":"क़लम यहाँ रखिए।"}
{"id":"probe-01","condition":"baseline","audio":"outputs/acceptance/original-01/audio.wav","text":"नमस्ते मित्र।"}
{"id":"probe-01","condition":"merged","audio":"outputs/acceptance/merged-01/audio.wav","text":"नमस्ते मित्र।"}
```

示例仅说明格式，不能替代真实录音的原转写。每个生成 probe 的 `baseline` 和 `merged` 必须成对、文本逐字相同；使用实际生成的完整 WAV。相同 TTS prompt、seed 和生成参数，以及 reference 的 held-out 说话人来源，都必须由操作者另行核对；此工具不验证这些条件。只接受非静音、单声道 PCM16、22,050 或 24,000 Hz、完整时长 1–20 秒、单文件 ≤2 MiB 的 WAV。SciPy 将完整音频重采样为 Whisper 的 16 kHz，不裁掉尾部。路径逃逸、越界 symlink、截断 WAV、超长文件等在模型加载之前停止。

```bash
# 默认仅 CPU。必须已有完整且 SHA 校验通过的 small.pt。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m hi_h200_native.asr_eval --project-root "$WORK" \
  --model-dir assets-local/whisper-small --execute \
  --input outputs/acceptance/asr-cases.jsonl \
  --output outputs/acceptance/asr-001
```

输出目录必须全新。执行禁止 Python IP 网络连接，直接使用已校验的本地权重路径，并在加载后再次复核权重哈希；实际 Whisper 模块路径必须属于版本已验证的安装分发。不传入目标文本作为提示。固定 Hindi/transcribe、temperature 0、beam 5、FP32、关闭前文条件化和词时间戳；全部条件用同一模型与参数。真实报告记录输入文件 hash、模型 SHA、Whisper 版本、参数、原转写、ASR 转写及编辑计数。需要 CUDA 时，先独立检查共享 GPU 资源，再显式加 `--device cuda --allow-cuda --gpu-id 0`；已初始化 CUDA context 的进程会被拒绝；本工具不会停止或调整 ComfyUI，也不预留 GPU。

## 解释分数

原始视图仅做 NFC 与空白合并，不使用 Whisper 的通用 normalizer。CER 按 Unicode code point 计数，包含规范后的空格；WER 按空白分词，均为编辑距离除以参考单位数，可大于 1。第二个视图仅把 Unicode `P*` 标点替换为空格；保留 Devanagari 组合标记、nukta、virama、ZWJ/ZWNJ、数字和货币，不展开数字、不改写词形。两个视图分别保留，不用后者掩盖原始错误。

先检查真实 `reference` 的转写和 WER/CER。如果 ASR 连该录音都不可靠，不能根据 generated 分数断言模型好坏。小样本不得设置未经校准的绝对通过阈值，也不得把训练损失下降或单次较低 WER 宣称为过拟合/发布门槛通过。`quality_gate_passed` 和 `manual_listening_completed` 始终为 false；人工听评、更多 held-out 文本和旧语言回归仍需另行完成。
