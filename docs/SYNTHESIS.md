# 官方 2.5 原模型 / Hindi 合并 GPT 离线合成验收

本项目的 `hi_h200_native.synth` 是原创的最小编排入口，实际合成调用固定版本官方 `indextts.infer_v2_5.IndexTTS2`，没有替代声学模型或模拟 trainer。默认仅做元数据与本地文件核验，不 import Torch、不初始化模型、不查询 CUDA、不下载、不安装依赖。本文档只说明已实现的入口；CPU 检查通过不等于完整预训练模型、FST 前端或 GPU 合成已验证。实际结果以每次运行的 `result.json` / `failure.json` 为准。

## 先检查环境，保持现有服务与 Torch 不变

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m hi_h200_native.synth --check-deps-only \
  --official-repo "$WORK/official"
```

此命令只检查模块是否可发现、已安装版本及固定官方 `pyproject.toml` 的约束，不初始化重模型；退出码 2 表示有缺失模块。它不证明二进制 ABI、所有传递 import 或 FST 构造可用。`runtime_import_verified=false` 是这个检查的正常结果。版本偏离单独列在 `version_mismatches`，不会静默安装或升级任何包。官方 Python 约束为 `>=3.10,<3.12`，本编排入口要求 Python ≥3.11。使用 Pod 既有 Python 3.12 / Torch 2.12 是超出官方约束的环境，必须单独记录真实运行结果。

实际 stock 路线需要的官方直接约束包含：`torch==2.8.*`、`torchaudio==2.8.*`、`transformers==4.52.1`、`numpy==2.2.6`、`librosa==0.10.2.post1`、`numba==0.63.0`、`descript-audiotools==0.7.2`、`munch==4.0.0`、`modelscope==1.27.0`、`safetensors==0.5.2`、`tokenizers==0.21.0`、`sentencepiece>=0.2.1`、`einops>=0.8.1`、`omegaconf>=2.3.0`、`openai-whisper>=20231117`、`tqdm>=4.67.1`、`requests>=2.28`、`fugashi>=1.2.0`、`unidic-lite>=1.0.0`。Linux 使用官方未固定版本的 `WeTextProcessing`（还需可用的 Pynini），Mac/Windows 使用 `wetext>=0.0.9`。`argbind`、`tiktoken`、PyYAML、SciPy、Hugging Face Hub、packaging 等传递依赖也会检查；没有给这些包编造官方版本锁。实际输出为当前平台的完整所选清单。

虽然只合成 Hindi，官方构造器仍无条件 import ModelScope，并初始化日语 Tagger 与中英 Normalizer。`use_qwen_emo=False` 不会消除该 ModelScope import，但不会加载 Qwen 权重。没有使用 NeMo 数字展开、Qwen、MaskGCT、DeepSpeed、自定义 CUDA kernel 或 torch.compile；不安装或显式启用可选 flash-attn 扩展，Torch 内置 SDPA 仍可能选择 Flash kernel。不需要为了这个入口安装官方所有 UI/训练/加速 extras。

缓存适配只继承官方 TextNormalizer 并改变真实中英 Normalizer 的 `cache_dir`；其他构造参数保留官方值，所有 FST/text 操作仍由安装的真实库执行。运行时检查真实 API 显式支持 `cache_dir`，否则停止。缓存只写新 run 的 `runtime/cache/xdg/tn/{zh,en}`，避免官方默认写源码目录或共享包目录。CPU 注入测试只证明参数与路径合同，未证明真实 FST 已构建，也未证明完整 stock 前端逐字节等价，报告中的 `stock_frontend_full_parity_verified` 始终为 false。

## 本地资产与独立目录

先通过 `hi_h200_native.assets --stage synth` 的显式容量/许可流程准备本地资产；这个入口本身不下载。固定 synth stage 为 20 个 payload 文件、7,082,513,550 bytes，额外的小型 manifest 不包含在该数字内。包括官方 2.5 GPT、codec、W2V、CAMPPlus、s2mel、情绪/说话人矩阵，以及固定 MIT BigVGAN 配置、生成器、LICENSE、README。入口逐文件校验 manifest 的 SHA256，再对照源码内固定公开 snapshot 的大小及 SHA256/Git blob SHA1；不凭目录名判断 2.5。

必须有三个独立目录，全部在本项目持久工作目录 `$WORK` 内：干净的固定官方源码 checkout、原始 synth 资产目录、`hi_h200_native.export` 完整产出的 merged bundle。merged 必须包含 `COMPLETE.json`，全部文件 hash、官方 commit、原始 GPT/config/tokenizer/LICENSE 身份相符。训练时较小资产 manifest 与 synth manifest 的附加文件不同是允许的；四个核心文件 hash 必须完全相同。不能使用 tiny random test bundle，不能修改或覆盖原始 GPT / ComfyUI 模型目录。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m hi_h200_native.synth \
  --workspace-root "$WORK" --official-repo "$WORK/official" \
  --model-dir "$WORK/assets/official-synth" \
  --model-manifest "$WORK/assets/official-synth/model-manifest.json" \
  --merged-bundle "$WORK/exports/hindi-gpt" \
  --prompt "$WORK/outputs/acceptance/reference.wav" \
  --text 'नमस्ते। यह हिन्दी भाषा का एक छोटा परीक्षण है।' \
  --variant merged --output-dir "$WORK/outputs/acceptance/hindi-merged-001"
```

这是默认 preflight，不创建输出目录、不加载模型。模型资产路径根据实际布局填写。参考音频必须有权使用、为 1–15 秒非静音单声道 PCM16 WAV；文本采用项目保守 NFC/空白规范化，保留 nukta、virama、ZWJ/ZWNJ。最小路线拒绝数字、货币与 `< > |` 注音/特殊标记。官方即使关闭 `text_normalization` 仍会替换部分标点；本入口先从固定源码读取映射，若会改变规范文本则停止，避免悄悄改变训练文本策略。仅支持一条、一个 segment、最多 160 字符 Hindi。

## 共享 GPU 上显式执行

在确认 GPU 真正空闲、剩余宿主/容器 RAM 与持久空间充足后，在上述命令加：

```bash
  --execute --allow-cuda --gpu-id 0 \
  --min-free-gpu-mib 65536 --min-free-ram-mib 32768 --max-utilization 5
```

数值只是保守实验起点，需根据真实资源确定，不是该模型显存/内存需求承诺。入口在模型构造前检查两组三次 `nvidia-smi` 与宿主/cgroup 可用 RAM；任一不满足即停止。观察到余量不等于预留 GPU，ComfyUI 之后仍可能抢资源；本入口不会停止、重启或修改现有服务。使用新进程、显式选择单卡；已有 CUDA context 会被拒绝。

实际构造使用 `use_bf16=True`、`device=cuda:0`，并显式关闭 Qwen、DeepSpeed、accel、自定义 CUDA kernel 和 compile。固定种子、贪心单 beam、严格确定性；遇到不支持确定性的真实算子即失败，不静默放宽。默认最多 256 个 mel codes，可用 `--max-mel-tokens` 调整但上限 512；达到官方截断警告不能算验收成功。新建 Python IP socket（含 TCP/UDP/监听端口）、DNS 与已知下载/迁移 helper 均被禁止，并为官方 `torch.load` 强制设置 `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`；这是进程保护，不是操作系统网络沙箱，仍应在离线环境运行。

先自行创建已被 `.gitignore` 排除的 `outputs/acceptance` 父目录，输出目录本身必须全新。结果与失败日志仅保留本地。记录每阶段是否开始/完成、输入 prompt/text hash、token hash、选择的配置与 GPT hash、seed、禁用开关、资源观测和输出 WAV/PCM hash。`audio.wav` 只在真实生成及非空 PCM 检查通过后发布；失败保留 `failure.json` 与可能的 partial，不能当作成功音频。先分别运行 `--variant original` 与 `--variant merged` 到不同新目录，使用相同 prompt/text/seed，再人工听评 Hindi 可懂度、发音、音色、截断与异常。WAV 存在不等于质量达标。

## 显式 KV cache A/B 实验

默认保留官方推理 wrapper 的 `kv_cache=True`。可选 `--no-kv-cache` 只把实际、来源已核验的官方 `GPT2InferenceModel` wrapper 现有布尔开关设为 `False`，改为完整上下文生成；不修改官方 positions 逻辑、权重或训练。它是独立实验开关，没有音质改善承诺。

真实官方 `generate` 在随机 tiny 配置中已观察到位置输入差异：cached 路线为 `[0]`、`[2]`、`[3]`，full-context 路线为 `[0]`、`[0,1]`、`[0,1,2]`。这仅证明所观察到的代码行为差异，不能认定为官方 bug，也不能认定为 Hindi 失败原因。

随后已在 H200 使用同一已知 TRAIN 文本、prompt、seed1234、max mel512，分别完成原模型与 v2 合并模型的 no-cache 真实 WAV 生成。两次运行均记录 `inference_cache.enabled=false`、`actual_wrapper_verified=true`，证明该输入下真实执行完成。相同 ASR 设置、去除 Unicode P 类标点评分口径的单句结果为：

| no-cache 路线 | CER | WER |
|---|---:|---:|
| 原模型 | 0.7916666667 | 1.0 |
| v2 合并模型 | 0.2361111111 | 0.5882352941 |

v2 合并模型与默认 KV 路线的该句 ASR 指标相同，未观察到单例 ASR 改善；音质改善仍未证实。该训练句诊断不证明 held-out 泛化，也不构成质量 gate 通过或失败修复的结论。默认仍为 `kv_cache=True`，详细结果与局限见 [VALIDATION.md](../VALIDATION.md)。

先按上一节检查共享资源。每组 A/B 固定同一 `variant`、权重、文本、prompt、seed 和全部生成参数，使用两个全新输出目录；不要同时运行以免争抢 GPU。以下操作模板只改变 cache 开关和输出目录，资产路径需替换为实际已核验的路径；示例文本不是上述 TRAIN 音频的实际转写：

```bash
common_synth_args=(
  --workspace-root "$WORK" --official-repo "$WORK/official"
  --model-dir "$WORK/assets/official-synth"
  --model-manifest "$WORK/assets/official-synth/model-manifest.json"
  --merged-bundle "$WORK/exports/hindi-gpt" --variant merged
  --prompt "$WORK/outputs/acceptance/reference.wav"
  --text 'नमस्ते। यह हिन्दी भाषा का एक छोटा परीक्षण है।'
  --seed 1234 --max-mel-tokens 512
  --execute --allow-cuda --gpu-id 0
  --min-free-gpu-mib 65536 --min-free-ram-mib 32768 --max-utilization 5
)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m hi_h200_native.synth "${common_synth_args[@]}" \
  --output-dir "$WORK/outputs/acceptance/hindi-merged-cache-001"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m hi_h200_native.synth "${common_synth_args[@]}" --no-kv-cache \
  --output-dir "$WORK/outputs/acceptance/hindi-merged-full-context-001"
```

如需原始模型对照，再固定 `--variant original` 做独立一组 A/B，不在同一组内更换权重。检查每次 `result.json` 的 `inference_cache`：默认 `enabled=true`，实验为 `enabled=false`、`experimental=true`。仅 metadata preflight 的 `actual_wrapper_verified=false` 不能证明实际模式已生效；真实 wrapper 核验后才记录 `actual_wrapper_verified=true`，同时保留 `upstream_positions_modified=false` 与 `quality_improvement_verified=false`。分别审听两份完整 WAV，并使用相同 ASR 模型/参数辅助比较；更低 ASR 错误率也不等于音质或发布门槛通过。

一个全局 merged GPT 会改变共享 LoRA 和输出 head 的行为，不能保证旧语言不变。默认继续为其他语言保留原始 GPT，把 Hindi 实验目录独立挂载；发布前还需独立旧语言回归与许可审核。此入口只生成本地验收文件，不发布权重或上传私人音频。

## 已有 CPU 覆盖

`python -m unittest discover -s tests -p 'test_native_synth.py' -v` 覆盖精确源码/前端映射、Unicode、真实文件 SHA 与 immutable metadata、merged inventory、真实 PCM WAV、真实 socket 阻断、资源阈值、无覆写和缓存参数合同。没有用 mock 合成器冒充真实模型；完整预训练推理和音质必须另行实跑并审听。

## 真实运行时导入预检

`--check-deps-only` 只检查模块和版本。若需要验证实际 ABI 与传递导入，可在独立环境运行以下命令；它会真实导入官方模块，禁止网络，不构造模型、Tagger 或 FST，也不初始化 CUDA。输出目录必须全新且父目录已存在，结果写入 `import-probe.json`。导入通过仍不等于合成通过。

```bash
python -m hi_h200_native.synth --probe-runtime-imports \
  --workspace-root "$WORK" --official-repo "$WORK/external/index-tts" \
  --output-dir "$WORK/outputs/synth-import-001"
```
