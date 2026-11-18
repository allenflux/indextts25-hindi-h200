# 独立加载 Hindi best100 实验 GPT

当前的 19 WAV 听评 ZIP 是音频与评测资料包，**没有模型权重，不能替换 IndexTTS 2.5**。候选替换的是官方 2.5 的 GPT 子模型，tokenizer、语义 codec、声学后端、声码器等仍使用匹配的官方资产。HF 公开包与本地原始导出格式不同：下文先给公开 **13 文件**下载/加载路线，后面的本地 bundle 路线只用于保留私有完成记录的原导出。

best100 已在 Linux/H200 上由真实官方 2.5 加载、合成并完成有限对照；正式质量 gate 仍为 false。当前质量门槛未过时，现网保留原版，候选在独立环境验收；推荐的后续单模型默认替换见 [Part10](SINGLE_MODEL_UPGRADE.md)。共享 attention 和两个 head 已合并，不能保证一个全局候选 GPT 对旧语言完全无影响。

RTX5090先完成过四核心节点的两份WAV；后续源码`792305841240072e5a9f2616eb9e387126cae5a2`已在隔离兼容服务完成原版→Hindi→原版三轮真实queue，保持原8节点/5连线，首尾原版WAV和PCM哈希一致。现网插件尚未由本项目部署兼容更新，质量gate仍为false。推荐的5090单模型默认替换、首次插件升级与维护回滚见 [Part10](SINGLE_MODEL_UPGRADE.md)；上述双目录三轮保留为历史验收记录。

新增目标是只保留一套候选运行目录、同T8插件升级后兼容旧模型名和空自定义路径。证据、离线原型、与既有双目录方案的区别及上线门槛见 [Part10：推荐单模型替换](SINGLE_MODEL_UPGRADE.md)。这是有条件方案；本项目尚未生产替换，旧语言质量门槛未通过。

## 文件与身份

| 项目 | 固定值 |
|---|---|
| 官方源码 | `index-tts/index-tts` commit `d9e41aac89fd00b3d71497fddb287b7f24613712` |
| 官方模型 | `IndexTeam/IndexTTS-2.5` revision `c39ce5ba981572cb187443877ff559dfb246ce63` |
| 原始官方 `gpt.pth` SHA256 | `43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85` |
| Hindi pilot best100 `gpt.pth` SHA256 | `423a8491b8856c7c3075254b12defc3ae47dbd9e0de285772523ddd4e9a97021` |
| Hindi best100 `gpt.pth` 实际字节数 | `3259538991` |
| 选择依据 | 固定 pilot 验证点中 step100 的 loss 最低；未用后续 test/ASR 重新选模 |

SHA256 来自已冻结、实际复核的文件。公开实验仓库已实际创建：[fluxa0223/IndexTTS-2.5-Hindi-best100-experimental](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental)。13文件权重包已通过网页上传，固定revision为`117c7922e4dcedc3f9885f3f6efa5be8f54c7919`。匿名核验确认公开状态、13文件加原`.gitattributes`的inventory、GPT LFS size/SHA及12个小文件实际GET/hash均匹配。首次核验的GPT部分仅检查LFS元数据；后续已在5090宿主用公开匿名下载脚本完成13文件下载，GPT完整流式SHA与冻结值一致。下载者仍须校验自己的本地完整文件，详细回执范围见[PUBLIC_RELEASE](PUBLIC_RELEASE.md)。仓库许可标记为 `other / bilibili-model-license`，指向固定官方模型许可；本项目工具的 MIT 许可不替代该许可。

## HF 公开 13 文件包：下载与独立加载

公开目录只含这 13 个普通文件：

```text
DATA_ATTRIBUTION.md
LICENSE
LICENSE_ZH.txt
MODEL_LICENSE
NON_ENDORSEMENT.txt
README.md
SHA256SUMS
config.yaml
evaluation.json
gpt.pth
manifest.json
multilingual_zh_ja_yue_char_del.tiktoken
release.json
```

它不包含原运行的私有 `COMPLETE.json` 或 `provenance.json`。**公开包必须用 `scripts/synth_public_release.py`；不能直接交给只接受本地原导出格式的 `python -m hi_h200_native.synth`。** 新入口复用相邻的审核脚本 `prepare_hf_release.verify_release`，检查固定 13 文件、hash、上游 pins 和 frozen best100 身份，再使用原离线合成门槛；不会伪造私有完成记录。需使用含 `scripts/` 的项目源码 checkout，wheel-only 安装不包含这些脚本。

### 下载到缓存，再复制精确清单

以下按已发布的固定40位commit匿名下载；完整payload匿名下载已另有5090宿主实测，本代码块供复现使用。运行示例后仍须通过本地13文件校验，不能用分支`main`或元数据commit代替该revision。公开读取不需要创建或发送新token。

从本项目源码根目录运行，先设置真实持久 `WORK`，缓存与最终公开目录必须分开：

```bash
export WORK="/path/to/persistent/indextts-hindi"  # 先改为实际持久目录
export HF_RELEASE_REVISION=117c7922e4dcedc3f9885f3f6efa5be8f54c7919
export HF_PUBLIC_CACHE="$WORK/cache/hf-public-downloads"
export HF_PUBLIC_DIR="$WORK/public/hindi-best100"
mkdir -p "$WORK/cache" "$WORK/public"

HF_HUB_OFFLINE=0 HF_HOME="$WORK/cache/hf-public-sdk-home" \
HF_XET_CACHE="$WORK/cache/hf-public-xet" HF_HUB_DISABLE_TELEMETRY=1 \
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import json
import os
from pathlib import Path
import re
import shutil
import sys
from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(Path.cwd() / "scripts"))
from prepare_hf_release import FILES, local, verify_release

repo = "fluxa0223/IndexTTS-2.5-Hindi-best100-experimental"
revision = os.environ["HF_RELEASE_REVISION"]
if not re.fullmatch(r"[0-9a-f]{40}", revision):
    raise ValueError("先填写实际权重发布回执中的完整 commit")
work = Path(os.environ["WORK"]).resolve(strict=True)
cache = local(work, os.environ["HF_PUBLIC_CACHE"], exists=False)
output = local(work, os.environ["HF_PUBLIC_DIR"], exists=False)
if output.exists() or not output.parent.is_dir():
    raise ValueError("公开目录必须全新，父目录须已创建")
if cache == output or cache.is_relative_to(output) or output.is_relative_to(cache):
    raise ValueError("HF 缓存和严格公开目录必须独立")
names = sorted(FILES)
assert len(names) == 13
info = HfApi(endpoint="https://huggingface.co", token=False).model_info(repo_id=repo, revision=revision)
if info.sha != revision:
    raise ValueError("远端不可变 revision 不匹配")
snapshot = Path(snapshot_download(repo_id=repo, repo_type="model",
    revision=revision, allow_patterns=names, cache_dir=cache, token=False, max_workers=2,
    endpoint="https://huggingface.co"))
if any(not (snapshot / name).is_file() for name in names):
    raise ValueError("该 revision 缺少完整 13 文件发布包")
output.mkdir(mode=0o700, exist_ok=False)
for name in names:
    with (snapshot / name).open("rb") as source, (output / name).open("xb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
checked = verify_release(output)
print(json.dumps({"status": checked["status"], "files": checked["files"],
    "gpt_sha256": checked["gpt_sha256"], "downloaded_revision": revision}))
PY
```

这是官方 [HF SDK snapshot 下载](https://huggingface.co/docs/huggingface_hub/v0.36.0/guides/download)加普通文件复制；相关 API 签名已在本地 `huggingface_hub 0.36.2` 核对；本代码块本身未执行，公开下载脚本的真实完整下载证据另见[发布记录](PUBLIC_RELEASE.md)。留出缓存与独立副本两份空间。下载进程的 HF/Xet 缓存也单独放在 WORK 内，不改共享认证目录；`token=False` 明确匿名读取。缓存 snapshot 中的文件可能是 blob symlink，示例只把其内容复制成 13 个普通文件；如果中断，残留目录不可当作完成包，也不覆写重用。

不要直接把 `hf download --local-dir ...` 的目标当作严格公开目录：HF 会添加 `.cache/huggingface/` 元数据，该目录会被精确 inventory 校验拒绝。远端 `.gitattributes`、额外本地配置、日志和临时文件也不复制进 13 文件目录；不要补入私有 `COMPLETE/provenance` 来迁就旧入口。

可另行只做本地验证：

```bash
python scripts/prepare_hf_release.py --work-root "$WORK" \
  --verify-release "$HF_PUBLIC_DIR"
```

该 verifier 的 `uploaded:false` 仅表示这次本地检查没有上传动作，不用于推断远端状态。

### 用公开包入口做 metadata preflight

仍须按后文准备固定源码和原始 20-file 官方合成资产；公开包不是整套声学模型。下面参数复用原 `synth`，仅入口不同，`--merged-bundle` 指向验证后的公开 13 文件目录：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" \
python scripts/synth_public_release.py \
  --workspace-root "$WORK" \
  --official-repo "$WORK/official" \
  --model-dir "$WORK/assets/official-synth" \
  --model-manifest "$WORK/assets/official-synth/model-manifest.json" \
  --merged-bundle "$HF_PUBLIC_DIR" \
  --variant merged \
  --prompt "$WORK/inputs/reference.wav" \
  --text 'नमस्ते। यह हिन्दी भाषा का एक छोटा परीक्षण है।' \
  --output-dir "$WORK/outputs/hindi-public-check-001"
```

默认只 metadata/hash，不导入 Torch、不加载模型或调用 CUDA。显式生成仍需在同一命令追加 `--execute --allow-cuda` 及 `--gpu-id`、`--min-free-gpu-mib`、`--min-free-ram-mib`、`--max-utilization`；其余离线、缓存隔离、绝对 GPT 配置、BF16/禁用加速与原入口一致。输出必须全新，资源门槛须按实际共享环境填写。本公开13文件格式已在H200实际完成独立stock构造和生成，见[发布记录](PUBLIC_RELEASE.md)；该结果不是ComfyUI节点验收或任意下载环境的兼容证明。

## 本地原始 bundle：保留原完成记录的后台路线

本项目已有本地原导出在 `WORK/exports/hindi-gpt-pilot-best100`，结构由 `COMPLETE.json` 决定，通常包括：

```text
hindi-gpt-pilot-best100/
  gpt.pth
  provenance.json
  COMPLETE.json
  LICENSE
  LICENSE_ZH.txt
  NON_ENDORSEMENT.txt
  README.md
  MODEL_LICENSE              # 以实际 COMPLETE 清单为准
```

`gpt.pth` 是完整官方格式的 `{"model": state_dict}`，已经合并 LoRA、语言/文字行和两个 head；它不是单独的 LoRA adapter，也不是可直接放到任意 LoRA 节点的文件。只有 `COMPLETE.json` 存在且清单全部 hash 通过才是完整 bundle。不要把训练完整 checkpoint、`adapter.pt`、音频 ZIP 或部分下载重命名成 `gpt.pth`。

## 准备独立布局

保留现有官方模型与 ComfyUI 目录，使用新的持久工作目录，例如：

```text
WORK/
  code/                         # 本项目工具
  official/                     # 干净的上述官方源码 checkout
  assets/official-synth/         # 原始官方完整合成资产，保留原 gpt.pth
    config.yaml
    model-manifest.json
    multilingual_zh_ja_yue_char_del.tiktoken
    gpt.pth
    codec.pth
    s2mel.pth
    wav2vec2bert_stats.pt
    feat1.pt
    feat2.pt
    hf_cache/...
  exports/hindi-gpt-pilot-best100/
  inputs/reference.wav
  outputs/                      # 新实验输出；父目录先创建
```

实际官方配置必须是 `version: 2.5`；目录叫“2.5”或节点名称相似都不能证明版本。`hf_cache` 仍需固定的 W2V-BERT、CAMPPlus 和 BigVGAN；本项目 `synth` 资产阶段共 20 个 payload 文件、7,082,513,550 bytes，下载前核实本地存储与许可，不拉整库，也不覆盖现有资产。资源清单与显式准备流程见 [合成指南](SYNTHESIS.md)。本页不自动下载模型或接受数据访问条件。

本项目单条 `synth` 入口要求解析后的源码、原始资产和候选目录均在所选 WORK 下且互不包含。原始模型保持只读；不要用改名、覆盖或 symlink 绕过目录检查。使用独立 Python 环境/受控 overlay；不要为了试用而升级现有 ComfyUI 的全局 Torch 或依赖。

## 本地原始 bundle 的核验与生成（不适用于 HF 13 文件包）

从本项目代码根目录运行，`WORK` 指向真实持久目录：

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src"

python -m hi_h200_native.synth --check-deps-only \
  --official-repo "$WORK/official"
```

依赖元数据预检不构造模型、不查询 CUDA。缺项、版本偏离和真实 stock import-only 检查方法见 [合成指南](SYNTHESIS.md)。即使只处理 Hindi，官方构造器也初始化日语、中英前端；不能只按目标语言猜安装依赖。

下面命令默认仅核验源码、全部本地资产、原/候选身份及输入，不初始化模型或生成 WAV：

```bash
python -m hi_h200_native.synth \
  --workspace-root "$WORK" \
  --official-repo "$WORK/official" \
  --model-dir "$WORK/assets/official-synth" \
  --model-manifest "$WORK/assets/official-synth/model-manifest.json" \
  --merged-bundle "$WORK/exports/hindi-gpt-pilot-best100" \
  --variant merged \
  --prompt "$WORK/inputs/reference.wav" \
  --text 'नमस्ते। यह हिन्दी भाषा का एक छोटा परीक्षण है।' \
  --output-dir "$WORK/outputs/hindi-best100-check-001"
```

参考音频须有权使用，是 1–15 秒非静音 mono PCM16 WAV。单条入口沿用保守 Hindi 规范，保留 nukta、virama、ZWJ/ZWNJ；不自动猜数字/货币读法，也不接受特殊 token/注音标记。输出目录必须全新。

确认共享 GPU 和宿主/容器内存余量后，在同一命令追加 `--execute --allow-cuda` 与明确的 `--gpu-id`、`--min-free-gpu-mib`、`--min-free-ram-mib`、`--max-utilization`。这些阈值必须由实际环境决定，不是显存保证。只运行一个实例；不要并发加载原/候选来争抢现有服务 GPU。成功后检查 `result.json` 的实际 GPT SHA、配置/输入/输出 hash 和开关；`failure.json` 或 partial WAV 不算成功。生成 WAV 也不自动通过音质 gate。

需要同条件原模型对照时，另开独立进程，把 `--variant` 改为 `original`，并使用新的输出目录；prompt、text、seed、cache mode 与生成参数保持一致。固定小清单的一次加载批量入口见 [合成指南](SYNTHESIS.md) 的有限验收说明。

## 实际怎样替换 GPT

公开与本地两种入口都不会修改原始 `config.yaml` 或 `gpt.pth`。它在新 run 的 `runtime/selected-config.yaml` 中保留官方完整配置，只把：

```yaml
gpt_checkpoint: /absolute/path/to/exports/hindi-gpt-pilot-best100/gpt.pth
```

指向候选文件，并以原始官方资产目录作为 `model_dir`。官方 [`IndexTTS2`](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/infer_v2_5.py) 使用 `os.path.join(model_dir, cfg.gpt_checkpoint)`；Linux 上绝对路径选择候选 GPT，其余相对资产仍从原始 `model_dir` 读取。因此 bundle 本身无需复制整套官方资产，也不能把仅有 GPT 的目录误作完整 `model_dir`。

真实调用参数是：

```python
IndexTTS2(
    cfg_path="/new/run/runtime/selected-config.yaml",
    model_dir="/read-only/official-synth",
    device="cuda:0", use_bf16=True,
    use_qwen_emo=False, use_deepspeed=False, use_accel=False,
    use_cuda_kernel=False, use_torch_compile=False,
)
```

这是已实现入口的参数说明，建议通过上述入口执行，以保留离线、hash、缓存隔离和资源门槛。官方原始加载 helper 使用 `strict=False`，所以“构造时没有崩溃”本身不是完整权重匹配的证据；本项目导出器另外验证全部官方 key/shape，已有 native 与 fresh stock 数值对照记录。不要忽略 missing/unexpected keys 日志，也不要靠手动宽松加载未知 checkpoint。

标准路径保持官方默认 `kv_cache=True`。`--no-kv-cache` 只用于显式记录的对照实验，不是本候选的必需安装步骤。DeepSpeed、accel、可选 flash-attn 扩展、自定义 CUDA kernel、compile、vLLM 等路径没有因此获得兼容证明。

## 已验证范围与质量边界

实际完整预训练模型训练、CPU checkpoint 转 adapter、完整 GPT 合并、H200 FP32 单 feature parity、Linux/H200 BF16 官方 stock WAV 和有限多语言 stress 均已有运行记录。实际 Pod 使用过 Python 3.12 / Torch 2.12 / Transformers 4.52.1；前两项超出官方 `pyproject` 的 Python `>=3.10,<3.12`、Torch `2.8.*` 约束，不能把这次可运行记录当作新的通用版本锁。Mac CPU 测试主要验证工具与 tiny 官方模型机制；没有证明 Mac/Windows/CPU/其他显卡的完整推理、速度或音质相同。

冻结 best100 的两条 Hindi test 严格 CER 仍为 **50%**；zh/ja 各一个 Hindi-prompt stress 样本的指标升高。样本有限、ASR 自身有误差、母语人工试听与完整旧语言回归未完成。详见 [有限验收](FINITE_ACCEPTANCE.md) 和 [验证记录](../VALIDATION.md)。Hindi 实验与原模型分开路由；生产默认继续使用原模型，不对所有语言全局替换。

## 原版节点的历史核查与后续RTX5090实测

早期只读核查的原版节点为`custom_nodes/comfyui-indextts25-t8`（相对ComfyUI根目录），仓库为[`T8mars/comfyui-indextts25-t8`](https://github.com/T8mars/comfyui-indextts25-t8/tree/1ff425ece89ef9824f58f0e4dbfb057991814e88)，固定HEAD `1ff425ece89ef9824f58f0e4dbfb057991814e88`，该阶段未改节点。**未经适配的这个固定版本存在HI入口和原始权重清单双阻塞，不能靠直接覆盖GPT解决。** 后续先取得四节点Hindi/base各一条的实际成功记录，之后已在RTX5090的6-5隔离服务完成原版→Hindi→原版三轮真实queue、最终WAV与同进程缓存复用验证；原8节点结构与5条连接保留。保持workflow结构、仅选择模型目录与language的路线、已过接口与未过质量边界见[RTX5090切换实测](RTX5090_SWITCHING.md)。未部署生产插件，也不据单个原版短句回切哈希相同推断所有文本或语言都无损。

1. **该原版标准生成入口不支持 HI。** 生成节点的语言选项及运行时白名单均只有 `ZH/EN/JA/ES/AR`；手传 `HI` 也会在取得模型前被拒绝。分段器的自动 token 上限也只覆盖这五种语言。其他 ASR/audio.cpp 节点出现的 `AUTO` 不是此标准生成路径的 Hindi 自动识别支持。[节点定义](https://github.com/T8mars/comfyui-indextts25-t8/blob/1ff425ece89ef9824f58f0e4dbfb057991814e88/nodes_v3.py#L2955)、[运行时白名单](https://github.com/T8mars/comfyui-indextts25-t8/blob/1ff425ece89ef9824f58f0e4dbfb057991814e88/runtime/inference_adapter.py#L100)、[分段器](https://github.com/T8mars/comfyui-indextts25-t8/blob/1ff425ece89ef9824f58f0e4dbfb057991814e88/runtime/text_planner.py#L19)。

2. **自定义路径不是自定义权重清单。** `custom_model_path` 优先选择解析后的独立目录，loader 随后始终按插件内置 `manifests/model_2_5.json` 检查文件大小；`verify_hashes=false` 仅省略 SHA 检查。其默认 `gpt.pth` 要求 `3259599833` 字节，而本候选为 `3259538991` 字节，直接换文件会被大小检查拒绝；开启 SHA 检查也不会匹配原模型 hash。插件清单还要求 `bpe.model` 和完整 Qwen 文件，不能把本项目不含 Qwen 的 20-file synth 布局或仅 GPT bundle 直接当作完整插件模型目录。不要通过篡改清单、填充字节或“下载/修复”来伪装成原始权重；后续集成须明确登记衍生模型身份及真实资产要求。[目录解析/验证](https://github.com/T8mars/comfyui-indextts25-t8/blob/1ff425ece89ef9824f58f0e4dbfb057991814e88/services/model_store.py#L115)、[固定模型清单](https://github.com/T8mars/comfyui-indextts25-t8/blob/1ff425ece89ef9824f58f0e4dbfb057991814e88/manifests/model_2_5.json)。

节点具有可用于后续独立集成的路径机制，但这些机制本身不消除上述阻塞：

- loader 的 node ID 是 `T8_IndexTTS25_ModelLoader`；实际运行加载插件内置的 `indextts.infer_v2_5.IndexTTS2`，以所选目录的 `config.yaml` 和该目录作为 `cfg_path/model_dir`。插件还传入 `use_fp16`、reference device/cache 等扩展参数，因此它不是本项目固定官方源码的逐字相同实现。[实际加载器](https://github.com/T8mars/comfyui-indextts25-t8/blob/1ff425ece89ef9824f58f0e4dbfb057991814e88/runtime/model_cache.py#L31)、[内置核心](https://github.com/T8mars/comfyui-indextts25-t8/blob/1ff425ece89ef9824f58f0e4dbfb057991814e88/indextts/infer_v2_5.py#L105)。
- `ModelHandle.cache_key` 包含 resolved model directory、设备、精度、reference/加速配置及 fingerprint。不同真实目录有不同 key；指向同一目录的 symlink 不构成目录隔离，也不能把该 key 当作模型 SHA 验证。[缓存身份](https://github.com/T8mars/comfyui-indextts25-t8/blob/1ff425ece89ef9824f58f0e4dbfb057991814e88/runtime/types.py#L29)。
- 精度 UI 支持 `auto/bfloat16/float16/float32`，`auto` 会按设备选择。若后续另行适配并验收，应先明确使用 `bfloat16`、`acceleration_mode=off`、旧 `use_cuda_kernel=false`，保持默认 `reuse_spk_cond_for_emo=false`，不使用情感文本额外加载 Qwen，并显式设置与本项目对照相同的 beam、repetition、normalization 等参数。节点默认采样参数与本项目并不相同；选择这些设置只是在缩小差异，不代表已验证。[精度/模式 UI](https://github.com/T8mars/comfyui-indextts25-t8/blob/1ff425ece89ef9824f58f0e4dbfb057991814e88/nodes_v3.py#L284)、[加速选择](https://github.com/T8mars/comfyui-indextts25-t8/blob/1ff425ece89ef9824f58f0e4dbfb057991814e88/runtime/acceleration.py#L144)、[采样参数](https://github.com/T8mars/comfyui-indextts25-t8/blob/1ff425ece89ef9824f58f0e4dbfb057991814e88/runtime/types.py#L80)、[情感文本的额外加载](https://github.com/T8mars/comfyui-indextts25-t8/blob/1ff425ece89ef9824f58f0e4dbfb057991814e88/runtime/inference_adapter.py#L174)。

早期该节点的 **CPU meta 张量接口核验已通过**：按真实构造签名使用 `spk_cond_mode=campplus`，按实际 bundle provenance 的 `checkpoint_container=model` 读取候选；期望与 checkpoint 均为 **456 tensors**，missing **0**、unexpected **0**、shape mismatch **0**，`load_state_dict(strict=True, assign=True)` 通过，`CUDA initialized=false`。节点两文件 working tree 干净且 Git blob 与固定 HEAD 一致；早期探针的参数/容器拼写错误已作为操作错误保留审计，不归因节点。[公开调用](https://github.com/T8mars/comfyui-indextts25-t8/blob/1ff425ece89ef9824f58f0e4dbfb057991814e88/indextts/infer_v2_5.py#L218)、[公开签名](https://github.com/T8mars/comfyui-indextts25-t8/blob/1ff425ece89ef9824f58f0e4dbfb057991814e88/indextts/gpt/model_v2.py#L287)。

上述CPU meta检查本身没有提交ComfyUI queue或执行GPU合成，只证明key/shape匹配，不解除原版的HI/固定清单双阻塞。后续四节点GPU成功是另一阶段的真实证据，范围见[RTX5090切换实测](RTX5090_SWITCHING.md)；它不等于完整8节点queue、同进程回切或音质通过。原始模型与冻结13文件发布包保持不动。

## 发布与许可

公开模型应提供同一候选 GPT 的 SHA、完整完成清单、来源/选择记录、实验性模型卡、上游模型完整许可与非背书声明；不要混入 optimizer、checkpoint、语料、私人音频、转写、prompt、凭据或私有报告。项目工具的 MIT 许可不覆盖官方模型与数据许可；不能因为已能加载就推断已获全部再分发权利。发布回执、许可审核和可公开内容由 [分发说明](DISTRIBUTION.md) 单独记录。
