# IndexTTS 2.5 Hindi 部署与 T8 升级

## 部署决策

**生产放行保持 HOLD。** 可在独立 5090 实例复现固定短句切换。拟上线方式为 HI 独立 worker，旧语种继续交给现有官方原版 worker。配置一起交付兼容 T8、已绑定的候选模型目录和原工作流副本。

| 路由 | 模型 | 状态 |
|---|---|---|
| ZH / EN / JA / ES / AR | 现有官方 IndexTTS 2.5 | 沿用当前生产实例和工作流 |
| HI | 由纯 LoRA step300 合并得到的完整 GPT，配套固定 2.5 资产 | 隔离复验；生产流量待签收 |

同一 RTX 5090 隔离进程已跑过原版 ZH → step300 HI → 原版 ZH。原图 8 个节点、5 条连线保留，真实 CUDA BF16 运行，首尾原版 WAV 和 PCM 一致。独立 Hindi 的完整 WAV 为原版 9/10、step300 7/10，阿拉伯语两条压力探针的字符编辑数由 5 增至 10。短句回切证据用于确认工程兼容，质量、并发和生产冷回滚分别验收。 [独立测试和 5090 证据](https://github.com/allenflux/indextts25-hindi-h200/blob/e4e0379083905e25016bb49b07cfd9be3d96b5dd/docs/HINDI_INDEPENDENT_ACCEPTANCE.zh.md)

## 工作流更改

保留原图 8 个节点的 ID、类型和连线。把下面字段合并到原 API 工作流的对应 `inputs` 中，其余输入复用现有模板。

```json
{
  "1": {
    "inputs": {
      "model_name": "IndexTTS-2.5",
      "custom_model_path": "/absolute/path/to/verified/step300-model",
      "download_missing": false
    }
  },
  "4": {
    "inputs": {
      "language": "HI",
      "text": "नमस्ते। यह हिन्दी भाषा का एक छोटा परीक्षण है।"
    }
  }
}
```

| 节点 | 具体操作 |
|---|---|
| 1 模型加载器 | HI 填**已绑定的完整候选目录**。隔离回切时填已核验的原版 2.5 目录。保留 `device=auto`、`precision=auto`、`acceleration_mode=off`、`use_cuda_kernel=false` 和 `persistent_reference_cache=true`。 |
| 2 参考音频 | `url` 仍按请求填入有使用权的参考音。控制器复验时使用固化的官方 PCM，并由隔离实例的 loopback 地址提供。 |
| 3 采样 | 沿用 `segmentation_mode=auto` 和模板的 `max_text_tokens_per_segment=120`。经审查的 T8 兼容副本在 auto 模式下对 HI 应用内部 60 token 上限。采样和停顿参数沿用原模板，业务长文本单独验收。 |
| 4 生成 | HI 路由同时填普通 Hindi 正文和 `language=HI`。插件处理语言前缀。隔离回切将语言恢复为对应的旧语种并同批切回原版路径。 |
| 5 输出 | 烟测使用独立文件名前缀和 `encrypt=false`，复核最终 WAV。生产输出和密钥仍按业务配置管理，加密流程按目标业务图复验。 |
| 6、7、8 | 保留兼容占位、备用情绪参考与情绪控制；固定验收图的节点 8 为 `mode=speaker`。 |

路由或工作流模板要同时设置节点 1 的路径和节点 4 的语言。T8 的语言字段负责文本处理，部署端负责模型选择；候选只接受发布时核定的目录。普通旧语请求继续使用现有工作流。HI worker 不可用时向调用方返回明确的 HI 不可用状态。H200 上手工比较 120 与 60 token 属于另一套推理入口，实测 60 的生成覆盖下降，HI auto 上限仍需在目标工作流的长文测试中验证。

## 插件升级与资产准备

**升级方式是生成固定 T8 兼容副本，再将其装入独立实例。** 上游 T8 固定到 `1ff425ece89ef9824f58f0e4dbfb057991814e88`，ComfyUI 固定到 `8ff6dc384ba5c410266b40e137799e049459d4f2`。脚本读取上游固定 Git 树，生成适配 `HI` 的节点与文本计划，保留旧语种配置。候选由 `T8_REVIEWED_CANDIDATE.json` 和固定清单选择，强制校验 26 项资产，包括辅助文件；自动下载入口关闭。节点名和连线与原工作流兼容。 [已实测的工具源码](https://github.com/allenflux/indextts25-hindi-h200/tree/d6527571d679ebc16e9caeae4de11d75834e0d0f)

从本次受控资产交接取得 step300 适配器和原版资产。公开 adapter ZIP 固定为历史 step100。适配器 5,931,907 字节，只供离线合并。节点 1 使用包含 26 项资产和登记标记的独立候选目录。保留原版 26 项目录供基准及回切，另备原生合并用的 16 项原始清单与对应资产。

| 输入或产物 | 固定身份 |
|---|---|
| 官方 IndexTTS 2.5 模型 | `IndexTeam/IndexTTS-2.5@c39ce5ba981572cb187443877ff559dfb246ce63` |
| 官方源码 | `d9e41aac89fd00b3d71497fddb287b7f24613712` |
| 原版 GPT SHA256 | `43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85` |
| 原始 16 项训练清单 SHA256 | `11dae1733635a0d46e7edbc7c5444900fed2bbb4450767aa4d2820b2221f0e6f` |
| step300 适配器 SHA256 | `035c586d265d3159d851ba59b3a6519d06e140b3894d50819c7ad93dc07ab782` |
| 完整 step300 GPT SHA256，3,259,538,991 字节 | `e2c526972292286b55829ab3ec0ba7d11259c172704d7ed871b4787abf4e00db` |
| 候选登记 SHA256 | `a84870771b2d2e1891db2d1f4b11b171901c88d7a90ab63003346d6dc55d0759` |

在 Linux 5090 验证环境设置这些规范绝对路径，路径各级均为真实目录。用本机已审核的 Python 解释器和离线依赖；导出另加既有 `PY_DEPS` 和 `PY_EXTRA` overlay。实际准备使用至少 16 GiB 存储余量和 32 GiB 可用 RAM 的 CPU 导出窗口。候选目录的 26 项文件合计 8,290,912,900 字节。将原版 T8 目录和参考 WAV 的变量分别指向 `WORK` 内已核验的实际位置。以下命令在同一个 Bash 会话依次执行。

```bash
set -euo pipefail
umask 077

COMFY_ROOT=/app/ComfyUI
WORK="$COMFY_ROOT/models/indextts25-hindi-acceptance-20261117"
RUN="$WORK/step300-validation-new"
CODE=/absolute/path/to/pinned-toolkit
PY=/absolute/path/to/reviewed/python3.12
PY_DEPS=/absolute/path/to/reviewed/py-deps
PY_EXTRA=/absolute/path/to/reviewed/py-extra
OFFICIAL=/absolute/path/to/pinned-index-tts
TRAINING_BASE=/absolute/path/to/original-training-assets
TRAIN_MANIFEST="$TRAINING_BASE/training-base-manifest.json"
T8_UPSTREAM="$COMFY_ROOT/custom_nodes/comfyui-indextts25-t8"
BASE_MODEL="$WORK/original-model"
ADAPTER=/absolute/path/to/step00000300.pt
PROMPT="$WORK/reference/voice_01-pcm16.wav"

BUNDLE="$RUN/export"
HI_MODEL="$RUN/model300"
T8_COMPAT="$RUN/t8-compat"
REG="$RUN/candidate.json"
STAGE="$RUN/stage"
REPORT="$RUN/report"
PORT=18193
ADAPTER_SHA=035c586d265d3159d851ba59b3a6519d06e140b3894d50819c7ad93dc07ab782
GPT_SHA=e2c526972292286b55829ab3ec0ba7d11259c172704d7ed871b4787abf4e00db
COMPLETE_SHA=48fe36e14057b6f0cddc42611ab8d7fea477d6ca0943eb2fd35090c661cb529c
PROVENANCE_SHA=4f98fcc004a4fdabb094b3ee6fae6dee92b248a8811d0fce693fe77cdc7e3d7b
REG_SHA=a84870771b2d2e1891db2d1f4b11b171901c88d7a90ab63003346d6dc55d0759

mkdir -m 700 "$RUN"
check_sha() {
  test "$(sha256sum -- "$2" | cut -d ' ' -f 1)" = "$1"
}
test "$(git -C "$CODE" rev-parse HEAD)" = d6527571d679ebc16e9caeae4de11d75834e0d0f
test "$(git -C "$OFFICIAL" rev-parse HEAD)" = d9e41aac89fd00b3d71497fddb287b7f24613712
test "$(stat -c %s "$ADAPTER")" = 5931907
check_sha "$ADAPTER_SHA" "$ADAPTER"
check_sha 11dae1733635a0d46e7edbc7c5444900fed2bbb4450767aa4d2820b2221f0e6f "$TRAIN_MANIFEST"

env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH="$CODE/src:$PY_DEPS:$PY_EXTRA" \
  "$PY" -B -m hi_h200_native.export \
  --official-repo "$OFFICIAL" --model-dir "$TRAINING_BASE" --model-manifest "$TRAIN_MANIFEST" \
  --adapter "$ADAPTER" --adapter-sha256 "$ADAPTER_SHA" --output "$BUNDLE" \
  --protect-root "$COMFY_ROOT/custom_nodes"

check_sha "$GPT_SHA" "$BUNDLE/gpt.pth"
check_sha "$COMPLETE_SHA" "$BUNDLE/COMPLETE.json"
check_sha "$PROVENANCE_SHA" "$BUNDLE/provenance.json"

"$PY" -B - "$REG" "$ADAPTER_SHA" "$GPT_SHA" "$COMPLETE_SHA" "$PROVENANCE_SHA" <<'PY'
import json
from pathlib import Path
import sys

filename, adapter, gpt, complete, provenance = sys.argv[1:]
record = {
    "format": "t8-reviewed-complete-gpt-v1",
    "candidate_id": "hindi-pure-pilot48-opt-step300",
    "repository": None,
    "revision": None,
    "training_profile": "attention_lora_only_v1",
    "adapter_sha256": adapter,
    "gpt_sha256": gpt,
    "gpt_bytes": 3259538991,
    "export_complete_sha256": complete,
    "export_provenance_sha256": provenance,
    "quality_gate_passed": False,
}
with Path(filename).open("xb") as stream:
    stream.write((json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
PY
check_sha "$REG_SHA" "$REG"

"$PY" -B "$CODE/scripts/check_t8_compat.py" inspect --node-root "$T8_UPSTREAM"
"$PY" -B "$CODE/scripts/check_t8_compat.py" prepare \
  --node-root "$T8_UPSTREAM" --destination "$T8_COMPAT" \
  --candidate-manifest "$REG" --candidate-manifest-sha256 "$REG_SHA"
"$PY" -B "$CODE/scripts/check_t8_compat.py" verify-copy \
  --node-root "$T8_COMPAT" --candidate-manifest "$REG" --candidate-manifest-sha256 "$REG_SHA"
"$PY" -B "$CODE/scripts/check_t8_compat.py" verify-assets \
  --node-root "$T8_COMPAT" --model-dir "$BASE_MODEL" --model-variant base \
  --candidate-manifest "$REG" --candidate-manifest-sha256 "$REG_SHA"
```

按原版清单组装候选。在新目录中复制 25 项原版文件和新 GPT，随后执行绑定。绑定成功会在候选根目录生成登记标记。

```bash
"$PY" -B - "$T8_COMPAT" "$BASE_MODEL" "$BUNDLE" "$HI_MODEL" <<'PY'
import json
from pathlib import Path, PurePosixPath
import shutil
import sys

plugin, base, bundle, candidate = map(Path, sys.argv[1:])
files = json.loads((plugin / "manifests/model_2_5.json").read_bytes())["files"]
if len(files) != 26 or "gpt.pth" not in files:
    raise ValueError("Unexpected model inventory")
candidate.mkdir(mode=0o700)
for name in sorted(files):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
        raise ValueError("Unsafe manifest path")
    source = (bundle if name == "gpt.pth" else base) / relative
    if source.resolve(strict=True) != source or not source.is_file():
        raise ValueError("Expected a canonical regular asset")
    destination = candidate / relative
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
PY

"$PY" -B "$CODE/scripts/check_t8_compat.py" bind-candidate \
  --node-root "$T8_COMPAT" --model-dir "$HI_MODEL" --workspace-root "$WORK" \
  --candidate-manifest "$REG" --candidate-manifest-sha256 "$REG_SHA"
```

## 5090 原图验收和隔离启动

用固定控制器做 CPU stage 和 GPU 烟测。stage 创建独立插件、数据库、输入输出和缓存目录；同实例只加载一份 T8。端口使用空闲的高位 loopback 端口。示例中的 ops 兼容参数只在宿主 `comfy/ops.py` 为已审三行变更、SHA256 为 `68611fba7c1d8750f6bd7266d5c6a40deb5c74ac8930f68c2572f4d314d299c8` 时使用；固定的原版宿主省略该参数，其他差异先复核。

```bash
export PYTHONDONTWRITEBYTECODE=1
"$PY" -B "$CODE/scripts/stage_comfy_validation.py" stage \
  --workspace-root "$WORK" --comfy-root "$COMFY_ROOT" --t8-root "$T8_COMPAT" \
  --audio-cache-root "$COMFY_ROOT/custom_nodes/comfyui-audio-cache" \
  --audio-crypt-root "$COMFY_ROOT/custom_nodes/comfyui-audiocrypt" \
  --destination "$STAGE" --port "$PORT" --preserve-reviewed-ops \
  --candidate-manifest "$REG" --candidate-manifest-sha256 "$REG_SHA"

"$PY" -B "$CODE/scripts/stage_comfy_validation.py" verify --stage-root "$STAGE" \
  --candidate-manifest "$REG" --candidate-manifest-sha256 "$REG_SHA"

PYTHONPATH="$CODE/src" "$PY" -B "$CODE/scripts/run_comfy_validation.py" \
  --workspace-root "$WORK" --stage-root "$STAGE" --base-model "$BASE_MODEL" \
  --hindi-model "$HI_MODEL" --prompt "$PROMPT" --report-dir "$REPORT" \
  --candidate-manifest "$REG" --candidate-manifest-sha256 "$REG_SHA" \
  --prompt-sha256 fd23ddac68f2db3710481c7cbfc1a50a117d9c38bc5c3ee51c33444f0a9b019a \
  --execute --allow-cuda --gpu-id 0 --min-free-gpu-mib 24576 --min-free-ram-mib 16384 \
  --max-utilization 10 --max-used-gpu-mib 24576 --startup-timeout 120 --run-timeout 600
```

`PROMPT` 使用受审官方示例的 PCM16 转换文件。先确认原服务队列空闲和实际 GPU 余量。结果验收核对 `result.json` 的原版、HI、回切三轮完成，候选 GPT SHA、实际 CUDA BF16、原版缓存复用、首尾 PCM 一致和自有进程退出。固定控制器的 loopback、输入约束及离线缓存用于隔离验收。运行工件和业务实例的发布配置分开保留，业务配置按其实际环境单独验收。

## 放行、部署和回滚

1. **签收**。登记候选 adapter/GPT、完整资产清单、兼容插件、工作流和路由版本。HI 先通过 ≥40 条、≥10 位新说话人、至少 10 条长文/跨段的双人母语盲评及严重缺陷检查。验证旧语请求仍命中原版。并发、冷启动和生产回滚按目标实际负载演练，保留原版快照和结果哈希。
2. **部署**。门槛签收后，在维护窗口准备独立 HI worker。将整个兼容插件目录安装到该 worker 自己的 `custom_nodes`，该进程只加载这一份 T8。将兼容插件、候选模型目录、工作流副本、路由配置和适用许可证作为同一版本发布。先跑旧语和 HI 的真实业务图，再开放内部白名单。首轮一实例或 ≤5% 流量，至少 24 小时并覆盖 100 条旧语和 20 条 HI 请求。
3. **停发**。候选身份错误、旧语误路由、OOM 或严重音频缺陷时立即暂停候选派单。连续两个五分钟窗口出现旧语 p95 超过原基线 1.20× 或错误率增加超过 1 个百分点时，同样回退。保留报告及失败、取消和超时记录。
4. **回滚**。将 HI 路由恢复到变更前状态并排空 HI worker。若服务层兼容更新已进入公共工作实例，按发布快照恢复原插件与原启动配置，冷启动该实例清除候选缓存。核验原 GPT SHA、固定 ZH 与业务关键旧语的输出、队列及进程状态。恢复接单目标 ≤15 分钟，以实际演练为准。

当前允许隔离复验。候选权重、训练语料、参考录音、原始日志及凭据保留在受控环境；公开交付保存代码、许可、哈希与脱敏结果。 [完整质量和回滚门槛](https://github.com/allenflux/indextts25-hindi-h200/blob/e4e0379083905e25016bb49b07cfd9be3d96b5dd/docs/REPLACEMENT_ACCEPTANCE.md#5-语音验收与放行门槛)
