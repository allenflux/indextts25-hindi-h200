# 历史方案：单模型默认替换（已撤回）

更新日期：2026-11-18（Asia/Saigon）。**用户已取消直接替换和单候选全局默认方向；本文件只保留旧方案、审计证据及未执行步骤的历史，不再作为部署建议。不要执行下文 B/B2 的替换或重启步骤。** 现行方向是 [重新训练 Hindi 专用纯 attention LoRA](HINDI_LORA.md)：原模型文件保留，全部 head/embedding 冻结；Hindi 用独立 adapter 实例，旧语言使用从未加载 adapter 的原实例。

旧 best100 含 head/embedding 行更新，不能去掉 A/B 后冒充原模型，也不能改名为新纯 LoRA。本文件的 CPU 原型、333 项测试与旧权重审计只属于历史路线；没有生产替换，也不构成新 profile 已训练、已推理或质量通过。

## 历史 10.1：原单模型方案与适用边界

工程上可以把合并后的 GPT 做成唯一默认模型，不需要运行时 LoRA 节点，也不依赖原版 GPT 驻留。声学后端等资产仍须匹配，完整 T8 目录是26项资产，候选清单payload共8,290,912,900 bytes（约8.29 GB），HF的13文件发布包本身不够。清单中25项与固定base的size/SHA身份相同，变化项是GPT；这不是按参数数量或音质计算的比例。保留旧选择名称只是兼容工作流字段，加载日志、manifest和缓存身份必须明确显示真实实验候选，不能伪装成官方权重。

**该单模型默认替换路径已撤回；以下保留当时的证据边界，不能承诺旧语言“大致一致”或无损。** 未改变的声学后端不代表最终声音不变；共享 GPT 的更新会影响其他语言。CPU原型检查、原候选HI已能生成、原版回切成功分别是不同证据。

## 历史 10.2：旧 best100 哪些保留、哪些改变

| 部分 | 实际情况 | 对“其他语言还是原模型”的含义 |
|---|---|---|
| 匹配的 codec、声学 CFM/DiT、BigVGAN、speaker/emotion 前端 | 沿用固定原版资产；未微调这些模块 | 后端相同，不保证它接收到的语义codes相同 |
| GPT attention | 24层的共享 `c_attn` / `c_proj` 已合并 rank8、alpha16 LoRA | ZH/EN/JA/ES/AR与HI都经过这些共享投影 |
| GPT语言 embedding | 训练只开放Hindi第17行 | 其他语言行受保护，也不能抵消共享层变化 |
| GPT文字 embedding | 训练实际出现的token行允许更新，可能含共用标点/token | 不能视为纯Hindi私有词表 |
| `text_head` / `mel_head` | 完整矩阵和bias参与训练并合并 | `mel_head`预测语义codes；旧语言也使用它 |

依据是实际[训练与合并实现](MODEL_RATIONALE.md)，不是仅凭LoRA通常如何工作推测。GPT文件大小接近、相同架构或大部分参数冻结均不能推导音质接近。

### 本轮真实权重审计

已对现有两份固定GPT执行一次只读CPU比较，不下载权重、不构造TTS、不执行GPU。先后均核验完整文件SHA和文件状态；使用 `weights_only=True`、CPU `mmap`、单线程和8MiB分块，按原始字节比较，不把正负零或NaN位模式归一化。审计源码固定为 [e193cab](https://github.com/allenflux/indextts25-hindi-h200/blob/e193cabca88bb1186ee1001820d5068c1b17b787/scripts/audit_merged_gpt.py)。

两份各456个张量，key/shape/dtype全部相同；**402个逐字节相同，54个有变化**：

| 实际分组 | 逐字节相同 | 有变化 |
|---|---:|---:|
| 24层共享attention投影weight | 0 | 48 |
| attention投影bias | 48 | 0 |
| 语义code head与文本辅助head（weight/bias） | 0 | 4 |
| 语言与文字embedding矩阵 | 0 | 2 |
| MLP、norm、位置、mel embedding与其余conditioning/state | 354 | 0 |

语言embedding共107行，仅Hindi第17行变化，其余106行逐字节相同；文字embedding共60,510行，69行变化、60,441行相同。审计只报告行计数，没有从这些行数推断它们全部属于Hindi或证明旧语言语义不变。张量数也不是参数数量/数值幅度占比，更不是音质百分比。

[完整只读审计JSON](acceptance/gpt-parameter-audit-20261118.json) 为2,723 bytes，SHA256 `d95482eb80d5434da7a9449f81738056121925600ebbf9bcedb253431625ea22`；已由正常Kite文件预览独立回读，本地同SHA。审计计时17.157秒（CPU，非推理耗时），`cuda_initialized=false`、`quality_verified=false`。结束时自建审计进程为空，原服务HTTP200、队列0/0；GPU仍654MiB已用、31,457MiB可用、0%利用率。未改生产权重、插件或服务。


## 历史 10.3：旧 best100 音频对照

以下严格CER从公开冻结 `evaluation.json` 重新核对，并与本地公开包 `SHA256SUMS` 一致，文件SHA256为 `4bf22df2ffd4e94139e490f6fcd179e752b78f1773143a3a140adb6feecfd139`。指标定义为原定canonical文本的NFC/空白规范化后codepoint编辑率，包含空格；不是母语人工评分。[固定HF评测文件](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental/blob/117c7922e4dcedc3f9885f3f6efa5be8f54c7919/evaluation.json)

| 语言 | 原版严格CER | 候选严格CER | 证据范围 |
|---|---|---|---|
| 中文ZH | 1/17 = 5.88% | 3/17 = 17.65% | 仅1例，Hindi参考音频 |
| 英语EN | 1/46 = 2.17% | 1/46 = 2.17% | 仅1例，Hindi参考音频 |
| 日语JA | 0/18 = 0% | 2/18 = 11.11% | 仅1例，Hindi参考音频 |
| 西语ES | 1/60 = 1.67% | 1/60 = 1.67% | 仅1例，Hindi参考音频 |
| 阿语AR | 0/41 = 0% | 0/41 = 0% | 仅1例，Hindi参考音频 |
| Hindi HI | 396/166 = 238.55% | 83/166 = 50% | 2位held-out speaker各1例；真人reference CER为31/166=18.67% |

中文、日文单例ASR指标变差是进一步检查的信号，不能自动认定TTS发音退化；其他三个单例相同也不能推广为普遍不变。旧五语言没有native-prompt校准或人工母语听评，Hindi reference不能校准其他语言。所有原版/候选输入、prompt、seed和设置配对已核对，每个case只执行一次；保留全部结果，没有按test重选best100。[完整有限验收](FINITE_ACCEPTANCE.md)

RTX5090三轮原版ZH→候选HI→原版ZH已真实完成，首尾原版WAV及PCM哈希相同；**这证明回到原模型恢复该固定短句，不证明候选用ZH输出与原版相同。** 本轮不把这项结果挪作单模型旧语言无损证据。[实际三轮回执](acceptance/rtx5090-switch-cycle-20261117.json)

## 历史 10.4：未部署的离线原型

已实现并CPU核验单候选原型：新工具从真实固定T8 `1ff425ece89ef9824f58f0e4dbfb057991814e88` 创建全新副本，`prepare`及`verify-copy`成功；标准默认路径、原模型名/空自定义路径、26项强制校验、legacy参数和6个禁下载入口均做过CPU检查。真实公开13文件目录作为负例被拒绝（缺少T8清单中的 `bpe.model` 等资产），没有把半套包当完整模型。

完整CPU suite为333/333通过、0跳过、37.916秒，对应新工具SHA前缀 `93ea6be1` 的快照。随后仅修正下载器候选身份文案和一处源文件行末空白；最终工具SHA256为 `8dee60ae1eb3f7e479364c58e2dc7c675cf6a46b49dad1895dcb42b5cee3dbc7`，相关22/22测试另通过、0.974秒，并对真实固定源码的fresh003完成校验。**没有把最终小幅改动的定向通过写成又完整重跑333项。** 权重审计器另经独立7/7真实tiny Torch CPU测试；这些测试不冒充GPU或音质验收。

新副本marker `T8_SINGLE_MODEL_PROTOTYPE.json` 明确 `cpu_prototype_only=true`、`gpu_validated=false`、`old_language_quality_verified=false`、`production_deployed=false`。此前已通过的GPU三轮属于另一个双目录兼容profile，新单候选默认profile尚未做GPU queue检查或生产部署。


原型准备器为 `scripts/prepare_t8_single_model.py`，只从固定T8源码创建新的独立代码副本，不部署服务、不下载模型、不执行GPU。与已做三轮GPU检查的双目录准备器分开；不能把此前GPU通过直接写成这个新默认路径已验。

目标行为如下：

- 原 `T8_IndexTTS25_ModelLoader` / Generate等节点类型、ID、连线及旧输入字段保留；默认 `model_name` 仍是 `IndexTTS-2.5`，空 `custom_model_path` 解析标准候选目录。
- 所有语言使用同一固定 best100 manifest与真实候选身份，不依据GPT大小回退原模型。额外搜索根或备份不能被静默发现成默认模型。
- `verify_hashes=false`、旧位置参数和跳过辅助资产均不能跳过完整26项校验；原GPT、缺失资产、同大小错误文件必须失败。
- 原工作流的 `download_missing=false` 保留。原型暂不支持自动下载/修复；为防误取原GPT，节点、下载CLI和直接下载入口均提前拒绝该操作。需要完整离线准备资产，不能称所有下载功能完全兼容。
- 日志明确记录候选repository、revision和GPT SHA；同路径升级必须维护重启到新进程，不能以旧内存或图缓存的输出当作新模型已加载。

```bash
python scripts/prepare_t8_single_model.py prepare \
  --node-root "$T8_SOURCE" --destination "$STAGED_T8_SINGLE" \
  --candidate-revision 117c7922e4dcedc3f9885f3f6efa5be8f54c7919
python scripts/prepare_t8_single_model.py verify-copy \
  --node-root "$STAGED_T8_SINGLE" \
  --candidate-revision 117c7922e4dcedc3f9885f3f6efa5be8f54c7919
python scripts/prepare_t8_single_model.py verify-assets \
  --node-root "$STAGED_T8_SINGLE" --model-dir "$CANDIDATE_DIR" \
  --candidate-revision 117c7922e4dcedc3f9885f3f6efa5be8f54c7919
```

`T8_SOURCE`为固定原始Git根；`STAGED_T8_SINGLE`为扫描路径外全新目录、父目录已存在；`CANDIDATE_DIR`为已有完整候选26资产目录。上述命令不准备生产路径或自动迁移任何文件；代码核验通过不等于资产或音质通过。

## 历史 10.5：已撤回的直接替换步骤（不执行）

以下仅存档旧方案的路径和步骤，不是现行操作授权。原路径针对当时已核验的 Linux/RTX5090 环境：ComfyUI根为 `/app/ComfyUI`，原T8目录为 `/app/ComfyUI/custom_nodes/comfyui-indextts25-t8`。完整候选来源是此前6-5验收目录；若换到另一台5090，先确认该目录实际存在、文件已迁移且26项哈希通过，不能假设另一台自动具备它。下面是操作说明，**本轮没有执行生产替换或重启**。

| 对象 | 5090 容器内路径 / 最终状态 |
|---|---|
| 已有完整候选源 | `/app/ComfyUI/models/indextts25-hindi-acceptance-20261117/portable-assets-001/models/hindi-best100-001` |
| 新候选唯一运行目录 | `/app/ComfyUI/models/TTS/IndexTTS-2.5` |
| 兼容插件部署目标 | `/app/ComfyUI/custom_nodes/comfyui-indextts25-t8`；只加载这一份T8 |
| 暂存与冷备 | `/app/ComfyUI/models/indextts25-single-model-staging-20261118`；放在TTS发现与custom_nodes扫描路径之外 |
| 工作流节点1 | `model_name="IndexTTS-2.5"`，`custom_model_path=""` |
| 工作流节点4 | Hindi选择 `language="HI"`；旧语言选择原语言，仍加载同一候选 |

### A. 准备并核验，不动运行中的服务

先核对剩余磁盘、RAM、真实GPU占用和服务队列；暂存完整候选至少需要约8.29GB额外payload空间，冷备及缓存另计。原宿主配置曾出现 `version:2.0`，不能只凭旧文件夹名直接沿用；必须使用匹配的官方2.5配置与全部资产。以下代码块在同一Shell按顺序执行，保留前面的环境变量。下列准备器不安装依赖、不下载模型、不调用CUDA，固定版本不匹配会停止，不应强行跳过。

```bash
set -euo pipefail
export COMFY_ROOT=/app/ComfyUI
export WORK="$COMFY_ROOT/models/indextts25-single-model-staging-20261118"
export T8_SOURCE="$COMFY_ROOT/custom_nodes/comfyui-indextts25-t8"
export STAGED_T8_SINGLE="$WORK/plugin-ready"
export CANDIDATE_DIR="$COMFY_ROOT/models/indextts25-hindi-acceptance-20261117/portable-assets-001/models/hindi-best100-001"
export MODEL_READY="$WORK/model-ready"

# WORK 必须是全新目录；已存在时停止，另选新名字，不覆盖历史结果。
mkdir -- "$WORK"
git clone https://github.com/allenflux/indextts25-hindi-h200.git "$WORK/toolkit"
git -C "$WORK/toolkit" checkout --detach 7836954d0f25ce2704be849aa8fa8247327383e9
python "$WORK/toolkit/scripts/prepare_t8_single_model.py" prepare \
  --node-root "$T8_SOURCE" --destination "$STAGED_T8_SINGLE" \
  --candidate-revision 117c7922e4dcedc3f9885f3f6efa5be8f54c7919
python "$WORK/toolkit/scripts/prepare_t8_single_model.py" verify-assets \
  --node-root "$STAGED_T8_SINGLE" --model-dir "$CANDIDATE_DIR" \
  --candidate-revision 117c7922e4dcedc3f9885f3f6efa5be8f54c7919
```

固定工具commit只取代码仓库（含已公开的小型演示文件），不拉大型模型或语料；当前目标T8须仍是已核固定源码，保留任何本地变更再走适配，不要执行强制reset。将校验过的26项资产复制到全新 `MODEL_READY`，可以使用下面的精确清单复制，避免带入源目录内无关文件；复制后再完整校验。该阶段不写生产目标目录：

```bash
PYTHONPATH="$WORK/toolkit/scripts" python - <<'COPY_MODEL'
import os, shutil
from pathlib import Path
from prepare_t8_single_model import REVISION, verify_copy, verify_assets
node = Path(os.environ["STAGED_T8_SINGLE"])
source = Path(os.environ["CANDIDATE_DIR"])
target = Path(os.environ["MODEL_READY"])
info = verify_copy(node, REVISION)
verify_assets(node, source, REVISION)
if target.exists() or target.is_symlink():
    raise SystemExit("MODEL_READY must be fresh")
target.mkdir(mode=0o700)
for name in info["manifest_data"]["files"]:
    destination = target / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / name, destination)
verify_assets(node, target, REVISION)
print("26 candidate assets copied and verified; production unchanged")
COPY_MODEL
```

新目录还须一并保留[固定公开权重包](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental/tree/117c7922e4dcedc3f9885f3f6efa5be8f54c7919)附带的完整 `LICENSE`、`LICENSE_ZH.txt`、`MODEL_LICENSE`、`DATA_ATTRIBUTION.md`、`NON_ENDORSEMENT.txt` 及模型卡；它们不是26项推理资产的替代品。插件副本保留自身许可证。HF13文件包只有GPT等发布内容，不能用它代替上述26项完整运行目录；本原型自动下载/修复保持禁用，原工作流的 `download_missing=false` 不变。

### B. 维护窗口内替换两个目录

1. 暂停向目标5090派新任务，确认running/pending为0，先备份当前T8、所有本地变更、原工作流JSON和原模型引用。冷备放到扫描/发现路径之外，不能把两份同名T8同时留在 `custom_nodes`。
2. 用该实例**现有且已核对的启动器**停止旧ComfyUI进程，确保没有模型请求仍在执行。不得热覆盖运行中的权重或插件。本项目未核定该实例的具体服务管理命令，因此不猜测 `kill`、Pod重建或启动命令。
3. 若生产目标 `/app/ComfyUI/models/TTS/IndexTTS-2.5` 已存在，把它完整移到冷备；再把已验 `MODEL_READY` 安装到该目标，保留整个目录结构、配置与许可。**不是只覆盖一个GPT，也不覆盖整个ComfyUI目录。** 原GPT不参与候选运行，冷备仅用于恢复。
4. 把旧T8移到冷备，将已验 `STAGED_T8_SINGLE` 安装为 `/app/ComfyUI/custom_nodes/comfyui-indextts25-t8`。不要复制验证stage/core/bootstrap到生产；保留既有Torch/CUDA/ComfyUI及已审ops修正。用同一准备器对最终插件运行 `verify-copy`、对最终模型目录运行 `verify-assets`，任一步失败保持停派单并恢复备份。
5. 通过既有启动器启动ComfyUI并刷新前端schema。确认 `HI` 出现在同一个生成节点的语言选项中；原节点类型、ID和5条连线不变。模型加载info必须显示真实best100 repository、固定revision `117c7922…` 和GPT SHA `423a8491…`，不能只看到旧的显示名称就认定加载成功。

### 历史 B2：原模型目录不换的 GPT/config 替换（不执行）

这是同一单模型方案的最小文件替换方式，可替代上面B的第3步；停服、兼容插件升级和验收步骤仍须执行。**不是在原插件上裸换GPT，也不是热覆盖。** 原模型目录名称和工作流路径字段保持不变，只保留候选作为运行模型，旧GPT放扫描路径外冷备。

适用条件是原目标目录中，除 `gpt.pth`、`config.yaml` 外其余24项已与候选清单逐项匹配。若配置本来就是匹配的官方2.5，只需更新GPT；本项目原宿主曾为 `version:2.0`，所以示例同时复制候选的 `config.yaml`。任何其余资产缺失或不匹配，都应先补齐完整目录并重新核验，不能只换GPT后关闭检查。

先在准备阶段只读核对这24项；这不是完整26项候选校验，也不修改原文件：

```bash
export MODEL_TARGET="$COMFY_ROOT/models/TTS/IndexTTS-2.5"
PYTHONPATH="$WORK/toolkit/scripts" python - <<'CHECK_UNCHANGED'
import os
from pathlib import Path
from prepare_t8_single_model import REVISION, compat, verify_copy
info = verify_copy(Path(os.environ["STAGED_T8_SINGLE"]), REVISION)
files = {name: data for name, data in info["manifest_data"]["files"].items()
         if name not in {"gpt.pth", "config.yaml"}}
assert len(files) == 24
compat.verify_assets(Path(os.environ["MODEL_TARGET"]), {"files": files})
print("24 retained assets match; GPT/config and full candidate checks still required")
CHECK_UNCHANGED
```

只有在目标ComfyUI进程已通过原启动器停止、队列清空、`MODEL_READY`全部26项已验且冷备目录全新的情况下，才执行下面的文件替换。先记录原文件身份；已有候选不能误作官方原版备份。此代码块只是部署说明，**本轮未在现网执行**：

```bash
set -euo pipefail
export DIRECT_BACKUP="$WORK/direct-cold-backup"
mkdir -m 700 -- "$DIRECT_BACKUP"
cp -p -- "$MODEL_TARGET/gpt.pth" "$DIRECT_BACKUP/gpt.pth"
cp -p -- "$MODEL_TARGET/config.yaml" "$DIRECT_BACKUP/config.yaml"
sha256sum "$DIRECT_BACKUP/gpt.pth" "$DIRECT_BACKUP/config.yaml" > "$DIRECT_BACKUP/SHA256SUMS"

# 避免覆盖已有暂存文件，包括悬空符号链接。
test ! -e "$MODEL_TARGET/.gpt.single-model-new"
test ! -L "$MODEL_TARGET/.gpt.single-model-new"
test ! -e "$MODEL_TARGET/.config.single-model-new"
test ! -L "$MODEL_TARGET/.config.single-model-new"
cp -- "$MODEL_READY/gpt.pth" "$MODEL_TARGET/.gpt.single-model-new"
cp -- "$MODEL_READY/config.yaml" "$MODEL_TARGET/.config.single-model-new"
mv -- "$MODEL_TARGET/.gpt.single-model-new" "$MODEL_TARGET/gpt.pth"
mv -- "$MODEL_TARGET/.config.single-model-new" "$MODEL_TARGET/config.yaml"
python "$WORK/toolkit/scripts/prepare_t8_single_model.py" verify-assets \
  --node-root "$STAGED_T8_SINGLE" --model-dir "$MODEL_TARGET" \
  --candidate-revision 117c7922e4dcedc3f9885f3f6efa5be8f54c7919
```

两次文件rename不是一个跨文件事务；若中途停止或完整校验失败，保持服务停止，从冷备恢复这两个文件并复核原SHA，不可带着半替换状态启动。完整校验成功后，按B第4–5步安装唯一兼容T8、核验最终插件/模型并启动；同时保留模型完整许可与归属。直接替换没有另外的LoRA文件，最终GPT SHA必须为 `423a8491b8856c7c3075254b12defc3ae47dbd9e0de285772523ddd4e9a97021`。

### C. 验收后恢复派单；失败时恢复冷备

用原工作流真实提交旧语言及HI短样本：节点1保持 `IndexTTS-2.5` 和空自定义路径，只按业务选择语言、文本与合法参考音频。检查真实queue history、最终WAV、候选SHA/revision、缓存指纹与显存；各语言都应使用候选，不能回退原GPT后冒充单模型通过。同路径升级必须使用新进程，不能拿旧缓存的结果当替换成功。

该新默认路径的GPU验收和10.6的旧语言质量门槛通过、再完成小流量检查后，才恢复候选默认派单。日常无需切模型目录或另挂LoRA节点；加载和质量未过时，现网默认仍保留原版。

回滚时暂停派单、通过同一启动器停止新进程，把原插件、原完整模型目录与原工作流引用从冷备恢复，再重启并核验。没有自动回原GPT；完全丢弃原文件会失去快速恢复来源，部署前须先明确冷备或原版重建来源。

## 历史 10.6：旧方案当时未满足的质量门槛

目前已有证据足以编写和CPU验证升级原型，**不足以批准所有旧语言的默认模型替换**。下一轮应在生成前冻结方案：五种原语言各选短句、数字/标点句、较长分段句各1条，共15对原版/候选；每种语言使用来源/使用权清楚的native-language参考，成对固定text、reference、seed和全部采样配置，不挑好例、不重选checkpoint。缺少合法母语参考时记录缺口，不能拿Hindi参考补齐后称母语回归。

这15对只是最低限度的定向诊断，不是统计上的普遍等价证明。保留所有音频；ASR同时检查合法真人参考以识别识别器偏差，并由相应母语审听者检查发音、漏字/重复、自然度、音色和句尾完整性。接受标准应在看新输出前固定；不能依据候选碰巧好看的样本临时放宽，也不能把参数相同比例、loss、CER单一数值或输出文件存在当作无损证明。

另需一次新单模型profile的原工作流RTX5090真实queue验收，包括原 `download_missing=false`、空路径、cache、正常音频URL，以及实际会用的加密/情绪策略。当前这些新profile的GPU/母语质量门槛均未通过。本轮不追加训练；保留冻结best100，先补证据再决定是否推广。
