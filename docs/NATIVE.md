# 原创原生 LoRA 路线

`hi_h200_native` 是独立原创实现，直接调用锁定官方模块，不需要 `itts25ft`。当前支持无padding的batch1、梯度累积、目标语言/文字行、GPT attention Conv1D LoRA，以及text/mel两种head。不要将它的adapter当作社区adapter、ComfyUI或vLLM的兼容格式。

已在CPython3.11/Torch2.8上对真实官方微型GPT做前向/反向、logits对齐、loss下降、完整adapter重载与停止/恢复测试。随机小权重、合成token/音频的这些测试**不证明预训练2.5权重、完整真实音频提取、Hindi音质或GPU可用性**。最终实际验证范围见VALIDATION.md。

## 环境与源码

在独立数据盘目录定义`WORK`和`REPO`，不要覆盖ComfyUI工程或原有模型。模型、缓存、语料、特征和run输出放在`WORK`中、仓库外；认证目录必须另行选择，禁止放入ComfyUI实际served的input/output/temp根内。若`WORK`本身是output持久卷，也不能使用`$WORK/hf-home`保存认证。容器overlay不能作为恢复位置；应确认持久卷及节点亲和/备份策略。

```bash
export WORK=/your/persistent/isolated/workspace
export REPO="$WORK/repo"
cd "$REPO"
export PYTHONPATH="$REPO/src"
export PYTHONDONTWRITEBYTECODE=1
# 只影响当前项目 shell/子进程，不修改现有服务环境；禁止间接 helper 回退到任意 pickle。
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=1
export TMPDIR="$WORK/tmp"
export XDG_CACHE_HOME="$WORK/cache"
# HF_HOME只在需要认证的进程中另行设置，见下方“Hindi 数据与特征”。
export HF_HUB_CACHE="$WORK/cache/hub"
export NUMBA_CACHE_DIR="$WORK/cache/numba"
export TIKTOKEN_CACHE_DIR="$WORK/cache/tiktoken"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
```

先阅读[依赖安全范围](../SECURITY.md)。`requirements/native.in`、`[native]`extra和旧CPU lock保留兼容性复现版本；其中Torch2.8存在可绕过`weights_only=True`的已知checkpoint漏洞，**不是推荐生产环境或未知checkpoint加载基线**。只复现该次测试时才按Python3.11记录创建隔离环境；不能把测试通过当成依赖安全证明。

真实权重任务优先使用经过实际CPU、tiny BF16及恢复gate的Torch>=2.10与匹配torchaudio；已有Pod的Torch2.12.1只读借用不代表旧pin或其他告警已经修复。任何新组合都须单独验证，不盲改全局环境。`requirements/native-cpu-macos.lock.txt`不能代替Linux/CUDA wheel锁；另存实际解析版本、wheel来源/哈希、驱动和CUDA记录。只读取固定官方哈希资产或本次自产且绑定身份的checkpoint；禁止不安全pickle回退。

官方代码单独checkout完整SHA `d9e41aac89fd00b3d71497fddb287b7f24613712`，保留其LICENSE和LICENSE_ZH.txt。原生加载器检查git身份和模块来源，不接受换名目录或任意remote code。以下不调用官方通用下载助手，避免下载不需要的推理后端/其他许可资产。

## 先规划资产，再显式下载

```bash
python -m hi_h200_native.assets --stage train
python -m hi_h200_native.assets --stage extract
```

默认只读公开元数据，固定官方HF端点、完整revision和精确文件允许列表，使用`token=False`。`train`选择5文件、**3,260,524,913 bytes**，用于已有特征时的GPT训练；`extract`选择13文件、**6,217,934,223 bytes**，增加codec、统计、w2v-BERT和CAMPPlus。它们都不是完整推理模型包；尚未包含波形合成验收所需全部后端。小文件API的Git blob SHA1不冒充SHA256，实际下载后另算SHA256。

```bash
python -m hi_h200_native.assets --stage extract   --destination "$WORK/assets/official-extract" --cache-dir "$WORK/cache/public-models"   --download --max-bytes 6300000000 --reserve-bytes 2147483648
```

目标目录必须全新，已存在就检查manifest而不是覆盖。工具先检查两份存储空间和保留量，只下载所列文件并核验大小/哈希/2.5配置，最后原子发布`model-manifest.json`。若发布后父目录fsync报错，目录可能已经存在，必须检查结果再决定后续操作，不能直接重试覆盖。模型文件、辅助模型卡和许可均保留。模型/衍生adapter的授权不变为MIT。

## Hindi 数据与特征

用户必须本人处理gated条款。登录只用独立`HF_HOME`，不要发送token或把它写进命令、配置、日志/仓库。认证前必须确认目录在ComfyUI实际served的input/output/temp之外；同UID的ComfyUI进程仍可读取0700/0600文件，这些权限不能充当HTTP文件路由或进程隔离。可使用专属临时认证目录，但先确认系统`/tmp`也不属于任何served根；缓存仍可由`HF_HUB_CACHE`留在`WORK`。迁移后旧served路径不得保留认证副本或指向新目录的符号链接。

只在需要Hub认证的项目进程中设置已确认的`AUTH_HOME`，并在导入Hub库前生效：

```bash
: "${AUTH_HOME:?Set the dedicated authentication directory outside all served roots}"
export HF_HOME="$AUTH_HOME"
```

临时认证目录没有持久性保证；容器被替换或临时盘丢失后，需要用户本人重新登录，不从聊天、日志或checkpoint恢复token。完整模型、特征和checkpoint已在持久盘时，离线训练及恢复不需要token；保持离线标志，不因认证丢失触发下载或不安全加载。若CLI提示没有Git credential helper，HF API登录仍可能已经成功；无需因此启用全局明文credential store。先核对实际账号及读取权限，再选择配置`Hindi`。

受限数据入口`hi_h200_native.hf_data`只针对锁定IndicVoices-R Hindi训练分片，默认元数据预检，实际下载需明确字节上限和本地rights record。选择最多两个固定训练分片，连同审计文档的字节预算最多1,000,000,000 bytes，不遍历/下载整个Hindi子集，不把官方test加入训练；边界和具体CLI见该模块`--help`。它会按明确speaker_id选短clip并转换为24k/mono/PCM16，保留源revision、行标识、原文与转换记录。许可记录是操作依据，不能称为独立核实所有个人声音权利。

候选publisher speaker上限默认仍为32；可显式设置`--max-candidate-speakers 128`，硬上限128，且不能小于目标speaker数。目标speaker仍为6–32，每位最多16条候选，最终每位至少4条完整clip；全局最多检查4096行。该开关只扩大有真实publisher ID的候选范围，不保证得到足够合格speaker；既有账号/权限/rights、固定分片与哈希、长度、Unicode/ASCII文字策略、音频及文本去重校验均不变，选定上限写入plan、import与下载审计。

显式128时，仅未完成speaker的PCM暂存理论上就可约1.84GB（128×15×20秒×24k×2字节），另有WAV容器、Parquet读取与其他缓存。运行前必须核对宿主/cgroup可用内存及余量；该数字不是总内存上界或资源充足承诺，默认32没有改变。

实际导入时，即使使用`--reuse-downloads`，CLI仍先检查既有账号、读取权限与Hub固定revision元数据，再复验本地分片完整哈希；它不是离线导入入口。Python级`import_parquets(..., max_candidate_speakers=128)`可读取调用者已经合法取得并核验来源、revision及哈希的本地固定分片bytes，不调用Hub，也不能证明认证已通过。不得将这个直接本地调用写成CLI认证成功，或用它代替用户本人处理访问条件与使用权利。

从一次真实训练的既存完整checkpoint选择候选adapter，见[从完整检查点选择候选 Adapter](CHECKPOINT_SELECTION.md)。该流程针对真实完整checkpoint，不是mock；原始源码、训练计划及其绑定身份保持不可变，不重新resume或改写历史记录。本次pilot的真实第100步完整checkpoint已在CPU转换为标准adapter，随后已实际完成完整GPT合并、H200单条feature的FP32 fresh-stock parity、真实WAV与配对ASR；详细结果及范围见[VALIDATION.md](../VALIDATION.md#64条pilot的500步训练与候选选择)。人工Hindi审听和旧语言验收仍未完成；更大候选池尚未导入或训练。

导入器若已生成`prepared`并完成校验，直接复用；不要重复运行prepare覆盖目录。自有raw JSONL可按以下布局使用CPU核心校验：

```bash
python -m hi_h200 prepare "$WORK/data/hindi/input.jsonl" --audio-root "$WORK/data/hindi" --output "$WORK/data/hindi/prepared"
python -m hi_h200 validate "$WORK/data/hindi/prepared" --audio-root "$WORK/data/hindi"
```

至少三位speaker且每人两条只是格式最低要求。小样本计划选择六位speaker、每人四条，从而产生16条train、4条val、4条内部test；不把它称为正式泛化评测。正式评测仍需更大held-out speaker集合。

将`configs/native-dataset.example.json`各复制一份train/val/test，改成真实路径及对应split，`model_manifest`指向资产助手输出；feature_root必须全新，manifest/provenance都放在其内。

```bash
python -m hi_h200_native.extract --config "$WORK/configs/train.dataset.json"
python -m hi_h200_native.extract --config "$WORK/configs/val.dataset.json"
python -m hi_h200_native.extract --config "$WORK/configs/test.dataset.json"
```

默认CPU且不自动下载。实际调用冻结的w2v-BERT(layer17+统计归一)、EnhancedCodec、CAMPPlus与GPT emotion encoder；完整严格受限权重加载，不保留随机初始化缺口。配方明确为24k直接重采样16k、参考截到15s、目标不截断，与官方推理的librosa重采样分支数值不等价，所以parity标志保持false。原生tokenizer使用NFC/空白策略和`<|hi|> `前缀；0/1保留token不被悄悄删除，出现时停止人工审查。

每个特征绑定源PCM/文字、七份prepared文件、词表、模型、源码、提取器recipe与源码SHA；加载时复算全部token、数组类型/形状/范围/有限值并拒绝陈旧/外来缓存。清单声明本身不能取代实际提取验收。

## 真实更新与中断恢复

复制`configs/native-train.example.json`，指向train/val数据配置。默认CPU fp32；CUDA需要一个明确可见GPU、`cuda:0`、BF16以及额外`--allow-cuda`。这些开关不证明共享资源安全，不申请或抢占GPU。H200起点仍是实验参数，不承诺吞吐/显存/质量。

```bash
# 用同一完整更新计划在第2次optimizer更新处协作停下。
python -m hi_h200_native.train --config "$WORK/configs/native.json" --stop-after-updates 2
# 选择实际生成的明确checkpoint和.sha256.json，不使用auto或缺失路径。
python -m hi_h200_native.train --config "$WORK/configs/native.json" --resume "$WORK/runs/native/checkpoints/step00000002.pt"
```

检查点文件名以实际输出为准。原生引擎保存完整可训练参数（包括两种head）、AdamW、scheduler、Python/NumPy/Torch CPU/CUDA RNG、epoch/permutation/cursor、配置/源码/环境/模型/数据身份。STOP/SIGINT/SIGTERM只设标志，在optimizer边界保存；不会在信号处理器里写文件。文件fsync后原子发布，resume缺失/哈希或身份不一致就失败，最终cursor保持真实值。SIGKILL/OOM/主机掉电仍只能依赖此前完整checkpoint，不能保证新保存。

先验收小样本过拟合，再固定计划执行STOP→resume→继续更新的真实往返，再做100次optimizer更新的资源/吞吐smoke。`updates.jsonl`区分compute/wall时间，计数实际micro_examples，CUDA采样是本进程allocator峰值，不能冒充整卡显存或纯kernel性能。继续使用Hindi/旧语言/导出重载评测门槛；源码小模型恢复通过不等于真实H200任务已验收。

## 独立保留的社区兼容路线

旧`check_backend.py`、`render_backend_config.py`及[H200参考指南](H200.md)仅用于外部社区接口，仍要求其明确许可。它们的配置、features、adapter和checkpoint都不能直接冒充原生格式。新原生路线不受该社区代码许可阻塞，但仍须遵守官方模型、辅助资产和数据许可。

## 导出可供官方加载的 GPT 权重

原生export会在新目录生成`gpt.pth`，容器为官方接受的`{"model": state_dict}`，合并attention LoRA并保留两种head、语言/文字行和所有官方参数/缓冲，不含optimizer或原始语料。

```bash
python -m hi_h200_native.export \
  --official-repo "$WORK/external/index-tts" --model-dir "$WORK/assets/official-extract" \
  --model-manifest "$WORK/assets/official-extract/model-manifest.json" \
  --adapter "$WORK/runs/native/adapters/ACTUAL_ADAPTER.pt" --adapter-sha256 ACTUAL_SHA256 \
  --output "$WORK/exports/hindi-gpt"
```

adapter路径和SHA必须来自本次真实产物；输出目录必须不存在。工具检查原始资产/adapter/受保护参数，按官方stock模型核对完整键和形状，附许可、非背书声明、哈希清单与COMPLETE标记。微型CPU模型的官方加载器重载和合并logits一致性已测试。每个新导出仍须独立验收；本次已完成范围见[VALIDATION.md](../VALIDATION.md)。

全局替换GPT权重会同时改变所有语言共享的参数，不能承诺旧语言逐位不变。保留原始2.5，Hindi在独立实例/目录加载；若选择一个合并文件处理所有语言，先完成固定zh/en/ja/es/ar回归评测。不要直接覆盖生产模型。
