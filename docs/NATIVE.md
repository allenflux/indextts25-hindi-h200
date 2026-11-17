# 原创原生 LoRA 路线

`hi_h200_native` 是独立原创实现，直接调用锁定官方模块，不需要 `itts25ft`。当前支持无padding的batch1、梯度累积、目标语言/文字行、GPT attention Conv1D LoRA，以及text/mel两种head。不要将它的adapter当作社区adapter、ComfyUI或vLLM的兼容格式。

已在CPython3.11/Torch2.8上对真实官方微型GPT做前向/反向、logits对齐、loss下降、完整adapter重载与停止/恢复测试。随机小权重、合成token/音频的这些测试**不证明预训练2.5权重、完整真实音频提取、Hindi音质或GPU可用性**。最终实际验证范围见VALIDATION.md。

## 环境与源码

在独立数据盘目录定义`WORK`和`REPO`，不要覆盖ComfyUI工程或原有模型。所有模型、缓存、认证、语料、特征和run输出放在`WORK`中、仓库外。容器overlay不能作为恢复位置；应确认持久卷及节点亲和/备份策略。

```bash
export WORK=/your/persistent/isolated/workspace
export REPO="$WORK/repo"
cd "$REPO"
export PYTHONPATH="$REPO/src"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$WORK/tmp"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/hf-home"
export HF_HUB_CACHE="$WORK/cache/hub"
export NUMBA_CACHE_DIR="$WORK/cache/numba"
export TIKTOKEN_CACHE_DIR="$WORK/cache/tiktoken"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
```

使用Python3.11创建隔离环境，安装`requirements/native.in`或本项目`[native]`extra。`requirements/native-cpu-macos.lock.txt`记录实际CPU测试环境，不能拿它代替Linux/CUDA wheel锁。GPU环境须独立保存解析版本、wheel来源/哈希、驱动和CUDA记录；不修改现有ComfyUI依赖。

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

用户必须本人处理gated条款。登录只用独立`HF_HOME`，不要发送token或把它写进命令、配置、日志/仓库。若CLI提示没有Git credential helper，HF API登录仍可能已经成功；无需因此启用全局明文credential store。先核对实际账号及读取权限，再选择配置`Hindi`。

受限数据入口`hi_h200_native.hf_data`只针对锁定IndicVoices-R Hindi训练分片，默认元数据预检，实际下载需明确字节上限和本地rights record。选择最多两个训练分片，不遍历/下载整个Hindi子集，不把官方test加入训练；边界和具体CLI见该模块`--help`。它会按明确speaker_id选短clip并转换为24k/mono/PCM16，保留源revision、行标识、原文与转换记录。许可记录是操作依据，不能称为独立核实所有个人声音权利。

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

adapter路径和SHA必须来自本次真实产物；输出目录必须不存在。工具检查原始资产/adapter/受保护参数，按官方stock模型核对完整键和形状，附许可、非背书声明、哈希清单与COMPLETE标记。微型CPU模型的官方加载器重载和合并logits一致性已测试，完整预训练与波形合成仍须另验。

全局替换GPT权重会同时改变所有语言共享的参数，不能承诺旧语言逐位不变。保留原始2.5，Hindi在独立实例/目录加载；若选择一个合并文件处理所有语言，先完成固定zh/en/ja/es/ar回归评测。不要直接覆盖生产模型。
