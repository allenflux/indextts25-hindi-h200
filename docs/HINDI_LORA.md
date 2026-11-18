# Hindi 专用纯 attention LoRA：新推荐方向

更新日期：2026-11-18。用户已取消直接替换完整 GPT / 单候选全局默认方案，要求**重新训练纯 LoRA，并同步 GitHub 与 Hugging Face**。当前采取：保留已核官方原始模型文件；Hindi 使用独立的原模型 + 新 attention-only adapter 实例；旧语言使用从未加载该 adapter 的原模型实例。新代码正在实现、CPU 验证与审查，**尚未完成这条路线的新 GPU 训练，没有新的纯 LoRA adapter、试听或质量通过结论，也没有新的 HF 权重发布结果**。

## 原理与冻结范围

GPT 将文字与语言条件映射为离散语音 semantic codes，后端 codec、CFM/DiT 与 BigVGAN 再生成波形。新实验只学习 GPT attention 的低秩增量；固定模型、词表、codebook 和声学后端，限制变化来源。Hindi 行17与 `<|hi|>` 已有接口槽位，但槽位存在不证明官方训练过 Hindi，也不保证冻结全部 embedding 后能取得足够质量。[固定架构与历史方法](MODEL_RATIONALE.md)

`training_profile="attention_lora_only_v1"` 的可训练集合严格限定为24层的 `c_attn` / `c_proj` 的 `lora_A`、`lora_B`，共48个模块、96个 A/B tensor。**全部语言和文字 embedding（包括 Hindi 第17行）、text_head、mel_head、其余 GPT 参数、speaker/emotion 前端及声学后端均冻结。** 不训练基础 attention weight/bias，也不把 head/embedding 权重塞进 adapter。

对于 attention 输入，LoRA 分支为 `base(x) + (alpha/r) × B(A(x))`；新推理路线在独立实例保留该运行时分支，主交付物不是合并后的 `gpt.pth`。示例配置为 rank8、alpha16、dropout0；这些是实验起点，不是官方 Hindi 配方或质量保证。理论参数量或文件小不能推导语音自然度、可懂度或训练 GPU 小时。

训练仍使用真实配对录音提取的冻结特征、batch1 与梯度累积，目标为 `CE_semantic_code + 0.2 × CE_text`。辅助 text head 虽冻结，损失仍可通过固定计算图给 A/B 传梯度；“冻结 head”不等于删除辅助 loss。只有 attention A/B 参与优化；若出现 head、embedding 或其他基础参数可训练，必须失败停止。

## 旧 best100 不能变成新纯 LoRA

| 项目 | 旧 best100 / legacy profile | 新 attention-only 路线 |
|---|---|---|
| 参数范围 | attention LoRA + Hindi/所见 text 行 + 两个完整 head | 仅 attention A/B；所有 head/embedding 冻结 |
| profile | `legacy_rows_heads_v1` | `attention_lora_only_v1` |
| 已有发布 | HF `117c7922e4dcedc3f9885f3f6efa5be8f54c7919` 为完整合并 GPT | 尚无新训练权重或新 HF 发布结果 |
| 是否可给旧包改名 | 不可 | 必须从已核原始 base 重新训练、单独保存与验收 |
| 旧语言处理 | 历史候选使用共享更新，有限 stress 不证明无损 | 使用从未加载 adapter 的独立原模型实例 |

旧 best100 的54个变化 tensor 包含48个共享 attention weight、4个 head tensor 和2个 embedding 矩阵。关闭、删掉或清零旧 adapter 的 A/B，**不能撤回 head/embedding 的变化**；已合并完整 GPT 也没有一个可直接关闭的 A/B 开关。不能拿旧 checkpoint、adapter 或完整 GPT 继续训练后改名为新纯 LoRA。旧训练、ASR、WAV 和 T8 三轮证据仍有效，但只属于旧路线。[旧参数审计与已撤回方案](SINGLE_MODEL_UPGRADE.md) · [历史 RTX 5090 验收](RTX5090_SWITCHING.md)

## 产物与身份 schema

新 profile 沿用原生 adapter 外层 schema，区别必须通过严格的 `adapter_spec` 与完整 tensor inventory 验证，不能仅凭文件名判断。当前代码约定的顶层字段如下；这是接口定义，**不是已经产生的新 adapter 文件**。

| 字段 | 要求 |
|---|---|
| `schema_version` | `1` |
| `kind` | `hi_h200_native_adapter` |
| `step` | 实际完成的 optimizer update，加载时须大于0；不能写假训练步数 |
| `base_identity` | 与实际训练所用官方源码、原始模型及其已核资产身份一致 |
| `adapter_spec` | `format="hi_h200_native_attention_lora_only_v1"`，`training_profile="attention_lora_only_v1"`，`heads=[]`、`text_rows=[]`、`frozen_embeddings=true`；另保留真实 rank/alpha/dropout、targets、model_dim、loss 等设置 |
| `trainable` | 仅 `base.gpt.h.{0..23}.attn.{c_attn,c_proj}.{lora_A,lora_B}`；生产结构严格48模块 / 96个 tensor，形状、dtype、有限值必须匹配 |

`adapter_spec.hindi_row=17` 只记录固定语言条件，**不表示该行可训练**。原生 loader 必须严格拒绝 legacy row/head adapter、额外 tensor、错误 base、错误 profile、缺项与非有限值；不使用宽松加载来“兼容”旧包。训练 checkpoint 含 optimizer/RNG 等私有恢复状态，不是发布 adapter；新旧 profile 禁止混用 resume。每次 adapter 发布记录独立 SHA256；基础 GPT 文件保持原始 SHA，不复制或覆盖为候选 GPT。

## 当前入口与进度

| 环节 | 当前状态 |
|---|---|
| 训练 profile 与冻结检查 | 原创实现已落代码，CPU 验证与审查中；不等于完整预训练模型 GPU 验证 |
| 独立配置 | [native-lora-only.example.json](../configs/native-lora-only.example.json)；默认 CPU/fp32、100 updates、累积4、lr3e-4、save10、val25，仅作实验起点 |
| 新一轮真实训练 / STOP→恢复 | 尚未完成；不能用旧 best100 的20→500恢复代替 |
| 新纯 LoRA adapter / 官方模型真实推理 | 尚无训练产物；新独立推理入口开发与 CPU 检查中，未 GPU 验收 |
| Hindi 母语、长句和旧语言隔离验收 | 未通过；不能提前给出质量或全局无损结论 |
| GitHub / HF 新产物同步 | 已有用户发布意图；待真实产物、许可/隐私审查与实际上传结果，当前不宣称完成 |

训练 CLI 沿用 `python -m hi_h200_native.train --config PATH`，它是**真实训练命令，不是 metadata dry-run**。可先看 `--help`；准备好已核原始 base、合法数据/特征、fresh 输出及完整计划后，才运行配置。示例默认 CPU/fp32 也会实际加载和训练，不是廉价 tiny 测试；CUDA 仍须明确设置 `loop.device="cuda:0"`、`loop.precision="bf16"` 并追加 `--allow-cuda`。没有新增下载器、Pod 分配或后台调度。

```bash
PYTHONPATH=src python -m hi_h200_native.train --help
# 以下会真实开始训练；本次文档更新没有执行它。
PYTHONPATH=src python -m hi_h200_native.train --config "$PURE_TRAIN_CONFIG"
```

新独立入口是 `hi_h200_native.adapter_synth`，当前仅接受 `--language hi`。默认只检查本地路径、资产/adapter SHA 和元数据，**不导入 Torch 或执行模型；tensor/profile 校验仍待 execute 阶段**。必须使用真实原始模型、实际训练 base manifest 与真实新 adapter，不能把旧 best100 填进去。当前 CLI 形状如下，尚无新 adapter 可据此完成实际生成：

```bash
PYTHONPATH=src python -m hi_h200_native.adapter_synth \
  --workspace-root "$WORK" --official-repo "$OFFICIAL_REPO" \
  --model-dir "$ORIGINAL_MODEL" --model-manifest "$SYNTH_MANIFEST" \
  --training-base-manifest "$TRAINING_BASE_MANIFEST" \
  --adapter "$PURE_ADAPTER" --adapter-sha256 "$PURE_ADAPTER_SHA256" \
  --prompt "$PROMPT_PCM16" --output-dir "$FRESH_OUTPUT" \
  --language hi --text 'नमस्ते।'
```

所有官方源码、原始模型、两份 manifest、adapter 和 prompt 都须位于明确的 `WORK` 内；synthesis manifest 须在原始模型目录内，adapter 须独立于原始模型和源码目录。该新入口没有可放宽此范围的 `--readonly-dependency-root` 参数。真实 `--execute` 还要求 `--allow-cuda`、明确 `--gpu-id` 与 GPU/RAM/利用率门槛、fresh 输出、固定官方资产和离线保护。实际加载应在私有 Hindi 实例先以 CPU FP32 严格校验 A/B，再构造官方推理 wrapper；**不写临时 merged GPT，不改原文件，也不接受现有旧语言服务对象**。接口仍在验证中，不能据默认 metadata 成功宣称已加载或生成。该入口是独立 Python 推理，**不是 T8 runtime adapter 支持，也不保证普通 ComfyUI LoRA 节点、PEFT、社区 adapter、DeepSpeed 或 vLLM 兼容**。

## 路由、关闭与恢复

- 旧语言请求始终走从未加载该 adapter 的原模型实例；保持原始文件身份、原路径与服务配置，不对同一个共享对象反复挂载/卸载。
- Hindi 请求只进入专用实例，严格核对原始 base 与 adapter；加载失败停止该私有任务，不污染旧语言实例，不静默使用旧 best100 或其他候选兜底。
- “关闭 Hindi adapter”在当前方案中指销毁私有 Hindi 进程；不承诺热卸载、同进程缓存回退或把完整候选转回原版。需要恢复旧语言服务时继续使用已核原实例。
- 进程隔离不等于无资源争用。新入口加载阶段会暂存原 TTS 构造对象和额外 CPU FP32 GPT；门槛是当时观察，不是资源预留，adapter 小也不能据此推断总 RAM/显存很小。两实例仍可能占同一卡；必须检查实际 GPU、RAM、持久存储和原服务队列，保留有界任务与仅清理自建进程的限制，不改变现有 ComfyUI 服务来抢卡。

attention 仍是共享结构；如果把新 adapter 加到其他语言请求中，也可能改变输出。“旧语言实例不加载”避免了 adapter 参数污染，但不能取代对实际路由、依赖、缓存和运行设置的检查，更不能从字节保护推出所有声音必然完全相同。

## 必须重新通过的 gate

1. **配置与可训练参数：** 从固定原始 GPT 开始，生产结构48模块/96 tensor；逐项确认全部 head/embedding、attention base、其他基础权重及声学资产未变；新旧 profile 与 checkpoint 互相误用必须失败。
2. **真实训练与恢复：** 先小批次，检查真实前向/反向与非零 A/B 更新，再做完整 checkpoint 的 STOP→新进程 resume，核 optimizer/scheduler/RNG 和计划身份；至少有界100 update smoke 与小样本过拟合检查。历史 mock/tiny 或旧 profile 恢复不算新路线通过。
3. **真实推理与关闭：** 原始 base + 新 A/B 在独立官方推理实例实际生成，记录 adapter/base SHA、实际 dtype/device、完整 WAV/PCM 与失败记录；分别核对不加载 adapter 的原实例，明确路由和生命周期。不拿随机初始化 A/B 的测试声称 Hindi 微调成功。
4. **语音验收：** speaker-disjoint VAL 选 checkpoint，冻结后再 test，不按 ASR 或试听挑好输出；Hindi 母语审听、ASR真人参考校准、长句/结束检查和合法 native-prompt 旧语对照。已有五语单例 Hindi-prompt stress 不能代替本轮验收；纯 LoRA 也不保证 Hindi 效果更好。
5. **发布审查：** 只白名单导出新 adapter、来源/冻结范围、模型卡、许可/归属、代码 commit 与校验清单。先逐项核对隐私与权利，再记录实际 Git/HF revision、远端 SHA 和公开状态；通过项、失败项、未验证项分别列明。

## GitHub / HF 同步与许可边界

GitHub 保存原创源码、配置、schema、模型卡及公开验证摘要；HF 保存**单独命名与版本登记的新纯 LoRA adapter**、固定 base 的依赖说明和完整适用许可。adapter 文件名、训练 profile、step、base SHA、adapter SHA 与对应代码 commit 必须相互一致。已有 `IndexTTS-2.5-Hindi-best100-experimental@117c7922e4dcedc3f9885f3f6efa5be8f54c7919` 保留原样并标为历史完整 GPT，不能通过覆盖、删 head 或重命名伪造新实验。当前没有新的 HF 仓库/权重发布结果可报告。

工具 MIT 不替代官方 Bilibili 模型条款、数据 CC-BY-4.0 或其他辅助资产许可；纯 adapter 也不自动豁免衍生分发义务。保留完整中英文协议、归属、变更和非背书说明，不代用户接受 gated 条款，不发布原参考、训练语料、私人转写、speaker/clip 身份、特征、恢复 checkpoint、日志或凭据。此前明确获准公开的两份四节点样本维持原范围，不自动授权本轮新生成音频。[分发说明](DISTRIBUTION.md) · [数据来源](DATA_SOURCES.md)

历史资料：[旧完整 GPT 安装](INSTALL_EXPERIMENTAL.md)、[已撤回的直接替换方案](SINGLE_MODEL_UPGRADE.md)、[旧 RTX 5090 三轮](RTX5090_SWITCHING.md)。这些链接只保留可追溯证据，不改变当前原模型保留、Hindi adapter 独立的新方向。
