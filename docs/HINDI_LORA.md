# Hindi 专用纯 attention LoRA：新推荐方向

更新日期：2026-11-18。用户已取消直接替换完整 GPT / 单候选全局默认方案，要求**重新训练纯 LoRA，并同步 GitHub 与 Hugging Face**。当前采取：保留已核官方原始模型文件；Hindi 使用独立的原模型 + 新 attention-only adapter 实例；旧语言使用从未加载该 adapter 的原模型实例。**新纯 LoRA 的 H200/BF16 run002 已真实完成第2步停止后跨进程恢复至100 updates，并产生 step100 adapter。** 完整文件大小、SHA 与受限 CPU tensor 实物审计均已核对；固定4例 TRAIN 的 before/after 机制 gate 已真实通过；冻结pilot候选已完成2/2句未合并native真实WAV及独立file/PCM回读；语音质量验收仍未完成。另一个从原始 base 独立开始的 pilot48 已真实完成 fresh2→新进程 resume100，按预声明的四个 VAL8 均值选出并冻结 step100；拟发布候选 SHA 为 `15c080715b3b50bf38b33a70f196a9c9e987446e0a6bb87b2e0c8b7835f57dbd`，不能与小集机制候选混用。独立[pure HF仓库](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-LoRA-experimental)已实际创建并回读Owner fluxa0223/Public，ZIP权重包已在固定revision `88e6297852699eb6dfbf6d4677446d37afcacf5f` 上传并经匿名完整下载核验；26文件在ZIP内。不能把发布包完整性通过写成质量或T8兼容通过。

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
| 已有产物 / 发布 | HF `117c7922e4dcedc3f9885f3f6efa5be8f54c7919` 为历史完整合并 GPT | pilot48/15c冻结候选已作为26文件ZIP上传HF固定88e62978 revision并匿名核验；run002/902d机制候选不是发布权重 |
| 是否可给旧包改名 | 不可 | 必须从已核原始 base 重新训练、单独保存与验收 |
| 旧语言处理 | 历史候选使用共享更新，有限 stress 不证明无损 | 使用从未加载 adapter 的独立原模型实例 |

旧 best100 的54个变化 tensor 包含48个共享 attention weight、4个 head tensor 和2个 embedding 矩阵。关闭、删掉或清零旧 adapter 的 A/B，**不能撤回 head/embedding 的变化**；已合并完整 GPT 也没有一个可直接关闭的 A/B 开关。不能拿旧 checkpoint、adapter 或完整 GPT 继续训练后改名为新纯 LoRA。旧训练、ASR、WAV 和 T8 三轮证据仍有效，但只属于旧路线。[旧参数审计与已撤回方案](SINGLE_MODEL_UPGRADE.md) · [历史 RTX 5090 验收](RTX5090_SWITCHING.md)

## 产物与身份 schema

新 profile 沿用原生 adapter 外层 schema，区别必须通过严格的 `adapter_spec` 与完整 tensor inventory 验证，不能仅凭文件名判断。当前代码约定的顶层字段如下。run002 的 adapter 已独立进行受限 CPU 实读：代码要求48模块/96 tensor，实物也确认为96个 A/B tensor、1,474,560参数。接口定义与实物审计是两项证据；它们都不代表语音质量通过。

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
| 纯 attention profile | 已用于完整预训练模型 H200/BF16 的 run002；所有 head/embedding/基础参数保持冻结的 profile 约束不变；adapter 实物 inventory 已独立 CPU 核对 |
| 仓库示例配置 | [native-lora-only.example.json](../configs/native-lora-only.example.json) 仍默认 CPU/fp32；示例参数不是本轮 GPU 私有计划的原样记录 |
| 新一轮真实训练 / STOP→恢复 | run002 第2步停止→新进程恢复至100，两个控制器均退出0、验证步数正确、自有进程均退出；总共100 updates，不是102 |
| 新纯 LoRA adapter | pilot48 冻结发布候选 step100，5,931,907 bytes / SHA `15c080715b3b50bf38b33a70f196a9c9e987446e0a6bb87b2e0c8b7835f57dbd`，完整文件 hash 已核；run002 的 `902d…cff1` 仅为另一个机制候选，其实物审计不能移记给新文件 |
| 固定 TRAIN 机制 gate | 已真实完成固定4例同例 before/after，mean ratio 0.7029568847≤0.9，模块 mechanism_gate_passed=true、frozen_state_unchanged=true；不是语音质量 gate |
| 独立推理 / Hindi 质量 / T8 | source31fd专用入口已在H200完成2/2句未合并native WAV，两个完整file/PCM均独立回读一致、guard exit0/自有子进程退出。未做人工/ASR/独立TEST/新5090/T8 pure验收，不代表原工作流兼容 |
| 独立 pilot48 | 原始 base 另起，fresh2 与新进程 resume100 均退出0、verified2/100、own exited=true；严格完成1..100，按预声明 VAL8 最低选100，candidate_frozen=true，TEST未用于选模 |
| GitHub / HF 同步 | 独立[pure权重固定revision](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-LoRA-experimental/tree/88e6297852699eb6dfbf6d4677446d37afcacf5f)已上传ZIP并匿名完整核ZIP/manifest/adapter SHA及26项CRC；网页封面09bc09ae与包内冻结卡a53分开，详见[发布包回执](acceptance/pure-lora-public-package-20261118.json) |

### pilot48：真实2→100恢复完成，已冻结step100

这是从原始 base 独立开始的纯 attention LoRA pilot，不从 run002 的过拟合 adapter 继续。训练源码仍为 `ed19c4a511be4baafe87b22f74adf6b192a0a55a`。复用旧 pilot64 的已授权、已提取且按 speaker 隔离的数据，没有在此步骤增加下载：

| 分组 | clips / speakers | 完整音频时长 |
|---|---|---|
| TRAIN | 48 / 12 | 491.029 s |
| VAL | 8 / 2 | 82.950 s |
| TEST | 8 / 2 | 69.924 s |

实际计划为 H200/BF16、rank8/alpha16/dropout0、累积4、lr1e-4、warmup10、max100、seed260618、save/val25；所有 head/embedding 与基础参数冻结。启动前已保存 `predeclared-selection.json`：只在25/50/75/100中按 **VAL8 mean loss 最低**选 checkpoint，完全相同则选较早 step；TEST/ASR 不参与选模。

| 实际阶段 | 完成与退出 | 控制器 wall |
|---|---|---|
| fresh2 | exit0 / verified2 / own process exited=true | 40.20832155458629 s |
| 新进程 resume100 | exit0 / verified100 / own process exited=true | 110.89273961819708 s |

optimizer updates 严格为1..100，共100次；不把 run002 的100次加进来成为200。两段 wall 包含各自加载、保存和验证，不是纯GPU训练时间或吞吐。pilot 训练进程 Torch peak CUDA allocated 为4,043,693,056 bytes、reserved为4,339,007,488 bytes，不是整卡总占用。

| 预声明 VAL step | 同一8例 VAL mean loss |
|---|---:|
| 25 | 8.110908448696136 |
| 50 | 7.306136906147003 |
| 75 | 7.053911030292511 |
| 100 | 6.989858150482178 |

按启动前规则选中100，已写入 `FROZEN_SELECTION`，`test_used_for_selection=false`、`candidate_frozen=true`；后续 TEST/ASR/试听不用于改选。新发布候选 actual adapter 为 **5,931,907 bytes**，SHA256 **`15c080715b3b50bf38b33a70f196a9c9e987446e0a6bb87b2e0c8b7835f57dbd`**，操作者完整字节读取与 trainer 回执一致；这不是下节 run002 的 `902d…cff1` 机制 adapter。该候选的首次真实推理 callback 已通过精确16项 base identity 与96个 A/B 的受限加载，构造48个 LoRA 的官方真实 wrapper；2/2句WAV已完成并独立回读，详见下节。HF ZIP已实际发布并核验；母语/ASR/独立TEST、新5090及T8 pure仍未通过。

run002 和该 pilot 的恢复均是实际跨进程继续训练，**没有与不中断连续运行逐位比较，不能宣称 bitwise 连续等价**。

### 冻结候选2/2句未合并native真实合成（功能验证）

使用评估源码 `31fd9fe11d84d5f328afa36c49f363dbe6f9cf24` 的独立 `adapter_synth`，在H200实际加载冻结pilot48候选 `15c…57dbd`。原模型磁盘GPT仍为 `43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85`，没有替换或写出临时mergedGPT。callback严格核对精确16项training identity与96个A/B后构造48模块的官方真实推理wrapper；报告 `adapter_loaded_without_merge=true`、`adapter_profile_verified=true`、`unmerged=true`、KV cache=true、`original_language_instance_touched=false`。

第一句是此前预固定的公开工程短句：**नमस्ते। यह हिन्दी भाषा का एक छोटा परीक्षण है।** 它不是独立held-out TEST质量样本。参考为固定官方Space `IndexTeam/IndexTTS-2-Demo@b01840e8e4fd9753743a6d0466cd73ae1d634a68` 的voice01，不是训练私有原音频：原FLOAT文件478,050 bytes、SHA `e33e6ee0107a1dd58e1d66dd90c13df3d55a8683047cc3d7ea206dad84ed3fc8`，仅转PCM16为234,160 bytes、SHA `fd23ddac68f2db3710481c7cbfc1a50a117d9c38bc5c3ee51c33444f0a9b019a`；没有crop或resample。

| 第一例实际输出 | 结果 |
|---|---|
| 格式 / 帧数 / 时长 | mono PCM16 / 22,050Hz / 84,480 frames / 3.8312925170068026 s |
| WAV bytes / SHA256 | 169,038 / `643b7e5f032039b353e8c935de85f7ce830e2bd29e6195dfb0d3bfa5ebd78810` |
| PCM SHA256 | `eaaccc3a8af22e0c48ce15c778162380999416739468538dfeeb804c3240a243` |
| 独立回读 | 完整文件与PCM哈希重新读取一致 |
| 生成计时 / guard wall | 5.084137551486492 s / 102.56857392378151 s |
| 控制器 | exit0 / own child exited=true |

生成计时与包含hash/加载等开销的guard wall不是同一种性能指标；不与旧T8四节点或三轮耗时混算。两份新WAV仅在受控环境保留，**未获额外公开授权，不上传音频或公开试听链接**。

这次使用greedy `do_sample=false`、beams1、repetition penalty1、text_normalization=false、max_mel256、max_text120、seed1234；旧图是sample=true、beams3、penalty10、normalization=true、max_mel1504。**这两句功能成功不是原工作流参数兼容、新5090或T8 pure集成通过。** 未人工审听、未ASR、未独立TEST，也未证明整句无截断。

第二句预固定为 **नमस्ते मित्र।**，已由同一冻结step100/15c候选实际生成，报告 `synthesis_executed=true`、`adapter_loaded_without_merge=true`、`adapter_profile_verified=true`、48模块wrapper、`unmerged=true`、KV=true：

| 第二例实际输出 | 结果 |
|---|---|
| 格式 / 帧数 / 时长 | mono PCM16 / 22,050Hz / 42,240 frames / 1.9156462585034013 s |
| WAV bytes / SHA256 | 84,558 / `562d2a8632777cc001b0d629ec56877a10abad35b4e418d35cd5b57be05542cd` |
| PCM SHA256 | `0d84b07822247419df5947ba7851064281207f3877b380c11f6424c863b591d0` |
| 独立回读 | 完整文件与PCM哈希重新读取一致 |
| 生成计时 / guard wall | 3.803127806633711 s / 98.87775631062686 s |
| 控制器 | exit0 / own child exited=true |

因此仅记 **2/2 native未合并工程句生成功能通过**，不记独立质量、原工作流、5090或T8 pure通过。两次合成结束后再次完整读取原GPT，SHA仍为43a8…原值；原ComfyUI HTTP200、队列0/0。自建训练与合成子进程的退出由控制器回执确认，不将尚未完成的额外独立PID存在性检查写成已通过。本轮未替换原模型、改动生产ComfyUI或安装系统依赖；健康快照不是后续共享资源保证。

实际运行时为Python3.12.3、Torch2.12.1、Transformers4.52.1、torchaudio2.11；部分版本与固定官方依赖约束不同。这是该次成功环境记录，**不是通用依赖锁或安全性保证**。已有依赖通过显式共享overlay与WORK内synth-overlay提供，missing=0，未安装依赖或修改系统；overlay可用也不等于任意机器下载即用。

### run002 的真实训练与恢复记录

训练源码固定 [ed19c4a511be4baafe87b22f74adf6b192a0a55a](https://github.com/allenflux/indextts25-hindi-h200/tree/ed19c4a511be4baafe87b22f74adf6b192a0a55a)。复用此前合法且按 speaker 隔离的 Hindi 小集，没有在此记录中扩集：

| 分组 | clips / speakers | 完整音频时长 |
|---|---|---|
| TRAIN | 16 / 4 | 82.007333 s |
| VAL | 4 / 1 | 25.664125 s |
| TEST | 4 / 1 | 32.419042 s |

本轮实际为 H200 / BF16、rank8、alpha16、dropout0、lr3e-4、warmup5、梯度累积1、计划 max100、seed260618、每10步保存、每25步验证。基础 GPT、全部 embedding/head 与声学模块冻结；不是示例默认 CPU 配置，也不沿用旧 best100 的行/head 训练状态。

| 阶段 | 实际结果 | 控制器 wall |
|---|---|---|
| fresh 至第2步停止 | exit0 / verified step2 / own process exited=true | 38.068683 s |
| 新进程 resume 至第100步 | exit0 / verified step100 / own process exited=true | 54.358446 s |

两段计时包含各自加载、保存与验证开销，不是纯训练吞吐或 GPU 小时承诺。训练进程 Torch peak CUDA allocated 为 **3,913,695,232 bytes**，reserved 为 **4,091,543,552 bytes**；这是对应进程的 Torch 指标，不是整卡总占用。末尾单个 TRAIN batch loss 为 `6.780472755432129`，VAL 4例均值为 `6.748236656188965`；不同 case、不同聚合口径的值不能拿来断言首尾下降，更不是固定 TRAIN 过拟合 gate 的替代品。

真实 step100 adapter 文件为 **5,931,907 bytes**，SHA256 **`902d0729e9b6ac340bfd6e0282294865b184dcc73a8a90dd711df310cdb6cff1`**；操作者对 actual file 的完整读取哈希与 trainer 回执一致。另以 `weights_only=True` 在 CPU 受限读取 actual adapter，读取前后完整 SHA 一致：24层 `c_attn` / `c_proj` 的 A/B 集合精确96个 tensor、1,474,560参数，全部 FP32 且 finite；step100、`heads=[]`、`text_rows=[]`、`frozen_embeddings=true` 与纯 profile 相符，CUDA 未初始化。这证明run002机制文件的inventory/身份；该文件未做独立合成，固定TRAIN比较见下节。pilot48发布候选另有前述两句真实合成证据，不能混用两个adapter。

### 固定4例 TRAIN 机制 gate（已实际完成）

评估使用独立源码 [31fd9fe11d84d5f328afa36c49f363dbe6f9cf24](https://github.com/allenflux/indextts25-hindi-h200/tree/31fd9fe11d84d5f328afa36c49f363dbe6f9cf24)，固定相同4例 TRAIN，分别比较原始 base 与 step100 adapter 的 teacher-forcing total loss，没有挑选较好 case 或换成 VAL/TEST：

| 指标 | 实际结果 |
|---|---|
| mean before loss | `9.192850112915039` |
| mean after loss | `6.462177276611328` |
| after / before | `0.7029568846698167`，满足预先固定 `≤0.9` |
| 模块 gate / 冻结状态 | `mechanism_gate_passed=true`、`frozen_state_unchanged=true`、`trainable_tensors=96` |
| 控制器结束 | child exit0、own process exited=true；wall 48.250421秒 |

这是**小集 TRAIN 的机制 gate，不是语音质量、独立泛化或真实合成验收**。比值来自相同固定4例的配对均值，不能与训练末单条 loss 或 VAL 均值混算。

操作者另行独立核对原 run identity 的 TRAIN canonical SHA 与模块 `dataset_identity_sha256` 完全一致，均为 `446283012f2b8ca24325610330101d7a2745c875d1b6476774a8bd694bf21083`，关联记为独立 operator-binding。模块本身的 `adapter_training_dataset_binding_verified=false` **仍保留 false，未篡改为工具自动证明**；adapter 本身不记录数据身份，外部关联核对与模块内部证据必须分开。

该 gate 通过后才启动上述独立 pilot48；fixed4 gate 的 step100 adapter 不是 pilot 的选中产物，不将两次 run 的步数、loss 或权重累计混用。pilot 已按上述独立回执完成并冻结step100；两个 run 的 checkpoint 与发布身份继续分开。

### 保留的失败与 CPU 验证边界

此前 failed001（源码前缀 `30cb446`）因 reentrant gradient checkpoint 与全部基础输入冻结导致梯度链断开，**0 optimizer updates、0 checkpoints**，已退出，控制器 wall 40.192秒；该失败目录保留，不覆写成成功。修复仅对 pure profile 改用 non-reentrant checkpoint，保留 checkpoint 机制、不解冻基础参数；生产 flag 的真实 tiny CPU 回归 **8/8** 与独立审查通过后才完成 run002。

最新修复后完整 CPU suite **369/369、43.927秒** 已通过：使用绝对官方源码路径、隐藏 CUDA、Hub 离线，覆盖 non-reentrant 修复、`pure_overfit` 及15项 eval guard 检查。评估工具源码已发布为 [31fd9fe11d84d5f328afa36c49f363dbe6f9cf24](https://github.com/allenflux/indextts25-hindi-h200/tree/31fd9fe11d84d5f328afa36c49f363dbe6f9cf24)；训练仍固定 ed19c4a，不改写训练来源。此前353/353、39.011秒只保留为修复前历史快照。完整 CPU 回归本身不是实际 GPU gate 或质量通过；另一次固定4例 GPU 机制结果已单列，仍不以文件大小、Git 状态或该 TRAIN gate 代替语音质量。

训练 CLI 沿用 `python -m hi_h200_native.train --config PATH`，它是**真实训练命令，不是 metadata dry-run**。可先看 `--help`；准备好已核原始 base、合法数据/特征、fresh 输出及完整计划后，才运行配置。示例默认 CPU/fp32 也会实际加载和训练，不是廉价 tiny 测试；CUDA 仍须明确设置 `loop.device="cuda:0"`、`loop.precision="bf16"` 并追加 `--allow-cuda`。没有新增下载器、Pod 分配或后台调度。

```bash
PYTHONPATH=src python -m hi_h200_native.train --help
# 以下会真实开始训练；本次文档更新没有执行它。
PYTHONPATH=src python -m hi_h200_native.train --config "$PURE_TRAIN_CONFIG"
```

新独立入口是 `hi_h200_native.adapter_synth`，当前仅接受 `--language hi`。默认只检查本地路径、资产/adapter SHA 和元数据，**不导入 Torch 或执行模型；tensor/profile 校验仍待 execute 阶段**。必须使用真实原始模型、实际训练 base manifest 与冻结 pilot48 adapter，不能把旧 best100 或 run002 机制候选填作发布候选。当前 CLI 形状如下。冻结pilot48候选已完成2/2句真实加载与未合并native WAV；默认metadata本身仍不完成tensor/profile加载检查，不能把仅preflight、文件SHA或TRAIN gate成功泛化成实际生成或质量通过：

```bash
PYTHONPATH=src python -m hi_h200_native.adapter_synth \
  --workspace-root "$WORK" --official-repo "$OFFICIAL_REPO" \
  --model-dir "$ORIGINAL_MODEL" --model-manifest "$SYNTH_MANIFEST" \
  --training-base-manifest "$TRAINING_BASE_MANIFEST" \
  --adapter "$PURE_ADAPTER" --adapter-sha256 "$PURE_ADAPTER_SHA256" \
  --prompt "$PROMPT_PCM16" --output-dir "$FRESH_OUTPUT" \
  --language hi --text 'नमस्ते।'
```

当前训练的精确 base manifest SHA 为 `11dae1733635a0d46e7edbc7c5444900fed2bbb4450767aa4d2820b2221f0e6f`，共16项（13项 payload 及生成辅助 metadata）。不能用 synth plan 替代它。操作者已新建本地兼容资产视图：24项中23个 hardlink 加一个独立复制的原 training asset-plan，保留原 training manifest 的精确 bytes；全部20项 synthesis payload 强制校验通过，未改 base_identity、源文件内容或权限，也未重新复制约7GB payload。hardlink 会改变 nlink/ctime，**不是写隔离**，不得修改共享 inode。已发布的26文件ZIP提供精确16项training manifest、三份生成辅助metadata及已审公开配置/词表/许可；不含原始GPT或声学等大权重，下载者仍须另备并核验完整原始/synthesis资产，不能把metadata包当作完整依赖。

所有官方源码、原始模型、两份 manifest、adapter 和 prompt 都须位于明确的 `WORK` 内；synthesis manifest 须在原始模型目录内，adapter 须独立于原始模型和源码目录。该新入口没有可放宽此范围的 `--readonly-dependency-root` 参数。真实 `--execute` 还要求 `--allow-cuda`、明确 `--gpu-id` 与 GPU/RAM/利用率门槛、fresh 输出、固定官方资产和离线保护。实际加载应在私有 Hindi 实例先以 CPU FP32 严格校验 A/B，再构造官方推理 wrapper；**不写临时 merged GPT，不改原文件，也不接受现有旧语言服务对象**。实际两句生成已单列，默认metadata成功仍不能单独证明已加载或生成。该入口是独立 Python 推理，**不是 T8 runtime adapter 支持，也不保证普通 ComfyUI LoRA 节点、PEFT、社区 adapter、DeepSpeed 或 vLLM 兼容**。

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

GitHub 保存原创源码、配置、schema、模型卡及公开验证摘要；HF 保存**单独命名与版本登记的新纯 LoRA adapter**、固定 base 的依赖说明和完整适用许可。adapter 文件名、训练 profile、step、base SHA、adapter SHA 与对应代码 commit 必须相互一致。已有 `IndexTTS-2.5-Hindi-best100-experimental@117c7922e4dcedc3f9885f3f6efa5be8f54c7919` 保留原样并标为历史完整 GPT，不能通过覆盖、删 head 或重命名伪造新实验。独立[HF固定权重revision](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-LoRA-experimental/tree/88e6297852699eb6dfbf6d4677446d37afcacf5f)已公开上传 `Hindi-LoRA-step100-public.zip`。6,941,415 bytes / SHA256 `1ede2c2ebc0a196d7cdb3013941487500d21f02269b790eab8cca8da6e9c29fd`，已匿名完整下载并核对26项CRC、manifest SHA和15c候选adapter SHA，匹配本地已审白名单包。26个文件位于ZIP内，不是仓库根目录的散装 `adapter.pt`。网页封面后于 `09bc09ae9720d16ed2580667968a42f2a8940a5d` 更新并回读下载链接；ZIP与包内冻结模型卡a53仍固定在88e62978，仓库封面不与包内README哈希混同。详见[发布包回执](acceptance/pure-lora-public-package-20261118.json)。

### 固定下载、解压与SHA校验

从[不可变ZIP链接](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-LoRA-experimental/resolve/88e6297852699eb6dfbf6d4677446d37afcacf5f/Hindi-LoRA-step100-public.zip)获取公开包。以下操作使用全新目录，先核对整个ZIP再解压；包内manifest、SHA256SUMS、adapter配置、精确base-metadata与完整许可应一并保留。

```bash
set -euo pipefail
mkdir hindi-lora-step100-88e62978
cd hindi-lora-step100-88e62978
curl --fail --location --output Hindi-LoRA-step100-public.zip \
  "https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-LoRA-experimental/resolve/88e6297852699eb6dfbf6d4677446d37afcacf5f/Hindi-LoRA-step100-public.zip"
printf '%s  %s\n' \
  "1ede2c2ebc0a196d7cdb3013941487500d21f02269b790eab8cca8da6e9c29fd" \
  "Hindi-LoRA-step100-public.zip" | shasum -a 256 -c -
mkdir package
unzip Hindi-LoRA-step100-public.zip -d package
cd package
shasum -a 256 -c SHA256SUMS
```

校验通过只证明该26文件发布包的内容身份。仍需单独准备匹配的完整原始base与synthesis大权重，保持精确training manifest字节及生成metadata；不能用HF下载后重写的synth plan代替。包内 `BASE_METADATA.md` 解释这些前提。下载包不含私人音频/数据/特征/optimizer日志，不表示质量、新5090、T8 pure或生产部署通过。

工具 MIT 不替代官方 Bilibili 模型条款、数据 CC-BY-4.0 或其他辅助资产许可；纯 adapter 也不自动豁免衍生分发义务。保留完整中英文协议、归属、变更和非背书说明，不代用户接受 gated 条款，不发布原参考、训练语料、私人转写、speaker/clip 身份、特征、恢复 checkpoint、日志或凭据。此前明确获准公开的两份四节点样本维持原范围，不自动授权本轮新生成音频。[分发说明](DISTRIBUTION.md) · [数据来源](DATA_SOURCES.md)

历史资料：[旧完整 GPT 安装](INSTALL_EXPERIMENTAL.md)、[已撤回的直接替换方案](SINGLE_MODEL_UPGRADE.md)、[旧 RTX 5090 三轮](RTX5090_SWITCHING.md)。这些链接只保留可追溯证据，不改变当前原模型保留、Hindi adapter 独立的新方向。
