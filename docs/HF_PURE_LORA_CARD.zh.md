---
license: other
license_name: bilibili-model-license
license_link: https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/c39ce5ba981572cb187443877ff559dfb246ce63/LICENSE
language:
  - hi
base_model: IndexTeam/IndexTTS-2.5
pipeline_tag: text-to-speech
datasets:
  - ai4bharat/indicvoices_r
tags:
  - indextts
  - hindi
  - lora
  - attention-only
  - native-adapter
  - experimental
---

# IndexTTS 2.5 Hindi 纯 attention LoRA：实验模型卡

**独立公开仓库已创建；下载以Files实际存在的不可变commit、文件与hash为准，本卡不作为上传证明。** [fluxa0223/IndexTTS-2.5-Hindi-LoRA-experimental](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-LoRA-experimental) 已实际创建并回读确认Owner为fluxa0223、Public；许可证元数据为other / bilibili-model-license、固定c39协议URL。仓库页面存在不表示adapter已经发布。pilot48 已完成并按预声明冻结step100，adapter大小与完整SHA已核；2/2句未合并native真实合成及file/PCM回读已核，独立质量评测未完成；打包源码已发布，实际造包、上传和远端验证仍以操作回执为准；保留历史 merged 仓库原样，不用改名或覆盖旧 best100 来冒充本实验。

这是从已核原始 IndexTTS 2.5 **重新训练**的 Hindi 专用 attention-only LoRA，不是官方 Hindi 模型，不是合并 GPT、完整推理包或生产质量保证。Hindi 使用独立原模型 + adapter 实例；旧语言使用从未加载该 adapter 的原模型实例。`quality_verified=false`；新 adapter 的 T8 runtime 集成尚未完成，不保证普通 ComfyUI LoRA 节点、PEFT 或其他社区格式兼容。

## 固定候选身份与当前发布状态

| 项目 | 已核身份 / 未完成项 |
|---|---|
| 训练 profile | `attention_lora_only_v1` |
| 独立 HF 仓库 | [fluxa0223/IndexTTS-2.5-Hindi-LoRA-experimental](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-LoRA-experimental)；Owner fluxa0223 / Public已核，模型卡不代替文件上传证明 |
| pilot48 训练源码 | `ed19c4a511be4baafe87b22f74adf6b192a0a55a` |
| 独立评估工具源码 | `31fd9fe11d84d5f328afa36c49f363dbe6f9cf24`；不冒充训练来源 |
| 验证与打包代码 | 推理/评估 `31fd9fe11d84d5f328afa36c49f363dbe6f9cf24`；[白名单打包器](https://github.com/allenflux/indextts25-hindi-h200/blob/e041f7cfde95932475628aedb03e64d62d059ad3/scripts/build_pure_lora_release.py) `e041f7cfde95932475628aedb03e64d62d059ad3`；源码发布本身不是造包/上传成功 |
| pilot 选中 checkpoint step | **100，已按预声明 VAL8 mean 最低冻结**；TEST/ASR未用于选择 |
| adapter 文件名 / bytes / SHA256 | 打包器白名单文件名 `adapter.pt`；实际选中adapter为 **5,931,907 bytes / `15c080715b3b50bf38b33a70f196a9c9e987446e0a6bb87b2e0c8b7835f57dbd`**，完整字节SHA已核，非run002机制候选 |
| HF 权重 revision / 远端 SHA | 固定Files中实际存在的不可变commit，并逐项核对 `adapter.pt` 与发布清单SHA；本卡不声明未经回执验证的远端revision或上传成功 |
| 真实原始 base + adapter 合成 | source31fd/H200已完成2/2句未合并native WAV，两个完整file/PCM均独立回读一致，两个guard exit0/自有子进程退出；仅功能验证 |
| Hindi / held-out TEST / 母语听评 | **未开展本轮独立TEST/ASR或母语听评，无本轮质量通过结论** |
| 新纯 LoRA T8 / 普通 ComfyUI LoRA | **未完成集成，不能填写为通过** |

模型卡文本、训练结束、CPU测试或文件能被读取都不能证明权重已经上传、接口已经生成音频或质量通过。发布时以最终白名单 manifest、完整 adapter SHA 和不可变 HF revision 为准。

## 模型、固定 base 与可训练范围

- 官方模型：[`IndexTeam/IndexTTS-2.5`](https://huggingface.co/IndexTeam/IndexTTS-2.5/tree/c39ce5ba981572cb187443877ff559dfb246ce63)，固定 revision `c39ce5ba981572cb187443877ff559dfb246ce63`。
- 官方源码：[`index-tts/index-tts`](https://github.com/index-tts/index-tts/tree/d9e41aac89fd00b3d71497fddb287b7f24613712)，固定 commit `d9e41aac89fd00b3d71497fddb287b7f24613712`。
- 原始 GPT SHA256：`43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85`；原模型文件不改写。
- 原创实现：[allenflux/indextts25-hindi-h200](https://github.com/allenflux/indextts25-hindi-h200)。未复制或导入无许可证的社区训练器。

只训练24层共享 attention 的 `c_attn` / `c_proj` A/B，共48模块、96 tensor、该 rank8 架构下1,474,560参数。**所有基础 attention weight/bias、语言/文字 embedding（包括 Hindi 行17）、text_head、mel_head、其余 GPT 与声学模块冻结。** Hindi 行17及 tokenizer 的 `<|hi|>` 是既有条件槽位，不表示该行更新或官方已训练/评测 Hindi。

训练使用 `CE_semantic_code + 0.2 × CE_text`。这里 `mel_head` 输出离散语义 code logits，不是连续 mel；冻结的辅助 text head 仍让梯度经过计算图传到 A/B，但其自身参数不更新。没有训练 codec、CFM/DiT、BigVGAN 或使用 ASR 奖励。纯 LoRA 限制变化来源，不保证 Hindi 发音、自然度或韵律一定改善。

## 专用原生 adapter 格式

当前外层 schema 为 `schema_version=1`、`kind="hi_h200_native_adapter"`，严格包含 `step`、`base_identity`、`adapter_spec`、`trainable`。纯 profile 要求：

- `adapter_spec.format="hi_h200_native_attention_lora_only_v1"`、`training_profile="attention_lora_only_v1"`。
- `heads=[]`、`text_rows=[]`、`frozen_embeddings=true`；Hindi 行17也冻结。
- `trainable` 仅含 `base.gpt.h.{0..23}.attn.{c_attn,c_proj}.{lora_A,lora_B}`；不得混入 head、embedding 或其他基础 tensor。
- 严格验证 base 身份、rank/alpha/dropout、目标模块、shape/dtype/finite 与完整 tensor inventory；不以宽松加载迁就错误 adapter。

这不是通用 PEFT `adapter_config.json` / safetensors 约定。当前 standalone runtime 直接保留未合并 A/B 分支，不输出临时 merged GPT；关闭 Hindi 功能指销毁私有 Hindi 进程，不承诺热卸载或同对象恢复。旧 best100 含 row/head 更新，删掉 A/B 无法还原这些变化，不能当作本模型的 adapter 或训练起点。

## 数据来源、拆分与处理

实际数据为 [AI4Bharat IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r)，revision `5f4495c91d500742a58d1be2ab07d77f73c0acf8`，仅 Hindi。只取得两份官方 **TRAIN** 分片，payload 合计 **946,963,686 bytes**；未下载全库或发布者官方 TEST。Rasa 未参与这次训练。

pilot48 使用此前已授权、已提取的 pilot64；本项目 train/val/test 全部由这两份官方 TRAIN 分片构建并按 speaker 隔离：

| 本项目分组 | Clips | Speakers | 音频秒数（约） |
|---|---:|---:|---:|
| TRAIN | 48 | 12 | 491.029 |
| VAL | 8 | 2 | 82.950 |
| TEST | 8 | 2 | 69.924 |

本项目 TEST 不是发布者官方 TEST。保留源文与保守 NFC/空白规范文本，保留 nukta、virama、组合标记及 joiner，不静默截断目标录音。24 kHz 到16 kHz的原生特征路径与 stock 的 librosa 路径存在数值前端差异；训练或单 feature 检查不能直接变成完整前端/声音等价性结论。

数据归属：AI4Bharat；Ashwin Sankar 等，2024，*IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS*，[论文](https://arxiv.org/abs/2409.05356)、[数据集固定 revision](https://huggingface.co/datasets/ai4bharat/indicvoices_r/tree/5f4495c91d500742a58d1be2ab07d77f73c0acf8)、[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)。发布包应附归属与修改说明，列明 Hindi 子集筛选、speaker 拆分、文本规范、音频转换、特征提取与 adapter 微调；不再分发原语料。原作者对采集同意的说明不等于本项目逐人核验所有声音人格权或其他用途授权，访问同意也不自动给予无限分发或克隆用途权利。

## 训练、恢复与预声明选模

pilot48 从已核原始 base 独立开始，**不是从 run002 的过拟合 adapter 继续**。H200/BF16、batch1、累积4、rank8/alpha16/dropout0、AdamW lr1e-4、warmup10、max100、seed260618、每25步保存/验证。所有 head/embedding 与基础权重冻结；不与旧 best100 的步数或状态累计。

启动前已保存预声明：只在25/50/75/100中按 **VAL8 mean loss 最低**选择，完全同分取较早 step；TEST、ASR 和试听不参与选模。fresh2 已实际 exit0 / verified2 / own process exited=true，wall40.20832155458629秒；新进程 resume100 也已 exit0 / verified100 / own process exited=true，wall110.89273961819708秒。optimizer updates严格1..100，不与run002相加；两段wall含加载、保存、验证，不是纯GPU时间。进程Torch peak CUDA allocated为4,043,693,056 bytes、reserved为4,339,007,488 bytes，不是整卡总占用。

| 固定 VAL step | 8例 mean loss |
|---|---:|
| 25 | 8.110908448696136 |
| 50 | 7.306136906147003 |
| 75 | 7.053911030292511 |
| 100 | 6.989858150482178 |

按预声明最低值选100，已记录 `FROZEN_SELECTION`、`test_used_for_selection=false`、`candidate_frozen=true`；后续不按TEST/ASR/试听改选。选中adapter实际5,931,907 bytes、SHA `15c080715b3b50bf38b33a70f196a9c9e987446e0a6bb87b2e0c8b7835f57dbd` 与trainer回执一致。旧run002的 `902d…cff1` 只是单独的小集机制候选，不是该发布产物。两句未合并native真实WAV均已执行并独立回读，2/2仅记工程功能，不等于独立TEST或质量通过。

此前 run002 小集已实际完成2→100恢复并生成待发布之外的机制候选；固定4例 TRAIN 的配对 loss 比值为0.7029568846698167≤0.9、冻结状态检查通过。它只证明那组训练例上的机制，不是 pilot 的 held-out 质量、合成或泛化验收。两次新进程恢复都是实际继续训练，**没有与不中断连续训练做逐位对照，不能宣称 bitwise 连续等价**。

修复后的完整CPU suite 为369/369、43.927秒（绝对官方源码路径、CUDA隐藏、Hub离线），覆盖 pure-only non-reentrant checkpoint 修复与评估 guard。CPU通过、实际训练、严格adapter加载及2/2句真实合成分别记录；独立TEST/ASR、母语质量与T8 pure仍未完成。

## 2/2例原始 base + 未合并 adapter 的真实工程验证

冻结pilot候选在H200/source31fd通过callback严格精确16项training identity与96A/B加载，实际构造48模块的官方推理wrapper。报告 `adapter_loaded_without_merge=true`、`adapter_profile_verified=true`、`unmerged=true`、KV cache=true、`original_language_instance_touched=false`；原GPT磁盘SHA仍为上述43a8…值，没有替换或临时mergedGPT。

预先固定的第一句公开工程文本为 **नमस्ते। यह हिन्दी भाषा का एक छोटा परीक्षण है।** 参考来自固定官方Space `IndexTeam/IndexTTS-2-Demo@b01840e8e4fd9753743a6d0466cd73ae1d634a68` voice01，而非训练私有原声：原FLOAT为478,050 bytes / SHA `e33e6ee0107a1dd58e1d66dd90c13df3d55a8683047cc3d7ea206dad84ed3fc8`，仅转PCM16后234,160 bytes / SHA `fd23ddac68f2db3710481c7cbfc1a50a117d9c38bc5c3ee51c33444f0a9b019a`，没有crop/resample。

| 第一例结果 | 已核实际值 |
|---|---|
| 格式 | mono PCM16 / 22,050Hz |
| bytes / frames / seconds | 169,038 / 84,480 / 3.8312925170068026 |
| WAV SHA256 | `643b7e5f032039b353e8c935de85f7ce830e2bd29e6195dfb0d3bfa5ebd78810` |
| PCM SHA256 | `eaaccc3a8af22e0c48ce15c778162380999416739468538dfeeb804c3240a243` |
| 独立回读 | 完整文件与PCM哈希一致 |
| 生成 / guard wall | 5.084137551486492 / 102.56857392378151秒；含不同阶段，不混作纯推理benchmark |
| 结束 | exit0、own child exited=true |

**新WAV保持私有，未获额外公开音频授权；本卡只列工程文本与统计，不附音频。** 第二句预固定 **नमस्ते मित्र।** 也已真实成功；step100/15c候选、严格profile加载、48模块未合并wrapper及KV=true均有实际回执。

| 第二例结果 | 已核实际值 |
|---|---|
| 格式 | mono PCM16 / 22,050Hz |
| bytes / frames / seconds | 84,558 / 42,240 / 1.9156462585034013 |
| WAV SHA256 | `562d2a8632777cc001b0d629ec56877a10abad35b4e418d35cd5b57be05542cd` |
| PCM SHA256 | `0d84b07822247419df5947ba7851064281207f3877b380c11f6424c863b591d0` |
| 独立回读 | 完整文件与PCM哈希一致 |
| 生成 / guard wall | 3.803127806633711 / 98.87775631062686秒 |
| 结束 | exit0、own child exited=true |

因此仅记 **2/2 native未合并工程句生成功能通过**。原GPT在两句结束后再次完整读取，仍为43a8…原值；原ComfyUI HTTP200/队列0/0。自有训练和合成子进程退出有控制器回执，不宣称未完成的额外独立PID存在性检查已通过。本轮没有替换原模型、改动生产ComfyUI或安装系统依赖；健康状态为结束时点观测。

本入口greedy do_sample=false、beams1、repetition penalty1、text_normalization=false、max_mel256、max_text120、seed1234；原T8图为sample=true/beams3/penalty10/normalization=true/max_mel1504。它们参数不同，**不得据这两句成功宣称原工作流兼容、新5090/T8 pure集成或任何音质通过**。尚未人工听评、ASR或独立TEST，也未证明句尾完整。

该次环境Python3.12.3、Torch2.12.1、Transformers4.52.1、torchaudio2.11，与固定官方部分依赖约束不同；这是实际环境记录，不是通用锁文件或安全性保证。已有依赖通过显式共享overlay与WORK内synth-overlay提供，missing=0、未安装依赖、未改系统；不能据此宣称任意环境已兼容。

## 当前加载前提：不能只下载官方 base 就宣称可运行

当前入口为 `python -m hi_h200_native.adapter_synth`，仅允许 `language=hi` 的独立实例；默认只做本地资产/adapter hash及元数据预检，tensor/profile 校验在显式 execute 的受限 CPU 加载阶段完成。真实生成还需完整原始 base、匹配源码及依赖、合法 PCM prompt、fresh 输出、显式 CUDA 授权与 GPU/RAM/利用率门槛。普通 T8 / ComfyUI LoRA 节点尚未接入这个原生格式。

**必须同时具备精确的 training-base manifest 与完整 synthesis 资产 manifest，二者不能互相替代。** adapter 的 `base_identity` 包含原始 training manifest 的文件 SHA 和各项身份；当前训练使用的16项 base manifest 含13项payload及 `asset-plan.json`、`w2v-bert-2.0.manifest.json`、`campplus.manifest.json` 三份本项目生成的辅助记录；这些不是声称可从上游HF直接下载的原文件。不能临时用 synth plan、只含官方下载文件的新 manifest 或 HF CLI 下载后的目录替换它，更不能删改 adapter 身份/关闭校验来强行加载。

精确training manifest SHA为 `11dae1733635a0d46e7edbc7c5444900fed2bbb4450767aa4d2820b2221f0e6f`；该16项记录含13项payload及生成辅助metadata。操作者已在本地准备24项兼容视图（23个hardlink与一个独立复制的原training asset-plan），保留原training manifest精确bytes，全部20项synthesis payload强制校验通过；没有改base_identity或源内容/权限，没有再复制约7GB payload。hardlink会改变nlink/ctime，**不是写隔离**，不得修改共享inode。

发布前仍须给出经过隐私/许可审核、与该身份匹配的实际 manifest 及所需辅助文件，或可确定性重建其相同身份的明确流程；本地视图不表示公开复现包已经发布。若这些前置尚未落实，须如实写为复现阻塞，**不能宣称仅运行 `hf download IndexTeam/IndexTTS-2.5` 就能得到本卡的可运行环境**。不得为了补 manifest 而上传整个工作区、原始特征、私有路径或训练记录；必要时先解决可公开复现的依赖身份，保持 strict loader fail-closed。

源、原始模型、两份 manifest、adapter、prompt 和输出均需按当前 CLI 的 WORK 边界准备；adapter 独立于原始模型/源码目录，原始 base 文件不能改写。加载期间可能暂存原 TTS 与额外 CPU FP32 GPT，进程隔离不预留 GPU/RAM，也不保证低显存或不与其他服务争抢。当前不能把“adapter小”当作总资源占用很小。

## 评测与使用边界

声称质量通过或正式推广前，应按冻结后的选中 checkpoint 做独立原始 base / adapter 对照：保留全部输入和输出、固定 text/prompt/seed/settings、检查 stop/max-code 截断与完整 WAV/PCM，再报告合法 held-out TEST、真人 reference 的 ASR 校准和 Hindi 母语听评。ASR误识别不必然是TTS错误，单例或纯 TRAIN 机制 gate 均不是质量证书。

旧语言请求应进入从未加载该 adapter 的原模型实例；不能把“回到原模型后同一短句相同”移作“候选对所有旧语言无损”。若把本 adapter 用到旧语言，它们也会经过共享 attention 增量，输出可能改变。旧 best100 的历史 ASR/T8三轮不转记为本卡新纯 adapter 的验收。

本pilot选模已冻结、候选文件大小与SHA已核，并已实际严格加载完成2/2句未合并native合成；当前未完成项包括独立音频TEST/ASR与母语质量、新5090/T8 pure runtime集成及生产部署。实验发布可以披露未完成项，但不等于正式 Hindi 支持、全局替换建议或所有旧语言无退化。

## 许可、分发与非背书

adapter 是原模型的衍生微调产物，适用 **bilibili Model Use License Agreement**，不是工具代码的 MIT。完整 [英文协议](https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/c39ce5ba981572cb187443877ff559dfb246ce63/LICENSE) 与 [中文协议](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/LICENSE_ZH.txt) 原文须随包保留；中文优先。本文摘要不替代完整条款。

本人或关联方上月 MAU 超过1亿，或上一年度收入超过人民币1亿元时，协议要求事先取得单独书面许可；其他用途、下游传递、第三方权利和分发义务仍依完整协议执行。此卡不公开个人财务/访问材料，也不将训练数据的访问许可冒充原模型权利人的普遍授权。

发布包必须携带完整 `LICENSE`、`LICENSE_ZH.txt`、`MODEL_LICENSE`、`DATA_ATTRIBUTION.md`、`NON_ENDORSEMENT.txt` 和模型卡/实际校验清单；按协议保留原始版权与规定的非背书声明。该实验及其修改不由原模型权利人或数据权利人背书、担保或保证；不能使用其名称暗示认可，具体声明以随包原文为准。完整中文/英文许可文件不能被模型卡的一段摘要代替。

公开白名单限最终 adapter、必要的已审来源/依赖与格式说明、模型卡、完整许可/归属、实际文件 hash 和可公开的汇总评测。**不上传原始 base 权重、原始/参考音频、数据分片、训练prompt或私人转写、speaker/clip身份、特征、optimizer/scheduler/RNG恢复状态、私有日志或认证凭据。** 生成音频如需另行公开，按其具体授权与来源审核，不因本 adapter 发布而自动获准；旧两份已授权样本维持其独立范围。

最终step、adapter SHA与两句native未合并功能结果已有实据，独立HF仓库和白名单打包器源码也已创建/发布。严格依赖公开包及权重下载状态以Files实际存在的不可变commit、完整清单与hash验证为准；本卡不作为上传证明，也不以仓库创建或代码发布代替造包、上传、远端SHA或未完成的独立质量验证。
