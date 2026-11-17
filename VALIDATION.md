# 实际验证记录

日期：2026-11-17。原创原生路线已打通真实官方2.5训练、停止/新进程恢复、完整GPT合并导出与stock推理。**Hindi成品目标尚未达到：单例ASR检查未见改善，人工Hindi试听及旧语言回归未完成，质量发布gate仍未通过。** 没有复制或导入未许可社区代码，没有创建HF模型仓库或上传权重。

## CPU实现与集成检查

本机CPython **3.11.15** / Torch **2.8.0** / torchaudio **2.8.0** / Transformers **4.52.1** / NumPy **2.2.6**，官方源码固定 `d9e41aac89fd00b3d71497fddb287b7f24613712`：**191/191通过，无跳过，35.857秒**。

| 范围 | 通过数 | 实际覆盖 |
|---|---:|---|
| 原始数据与外部兼容预检 | 59 | WAV/Unicode/路径/泄漏；外部checkpoint为合成状态与mock安全加载，未执行社区后端 |
| 原生模型 | 9 | 官方随机微型GPT前向/反向、AdamW、生成logits对齐、LoRA/冻结行 |
| 原生数据 | 12 | 实体WAV/NPZ、provenance/token/speaker/hash；示例配置走真实tiny资产与生产schema |
| 原生提取组件 | 19 | CAMPPlus、tiny codec、GPT emotion、Seamless前端的实际CPU算子；随机权重/合成音频 |
| 原生恢复 | 10 | 实际Torch文件、STOP/新进程恢复、参数/AdamW/scheduler/RNG/cursor对照、SIGTERM边界保存 |
| 原生CLI/集成 | 8 | NativeFeatureDataset、官方tokenizer类与微型GPT真实更新；明确合成特征夹具 |
| 公开资产助手 | 10 | 固定元数据、小型真实文件、SHA/预算/隔离缓存/原子发布与synth来源清单 |
| Hindi导入器 | 23 | 实际小Parquet、48k→24k、lazy分片、授权/ID/预算、完整utterance与复用哈希；网络mock |
| 合并GPT导出 | 6 | 微型模型合并后由fresh stock严格加载，logits/loss对照 |
| Cross-entropy | 3 | flatten前后loss/梯度、token轴边界、实际微型模型梯度对照 |
| 独立tiny gate | 6 | 三进程CPU连续/STOP/恢复与资源/设备/输出门槛 |
| 合成入口 | 11 | 资产/前端/PCM、merged身份、socket阻断、缓存/资源与无覆写门槛 |
| ASR与评分 | 13 | 校验/预算/离线入口、音频与文本评分边界；不以mock结果代替实际转写 |
| stock parity | 2 | 独立stock logits、严格合并加载与目标/数值比较门槛 |

同一H200宿主的既有环境加隔离overlay中，**189/189 CPU测试通过，无跳过，29.589秒**，不含后加的两项parity测试：CPython **3.12.3** / Torch **2.12.1+cu130** / torchaudio **2.11（cu130）** / Transformers **4.52.1**。这偏离官方Python和Torch/torchaudio锁定版本，记录实际验证范围，不作为任意环境兼容性保证。没有为测试修改ComfyUI全局运行库。

CPU数学/模块测试主要使用随机微型权重与合成输入。另有H200随机微型官方模型的BF16连续6次更新，对照STOP于3次后新进程再更新3次的gate通过；此机制对照与下述完整预训练模型证据分开记录。

## 官方资产与真实Hindi数据

官方模型固定 `c39ce5ba981572cb187443877ff559dfb246ce63`，源码完整LICENSE及LICENSE_ZH保留。官方和辅助资产已实际完整下载并核对manifest哈希：

| 阶段 | payload文件数 | payload字节数 |
|---|---:|---:|
| extract | 13 | 6,217,934,223 |
| synth | 20 | 7,082,513,550 |

synth包含extract资产，两行不能相加当作唯一下载量；不包含缓存副本和小型manifest。保留许可/模型卡，区分Git blob SHA1与实际文件SHA256，不凭目录名判断模型版本。

用户本人处理访问授权后，准备了 **24 clips / 6 speakers** 的真实Hindi小批次。只选完整utterance且不超过10秒，不截断长音频来满足上限：

| Split | Clips | Speakers | 音频秒数 |
|---|---:|---:|---:|
| train | 16 | 4 | 82.0073 |
| val | 4 | 1 | 25.6641 |
| test | 4 | 1 | 32.4190 |

完整预训练编码器的实际特征与provenance用于此次原生训练。val/test各只有一位speaker，当前小集合只能支持有限验收，不能给出正式泛化结论。私人音频、转写、特征与访问记录留在本地受控目录。

## v1完整预训练训练与资源实测

H200上使用完整官方2.5模型完成 **BF16 100次optimizer更新 / 400个microbatch**。真实检查点链路为第2步 → 新进程至第20步 → 新进程至第100步；controller与monitor退出码均为 **0**。首5次loss均值 **9.21582865715**，末5次均值 **6.51796336174**。这证明实际更新和恢复链路运行，loss下降不能代替语音质量验收。

稳定窗口为更新 **21–100**：

| 指标 | 实测 |
|---|---:|
| 更新计算段P50 / P95 | 0.2481003413 / 0.4413370118 秒 |
| 完整更新wall P50 / P95 | 3.0103580970 / 6.5492602885 秒 |
| optimizer更新吞吐 | 0.3287805334 updates/s |
| 本进程Torch CUDA allocated峰值 | 6,179,036,160 bytes，约5.75 GiB |
| 整设备显存采样最大值 | 9990 MiB，共510个样本，包含ComfyUI等占用 |

wall区间包含保存/验证开销；计算段不是纯kernel性能承诺。整设备采样不能当作本训练进程的显存峰值，也不能当作显存预留或其他负载的最大需求。

## 完整导出、stock parity与真实波形

第100步完整GPT合并导出已完成，包含两个输出head及全部原生可训练状态。完整预训练模型上，用 **1条train feature** 将native adapter路径与fresh stock模型严格加载的无LoRA合并路径独立`get_logits`比较，固定 **rtol=atol=1e-4** 通过：

| 比较 | 实测 |
|---|---:|
| code logits最大绝对差 | 9.965896606e-5 |
| text logits最大绝对差 | 1.36375427246e-4 |
| loss差 | 4.76837158e-7 |
| 两路径的code/text argmax；两路径的targets | 分别逐项相等 |

该判定同时使用相对与绝对容差，**不表示所有绝对差都小于1e-4**；单条feature也不代表整个数据集完全等价。

真实官方 `indextts.infer_v2_5.IndexTTS2` 分别加载原模型和完整合并模型，以相同prompt、文字、seed实际生成WAV：原模型 **2.8328344671秒**，合并模型 **3.0302040816秒**。原模型SHA未改变，ComfyUI健康检查返回HTTP **200**。这些记录证明训练/导出/推理路径已接通，WAV存在及服务健康响应不能替代音质或旧语言回归。

## 同口径ASR检查与质量结论

公开官方small ASR权重 **483,617,219 bytes** 已实际下载并核对固定SHA，在本地GPU做真实转写；此次共3个cases，即1例参考音频、1例基线生成和1例合并生成。统一采用去除Unicode P类标点的评分口径：

| 用途 | 样本范围 | CER | WER |
|---|---|---:|---:|
| 参考音频校准 | 1例 | 0.1714285714 | 0.3333333333 |
| 原模型生成 | 1条4词的新句 | 0.7222222222 | 1.0 |
| v1合并模型生成 | 同一句、同prompt/text/seed | 0.7222222222 | 1.0 |

该单例未见改善，**质量gate仍未通过**。ASR自身存在校准误差，单句结果不能代表整体Hindi质量；人工Hindi试听、发音/韵律/相似度、多句覆盖和旧语言回归均未完成。不发布原始音频或转写来展示结果。

## 当前实验与仍需验收

独立的 **fresh v2** 从原始base开始，使用同16条train样本，计划`max_updates=1000`、学习率`3e-4`、warmup25、save25；已在 **200次更新处成功停止，退出码0，checkpoint与adapter已保存**。首5次loss均值 **9.2006926**，末5次 **0.161443803**；step200的`val_loss=8.518435359`，仅评估2个examples。训练loss与有限验证结果差距明显，表现为训练样本记忆，尚未证明泛化。step200完整合并导出已完成，状态`complete_native_gpt_bundle`，共7个文件；完整GPT SHA256为`47be5a16a441a6f96f3b8a9c1c1f0b10ef7b0ca18ed3123baaa8d26d663fc408`。训练句基线/合并模型WAV及ASR仍待验收，不提前计推理或质量成功。v2不是从v1续训，两次run不能合计成一个300步模型。

发布前仍需：更充分的Hindi/旧语言 held-out评测、人工审听与故障分析、验证ASR口径及其误差、真实模型卡与权利审核。小数据loss下降、stock parity和真实WAV均不自动通过质量发布gate。

## 共享资源、依赖与权限边界

本机文件位于CodexData，Pod输出在独立持久工作目录；未主动停止、重启或修改现有ComfyUI服务、全局代码、模型或Python包。此前曾观测到Pod对象被外部重建，原因未确认，不能排除间接资源竞争；后续实际步骤使用重新确认的运行资源，健康HTTP200也不证明过去或未来绝无影响。继续实验仍需实时检查GPU、宿主/cgroup内存和持久状态。

核心运行依赖为空；训练为可选extra。历史CPU lock不是Linux/CUDA锁或安全更新基线；依赖告警、可信checkpoint与认证隔离见 [SECURITY.md](SECURITY.md)。不把`weights_only=True`当sandbox，不接收未知checkpoint，也不自动升级现有服务库。原生路线不依赖社区后端；外部兼容路线保留许可fail-fast。完整合成操作见 [SYNTHESIS.md](docs/SYNTHESIS.md)，发布门槛见 [EVALUATION.md](docs/EVALUATION.md)。
