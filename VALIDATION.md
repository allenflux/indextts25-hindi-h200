# 实际验证记录

日期：2026-11-17。本机源码、官方checkout、CPU测试环境、缓存和临时文件位于已核实的CodexData挂载；Pod资产、数据与特征使用独立持久工作目录。官方源码完整LICENSE及LICENSE_ZH保留在独立checkout。没有复制或导入未许可社区实现，未主动停止、重启或修改现有ComfyUI服务、全局代码、模型或Python包。资源竞争与Pod重建的限制另见下文。

## 本机完整CPU套件

CPython **3.11.15** / Torch **2.8.0** / torchaudio **2.8.0** / Transformers **4.52.1** / NumPy **2.2.6**，官方源码固定 `d9e41aac89fd00b3d71497fddb287b7f24613712`：**175/175通过，无跳过，31.699秒**。

| 范围 | 通过数 | 实际内容 |
|---|---:|---|
| 原始数据与外部兼容预检 | 59 | 合成WAV、输入/路径/Unicode/泄漏、合成checkpoint状态与mock安全加载；未执行社区后端 |
| 原生模型 | 9 | 真实官方微型UnifiedVoice前向/反向、20次AdamW更新loss下降、生成前向logits对齐、LoRA/冻结行 |
| 原生数据 | 12 | 真实NPZ与tensor、provenance/token/speaker/hash；示例配置经实际官方tiny检查点与byte-BPE进入生产dataset校验，缺schema版本被dataset/extract双入口拒绝 |
| 原生提取组件 | 19 | 实际CAMPPlus+fbank、tinyEnhancedCodec量化/解码、GPT emotion与Seamless前端；随机权重/合成音频 |
| 原生恢复 | 10 | 实际Torch文件、微型GPT更新、STOP/恢复与连续训练参数/AdamW/scheduler/RNG/cursor相同、SIGTERM安全边界保存 |
| 原生CLI/集成 | 8 | 实际NativeFeatureDataset、官方tokenizer类、微型GPT执行更新；特征明确为手工合成夹具 |
| 公开资产助手 | 10 | mock元数据与小型真实文件、SHA/预算/缓存隔离/原子发布；包括synth阶段与独立BigVGAN来源清单 |
| Hindi导入器 | 23 | 真实小Parquet、48k→24k转换、lazy分片、授权/ID/预算、完整utterance时长上限与复用哈希；套件网络使用mock |
| 合并GPT导出 | 6 | 微型官方模型训练后合并，由fresh stock模型及官方checkpoint loader严格加载，logits/loss在容差内相同 |
| Cross-entropy | 3 | flatten前后loss与梯度、token轴边界、真实官方微型模型全部可训练梯度对照 |
| 独立tiny gate | 6 | 真实三进程CPU连续/STOP/恢复对照、RNG与浮点容差、资源/设备/输出限制 |
| 合成入口 | 10 | 精确资产/源码/前端、真实PCM与socket阻断、merged身份、资源门槛、缓存隔离；没有用mock合成器冒充波形生成 |

完整套件之后的两处合成入口guard小修，另有该模块 **10/10** 定向测试通过。完整套件记录与定向记录分别保留，不能据此推断完整预训练合成已成功。

这些CPU检查使用很小的随机模型及合成输入，验证数学、模块、优化器、数据边界和状态恢复。它们不证明预训练2.5的Hindi更新或音质。旧社区检查点工具的mock测试与原生真实Torch往返分别列明。官方模块的weight_norm弃用、GenerationMixin和tokenizer文件句柄警告没有被解释为完整`.generate()`/波形路径通过。

## Pod CPU与H200机制验证

在Pod既有环境及隔离overlay中，较早代码集的 **148/148 CPU测试通过，19.689秒**：CPython **3.12.3** / Torch **2.12.1+cu130** / torchaudio **2.11（cu130）** / Transformers **4.52.1**。这是超出官方Python范围、且Torch/torchaudio偏离官方锁定版本的实际记录，不等同于当前175项套件或完整模型推理兼容性证明。没有为了测试更换ComfyUI全局运行库。

H200上已实际运行官方随机微型模型的 **BF16连续6次更新**，与 **STOP后3次更新检查点 → 新进程恢复再更新3次** 的对照gate通过。这里有真实CUDA前向/反向、optimizer更新、磁盘checkpoint与跨进程恢复；不是预训练Hindi模型更新，也不是100-step吞吐或语音质量验收。

## 已完成的官方资产与真实小批次

官方HF模型固定 `c39ce5ba981572cb187443877ff559dfb246ce63`。以下资产已实际完整下载，并按manifest核对文件哈希，不仅是元数据检查：

| 阶段 | payload文件数 | payload字节数 | 状态 |
|---|---:|---:|---|
| extract | 13 | 6,217,934,223 | 完整下载与哈希清单核验完成 |
| synth | 20 | 7,082,513,550 | 完整下载与哈希清单核验完成 |

synth包含extract资产；表中数字不能相加当作唯一下载量，也不包含缓存副本和小型manifest。保留官方与辅助模型许可/模型卡。小文件Git blob SHA1和下载后SHA256有明确区分；目录名或完整清单不能代替config版本验证，`version: 2.0`回归用例仍会失败。完整资产存在不代表所有合成模块都已实际加载。

用户本人处理HF访问授权后，已准备真实Hindi小批次 **24 clips / 6 speakers**，保留完整utterance，只接受不超过10秒的样本，未截断长音频来满足上限：

| Split | Clips | Speakers | 音频秒数 |
|---|---:|---:|---:|
| train | 16 | 4 | 82.0073 |
| val | 4 | 1 | 25.6641 |
| test | 4 | 1 | 32.4190 |

已在H200上用完整预训练官方编码器实际提取 **train 16条特征及provenance**。这证明该小批次训练特征提取已运行，不证明预训练optimizer更新已运行。val/test各只有一位speaker，适合最小隔离与冒烟检查，不足以给出正式泛化结论。真实来源、访问、数据哈希和运行证据保存在本地受控记录，不随代码发布私人音频/特征。

## 中断与共享资源限制

完成train特征提取后，观测到原Pod对象被外部重建。本任务没有执行该Pod或ComfyUI的停止、重启或配置修改；**原因未确认，不能排除本任务与现有服务的间接资源竞争**。不将这次事件写成与任务无关，也不把GPU空闲快照解释为显存预留。重新运行前仍需核实Pod身份、持久目录、宿主/cgroup内存、GPU实时余量和现有负载；迁移或后续执行尚未成功的步骤不能提前计入结果。

## 包、依赖与权限范围

核心运行依赖为空；训练为可选extra。`requirements/native-cpu-macos.lock.txt`保留历史CPU复现环境，不是Linux/CUDA锁或安全更新基线；已知依赖告警、受控checkpoint范围及认证隔离见 [SECURITY.md](SECURITY.md)。不把`weights_only=True`当sandbox，不接收未知checkpoint，不手动忽略告警，也不自动升级现有服务运行库。此前wheel与CLI构建检查已通过；原生版本最终构建以对应提交核验记录为准。

原生路线直接使用允许许可的固定官方接口，不依赖社区后端；外部兼容路线仍保留许可fail-fast。数据访问由用户本人处理，不代为接受gated条件、不猜speaker身份。尚未创建HF模型仓库或上传权重。

## 尚未完成的真实训练与发布验收

1. val/test完整预训练特征提取与对应provenance核验。
2. 预训练2.5的实际optimizer更新、小样本过拟合及真实训练停止/恢复。
3. 100次optimizer更新的实际显存/吞吐与完整共享资源观察。
4. 完整预训练合并GPT、原模型/合并模型的真实WAV合成与人工审听，见 [SYNTHESIS.md](docs/SYNTHESIS.md)。
5. Hindi可懂度/发音/音色/故障率，以及zh/en/ja/es/ar旧语言回归。小模型merge数学对照不替代这些步骤。
6. 数据/模型权利审核、实际模型卡与发布门槛。没有用mock恢复、元数据检查或随机模型gate替代上述验收，不承诺GPU小时或音质。
