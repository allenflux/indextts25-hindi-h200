# 实际验证记录

日期：2026-11-17。源码、模型源码checkout、测试环境、缓存和临时文件位于已核实的CodexData挂载；现有ComfyUI代码、模型和Python包未改动。官方源码完整LICENSE及LICENSE_ZH保留在独立checkout。没有复制或导入未许可社区实现。

## 本机完整CPU套件

CPython3.11.15 / Torch2.8.0 / torchaudio2.8.0 / Transformers4.52.1 / NumPy2.2.6：**148/148通过，无跳过**。

| 范围 | 通过数 | 实际内容 |
|---|---:|---|
| 原始数据/社区接口预检/社区checkpoint检查 | 59 | 合成WAV、输入/路径/Unicode/泄漏、合成状态与mock安全加载；未执行社区后端 |
| 原生模型 | 9 | 真实官方微型UnifiedVoice前向/反向、20次AdamW更新loss下降、生成前向logits对齐、LoRA/冻结行 |
| 原生数据 | 11 | 真实NPZ、严格provenance/token/speaker/hash；测试tokenizer/资产身份为显式夹具 |
| 原生提取组件 | 19 | 实际CAMPPlus+fbank、tinyEnhancedCodec量化/解码、GPTemotion和Seamless前端，随机权重/合成音频 |
| 原生恢复 | 10 | 实际Torch文件、真实微型GPT更新、STOP/恢复与连续训练参数/AdamW/scheduler/RNG/cursor相同，SIGTERM边界保存 |
| 原生CLI/集成 | 8 | 实际NativeFeatureDataset、官方tokenizer类、微型GPT执行真实更新；明确手工合成特征 |
| 公开资产助手 | 8 | mock元数据、真实小文件/哈希/预算/原子发布，无大模型下载 |
| Hindi导入器 | 18 | 真实小Parquet与48k→24k转换、lazy分片和授权/ID/预算；网络mock，无真实语料 |
| 合并GPT导出 | 6 | 实际官方模型训练后合并，由freshstock模型及官方checkpoint loader严格加载，logits/loss容差内相同 |

这些是**真实模型机制/模块测试**，模型尺寸很小、权重随机，输入为合成token/音频。它们不是完整预训练2.5、真实Hindi训练或语音质量的证据。旧社区检查点工具的mock测试与原生真实Torch往返分别列明。

官方模块发出weight_norm弃用、GenerationMixin及tokenizer文件句柄警告；不影响已执行的forward测试，但没有据此宣称完整`.generate()`/波形路径验证成功。

## 已核查的源与资产接口

- 官方源码固定`d9e41aac89fd00b3d71497fddb287b7f24613712`；真实导入`indextts.gpt.model_v2.UnifiedVoice`，无替代模型stub。
- 官方HF模型固定`c39ce5ba981572cb187443877ff559dfb246ce63`；实际公开元数据匹配5文件3,260,524,913B的GPT阶段，13文件6,217,934,223B的提取阶段。小文件Gitblob SHA1与下载后SHA256区分。
- 目录名不证明2.5；完整哈希清单匹配但configversion2.0的用例仍明确失败。
- 原生路线不依赖社区后端；外部兼容路线仍因缺许可保留fail-fast。
- 模型/数据下载是否完成、完整权重/音频/GPU是否运行，以本地实际运行记录和后续验收为准；此处不把元数据检查当下载完成。

## 包与环境

核心运行依赖为空；训练为可选extra。实际CPU解析清单见`requirements/native-cpu-macos.lock.txt`，不等于Linux/CUDA锁。构建依赖已升级setuptools84.1.0覆盖此前3项实际告警；不手动忽略告警，GitHub后台状态另以返回为准。此前wheel构建及CLI检查通过；原生版本的最新构建结果随提交核验。

## 真实训练/发布仍需通过

1. 完整模型及辅助资产哈希/严格加载、实际Hindi音频提取与人工对齐检查。
2. 当前授权认证、明确Hindi配置和小分片预算，保留实际来源/许可；不替用户接受gated条件、不猜speaker身份。
3. 独立持久目录、实际GPU/运行环境检查；随机微型CPU测试不能代替预训练H200停止/恢复、100-update显存/吞吐。
4. 完整合并GPT的实际合成、Hindi及zh/en/ja/es/ar回归；合并共享权重不能保证旧语言无变化。
5. 数据/模型权利审核、真实模型卡和波形/导出验收后才发布权重。没有创建HF模型仓库或上传权重。
