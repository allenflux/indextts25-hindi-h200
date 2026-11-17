# Hindi 与旧语言评测 / 发布门槛

本页的训练、VAL与早期ASR记录属于此前阶段；后续冻结test及五种旧语言Hindi-prompt stress的完整结果见 [FINITE_ACCEPTANCE](FINITE_ACCEPTANCE.md)。质量发布gate仍未通过。
最新64条/16人pilot的best100已完成真实BF16训练后的标准adapter转换、7文件完整GPT合并、H200单feature FP32 parity、两句stock WAV、配对审计与六cases GPU ASR，流水线退出码0。**Hindi训练、导出和生成已实现，小样本ASR改善；正式Hindi质量与发布gate仍未通过。** 人工Hindi试听、旧语言回归和gated数据/模型发布权利审核尚未完成，保留实验标识，原模型未替换，HF未发布。本机CPU检查200/200通过、无跳过（32.050秒），机制检查不替代语音质量验收。

## 最新candidate实测

统一以固定官方small ASR权重、去除Unicode P类标点的口径评分：

| 样本 | 原模型CER / WER | pilot best100 CER / WER | 范围 |
|---|---|---|---|
| 新4词句 | 0.7222222222 / 1.0 | 0.2222222222 / 0.75 | 同prompt/text/seed与生成设置 |
| 未参与训练的VAL句 | 0.7448979592 / 1.0 | 0.3469387755 / 0.75 | 完整合法clip；按预设元数据规则选择 |

六个ASR cases包含两条真人reference和上述四条生成。既有校准reference为CER **0.1714285714** / WER **0.3333333333**；本次VAL真人reference为CER **0.1734693878** / WER **0.7083333333**。ASR与归一自身误差明显，不能用单一WER阈值直接认证生成可用/不可用，也不能由两个生成例推出整体Hindi泛化。

两个generation pairs已审核normalized text SHA、prompt WAV、seed、flags、生成设置和token IDs一致，cache保持True。best100的真实stock WAV分别为 **2.6356462585秒 / 7.5464852608秒**，VAL原模型基线为 **6.1881179138秒**。review pack已在Pod本地创建并在Kite Files确认ZIP `WORK/outputs/review/pilot-best100-review.zip`及文件夹`WORK/outputs/review/pilot-best100/`，压缩包1,028,130 bytes，包内10个文件（5 WAV、2 texts、metrics、中文README、MANIFEST）。未下载到Mac、未公开；人工试听尚未完成。

完整预训练FP32 parity在H200使用独立fresh stock `get_logits`，比较native adapter与无LoRA完整合并GPT。固定`rtol=atol=1e-4`通过，两路径targets、code/text argmax分别逐项完全相等。code/text logits最大绝对差分别约 **5.34e-5 / 1.89e-4**，loss差 **4.76837158e-7**。该判定结合相对与绝对容差，不表示所有绝对差小于1e-4；单条train feature不代表整个数据集等价。

## 数据、训练与选择边界

pilot离线复用两份合法已有Parquet，新增网络下载为0：64 clips / 16 speakers，完整utterance max20秒，不截断。保留`prepared-frozen` seed78（0..4095中保留全部交集旧split的最小值），仅依据speaker ID选择，没有使用ASR结果；5个交集旧speaker固定，另1个旧speaker不在pilot。train为48条/12人/491.0290416667秒，val为8条/2人/82.9501666667秒，test为8条/2人/69.9240833333秒；三split特征全部实际提取。

从原始base独立开始，完整预训练2.5 BF16、LoRA rank8/alpha16，学习率1e-4、warmup25、save25，计划max1000。真实第20步checkpoint→新进程第500步，controller退出码0。恢复段480次更新wall耗时364.5355719514秒（含加载、验证与checkpoint），约1.3167 updates/s；本进程Torch CUDA allocated峰值约5.77 GiB，整设备采样9868 MiB包含ComfyUI，不能归为本进程。

首5次loss均值9.3807→末5次1.2382，但固定5个验证点（每次全部8例）在100/200/300/400/500步分别为6.6251/6.8917/7.0331/7.2082/7.3474。验证loss持续上升，已停止扩大该run，选择验证最低的 **100步**，不选last500，**未按ASR挑checkpoint，不用test调参**。VAL与train按speaker隔离，但参与选模，最终test尚未质量验收。真实CUDA complete checkpoint在CPU转换为标准adapter，tensor validation与加载校验通过；转换不恢复训练状态、不执行训练，来源校验保留在`selection.json`，再完成上述实际合并和推理。

VAL生成案例从时长1–12秒、文字≤120 tokens、prompt时长1–15秒的元数据eligible集合取最短者，未按ASR或试听选择。曾有prompt15.724秒的候选在生成前超限弃用；最终使用同speaker不同clip配对的完整合法音频，不截断。

v1、v2、pilot均从原始base独立开始，不能合计为一个800步模型。早期v1新句ASR未改善，v2只有已知TRAIN句改善；noKV没有改善该v2单例的合并ASR，默认cache保持True。单例emotion-conditioning teacher-forcing诊断也不能确定整体Hindi失败原因。完整早期对照、资源窗口和CPU范围见 [VALIDATION](../VALIDATION.md)。

## 仍需通过的发布gate

| 维度 | 当前状态 | 需要补齐 |
|---|---|---|
| Hindi自然度/发音/韵律 | 未人工评测 | 至少两位Hindi评审盲听，记录分歧 |
| 词类/姓名/数字/币种覆盖 | 未系统评测 | nukta/virama、连写、鼻音、借词和朗读形式 |
| 说话人相似度与故障率 | 未通过 | 相似度及截断/重复/静音等统计 |
| zh/en/ja/es/ar回归 | 未测 | 固定旧语言集与原模型对照 |
| 多speaker held-out质量 | 仅少量ASR例 | 更多独立speaker、多句与ASR误差分析 |
| 数据/模型发布权利与模型卡 | 未完成 | gated使用/再发布条件、许可与真实限制审核 |

本次所有GPU任务均已结束，持久运行状态为selected100、active training=false、quality_gate_passed=false，原模型未替换。pilot val/test各只有两位speaker，正式泛化结论仍需更充分的独立样本。保留固定测试集、speaker隔离、归一规则、prompt/text/seed、ASR/声纹revision与复现哈希。loss下降、WAV存在及stock parity都不自动通过质量gate；尚未创建HF模型仓库或上传权重。

## legacy外部兼容路线

旧外部参考trainer的`train_heads=true`同时训练mel/text head，却在adapter导出中遗漏text head，且缺少明确许可证。此风险只属于保留的legacy兼容路线；当前原生实现保存两个head及全部可训练状态，已完成实际完整合并、parity与WAV。原始模型只读、语言adapter和合并模型各自版本化；并发使用隔离实例。发布时保留非官方声明、原始许可、来源与已知限制，不附私人数据或训练prompt。
