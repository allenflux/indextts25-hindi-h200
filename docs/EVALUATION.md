# Hindi 与旧语言评测 / 发布门槛

原生路线已完成v1真实训练、恢复、完整合并和stock WAV生成，**Hindi成品与质量发布gate仍未通过**。固定测试集、speaker隔离、文本归一、prompt/text/seed、ASR/声纹模型及revision后再比较；不以训练loss、导出成功或WAV存在宣称音质达标。

## 当前实测

| 维度 | 原模型 | native v1合并GPT | 判定与范围 |
|---|---|---|---|
| 单句Hindi ASR | CER 0.7222222222 / WER 1.0 | CER 0.7222222222 / WER 1.0 | 同ASR、同prompt/text/seed的一条4词新句，未见改善；不代表整体质量 |
| 实际WAV | 2.8328344671秒 | 3.0302040816秒 | 固定官方IndexTTS2真实生成；尚未人工Hindi审听 |
| 自然度/发音/韵律 | 未评 | 未评 | 需至少两位Hindi评审盲测、报告分歧 |
| 词类/姓名/数字/币种覆盖 | 未评 | 未评 | 逐项人工检查nukta/virama、连写、鼻音、借词与朗读形式 |
| 说话人相似度与故障率 | 未评 | 未评 | 两个WAV不能证明完整相似度或截断/重复/静音统计通过 |
| zh/en/ja/es/ar回归 | 未测 | 未测 | 原模型SHA未改不等于合并模型无回归 |
| 导出状态一致性 | 固定原始base | 单条train feature FP32 parity通过 | 两个head完整合并、fresh stock严格加载；不自动推出质量通过 |
| 训练/资源机制 | — | BF16 100 optimizer更新 / 400 microbatch，真实跨进程恢复 | wall吞吐约0.3287805334 updates/s；资源与范围见 [VALIDATION](../VALIDATION.md) |

公开官方small ASR权重已按固定SHA核验并实际GPU转写；本次总共3个cases，仅1例是参考音频校准，另外两例分别是基线与合并生成。去除Unicode P类标点的统一口径下，1例参考音频校准为 **CER 0.1714285714 / WER 0.3333333333**；ASR本身存在误差。单条新句的原模型/合并模型结果相同，不能将这次检查解读成Hindi已改善或对所有句子无改善。没有发布私人音频或原始转写。

完整预训练FP32 parity使用独立stock `get_logits`，比较native adapter与无LoRA严格加载的完整合并GPT。固定`rtol=atol=1e-4`通过，两路径的code/text argmax分别逐项相等，teacher-forcing targets也逐项相等；最大绝对差分别为`9.965896606e-5`和`1.36375427246e-4`，loss差`4.76837158e-7`。判定结合相对与绝对容差，不表示全部绝对差小于1e-4，也不表示整个数据集等价。

## 扩大训练与发布门槛

小样本gate先从train取16–64条有授权、已听检的完整短语音，每speaker至少两条，并保留独立speaker验证。v1首5步平均loss从9.2158降至末5步6.5180，但新句ASR未改善，尚不能认定发音/对齐过拟合gate通过。检查文本前端、标签、梯度与固定样本生成，保留可复核的失败结果；最终test不反复调参。

100-step机制与资源实测已运行；它不替代质量gate。本进程Torch CUDA allocated峰值约5.75 GiB，整设备采样9990 MiB包含ComfyUI，不能归为本训练进程或当作预留显存。继续实验须按共享GPU实时余量与宿主/cgroup内存检查执行。

fresh v2使用同16条train样本，从原始base开始独立实验：计划max1000、学习率3e-4、warmup25、save25；已在200次更新处成功停止，退出码0，checkpoint与adapter已保存。首5次平均loss **9.2006926 → 末5次0.161443803**，但step200验证loss为 **8.518435359**，仅2个examples。训练样本记忆明显，尚未证明泛化；step200完整合并导出已完成（`complete_native_gpt_bundle`，7个文件）；训练句基线/合并模型WAV及ASR仍待验收，不提前计推理或质量成功。它不是v1续训，两run不能合计为一个300步模型，低训练loss也不能自动通过质量门槛。

发布前必须补齐：多句held-out Hindi评测与ASR误差分析、至少两位Hindi评审的盲听、发音/词类/相似度与故障率、旧语言回归、干净环境加载、真实模型卡、数据/模型权利审核与复现哈希。当前val/test各只有一位speaker，正式泛化结论需要更充分的独立speaker与样本量。尚未创建HF模型仓库或上传权重。

## legacy外部兼容路线

旧外部参考trainer的`train_heads=true`同时训练mel/text head，却在其adapter导出中遗漏text head；它还缺少明确许可证。这个风险只属于`scripts/check_backend.py`等保留的legacy兼容路线，不能套到当前native完整合并结果上。原生路线已保存两个head及全部可训练参数，并完成上述实际parity和WAV检查；质量gate仍须独立通过。

原始模型保持只读，语言adapter与合并模型各自版本化，旧语言继续保留原始路径；并发使用隔离进程/实例。发布时保留非官方声明、原始许可、来源与已知失败范围，不附带私人数据或训练prompt。
