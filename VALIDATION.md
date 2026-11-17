# 实际验证记录

日期：2026-11-17。原创原生路线已完成真实官方2.5 BF16训练、新进程恢复、完整GPT合并与stock Hindi生成。**最新pilot best100已完成转换、合并、H200 FP32 parity、两句真实WAV、配对审计与六cases GPU ASR，整体流程退出码0；两个生成样本的ASR较基线改善。正式Hindi质量仍未认证：人工Hindi试听、旧语言回归和gated数据/模型发布权利审核未完成，质量发布gate仍未通过。** 原模型未替换，没有复制或导入未许可社区代码，没有创建HF模型仓库或上传权重。

## CPU实现与集成检查

本机Mac的CPython **3.11.15** / Torch **2.8.0** / torchaudio **2.8.0** / Transformers **4.52.1** / NumPy **2.2.6**，官方源码固定 `d9e41aac89fd00b3d71497fddb287b7f24613712`：**200/200通过，无跳过，32.050秒**。

| 范围 | 通过数 | 实际覆盖 |
|---|---:|---|
| 原始数据与外部兼容预检 | 59 | WAV/Unicode/路径/泄漏；外部checkpoint为合成状态与mock安全加载，未执行社区后端 |
| 原生模型 | 9 | 官方随机微型GPT前向/反向、AdamW、生成logits对齐、LoRA/冻结行 |
| 原生数据 | 12 | 实体WAV/NPZ、provenance/token/speaker/hash；示例配置走真实tiny资产与生产schema |
| 原生提取组件 | 19 | CAMPPlus、tiny codec、GPT emotion、Seamless前端的实际CPU算子；随机权重/合成音频 |
| 原生恢复 | 10 | 实际Torch文件、STOP/新进程恢复、参数/AdamW/scheduler/RNG/cursor对照、SIGTERM边界保存 |
| 原生CLI/集成 | 8 | NativeFeatureDataset、官方tokenizer类与微型GPT真实更新；明确合成特征夹具 |
| 公开资产助手 | 10 | 固定元数据、小型真实文件、SHA/预算/隔离缓存/原子发布与synth来源清单 |
| Hindi导入器 | 27 | 实际小Parquet、48k→24k、lazy分片、授权/ID/预算、完整utterance与复用哈希；网络mock |
| 合并GPT导出 | 6 | 微型模型合并后由fresh stock严格加载，logits/loss对照 |
| complete checkpoint转adapter | 3 | 实际微型训练checkpoint转换后fresh加载logits一致；路径/无覆写/plan身份与默认无Torch导入门槛 |
| Cross-entropy | 3 | flatten前后loss/梯度、token轴边界、实际微型模型梯度对照 |
| 独立tiny gate | 6 | 三进程CPU连续/STOP/恢复与资源/设备/输出门槛 |
| 合成入口 | 11 | 资产/前端/PCM、merged身份、socket阻断、缓存/资源与无覆写门槛 |
| ASR与评分 | 13 | 校验/预算/离线入口、音频与文本评分边界；不以mock结果代替实际转写 |
| stock parity | 2 | 独立stock logits、严格合并加载与目标/数值比较门槛 |
| KV/cache对照实验 | 2 | 真实tiny `.generate()`位置行为、权重不变与控制边界；H200单例实测另列 |

同一H200宿主的既有环境加隔离overlay中，**189/189 CPU测试通过，无跳过，29.589秒**，这是较早的189项测试集，不含后加的parity 2项、KV/cache 2项、checkpoint转换3项和Hindi导入器4项：CPython **3.12.3** / Torch **2.12.1+cu130** / torchaudio **2.11（cu130）** / Transformers **4.52.1**。这偏离官方Python和Torch/torchaudio锁定版本，记录实际验证范围，不作为任意环境兼容性保证。没有为测试修改ComfyUI全局运行库。

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

另已离线准备 **64 clips / 16 speakers** 的max20秒pilot，直接复用两份已合法下载的Parquet，新增网络下载为0；仍保留完整utterance，不截断。默认重新拆分会改变旧speaker归属，该结果已弃用；保留的`prepared-frozen`使用seed **78**，它是0..4095中保持全部交集旧split的最小seed，仅依据speaker ID选择，没有使用ASR结果。5个重叠旧speaker的归属固定，另1个旧speaker未出现在pilot中。

| frozen pilot split | Clips | Speakers | 音频秒数 |
|---|---:|---:|---:|
| train | 48 | 12 | 491.0290416667 |
| val | 8 | 2 | 82.9501666667 |
| test | 8 | 2 | 69.9240833333 |

该pilot的 **train48 / val8 / test8特征已全部实际提取**，随后完成下述独立BF16训练与真实新进程恢复；它不是v1/v2续训，三个run不能合计为一个800步模型。


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

## v1完整导出、stock parity与真实波形

第100步完整GPT合并导出已完成，包含两个输出head及全部原生可训练状态。完整预训练模型上，用 **1条train feature** 将native adapter路径与fresh stock模型严格加载的无LoRA合并路径独立`get_logits`比较，固定 **rtol=atol=1e-4** 通过：

| 比较 | 实测 |
|---|---:|
| code logits最大绝对差 | 9.965896606e-5 |
| text logits最大绝对差 | 1.36375427246e-4 |
| loss差 | 4.76837158e-7 |
| 两路径的code/text argmax；两路径的targets | 分别逐项相等 |

该判定同时使用相对与绝对容差，**不表示所有绝对差都小于1e-4**；单条feature也不代表整个数据集完全等价。

真实官方 `indextts.infer_v2_5.IndexTTS2` 分别加载原模型和完整合并模型，以相同prompt、文字、seed实际生成WAV：原模型 **2.8328344671秒**，合并模型 **3.0302040816秒**。原模型SHA未改变，训练所在Pod的ComfyUI健康检查返回HTTP **200**。这些记录证明训练/导出/推理路径已接通，WAV存在及服务健康响应不能替代音质或旧语言回归。

## 最新best100的同口径ASR与真实波形

公开官方small ASR权重 **483,617,219 bytes** 已实际下载并核对固定SHA，实际GPU转写统一采用去除Unicode P类标点的评分口径。最新流程共 **6 cases**：两条真人参考、新4词句及未参与训练的VAL句各自的原模型/微调生成；不把总cases数当参考样本数。

| 用途 | CER | WER |
|---|---:|---:|
| 既有校准真人reference（1例） | 0.1714285714 | 0.3333333333 |
| 本次VAL真人reference（1例） | 0.1734693878 | 0.7083333333 |
| 原模型：新4词句 | 0.7222222222 | 1.0 |
| pilot best100：同一新4词句 | 0.2222222222 | 0.75 |
| 原模型：未参与训练的VAL句 | 0.7448979592 | 1.0 |
| pilot best100：同一VAL句 | 0.3469387755 | 0.75 |

两个generation pairs已核对normalized text SHA、prompt WAV、seed、禁用开关、生成设置和token IDs一致，`cache=True`。best100合并模型生成的两段真实stock WAV时长分别为 **2.6356462585秒 / 7.5464852608秒**，VAL原模型基线为 **6.1881179138秒**。checkpoint100来自5个固定验证loss点中的最低点，未按ASR挑选；VAL生成样本也未按ASR或试听筛选。

这是小样本ASR改善，尚不代表整体泛化或质量发布gate通过。VAL真人参考的WER本身约0.7083，归一与ASR误差明显，不能直接以单一WER阈值判定生成语音不可用或通过。人工Hindi试听、发音/韵律/相似度、多句覆盖和旧语言回归仍未完成。review pack已在Pod本地真实创建，Kite Files中确认ZIP及文件夹`WORK/outputs/review/pilot-best100/`存在；`WORK/outputs/review/pilot-best100-review.zip`大小为 **1,028,130 bytes**，包内共10个文件：5 WAV、2 texts、metrics、中文README、MANIFEST。它未下载到Mac、未公开；不把本地整理误写为交付或发布私人音频/转写。

早期对照保留为诊断记录，不能替代最新candidate或扩充样本量：v1新4词句原模型/合并均为CER0.7222222222、WER1.0；v2已知TRAIN句默认KV从CER/WER 0.75/0.9411764706降至0.2361111111/0.5882352941，只证明训练样本拟合。其noKV基线为0.7916666667/1.0，合并分数与默认KV相同；train-target参考为0.125/0.4117647059。

## 有界KV与情感conditioning诊断

H200上已实际完成同一已知TRAIN句的noKV/full-context基线和v2合并WAV，两次执行退出码均0，运行时wrapper核验通过且`cache_enabled=false`。文字、prompt、seed和max mel512相同，其余禁用开关相同；两个参考音频的ASR分数未变。合并模型的本例ASR分数与默认KV相同，关闭KV没有改善该诊断。不能把tiny模型的位置差异直接归因为Hindi失败，单例也不足以确认或排除普遍KV问题；默认cache继续保持True。

另在CPU上使用完整预训练v2，对一条已知TRAIN feature仅切换情感向量做teacher-forcing诊断：

| 情感向量 | code CE（约） | 总loss（约） | 首个gold token概率 / argmax |
|---|---:|---:|---|
| target emotion | 0.00373 | 0.12953 | 0.9988 / 正确 |
| prompt emotion | 0.01637 | 0.15231 | 0.9792 / 正确 |

这只说明该例的conditioning变化影响loss，不能确定整体Hindi问题原因，不能替代未知文本生成或泛化评测。

## 早期v2训练诊断

独立的 **fresh v2** 从原始base开始，使用同16条train样本，计划`max_updates=1000`、学习率`3e-4`、warmup25、save25；已在 **200次更新处成功停止，退出码0，checkpoint与adapter已保存**。首5次loss均值 **9.2006926**，末5次 **0.161443803**；step200的`val_loss=8.518435359`，仅评估2个examples。训练loss与有限验证结果差距明显，表现为训练样本记忆，尚未证明泛化。step200完整合并导出已完成，状态`complete_native_gpt_bundle`，共7个文件；完整GPT SHA256为`47be5a16a441a6f96f3b8a9c1c1f0b10ef7b0ca18ed3123baaa8d26d663fc408`。随后已用stock默认KV分别成功生成该已知TRAIN句的基线/合并WAV，时长 **5.1083900227秒 / 5.5031292517秒**；seed1234、相同prompt/text，prompt为同speaker的另一clip，max mel512，其余禁用开关相同。上述ASR只显示该训练句诊断改善，质量gate仍未通过。默认KV与noKV的H200单例推理均已运行；关闭KV没有改善本例合并模型ASR，默认cache保持True。v2不是从v1续训，两次run不能合计成一个300步模型。

## 64条pilot的500步训练与候选选择

pilot从原始base独立开始，完整预训练2.5、BF16、LoRA rank8 / alpha16，学习率`1e-4`、warmup25、save25，计划最大1000步；每100步验证，最多8例。先完成20次fresh更新，再由新进程从真实checkpoint恢复至 **500次optimizer更新**；controller退出码 **0**、stop reason为`None`，确认提供了resume。fresh前20步耗时 **58.8731450606秒**，恢复后的480次更新wall耗时 **364.5355719514秒**，包含加载、验证和checkpoint；首5次loss均值 **9.3806640625**，末5次 **1.2381711557507515**。

以下为 **pilot500** 的独立观测，不与v1资源窗口混用：

| pilot指标 | 实测 |
|---|---:|
| 恢复段480次更新的effective wall吞吐 | 1.3167439255157926 updates/s，含加载、验证和checkpoint |
| 更新21–500计算段P50 / P95 | 0.4813607325774111 / 0.5934735553637137 秒 |
| 更新21–500 wall P50 / P95 | 0.4846375621855259 / 0.7033535651862621 秒 |
| 本进程Torch CUDA allocated峰值 | 6,194,198,016 bytes，约5.77 GiB |
| 整设备327个样本中的显存最大占用 / 最小free | 9868 / 133903 MiB，包含ComfyUI等占用 |

整设备数据不代表本训练进程的显存峰值；采样余量也不是未来可用显存保证。计算段和更新窗口统计不等于含加载与保存的整段wall吞吐。

| 验证步数 | val loss | examples |
|---|---:|---:|
| 100 | 6.625134348869324 | 8 |
| 200 | 6.891722023487091 | 8 |
| 300 | 7.033073604106903 | 8 |
| 400 | 7.208151876926422 | 8 |
| 500 | 7.34740549325943 | 8 |

训练loss下降而验证loss在已记录的第100步最低后持续上升，不能据此宣称泛化提高。已停止扩大该run，**不选择last500**；已将真实CUDA训练的第100步complete checkpoint在CPU上严格转换为标准adapter，状态`selected_checkpoint_adapter_complete`，selected step为100；tensor validation与adapter加载校验均通过。转换过程没有恢复训练状态，也没有执行训练；来源checkpoint与adapter的完整SHA256另存于`selection.json`，不把转换称为再次恢复训练。随后已实际完成 **7文件完整GPT合并、H200完整FP32 parity、两句stock WAV、配对元数据审计与六cases真实GPU ASR**，整体流水线退出码0。原生adapter与fresh stock无LoRA完整合并模型的单条train feature比较，固定`rtol=atol=1e-4`通过：

| best100 parity比较 | 实测 |
|---|---:|
| code logits最大绝对差 / shape | 5.340576171875e-5 / (1, 8194, 278) |
| text logits最大绝对差 / shape | 约1.89e-4 / (1, 60510, 173) |
| loss差 | 4.76837158e-7 |
| 两路径targets；两路径code/text argmax | 分别逐项完全相等 |

这是相对与绝对容差合用的通过，**不表示所有绝对差小于1e-4**，也不表示整个数据集完全等价。VAL与train按speaker隔离，但参与选模，最终test尚未质量验收；候选仅按固定验证loss选择，不使用ASR或test挑checkpoint；原模型未替换，完整模型SHA等来源保留在artifact完成记录。

这次VAL生成样本从时长1–12秒、文字不超过120 tokens、prompt时长1–15秒的元数据eligible集合中选最短者，未按ASR或试听选择。第一次候选prompt为15.724秒，超限后在生成前弃用；最终VAL target为6.703秒、92个tokens，prompt为同speaker另一条8.074秒clip；两者均完整合法，不截断。

发布前仍需：更充分的Hindi/旧语言 held-out评测、人工审听与故障分析、验证ASR口径及其误差、真实模型卡及gated数据/模型发布权利审核。小数据loss下降、stock parity和真实WAV均不自动通过质量发布gate。

## 共享资源、依赖与权限边界

本次全部GPU job已结束。持久`RUN_STATE.json`已保存源代码commit `222daad01275fbb998df1ac224cbdcdc46a7ad8a`、selected step100、`active_training=false`与`quality_gate_passed=false`，并明确200项CPU检查来自Mac环境。最终仅在训练所在Pod的localhost直接健康检查返回 **HTTP200**，H200整设备快照为 **2753 MiB已用、GPU utilization 0%**；这是结束时快照，不是本训练进程峰值或未来余量保证。

本机文件位于CodexData，Pod输出在独立持久目录；两台已用共享pod的ComfyUI均未被我们停止、重启或修改，全局代码、模型及Python包亦未被修改。此前访问异常的原Pod仅由Kite UI确认已恢复 **Ready1/1、restarts0**；此前退出码137的旧terminal已失效，异常原因仍未证实，不能直接归因OOM/重启或排除间接资源竞争。健康响应不证明过去或未来绝无影响。

核心运行依赖为空；训练为可选extra。历史CPU lock不是Linux/CUDA锁或安全更新基线；依赖告警、可信checkpoint与认证隔离见 [SECURITY.md](SECURITY.md)。不把`weights_only=True`当sandbox，不接收未知checkpoint，也不自动升级现有服务库。原生路线不依赖社区后端；外部兼容路线保留许可fail-fast。完整合成操作见 [SYNTHESIS.md](docs/SYNTHESIS.md)，发布门槛见 [EVALUATION.md](docs/EVALUATION.md)。


## 后续有限test与旧语言prompt stress（源码d3bb1f5）

本节追加后续实测，不改写上文source222daad及200项CPU检查的历史范围。工具源码`d3bb1f575c68e227cd1e442fc6586339a6459d83`已公开推送并核对远端SHA；Mac完整CPU suite **222/222通过、0跳过、34.351秒**，新增批量生成与有限ASR实现已独立审查。

冻结best100，Hindi test两speaker各一例、五旧语言各一例，使用Hindi prompt；各case原模型/候选仅执行一次，不训练、不重选case或checkpoint。原模型7/7与best100 7/7真实stock生成、14条均frontend verified；7对text/prompt SHA、seed、settings、stock token IDs/language slot一致，14 WAV哈希审计通过。两个variant及ASR三个子进程退出码均0、无停止原因；时长分别 **84.81 / 79.06 / 20.21秒**，16/16 GPU ASR（14生成+2真人reference）无覆盖缺口。全部样本保留，无retry/reselection。

本轮为canonical authored/test text的 **strict** 指标，包含空格，与上文去标点分数不能直接混算。Hindi两例的微平均严格CER：真人reference **31/166=0.186747**，原模型 **396/166=2.385542**，best100 **83/166=0.500000**；严格WER分别为 **14/32=0.4375、146/32=4.5625、27/32=0.84375**。候选有方向性改善，但CER仍50%，且真人reference本身ASR误差较高，不能据此判定Hindi质量通过。

旧语言每种一例、均为Hindi-prompt stress，严格CER原模型→best100为：zh **1/17→3/17**，en **1/46→1/46**，ja **0/18→2/18**，es **1/60→1/60**，ar **0/41→0/41**。中文、日文单例指标升高；缺母语试听、native-prompt与本语言校准，不能保证无退化或称正式旧语言回归。完整匿名逐例统计与限制见 [FINITE_ACCEPTANCE](docs/FINITE_ACCEPTANCE.md)。

整设备峰值原模型 **9350 MiB/76样本**、候选 **9486 MiB/71样本**，监控的total-used最小值 **134421/134285 MiB**，未扣NVML reserved，不是实际`memory.free`，包含ComfyUI，不能归为本进程。结束时nvidia-smi实际`memory.free`为 **140404 MiB**，设备 **2753 MiB、0%**，本项目Python进程NONE；训练所在Pod localhost ComfyUI **HTTP200**。原模型与best100 GPT哈希最终复核未变，运行状态`active_training=false`、`quality_gate_passed=false`。人工Hindi/母语试听、native回归及发布权利gate仍未完成；未生产替换、未HF发布。

另已离线构建并检查wheel：**121451 bytes**、21个Python源码文件与源码逐一相同、两个CLI `--help`通过，SHA256为`dc900c04b7942c53d7f33e6fbaa06d0dc9d167da4ccf760aa9244626f8bbc622`。使用已有setuptools **84.1.0**直接调用backend；原native环境没有pip/build CLI，没有为打包修改全局依赖。GitHub推送反馈 **52项已知依赖告警（23 high / 23 moderate / 6 low）**，这里只记录平台报告，未逐条独立审计，也未升级生产依赖。
