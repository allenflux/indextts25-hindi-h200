# IndexTTS 2.5 Hindi 换模验收方案 v1.0

状态为待执行，现有纯 LoRA 已通过 H200 原生加载和两句工程样本合成。RTX 5090 上的同工作流验收、独立 Hindi 听评和发布后的回滚演练尚待完成。本文给出通过条件和操作顺序，适用于准备换模和审签。

## 1. 选定换模范围

| 方式 | 工作流与实际加载 | 放行条件 |
| --- | --- | --- |
| **A（推荐）：只给 HI 切换候选** | 兼容更新同一个 T8 插件，保留 8 个节点 ID、类型和 5 条连线。旧语言继续从原路由加载官方 GPT，`language=HI` 由独立原生实例加载固定原版和新 adapter。具备候选注册后，也可让 HI 实例加载由该 adapter 导出的完整 GPT。 | 原工作流、旧语隔离、Hindi 质量、5090 资源和回滚均通过。最初只开放已覆盖的工作流参数与流量。 |
| **B：固定旧加载入口，直接换完整 GPT** | 保留原节点、原 `model_name=IndexTTS-2.5` 和空 `custom_model_path`。兼容插件把同一逻辑入口解析到已验收的候选版本目录，启动时加载完整 `gpt.pth`。所有语言共用候选 GPT。 | 除功能、Hindi 质量、性能和回滚外，ZH/EN/JA/ES/AR 各自的母语质量非劣效都要通过。放行时冷切整个发布版本，保留原版用于回滚。 |

B 的注意力权重是语言共享的。冻结 embedding、输出 head 和声学资产只能限定改动范围，不能证明旧语输出等价。A 通过也不能替 B 放行。任何方式都需要 T8 兼容更新才能让同一生成节点接受 `HI`。基线 T8 的枚举和文件清单会拒绝这种请求或候选，关闭界面上的 `verify_hashes` 不能替代更新。

历史合并 best100（HF `117c7922…`）在 5090 的三轮切换只是工程先例。其 GPT 还修改过语言/文本 embedding 与输出 head，和下面锁定的纯 LoRA 属于不同候选，其音频、SHA 和测试不能计入本次。

## 2. 冻结候选与验收资产

| 项目 | 固定身份 |
| --- | --- |
| 官方模型 | `IndexTeam/IndexTTS-2.5@c39ce5ba981572cb187443877ff559dfb246ce63`，实际 `config.yaml` 必须为 2.5 |
| 原始 GPT SHA256 | `43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85` |
| 官方代码 | `index-tts/index-tts@d9e41aac89fd00b3d71497fddb287b7f24613712` |
| 纯 LoRA | pilot48 step100，`attention_lora_only_v1`，48 个 attention 投影、96 个 A/B 张量。adapter SHA256 `15c080715b3b50bf38b33a70f196a9c9e987446e0a6bb87b2e0c8b7835f57dbd` |
| adapter 发布包 | HF `88e6297852699eb6dfbf6d4677446d37afcacf5f`，ZIP 6,941,415 bytes，SHA256 `1ede2c2ebc0a196d7cdb3013941487500d21f02269b790eab8cca8da6e9c29fd`。[固定下载](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-LoRA-experimental/resolve/88e6297852699eb6dfbf6d4677446d37afcacf5f/Hindi-LoRA-step100-public.zip) |
| 原训练清单 | 取 ZIP 中 `base-metadata/model-manifest.json` 的原始字节，SHA256 `11dae1733635a0d46e7edbc7c5444900fed2bbb4450767aa4d2820b2221f0e6f`。按清单核全部 16 项，其中包括三份生成辅助记录。完整 synthesis 清单另保留。 |
| T8 基线 | `T8mars/comfyui-indextts25-t8@1ff425ece89ef9824f58f0e4dbfb057991814e88`。单独记录目标机实际源码、局部改动和最终兼容改动 commit。 |
| 新完整 GPT | 从上面的原版和 adapter 导出后登记实际 SHA、字节数、provenance SHA、候选清单 SHA。这些字段均待实际导出后填写。 |

在隔离持久目录准备两个版本。原版、候选、工作流 JSON、参考集、软件依赖、采样设定、性能指标和样本分配在第一条正式生成之前冻结。保留原版完整目录和插件的可启动备份。记录 RTX 5090、驱动、Torch/CUDA、ComfyUI、T8、其他音频节点的实际版本与 GPU 编号。按真实模型目录的字节数预留两套资产、冷备、导出及日志的磁盘空间。原目录曾出现 `version: 2.0`，先读取配置与全量哈希。

安装前核整包 SHA，再解压到新目录并执行 `shasum -a 256 -c SHA256SUMS`。完整原版合成资产需另备。训练构造使用原 16 项训练清单，原生生成使用完整合成清单，T8 使用当前兼容插件选择的模型清单。原生推理的清单不能代替 T8 的模型清单。T8 基线列出 26 项模型资产，按目标插件的实际清单验证数量、大小、完整 SHA 和许可。可用官方下载说明在 [原生加载指南](https://github.com/allenflux/indextts25-hindi-h200/blob/2c804756d7ac786a803dae7694cff9d26b16bb7d/docs/HINDI_LORA.md#保留的失败与-cpu-验证边界)。

## 3. 构建可替换件并验收加载

### A. 保留原版权重，按 HI 加载

用现有 `hi_h200_native.adapter_synth` 在隔离进程作预检和 H200 原生对照，真实执行使用独立输出和有界 GPU 控制器。该入口接收 `--training-base-manifest`、`--model-manifest`、`--adapter` 和固定 SHA，样例及已验证参数范围见上面的原生加载指南。

在兼容 T8 中让 HI 在原模型 cache 获取前进入独立 worker。显式传递和验证采样、情绪参考、时长、分段、停顿、文本规范化、重试和取消，输出符合原图的 `AUDIO`。当前独立 CLI 是受限原生推理，不能直接当该 worker。保持 `use_accel` / DeepSpeed 关闭，直到单独验收。工作流请求失败时将错误返回给调用者，并保持旧语言路由可用。缓存身份覆盖实际 base、适配器、tokenizer、运行时、设备和精度。参考缓存也须区分预处理和模型身份。单卡并发在压测通过前串行运行。

### B. 导出完整 GPT，再用 T8 加载

当前 CPU 导出 CLI 可接收纯 profile。下面执行会读取原始权重并在一个全新目录写出完整 GPT。先确认工作区可用 RAM 建议至少 32 GiB，以及磁盘余量，初次导出时记录实际用量。所有变量用已核绝对路径，`LIVE_MODEL_DIR` 和 `LIVE_T8_DIR` 分别为保护的生产资产与插件目录，`TRAINING_BASE_MANIFEST` 是原 16 项清单，`MERGED_BUNDLE` 必须是已存在父目录下尚不存在的独立路径。运行环境使用项目已审 Torch 与依赖。

```bash
set -euo pipefail
PURE_ADAPTER_SHA256=15c080715b3b50bf38b33a70f196a9c9e987446e0a6bb87b2e0c8b7835f57dbd
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" \
python -m hi_h200_native.export \
  --official-repo "$OFFICIAL_REPO" --model-dir "$ORIGINAL_MODEL" \
  --model-manifest "$TRAINING_BASE_MANIFEST" --adapter "$PURE_ADAPTER" \
  --adapter-sha256 "$PURE_ADAPTER_SHA256" --output "$MERGED_BUNDLE" \
  --protect-root "$LIVE_MODEL_DIR" --protect-root "$LIVE_T8_DIR"
```

1. **完整性。** 检查 `COMPLETE.json`、`provenance.json`、`gpt.pth` 与许可。用受限 CPU `weights_only=True` 独立加载，严格验证 `{"model": state_dict}`、官方模型键/形状/dtype 和全部有限值。对原 GPT 比较逐张量字节，差异仅允许出现在 24 层 `c_attn.weight` 和 `c_proj.weight` 共 48 个键，至少一个键确实变化，且各目标的数值与独立计算的 `W + (alpha/r) × AᵀBᵀ` 相符（官方 Conv1D 权重按输入×输出存储）。embedding、head、bias 与其他张量均应逐字节相同。再校验原 GPT 整包 SHA，重新核对候选 SHA。现有 `audit_merged_gpt.py` CLI 固定旧 best100，须先新增新版候选审计入口或在受审工具中独立完成上述比较。
2. **真实前向等价性。** 在固定真实 Hindi 短、中、长 3 个特征样本上比较原版＋A/B 与新完整 GPT。用 FP32、同设备、同输入、同 seed，关闭 TF32。按现有 `parity.py` 固定标准 `rtol=atol=1e-4`，文本和语义 logits、loss 均在容差。两路预测 argmax 逐位一致，两路标签 target 逐位一致。当前 `parity.py` 的执行构造未传入 `training_profile`，需要先支持纯 profile 并测其 fail-closed 行为。在报告中记录实际 profile、adapter spec SHA 和 3 个样本各自的比较结果。默认预检会写新 JSON 并可导入 Torch，但不执行真实前向。现有纯 LoRA 控制器的模块白名单不包含 parity 或 stock synth，GPU 跑这些用例前需准备受审的有界控制器。
3. **独立生成。** 用原生 `hi_h200_native.synth --variant merged` 加载原版合成资产、完整合成清单和新 `--merged-bundle`。先在独立进程以和未合并一致的 greedy 参数做同参考短句对照，再验证实际工作流参数。两条路径分别登记真实 checkpoint、设备、dtype、token/停止原因和 WAV 证据。比较语义序列、句尾、时长与自然度，任何差异需解释并独立通过 T8 与听评。原生最小 CLI 的采样和长度上限不代替 T8 验收。
4. **T8 完整候选目录。** 以已验的 2.5 资产组装新版本目录，只更换拟发布 GPT。为新候选登记真实文件大小、完整 SHA、来源 adapter 和新清单。保留基线清单，按显式候选身份路由并核完整 SHA。新 GPT 可能与其他候选同大小，文件大小只参与校验。现有 `check_t8_compat.py`、`prepare_t8_single_model.py` 以及历史 5090 控制器绑定 `117c7922…`。复用其隔离、观察和清理逻辑需先增加新候选，旧哈希/旧清单不能代用。完整候选目录通过检查后，再测试原 `model_name`、空 `custom_model_path` 和关闭自动下载的真实加载路径。

## 4. RTX 5090 同工作流验收

使用独立服务、独立端口、独立数据库和 cache。原队列先排空，验证窗口暂缓新派单。准入起点为实际 GPU 空闲显存 ≥24 GiB、宿主可用 RAM ≥16 GiB、GPU 利用率 ≤10%，与本机资源和服务基线一起审查。首次仅验证单路，在确认容量后再测计划并发。控制器应持续检查占用、超时和共享任务，只终止测试进程。双模型共存若逼近预算，先用分进程串行方案，随后用真实计划并发重测。

从实际 API 工作流复制成脱敏验收图，固定 `class_type`、节点 ID、连线和输出节点，确认同一 `T8_IndexTTS25_Generate` 提供 HI。在正向对照中只变更语言、文本、授权参考和有意选择的发布版本。使用合法参考音频，固定实际输入 bytes、seed 及所有字段。关闭整图缓存验证依赖真实执行，同时按生产的引用缓存、驻留、精度和加速选项验收。旧图的短句参数含 sample、beams 3、repetition penalty 10、norm 和 max mel 1504，不能用当前原生短样本的 greedy 代替。若实际业务开启加密，另用测试密钥校验输出可解密，实际密钥不出现在报告。

| 编号 | 场景 | 必须记录的通过条件 |
| --- | --- | --- |
| E1 | 原版 ZH，冷启动两次 | 实际加载原官方 SHA，队列与 history 成功。记录基线自然波动。同设备同环境的确定性用例，WAV 解码后 PCM 一致。 |
| E2 | A 走 ZH → HI → ZH；B 走原版 → 候选 → 原版冷启动 | 原图 8 个节点及 5 条连线一致，已连接的 `1/2/3/4/5/8` 真正执行。最终 WAV 有效，分段无错，选择的 base/adapter/GPT SHA 与路由一致。A 的旧语前后、B 回滚后在确定性用例 PCM 一致。B 候选阶段每种语言都应加载候选。 |
| E3 | HI 中长文、停顿、数字货币、情绪参考、音色复用 | 至少一条跨实际分段边界，记录分段 token 计数、停止原因和最终 PCM。句尾无丢失，speaker/emotion 配置真实生效。对上线范围中的每组选项各有证据。 |
| E4 | 错 SHA、缺资产、同名目录或旧缓存、不支持的参数、错误输入、取消和超时 | 返回明确错误和可定位阶段。缺资产与错误 SHA 在加载前失败，禁用自动下载或静默回退。候选加载前后核对真实 SHA 与缓存身份。在隔离测试中故意取消 HI 后立即发旧语任务，旧语仍成功，队列清理和进程退出均有证据。 |
| E5 | 每路 3 次冷启动、至少 30 请求热态，各跑 1 路及计划实际并发 | 保留错误、取消、OOM 全部记录。要求无 OOM/意外失败，旧语成功率无下降。相对同图原路由的旧语 p95 端到端延迟 ≤1.20×，HI p95 达到上线前冻结的同业务长度预算。持续 GPU 空闲余量 ≥4 GiB，确认峰值与结束释放，所有阈值按冻结版本录表。 |

计时同时记排队、加载、合成、整图和 RTF，报告设备峰值与进程峰值的口径。生成成功须有 history 成功、可解码音频及完整文件/PCM SHA 三类证据。短句哈希相等只覆盖对应确定性用例。

## 5. 语音验收与放行门槛

在看到输出前冻结样本清单、参考使用权、两名盲评人和下面的门槛。下面为项目的建议上线门槛，任何调整在执行前登记。失败样本、超时与空音频都计入，候选保持 step100。修改模型、分段策略或兼容代码后对受影响的 gate 重跑。

| 分组 | 最小覆盖 | 判定 |
| --- | --- | --- |
| P0 诊断 | 现有冻结 Hindi test 8 条/2 说话人，加两条工程句及至少一条跨段长句。 | 全量同时生成并检查。用于早停和定位。两位说话人的规模不足以批准生产。 |
| P1 Hindi | ≥40 条独立目标、≥10 名未参与训练或选模的说话人，至少 10 条长文/跨段；覆盖数字、日期、金额、标点、天城文与常用缩写。参考与同 speaker 目标是不同 clip。 | 至少 95% 条目达到可懂性 ≥4/5（两名 Hindi 母语评审，分歧复判），自然度和音色保持均值各 ≥3.5/5。整批零严重缺陷，包括错语言、关键数字错误、整句漏读、重复片段、句尾截断、异常静音或明显破音。 |
| P2 旧语 | ZH/EN/JA/ES/AR 每语 ≥20 对、≥5 名独立说话人，同语言和合法母语参考，含短句、数字、长分段。对照使用同文本、同参考、同 seed 与相同采样。 | A 核实际路由与确定性回归。B 每语需独立过门槛。两名该语言的盲评人对可懂性、自然度、音色评分，均值下降各 ≤0.25/5，且不存在候选新增严重缺陷。用说话人为重采样单位计算差值区间。说话人不足或任一指标的单侧 95% 下降上界超过 0.25 时，补足独立样本后重测，期间保持待验收。未覆盖语言保持原路由，B 等待补齐。 |

ASR 固定版本、语言和文本规范，先用合法真人参考校准，保留每条转写与编辑数。中文/日语优先 CER，Hindi/其他语言报告定义清楚的 CER/WER 与数字切片。P2 的默认辅助报警线为每语候选较原版微平均 CER 绝对上升超过 2 个百分点，触发复听和归因。ASR 不足以判定发音与音色。真人校准质量不足或出现人机冲突时标记不确定并复核，证据不足时暂停放行。评审保留全量盲评编号和原始评分，GitHub/HF 仅发布获授权、脱敏的汇总。

## 6. 上线、停止与回滚

1. **签收。** 核对固定 candidate/GPT、T8、工作流、完整资产清单、测试计划与报告哈希。对应 A 或 B 所有前序 gate 状态必须是 PASS。保存原发布版本、原启动器配置、原插件与可恢复的路由快照，提前在隔离环境演练下述回滚。
2. **部署。** 由值班人在维护窗口暂停向目标实例派单并等队列为零。A 部署固定兼容插件与 HI worker 配置，先用内部白名单把 HI 指到候选。B 把独立候选版本目录及插件作为一份发布，在目标进程停止后通过部署配置一次激活，保留原 `IndexTTS-2.5` 逻辑入口。若环境只能用物理默认目录，先停进程，再以已验证的完整目录切换并保留同文件系统的原版目录。模型和插件均复核后才启动，避免在线覆盖 `gpt.pth`、逐文件热写和半更新状态。
3. **放量。** 启动后先走一条旧语和一条 HI 的真实业务图，核 /queue、目标 SHA、首尾音频和报警。沿用单实例或 ≤5% 流量，至少观察 24 小时和 100 条旧语、20 条 HI 请求。流量不足则保持阶段并补足，故障计入总分母。只有门槛满足才扩大。用实际业务时延预算约束 HI，旧语 p95 回归和 4 GiB 显存余量延续 E5。
4. **立刻停止。** 候选身份不匹配、任何 OOM、旧语误路由或严重音频缺陷即停新派单并回退。连续两个五分钟窗口旧语 p95 超过基线 1.20× 或错误率较基线上升超过 1 个百分点，同样回退。未达到最低样本数先观察原始计数，严重事件不等比例。
5. **回滚。** A 关 HI 路由，排空并关闭该 worker；若兼容插件影响旧语，将插件和配置回到原版本并冷启动。B 完整恢复原插件和原模型发布入口，冷启动清除候选缓存。用冻结基准 ZH 和实际业务的关键旧语跑真实图，回读原 GPT SHA、音频和队列，确认测试/候选进程退出。目标为启动回滚到恢复接单 ≤15 分钟，以演练实测为准。若超时继续隔离该实例，流量转健康原版并记录事故。

## 7. 交付与证据同步

每个 gate 的验收记录包含 `route`、候选与原版 SHA、插件/运行时 revision、测试计划 SHA、样本对数、门槛、实际值、失败数、报告哈希、负责人和 `PENDING/PASS/FAIL`。回滚另记开始/结束时间、恢复版本、冒烟任务与残留进程。每次运行写新记录，历史原始结果保留。

GitHub 记录冻结代码、兼容实现、验收方案和脱敏报告。Hugging Face 同步同一版验收方案和候选权重的不可变 revision、大小、SHA、许可、已通过项与待验项。现有 ZIP 始终对应已公开的纯 adapter。只有真实导出、审查模型/数据分发条件并验收后，才为新的完整 GPT 建立独立权重版本和模型卡。保留完整 Bilibili 模型许可、数据集归属、变更与非背书声明。日志、训练语料、参考音频与未获公开授权的试听文件保留在受控环境。

| 现状 | 状态 |
| --- | --- |
| 固定纯 adapter 公开 ZIP 与 H200 两句未合并合成 | 已核验 |
| CPU 纯 profile、tiny 官方模型导出/重载 | 8 项现有测试通过，使用小型随机模型 |
| 真实纯候选完整 GPT 导出、独立 CPU 对比与真实前向等价性 | 待执行 |
| 新候选 T8 兼容实现与 RTX 5090 原工作流 | 待实现与实测 |
| Hindi 独立听评、每语质量回归、canary 与回滚 | 待执行 |
| 生产换模结论 | 待验收和签收 |

[纯 LoRA H200 实测证据](https://github.com/allenflux/indextts25-hindi-h200/blob/2c804756d7ac786a803dae7694cff9d26b16bb7d/docs/acceptance/pure-lora-pilot48-20261118.json) · [公开权重完整性回执](https://github.com/allenflux/indextts25-hindi-h200/blob/2c804756d7ac786a803dae7694cff9d26b16bb7d/docs/acceptance/pure-lora-public-package-20261118.json) · [历史 5090 切换证据](https://github.com/allenflux/indextts25-hindi-h200/blob/2c804756d7ac786a803dae7694cff9d26b16bb7d/docs/RTX5090_SWITCHING.md) · [官方模型许可](https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/c39ce5ba981572cb187443877ff559dfb246ce63/LICENSE)
