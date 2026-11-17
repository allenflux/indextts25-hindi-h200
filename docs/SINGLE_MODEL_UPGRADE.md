# 11. 单模型默认替换：证据、升级方案与上线门槛

更新日期：2026-11-18（Asia/Saigon）。用户最新目标是只部署一套我们的完整候选模型，给同一 T8 插件做兼容升级；保留原工作流节点、连线、`model_name="IndexTTS-2.5"` 和空 `custom_model_path`，仅在业务需要时选择 `language=HI`。本节取代第10节“双目录日常路由”作为拟推广目标；第10节保留已验双目录方案与历史回滚说明。**本轮收集证据并准备离线原型，没有部署、重启或替换现网。**

## 11.1 当前可以下什么结论

工程上可以把合并后的 GPT 做成唯一默认模型，不需要运行时 LoRA 节点，也不依赖原版 GPT 驻留。声学后端等资产仍须匹配，完整 T8 目录是26项资产，候选清单payload共8,290,912,900 bytes（约8.29 GB），HF的13文件发布包本身不够。清单中25项与固定base的size/SHA身份相同，变化项是GPT；这不是按参数数量或音质计算的比例。保留旧选择名称只是兼容工作流字段，加载日志、manifest和缓存身份必须明确显示真实实验候选，不能伪装成官方权重。

**目前证据不足以承诺候选对原支持语言“大致一致”或无损，因此方案是有条件升级，尚不建议全量替换生产默认。** 未改变的声学后端不代表最终声音不变；共享 GPT 的更新会影响其他语言。CPU原型检查、原候选HI已能生成、原版回切成功分别是不同证据。

## 11.2 哪些保留、哪些改变

| 部分 | 实际情况 | 对“其他语言还是原模型”的含义 |
|---|---|---|
| 匹配的 codec、声学 CFM/DiT、BigVGAN、speaker/emotion 前端 | 沿用固定原版资产；未微调这些模块 | 后端相同，不保证它接收到的语义codes相同 |
| GPT attention | 24层的共享 `c_attn` / `c_proj` 已合并 rank8、alpha16 LoRA | ZH/EN/JA/ES/AR与HI都经过这些共享投影 |
| GPT语言 embedding | 训练只开放Hindi第17行 | 其他语言行受保护，也不能抵消共享层变化 |
| GPT文字 embedding | 训练实际出现的token行允许更新，可能含共用标点/token | 不能视为纯Hindi私有词表 |
| `text_head` / `mel_head` | 完整矩阵和bias参与训练并合并 | `mel_head`预测语义codes；旧语言也使用它 |

依据是实际[训练与合并实现](MODEL_RATIONALE.md)，不是仅凭LoRA通常如何工作推测。GPT文件大小接近、相同架构或大部分参数冻结均不能推导音质接近。

### 本轮真实权重审计

已对现有两份固定GPT执行一次只读CPU比较，不下载权重、不构造TTS、不执行GPU。先后均核验完整文件SHA和文件状态；使用 `weights_only=True`、CPU `mmap`、单线程和8MiB分块，按原始字节比较，不把正负零或NaN位模式归一化。审计源码固定为 [e193cab](https://github.com/allenflux/indextts25-hindi-h200/blob/e193cabca88bb1186ee1001820d5068c1b17b787/scripts/audit_merged_gpt.py)。

两份各456个张量，key/shape/dtype全部相同；**402个逐字节相同，54个有变化**：

| 实际分组 | 逐字节相同 | 有变化 |
|---|---:|---:|
| 24层共享attention投影weight | 0 | 48 |
| attention投影bias | 48 | 0 |
| 语义code head与文本辅助head（weight/bias） | 0 | 4 |
| 语言与文字embedding矩阵 | 0 | 2 |
| MLP、norm、位置、mel embedding与其余conditioning/state | 354 | 0 |

语言embedding共107行，仅Hindi第17行变化，其余106行逐字节相同；文字embedding共60,510行，69行变化、60,441行相同。审计只报告行计数，没有从这些行数推断它们全部属于Hindi或证明旧语言语义不变。张量数也不是参数数量/数值幅度占比，更不是音质百分比。

[完整只读审计JSON](acceptance/gpt-parameter-audit-20261118.json) 为2,723 bytes，SHA256 `d95482eb80d5434da7a9449f81738056121925600ebbf9bcedb253431625ea22`；已由正常Kite文件预览独立回读，本地同SHA。审计计时17.157秒（CPU，非推理耗时），`cuda_initialized=false`、`quality_verified=false`。结束时自建审计进程为空，原服务HTTP200、队列0/0；GPU仍654MiB已用、31,457MiB可用、0%利用率。未改生产权重、插件或服务。


## 11.3 已有真实音频对照说明了什么

以下严格CER从公开冻结 `evaluation.json` 重新核对，并与本地公开包 `SHA256SUMS` 一致，文件SHA256为 `4bf22df2ffd4e94139e490f6fcd179e752b78f1773143a3a140adb6feecfd139`。指标定义为原定canonical文本的NFC/空白规范化后codepoint编辑率，包含空格；不是母语人工评分。[固定HF评测文件](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental/blob/117c7922e4dcedc3f9885f3f6efa5be8f54c7919/evaluation.json)

| 语言 | 原版严格CER | 候选严格CER | 证据范围 |
|---|---|---|---|
| 中文ZH | 1/17 = 5.88% | 3/17 = 17.65% | 仅1例，Hindi参考音频 |
| 英语EN | 1/46 = 2.17% | 1/46 = 2.17% | 仅1例，Hindi参考音频 |
| 日语JA | 0/18 = 0% | 2/18 = 11.11% | 仅1例，Hindi参考音频 |
| 西语ES | 1/60 = 1.67% | 1/60 = 1.67% | 仅1例，Hindi参考音频 |
| 阿语AR | 0/41 = 0% | 0/41 = 0% | 仅1例，Hindi参考音频 |
| Hindi HI | 396/166 = 238.55% | 83/166 = 50% | 2位held-out speaker各1例；真人reference CER为31/166=18.67% |

中文、日文单例ASR指标变差是进一步检查的信号，不能自动认定TTS发音退化；其他三个单例相同也不能推广为普遍不变。旧五语言没有native-prompt校准或人工母语听评，Hindi reference不能校准其他语言。所有原版/候选输入、prompt、seed和设置配对已核对，每个case只执行一次；保留全部结果，没有按test重选best100。[完整有限验收](FINITE_ACCEPTANCE.md)

RTX5090三轮原版ZH→候选HI→原版ZH已真实完成，首尾原版WAV及PCM哈希相同；**这证明回到原模型恢复该固定短句，不证明候选用ZH输出与原版相同。** 本轮不把这项结果挪作单模型旧语言无损证据。[实际三轮回执](acceptance/rtx5090-switch-cycle-20261117.json)

## 11.4 单模型升级的离线工程原型

已实现并CPU核验单候选原型：新工具从真实固定T8 `1ff425ece89ef9824f58f0e4dbfb057991814e88` 创建全新副本，`prepare`及`verify-copy`成功；标准默认路径、原模型名/空自定义路径、26项强制校验、legacy参数和6个禁下载入口均做过CPU检查。真实公开13文件目录作为负例被拒绝（缺少T8清单中的 `bpe.model` 等资产），没有把半套包当完整模型。

完整CPU suite为333/333通过、0跳过、37.916秒，对应新工具SHA前缀 `93ea6be1` 的快照。随后仅修正下载器候选身份文案和一处源文件行末空白；最终工具SHA256为 `8dee60ae1eb3f7e479364c58e2dc7c675cf6a46b49dad1895dcb42b5cee3dbc7`，相关22/22测试另通过、0.974秒，并对真实固定源码的fresh003完成校验。**没有把最终小幅改动的定向通过写成又完整重跑333项。** 权重审计器另经独立7/7真实tiny Torch CPU测试；这些测试不冒充GPU或音质验收。

新副本marker `T8_SINGLE_MODEL_PROTOTYPE.json` 明确 `cpu_prototype_only=true`、`gpu_validated=false`、`old_language_quality_verified=false`、`production_deployed=false`。此前已通过的GPU三轮属于另一个双目录兼容profile，新单候选默认profile尚未做GPU queue检查或生产部署。


原型准备器为 `scripts/prepare_t8_single_model.py`，只从固定T8源码创建新的独立代码副本，不部署服务、不下载模型、不执行GPU。与已做三轮GPU检查的双目录准备器分开；不能把此前GPU通过直接写成这个新默认路径已验。

目标行为如下：

- 原 `T8_IndexTTS25_ModelLoader` / Generate等节点类型、ID、连线及旧输入字段保留；默认 `model_name` 仍是 `IndexTTS-2.5`，空 `custom_model_path` 解析标准候选目录。
- 所有语言使用同一固定 best100 manifest与真实候选身份，不依据GPT大小回退原模型。额外搜索根或备份不能被静默发现成默认模型。
- `verify_hashes=false`、旧位置参数和跳过辅助资产均不能跳过完整26项校验；原GPT、缺失资产、同大小错误文件必须失败。
- 原工作流的 `download_missing=false` 保留。原型暂不支持自动下载/修复；为防误取原GPT，节点、下载CLI和直接下载入口均提前拒绝该操作。需要完整离线准备资产，不能称所有下载功能完全兼容。
- 日志明确记录候选repository、revision和GPT SHA；同路径升级必须维护重启到新进程，不能以旧内存或图缓存的输出当作新模型已加载。

```bash
python scripts/prepare_t8_single_model.py prepare \
  --node-root "$T8_SOURCE" --destination "$STAGED_T8_SINGLE" \
  --candidate-revision 117c7922e4dcedc3f9885f3f6efa5be8f54c7919
python scripts/prepare_t8_single_model.py verify-copy \
  --node-root "$STAGED_T8_SINGLE" \
  --candidate-revision 117c7922e4dcedc3f9885f3f6efa5be8f54c7919
python scripts/prepare_t8_single_model.py verify-assets \
  --node-root "$STAGED_T8_SINGLE" --model-dir "$CANDIDATE_DIR" \
  --candidate-revision 117c7922e4dcedc3f9885f3f6efa5be8f54c7919
```

`T8_SOURCE`为固定原始Git根；`STAGED_T8_SINGLE`为扫描路径外全新目录、父目录已存在；`CANDIDATE_DIR`为已有完整候选26资产目录。上述命令不准备生产路径或自动迁移任何文件；代码核验通过不等于资产或音质通过。

## 11.5 条件满足后如何上线

以下是拟议的受控推广流程；恢复业务默认必须同时满足11.6预先冻结的质量门槛。隔离或小流量的queue功能成功不能替代质量验收；门槛未过时，现网默认继续保留原版。

1. 在维护前完成代码/26项资产校验、许可归属和输入检查。部署时只保留一个活动T8插件；原插件和原GPT若保留冷备，应放到扫描与模型发现路径外，不需要双模型常驻。
2. 暂停目标实例新派单、等待running/pending为0，核实际GPU/RAM/磁盘；用已核对的现有服务管理方式停止旧进程。不得向仍运行的服务原地覆盖权重或插件。
3. 把已验候选完整目录安装到该实例标准 `models/TTS/IndexTTS-2.5`，同一T8目录安装兼容副本，再由既有启动器启动并刷新schema。不是只扔入一个GPT或覆盖整个ComfyUI；不全局升级Torch/驱动，也不复制隔离验收bootstrap到生产。
4. 用原工作流、空custompath和旧model_name真实提交旧语言及HI短样本，核定最终WAV、实际候选SHA/revision、队列history和缓存指纹；候选→候选跨语言切换全过程不应加载原GPT。只有这步完成才可称新默认路径已GPU验收。
5. 质量门槛通过且小流量验收完成后再恢复候选默认派单。业务请求只按语言改 `language`（HI/原语言），不再日常切换两个模型目录；正常业务text/reference仍按原工作流提供。

若加载失败或质量不达标，暂停派单并通过既有启动器停止新进程，恢复已核原插件、原模型目录与原工作流引用后重启并验证。没有自动回原模型；只保留一个候选且完全丢弃原文件，会失去快速回滚，必须先解决恢复来源。

## 11.6 还缺什么证据才适合全量替换

目前已有证据足以编写和CPU验证升级原型，**不足以批准所有旧语言的默认模型替换**。下一轮应在生成前冻结方案：五种原语言各选短句、数字/标点句、较长分段句各1条，共15对原版/候选；每种语言使用来源/使用权清楚的native-language参考，成对固定text、reference、seed和全部采样配置，不挑好例、不重选checkpoint。缺少合法母语参考时记录缺口，不能拿Hindi参考补齐后称母语回归。

这15对只是最低限度的定向诊断，不是统计上的普遍等价证明。保留所有音频；ASR同时检查合法真人参考以识别识别器偏差，并由相应母语审听者检查发音、漏字/重复、自然度、音色和句尾完整性。接受标准应在看新输出前固定；不能依据候选碰巧好看的样本临时放宽，也不能把参数相同比例、loss、CER单一数值或输出文件存在当作无损证明。

另需一次新单模型profile的原工作流RTX5090真实queue验收，包括原 `download_missing=false`、空路径、cache、正常音频URL，以及实际会用的加密/情绪策略。当前这些新profile的GPU/母语质量门槛均未通过。本轮不追加训练；保留冻结best100，先补证据再决定是否推广。
