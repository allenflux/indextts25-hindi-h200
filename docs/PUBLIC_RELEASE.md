# Hindi best100 公开实验模型发布记录

本页记录公开格式的实际验证与发布进度。冻结best100仍是**小数据实验候选**：两条Hindi test生成的严格CER为50%，中文/日文单例Hindi-prompt stress指标升高，人工母语听评与生产质量gate未通过。用户已明确授权实验公开；该授权不把候选变成生产替换或官方Hindi模型。方法与限制见 [有限验收](FINITE_ACCEPTANCE.md)、[模型卡](MODEL_CARD.zh.md) 和 [分发边界](DISTRIBUTION.md)。

## 当前发布状态

| 环节 | 已实际确认 | 尚待确认 |
|---|---|---|
| GitHub代码 | 公开源码`36dffe21c0793b1544898e6bcb11f1d34af542a3`已推送并核对远端SHA | — |
| CPU与审查 | Mac完整suite **246/246通过、0跳过、37.376秒**；独立审查通过 | 不代表任意环境兼容或音质通过 |
| HF发布 | 冻结13文件网页发布；匿名元数据/12小文件核验及后续完整SDK下载均通过 | 下载与哈希通过不等于质量或节点兼容通过 |
| 本地公开包 | 精确13文件目录准备、完整ZIP与本地payload验证通过 | 下载者仍应独立校验本地完整文件 |
| 公开格式加载 | metadata preflight退出0，H200实际完整stock加载和生成退出0 | 不代表ComfyUI节点可直接替换 |

HF实验权重已发布至 [fluxa0223/IndexTTS-2.5-Hindi-best100-experimental](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental/tree/117c7922e4dcedc3f9885f3f6efa5be8f54c7919)，网页上传commit为 **`117c7922e4dcedc3f9885f3f6efa5be8f54c7919`**。首次匿名核验退出0，状态`verified_remote_metadata_and_12_small_files`：`private=false`，14项inventory为冻结13文件加原`.gitattributes`，没有额外私人产物。

首次核验的GPT远端LFS元数据为 **3,259,538,991 bytes**、SHA256 `423a8491b8856c7c3075254b12defc3ae47dbd9e0de285772523ddd4e9a97021`，与冻结候选一致；12个小文件实际匿名GET并逐hash匹配，文件内容共 **963,442 bytes**。另有API响应 **2,585 bytes**，helper的`total_verified_payload_bytes=966027`是两者合计，不是12个模型文件的大小。该次只核验GPT LFS元数据，未下载其完整内容。

后续已在5090宿主的全新独立工作目录使用[`scripts/download_public_release.py`](../scripts/download_public_release.py)（源码`4c9dd0e10790530f31b756d991981bd4d3116610`）匿名SDK下载固定HF revision `117c7922e4dcedc3f9885f3f6efa5be8f54c7919`。实际完成回执为`complete=true`、13文件，GPT为 **3,259,538,991 bytes**，完整流式重算SHA与上述冻结值一致；首次确认完成时，距任务启动约 **504秒**（未测量精确下载耗时）。这是完整payload下载/校验的实际证据，未使用token、未触及GPU或启动训练；不再仅有LFS元数据核验。

历史`469daf1093966c58827dff5c7525e18e57000cb1`仅提交pre-release说明元数据，不是权重下载revision。此次上传使用冻结13文件原样README（8747 bytes），远端12个小文件hash已与原manifest匹配；未更改长期模型卡或冻结发布包。

此前Chrome显示完成的本机ZIP仅2,277,806,075字节，整包SHA不符且缺EOCD；原不完整文件及后续截断片段均保留未改。按用户明确授权，仅补缺失字节并在CodexData另建完整副本：10个canonical parts均通过真实payload校验，固定偏移合并成功；新ZIP实际读回 **3,260,503,973 bytes**，SHA256为`b0df1c597222b78a4aa87cc6ee9a4d167ad12aa91706cff651bd403ca64618db`。恢复中两份100 MiB下载块也截断，保留正确前缀，仅补455187/504339字节尾部；其他缺失块经用户指定中转，各外包带完整许可、归属和模型卡，不含私人语料。

完整ZIP的13个普通ZIP_STORED文件通过安全路径与CRC检查；流式复制至全新目录后，`prepare_hf_release --verify-release`实际返回`status=verified_public_release`、`files=13`，GPT SHA与冻结best100一致，`checkpoint_step=100`、`quality_gate_passed=false`、`uploaded=false`。这是上传前的本地验证回执，`uploaded=false`只表示该次本地检查没有执行上传；随后已由用户网页手选13文件并完成上述HF发布。此前自动chooser停重试、CLI未启动，上传成功由网页commit与匿名远端核验另行确认。

[恢复工具](../scripts/recover_public_download.py)仅负责本地固定偏移的分段与新副本合并，不负责网络或登录；它强制检查源包、已下前缀和最终整包的固定SHA，并预检磁盘空间。恢复工具7项、本地接收器9项小文件检查分别通过；它们与此次真实大包读回/解包验证分开记录，不是模型质量或新增GPU训练证据。[该版本Kite处理器](https://github.com/kite-org/kite/blob/7eddcc69fbadf0772cd187b9b75cba7b83cab1f9/pkg/resources/pod_handler.go#L401-L443)未实现HTTP Range，补尾采用独立小包；当前没有证据确定原传输截断的原因。 源端中转batch 7/7无错误，我方控制器已不活跃，ComfyUI健康检查HTTP200；恢复过程没有新增GPU工作。

## 13文件格式的真实集成

公开候选`gpt.pth`为 **3,259,538,991 bytes**，冻结SHA256为`423a8491b8856c7c3075254b12defc3ae47dbd9e0de285772523ddd4e9a97021`。公开目录只保留已审核的13文件白名单，包括模型、固定配置/tokenizer、许可、模型卡、清单及非敏感评测摘要；不包含原始/生成音频、语料、私人文本、特征、私有provenance、optimizer/RNG或凭据。精确清单与下载方式见 [安装说明](INSTALL_EXPERIMENTAL.md#hf-公开-13-文件包下载与独立加载)。

[`scripts/synth_public_release.py`](../scripts/synth_public_release.py)已实际验证这个13文件目录：metadata模式退出0，H200 BF16完整stock构造与生成退出0，耗时 **74.64秒**，真实WAV为 **2.3452154195秒**。无stop reason、没有新训练，训练所在Pod localhost的ComfyUI健康检查为 **HTTP200**。本次复用此前的新短句，只证明公开格式加载与推理接通，不新增独立质量样本，不重选best100，也未提交ComfyUI queue。

公开入口复用既有离线、隔离缓存、资源和输入门槛，将候选GPT路径设为独立绝对配置，并绑定固定官方20文件合成资产。**13文件GPT公开包不是完整声学模型包**；它不含本地原始导出的私有`COMPLETE.json`/`provenance.json`，入口不会伪造这些记录。不能将它直接交给仅接受原始本地bundle格式的旧入口。操作与两种格式的边界见 [安装说明](INSTALL_EXPERIMENTAL.md)，导出流程见 [EXPORT](EXPORT.md)。

## 当前ComfyUI节点的实际阻塞

实际检查的T8节点固定版本为`1ff425ece89ef9824f58f0e4dbfb057991814e88`。其CPU meta结构检查已通过：**456/456 tensors**，missing、extra及shape mismatch均为0，strict加载通过。该检查只证明CPU meta设备上的key/shape兼容，没有运行节点GPU合成或ComfyUI queue。

两项独立阻塞仍存在：UI/运行时语言白名单拒绝`HI`；节点的原始GPT固定大小与哈希清单拒绝候选。关闭SHA检查也不解除大小检查。因此，**不能直接覆盖当前ComfyUI原GPT**；独立stock公开加载成功不解除这些节点限制。原始模型保持不动；节点验证准备仍在进行，尚未运行节点GPU合成。目标保持workflow结构，language参数可调整，不能据HF发布推断节点集成已完成。

另对该宿主原资产目录的26项作只读哈希审计：25项匹配，唯一差异为`config.yaml`，语义差异仅`version: 2.0`到`2.5`。隔离测试目录改用已固定的官方2.5配置，原目录未改；此准备检查不构成节点GPU成功证据。

## 发布收口边界

已完成首次owner/公开状态/revision/inventory、GPT LFS及12小文件核验，随后又完成独立匿名SDK的13文件完整下载与GPT流式哈希校验。实验标识、Hindi 50% test CER、旧语言stress差异及人工未完成等限制必须随权重保留。

先前公开包只读收尾检查确认训练与stock合成任务均已退出，ComfyUI健康检查返回 **HTTP200**；节点验证准备仍在进行，GPU合成尚未执行。

此前两份私有评审/许可ZIP已由用户自行下载到Mac并逐hash验证，仍不公开，不传HF或Library；它们与当前13文件公开模型目录分开。当前未改全局依赖、未替换生产模型、未接受新数据访问条款，也不公开认证材料、私人音频/文本或财务确认。

## 固定许可的逐项复核（2026-11-17）

另行只读核查了官方[新revision的LICENSE](https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/2d8d061a6e922c94c3a5d3f81c986e9e2e260c76/LICENSE)。它与本候选实际固定base `c39ce5ba981572cb187443877ff559dfb246ce63` 的[LICENSE](https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/c39ce5ba981572cb187443877ff559dfb246ce63/LICENSE)逐字节相同，并未新增正文条件。新revision仅作比较证据；没有替换冻结13文件包中的许可或改变本候选的base来源。

| 官方来源/实际文件 | 字节数 | SHA256 | 比较结果 |
|---|---:|---|---|
| HF `2d8d061…` LICENSE | 10,553 | `cc7da9ea0f8a97ef15ab3bf0389e636ce79ffca1aef5489520796ac87d87a87b` | 与HF `c39ce5b…`完全相同 |
| HF `c39ce5b…` LICENSE / 包内MODEL_LICENSE | 10,553 | `cc7da9ea0f8a97ef15ab3bf0389e636ce79ffca1aef5489520796ac87d87a87b` | 固定模型许可 |
| [源码d9e41aa…英文LICENSE](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/LICENSE) / 包内LICENSE | 10,554 | `53875532237e0a97b17a41721556ff57e93193fc566e48c3a0febe6e35bb5343` | 比HF许可只多末尾LF，正文相同 |
| [源码d9e41aa…中文LICENSE_ZH.txt](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/LICENSE_ZH.txt) / 包内同名文件 | 7,336 | `366175ecf329274358801b8365922677c52f25e643dcf00f0bc15857b5b2777f` | 固定中文原文 |

本机已独立读盘复核上述三份包内许可及251字节的`NON_ENDORSEMENT.txt`；非背书文件SHA256为`ff80f462446489711a2ae9c828b1d933d6224261225b154df8265c9b86ae48d6`，内容与4.1(a)规定的完整英文声明完全匹配。实际13文件包也已由准备工具按固定上游pins及逐文件哈希通过校验。

- **3.4(a/b)：** 完整协议及版权材料随包保留，协议约束下游；后续分发者仍须通过适当条款落实这些义务。文件核验不保证所有未来下游行为。
- **3.4(c)、4.1(a/b)：** 当前为原模型自身的微调衍生品，完整非背书声明已附带；不因此获得改进任意其他商业AI的许可。引入第三方数据/权重仍须具备相应用途授权。
- **第9节及中文2.2：** 中文优先；上月MAU超过1亿或上年收入超过人民币1亿时须单独许可。商业条款适用性的确认保存在私下审核，不披露个人财务信息。

实际`DATA_ATTRIBUTION.md`为966字节，保留AI4Bharat、作者/论文、数据来源、固定revision、CC-BY-4.0链接及选择/重采样/文本规范化/微调改动。该[版权许可](https://creativecommons.org/licenses/by/4.0/legalcode.en)提供有条件分享与改作依据，不涵盖全部隐私及人格权。已有操作者本人处理访问条件、认证读取成功的记录；固定数据卡未发现`extra_gated_*`字段，不等于平台没有额外条件。访问成功不能代替分发权审核，也不能在缺乏具体限制证据时反推一律禁止公开实验GPT或必须另向作者申请许可。

本轮只分发GPT、固定配置/tokenizer及许可/卡/公开摘要；W2V、CAMPPlus、BigVGAN权重没有附带，仍从固定官方base资产引用并遵守各自适用条款。这不额外授予辅助权重或声音的使用权。当前未发现新revision带来的新增许可阻塞；本次上传与远端核验范围按上文实测记录，人工听评和生产质量gate独立记录。
