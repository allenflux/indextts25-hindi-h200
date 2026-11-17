# 原生GPT导出与实验性 Hugging Face 发布

用户已明确要求将当前候选上传GitHub和Hugging Face；本次已发布**清楚标记experimental的冻结best100**。这是实验产物公开，不等于Hindi质量、旧语言或生产部署验收通过。公开[HF仓库](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental)已由用户网页上传13文件，固定revision为`117c7922e4dcedc3f9885f3f6efa5be8f54c7919`；匿名核验已通过远端inventory、GPT LFS元数据和12个小文件实际GET/hash。[最新实测](../VALIDATION.md)、[有限test与旧语言stress](FINITE_ACCEPTANCE.md)、[分发边界](DISTRIBUTION.md)

## 当前实际候选

最新pilot从原始官方2.5独立训练，按固定验证loss选择100步；500步的后续过拟合状态保留，不用test重选。完整checkpoint转标准adapter、7文件完整GPT合并、H200单feature FP32 parity及真实stock生成已完成。后续冻结test严格CER仍为50%，中文/日文单例Hindi-prompt stress出现识别差异，人工母语试听未完成。

原生导出保留text/mel两个head和全部可训练状态，原始模型保持独立。标准adapter、完整GPT、训练checkpoint用途不同，不能互相当作恢复文件，也不与社区格式互通。早期v1/v2只是历史诊断；不能用它们的状态替代当前best100或合并累计训练步数。

## 另建干净发布目录

保留原始bundle、checkpoint及其完成记录不动；不要将整个workspace、runs、features或数据目录上传。逐文件建立公开白名单：

- 已验证且SHA与冻结best100一致的完整`gpt.pth`；如发布adapter，须明确为本项目原生格式并单独核验。
- 完整原始版权与`LICENSE`、`LICENSE_ZH.txt`、`MODEL_LICENSE`、`NON_ENDORSEMENT.txt`。
- 将已审阅的[中文模型卡模板](MODEL_CARD.zh.md)复制为发布目录`README.md`，保留全部已知失败与评测范围。
- 仅包含可公开字段的来源/配置/哈希摘要、实际文件清单；如另附配置或tokenizer，核对固定来源及其分发条件。它们不能把一个GPT文件变成完整推理包。

原始导出元数据可能包含本地路径或数据身份，不可不经审核直接上传。只保留公开必要字段，移除私人路径、clip/speaker标识和访问材料；清理后为发布目录**另建**哈希清单，不伪称改变后的元数据仍是原bundle的COMPLETE。GPT二进制必须与冻结候选哈希一致。

公开包不包含原始或生成音频、语料分片、提示、私人转写/manifest/特征、optimizer/RNG恢复状态、认证凭据或无许可代码。用户私下听评包与公开模型仓库分开处理，不能依赖`.gitignore`代替白名单审核。

## 发布条件与真实阻塞

实验公开可在适用许可、明确发布意图及必要审核满足后进行；人工质量/生产推荐gate未过时，须醒目标明限制，不能宣称正式Hindi支持、可安全生产替换或旧语言无退化。**不能只因质量gate未过就把用户已授权的诚实实验发布写成绝对禁止。**

许可条件见[DISTRIBUTION](DISTRIBUTION.md)：核对商业适用、必要第三方权利，保留完整协议/版权/非背书声明，并以适当条款约束下游。商业条款适用性的确认保存在私下审核，不公开用户财务信息；不重复索取已完成的事实确认。数据访问已同意不是所有分发权的证明，缓存卡缺少额外gated字段也不证明平台无其他条件；未观察到禁止条款时，不虚构新禁令或自动要求作者另行许可。

实际阻塞应具体记录：例如认证缺失/owner不匹配、平台拒绝上传、触发许可证条件但缺必要许可、发布包含私人材料或哈希不一致。未知事实列为待核，不能写成已触发禁令或已获全部授权。

## GitHub与HF操作顺序

GitHub保存非敏感原创代码、模型卡、安装说明和验证摘要，并链接实际HF产物；不把训练数据或多GB模型混入普通代码提交。

操作记录（2026-11-17）：已通过网页向[`fluxa0223/IndexTTS-2.5-Hindi-best100-experimental`](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental/tree/117c7922e4dcedc3f9885f3f6efa5be8f54c7919)上传冻结13文件，commit为`117c7922e4dcedc3f9885f3f6efa5be8f54c7919`。匿名核验确认`private=false`、14项inventory（13文件加原`.gitattributes`）；首次核验实际GET的12个小文件为963,442 bytes，另API响应2,585 bytes，helper总计966,027 bytes；GPT当时只按LFS元数据核对size/SHA。历史`469daf1093966c58827dff5c7525e18e57000cb1`仅为说明元数据commit，不是权重下载revision。完整核验范围见[PUBLIC_RELEASE](PUBLIC_RELEASE.md)。

创建与上传使用用户本人网页登录会话，没有借用Pod默认其他账户或改共享凭据。长期模型卡及冻结13文件保持原样；私人听评ZIP、训练数据和认证材料未上传。

按HF官方[上传指南](https://huggingface.co/docs/huggingface_hub/guides/upload)与[仓库管理](https://huggingface.co/docs/huggingface_hub/guides/repository)接口，只提交独立目录的明确白名单；不要设置删除远端不相关文件的参数。[模型卡元数据](https://huggingface.co/docs/hub/model-cards#specifying-a-license)使用自定义bilibili许可，不标MIT。

首次远端核验退出0，状态为`verified_remote_metadata_and_12_small_files`，当时未下载完整GPT。后续已在5090宿主的新独立工作目录使用公开`download_public_release.py`（源码4c9dd0e），匿名SDK固定上述HF revision完成13文件下载，回执`complete=true`；GPT 3,259,538,991 bytes的完整流式SHA与冻结423a8491…值匹配，首次确认完成时，距任务启动约504秒（未测量精确下载耗时），未使用token或GPU。详情见[PUBLIC_RELEASE](PUBLIC_RELEASE.md)。人工Hindi听评与完整旧语言回归继续独立记录；当前固定ComfyUI版本的HI/原始文件清单双阻塞仍在，节点验证准备进行中，尚未运行节点GPU合成。
