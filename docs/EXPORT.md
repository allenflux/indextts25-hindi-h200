# 原生GPT导出与实验性 Hugging Face 发布

用户已明确要求将当前候选上传GitHub和Hugging Face；本次准备发布**清楚标记experimental的冻结best100**。这是实验产物公开，不等于Hindi质量、旧语言或生产部署验收通过。公开[HF仓库](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental)已在用户本人网页登录会话中以`fluxa0223`为owner创建；当前仅有初始README，**权重尚未上传**。后续上传成功以实际回执、文件清单与远端哈希核验为准。[最新实测](../VALIDATION.md)、[有限test与旧语言stress](FINITE_ACCEPTANCE.md)、[分发边界](DISTRIBUTION.md)

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

操作记录（2026-11-17）：HF公开目标已创建为[`fluxa0223/IndexTTS-2.5-Hindi-best100-experimental`](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental)，已通过普通网页编辑器提交含“权重尚未上传”提示的完整模型卡，commit为`469daf1093966c58827dff5c7525e18e57000cb1`；匿名复核仍只有README和`.gitattributes`，权重上传待完成。该commit仅是模型说明，不是权重下载revision。此操作状态不写入长期模型卡正文；模型卡引导读者以实际Files、不可变revision及manifest核对产物。创建使用用户本人的现有网页登录会话；没有借用Pod默认的其他只读账户，也未改动共享凭据。接下来仅向该明确目标上传经审核的权重与公开白名单，不读取、打印或向聊天索取token，不覆盖不相关内容。任何中转也仅限已授权的模型发布文件，不包含私人听评ZIP或训练数据。

按HF官方[上传指南](https://huggingface.co/docs/huggingface_hub/guides/upload)与[仓库管理](https://huggingface.co/docs/huggingface_hub/guides/repository)接口，只提交独立目录的明确白名单；不要设置删除远端不相关文件的参数。[模型卡元数据](https://huggingface.co/docs/hub/model-cards#specifying-a-license)使用自定义bilibili许可，不标MIT。

最后核对远端owner、公开状态、提交SHA、逐文件大小/哈希、完整许可和实际模型卡。没有成功回执与远端核验，不称已上传或可下载；保留失败原因和本地发布包。人工Hindi听评及完整旧语言回归继续单独记录，代码/ASR不能冒充完成。
