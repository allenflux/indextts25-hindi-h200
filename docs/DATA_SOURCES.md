# 候选 Hindi 数据来源：只做访问预检

本次没有登录 Hugging Face、接受任何 gated 条款、读取语料行或下载音频。以下只核查公开模型卡/数据配置，不能据此宣称许可已经获得或音频质量、子集大小、目录存储架构经过验证。

| 来源 | 人工访问前置条件 | 公开元数据中 Hindi 配置 | 本项目处理前的阻塞 |
|---|---|---|---|
| [AI4Bharat Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) | 你本人登录 HF，阅读并接受共享账户信息/访问条件；不能由本项目代替接受 | config 名 `Hindi`，公开配置列出 Hindi train/test shards | 可见 schema 没有 speaker_id；须取得发布方可信说话人映射后，才能保证按 speaker 隔离 |
| [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r) | 同样需你本人登录并阅读/接受 gated 条件 | config 名 `Hindi`；公开 schema 有 speaker_id，audio 标注48 kHz | 实际行、speaker稳定性、文本/音频对齐与具体许可仍需有权访问后验证；先筛 Hindi、小样本听检再离线转码 |

Rasa 全库标称约 **388 GB**，不是本项目测量值。不要默认下载全库，不声称 Hindi 子集大小或流式/底层缓存架构已经验证。访问授权完成后，先用发布方当前文档核对 config/paths，明确只选 `Hindi`、限定 shard/行数和本地配额，再允许下载。任何数据加载工具都可能访问元数据或缓存 shard；不要把 streaming 一词当成零下载保证。

预检顺序：

1. 在浏览器打开上述官方数据页，由你本人核对并处理账户信息共享和访问条件。页面显示 gated 时停在这里；不尝试绕过，也不自动接受。
2. 记录 dataset repo 的不可变 revision、LICENSE/模型卡版本、允许用途、说话人同意依据。核查数据标注是否足以满足你的用途；仓库可访问不等于所有后续用途都获授权。
3. 对已授权的 Hindi 小批次核查 schema、sample rate、原始/规范文本、说话人键与源分割，再映射成本项目 raw JSONL。Rasa 不得用 gender、style、文件名猜测 speaker 身份；缺少可信映射就停止训练准备。
4. 原始数据页的 train/test 不保证符合本项目 speaker 隔离。检查全部 speaker 重叠，保留官方 test；若有重叠按明确策略重建 holdout 并记录差异，不能让 test 音频参与 prompt 或调参。
5. 保存 source/revision/原始 clip ID/许可证据索引在本地审计记录；音频转为本项目所需 PCM 格式后保留原始文件哈希与转换参数。最后运行 prepare/validate，再决定是否扩大。

目前没有内置 gated 下载器，也没有私人数据样例。全量容量预算、Hindi 子集大小和读取成本均待你完成访问预检后实测。

完成浏览器人工授权后，可按 HF 官方 [auth_check API](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.auth_check) 使用 `HfApi.auth_check(repo_id, repo_type="dataset")` 做只验证读取权限的预检；遇 GatedRepoError 停止，不自动接受条款。Rasa 默认配置可能跨语言，因此必须显式使用 `Hindi`，不能依赖默认 config。
