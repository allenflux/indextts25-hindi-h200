# 未来导出与 Hugging Face 发布

本次只发布原创训练准备代码到 GitHub；没有训练完成的模型或 adapter，没有创建 Hugging Face 仓库，也没有上传权重。未来 HF owner 为 **fluxa0223**。公开发布前必须先通过 [评测与发布门槛](EVALUATION.md)，不能用这个说明替代真实训练结果。

外部社区参考后端训练两个 heads，却不在其 adapter 中导出 text_head；该限制仅适用于外部路线。原生路线已保存/合并两个head，但仍需要真实预训练/合成/质量验收。必须在有授权的实现中解决保存/加载完整性，再对训练态与干净环境重载态进行固定 prompt/seed/token/logit/试听对照。不要直接把当前任意训练 checkpoint 标为可用 Hindi adapter。

原始模型保留只读，Hindi adapter 单独版本化。导出后在 `$HI_H200_WORK/exports/hindi-VERSION` 创建全新白名单 bundle，仅放已审核的适配器参数、机器可读配置/原始模型 revision及必要哈希、完整许可证和声明、评测报告与模型卡。实际导出文件名须以锁定后端产物为准；本项目没有虚构现成 adapter。

模型卡应说明：非官方实验、Hindi 数据来源与许可、训练/评测范围、基线和 adapter 加载方式、所用原始权重与代码 revision、硬件/实际优化步数、真实资源记录、旧语言回归、已知失败场景、提示音频/个人数据不随模型发布。上游要求的原始版权/许可证、非背书声明与下游条款需随衍生物保留。原始权重是否允许重新分发需另外核实；优先只发布允许分发的 adapter。

最终在 HF 的目标可命名为 `fluxa0223/indextts25-hindi-adapter`，届时重新检查名称是否被占用、owner身份、公开权限与用户发布意图。使用 HF 官方 [上传指南](https://huggingface.co/docs/huggingface_hub/guides/upload) 与 [仓库管理指南](https://huggingface.co/docs/huggingface_hub/guides/repository) 的当前接口；只上传上述全新 bundle 的明确白名单文件。不要直接 upload_folder 整个项目、runs、features、checkpoints或数据目录。

上传前逐个审核文件及 git/LFS 历史，不含：私人语音/文字/manifest/特征/提示、访问token、授权证据中的个人信息、optimizer状态或无许可复制的代码。新仓库中不能保留曾提交的敏感历史；发现问题先停下处理。令牌仅使用平台凭据机制，禁止写进命令、文档或仓库。没有上传操作的返回结果就不能写“已上传”。

HF 上传不能依赖本地 .gitignore 防止泄漏；使用独立干净目录及 `allow_patterns` 精确相对文件白名单，参考 [upload_folder API](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder)。不设置删除远端文件的参数，发布时先核实 `whoami` 与目标 owner。

若未来使用参考 `extract_adapter.py`，它的默认语言/前端为Turkish，Hindi必须显式指定 `--lang hi --normalizer none --case none`；其他必要参数以锁定脚本 --help 为准。该提取器读取checkpoint使用不安全pickle兼容模式，因此只允许你自己训练且经过哈希核验的可信本地文件，禁止加载互联网上未知checkpoint。导出文件不是断点恢复checkpoint，且写入不是原子替换；先写独立本地目录、核验完成后再考虑发布。

原创原生adapter由`hi_h200_native`保存全部可训练张量，包括text/mel两个head；安全加载器绑定原始模型/LoRA规格并校验SHA。微型CPU模型导出/重载已测试，但真实预训练模型的波形生成、Hindi与旧语言质量仍待验收，不能据此直接发布。原生adapter不是社区格式，原生checkpoint与社区checkpoint不能互相resume。
