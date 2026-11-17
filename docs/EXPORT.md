# 原生导出与未来 Hugging Face 发布

早期阶段仅发布原创训练准备代码；当前已在本地完成v1真实100次optimizer更新、完整GPT合并导出、stock单条feature FP32 parity和真实WAV生成，另有独立fresh v2在200次更新处保存checkpoint与adapter。**Hindi质量gate仍未通过**，这些产物是实验结果，不能称为通过验收的Hindi模型。实际范围与已知失败见 [VALIDATION.md](../VALIDATION.md)。没有创建Hugging Face模型仓库，也没有上传权重；未来HF owner为 **fluxa0223**。公开发布前必须通过 [评测与发布门槛](EVALUATION.md)。

外部社区参考后端训练两个heads，却不在其adapter中导出text_head；该限制仅适用于外部路线。原生v1已完整保存/合并两个head，完成真实预训练模型的严格重载、单feature logits对照与stock WAV生成。保存/加载一致性和WAV存在不能代替Hindi质量、人工试听或旧语言回归；不要把当前实验checkpoint标为已通过质量验收的Hindi adapter。

原始模型保留只读，Hindi adapter 单独版本化。导出后在 `$HI_H200_WORK/exports/hindi-VERSION` 创建全新白名单 bundle，仅放已审核的适配器参数、机器可读配置/原始模型 revision及必要哈希、完整许可证和声明、评测报告与模型卡。实际导出文件名须以锁定后端产物为准；本项目没有虚构现成 adapter。

模型卡应说明：非官方实验、Hindi 数据来源与许可、训练/评测范围、基线和 adapter 加载方式、所用原始权重与代码 revision、硬件/实际优化步数、真实资源记录、旧语言回归、已知失败场景、提示音频/个人数据不随模型发布。上游要求的原始版权/许可证、非背书声明与下游条款需随衍生物保留。原始权重是否允许重新分发需另外核实；优先只发布允许分发的 adapter。

最终在 HF 的目标可命名为 `fluxa0223/indextts25-hindi-adapter`，届时重新检查名称是否被占用、owner身份、公开权限与用户发布意图。使用 HF 官方 [上传指南](https://huggingface.co/docs/huggingface_hub/guides/upload) 与 [仓库管理指南](https://huggingface.co/docs/huggingface_hub/guides/repository) 的当前接口；只上传上述全新 bundle 的明确白名单文件。不要直接 upload_folder 整个项目、runs、features、checkpoints或数据目录。

上传前逐个审核文件及 git/LFS 历史，不含：私人语音/文字/manifest/特征/提示、访问token、授权证据中的个人信息、optimizer状态或无许可复制的代码。新仓库中不能保留曾提交的敏感历史；发现问题先停下处理。令牌仅使用平台凭据机制，禁止写进命令、文档或仓库。没有上传操作的返回结果就不能写“已上传”。

HF 上传不能依赖本地 .gitignore 防止泄漏；使用独立干净目录及 `allow_patterns` 精确相对文件白名单，参考 [upload_folder API](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder)。不设置删除远端文件的参数，发布时先核实 `whoami` 与目标 owner。

若未来使用参考 `extract_adapter.py`，它的默认语言/前端为Turkish，Hindi必须显式指定 `--lang hi --normalizer none --case none`；其他必要参数以锁定脚本 --help 为准。该提取器读取checkpoint使用不安全pickle兼容模式，因此只允许你自己训练且经过哈希核验的可信本地文件，禁止加载互联网上未知checkpoint。导出文件不是断点恢复checkpoint，且写入不是原子替换；先写独立本地目录、核验完成后再考虑发布。

原创原生adapter由`hi_h200_native`保存全部可训练张量，包括text/mel两个head；受限加载器绑定原始模型/LoRA规格并校验SHA，`weights_only=True`不是sandbox。微型CPU模型与真实v1预训练模型的导出/重载、单feature parity及stock波形生成已完成，详细范围见 [VALIDATION.md](../VALIDATION.md)。Hindi人工听评、更多held-out覆盖与旧语言回归仍未完成，质量gate未通过，不能据此直接发布。原生adapter不是社区格式，原生checkpoint与社区checkpoint不能互相resume。
