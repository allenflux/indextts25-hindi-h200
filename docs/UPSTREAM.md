# 上游版本、接口与授权记录

核查日期：2026-11-17。`upstream.lock.json` 是源版本与接口审计的锁，不是完整 CUDA/Python 二进制环境锁。更换任意上游 commit 后必须重新审计，不能只改 SHA 绕过字段检查。

| 来源 | 固定 commit | 作用 | 授权状态 |
|---|---|---|---|
| [官方 IndexTTS](https://github.com/index-tts/index-tts/tree/d9e41aac89fd00b3d71497fddb287b7f24613712) | `d9e41aac89fd00b3d71497fddb287b7f24613712` | 模型、2.5 tokenizer/推理后端 | 自定义 bilibili Model Use License Agreement，需自行阅读 |
| [社区训练接口参考](https://github.com/oberkayb/index-tts-2.5-ft/tree/abdcbda0f6624d87134da8d1f4d482edac4a2fb9) | `abdcbda0f6624d87134da8d1f4d482edac4a2fb9` | GPT LoRA 与特征训练接口 | 锁定树没有 LICENSE；本仓库不复制源码或配置，不提供许可 |

官方 [tokenizer](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/utils/tokenizer.py) 的语言表按插入顺序编号，Hindi `hi` 的零基索引是 17，已分配 `<|hi|>`。官方 [README](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/README.md) 并未把 Hindi 列为经过训练验证的语言。不能据语言槽位推断 Hindi 质量。

社区接口事实来自固定版本 [train.py](https://github.com/oberkayb/index-tts-2.5-ft/blob/abdcbda0f6624d87134da8d1f4d482edac4a2fb9/scripts/train.py)、[配置参考](https://github.com/oberkayb/index-tts-2.5-ft/blob/abdcbda0f6624d87134da8d1f4d482edac4a2fb9/configs/turkish.yaml)、[lang.py](https://github.com/oberkayb/index-tts-2.5-ft/blob/abdcbda0f6624d87134da8d1f4d482edac4a2fb9/itts25ft/lang.py)、[textfront.py](https://github.com/oberkayb/index-tts-2.5-ft/blob/abdcbda0f6624d87134da8d1f4d482edac4a2fb9/itts25ft/textfront.py)、[check_setup.py](https://github.com/oberkayb/index-tts-2.5-ft/blob/abdcbda0f6624d87134da8d1f4d482edac4a2fb9/scripts/check_setup.py)、[pipeline README](https://github.com/oberkayb/index-tts-2.5-ft/blob/abdcbda0f6624d87134da8d1f4d482edac4a2fb9/pipeline/README.md) 与 [pipeline 配置说明](https://github.com/oberkayb/index-tts-2.5-ft/blob/abdcbda0f6624d87134da8d1f4d482edac4a2fb9/pipeline/pipeline.example.toml)。本仓库只按接口事实编写原创工具。

- train CLI 使用 `--config FILE`，但其 required 参数在加载配置之前检查，因此仍须加 `--output-dir DIR`。未知 YAML 字段会被静默忽略；本项目生成器检查全部配置 key。
- `--normalizer none --case none --no-pronunciation` 用于已经人工规范化的 Hindi 文字。仍会经过官方标点映射，冒号等可能变化；训练/推理须保持一致。
- 分别预处理三个已拆分的 raw manifest，均用 `--val-size 0 --val-speakers 0`，避免社区脚本重排 speaker；各输出全量文件都叫 `utterances_train.jsonl`。工具会比对每一条 expected ID，缺失、重复或额外样本均失败，不能只看进程退出码。
- 上游 `id` 会用于特征文件路径，且缓存按 id 重用。必须限制 ID，任何文字、参数、模型 revision 或音频变化都使用全新 feature 目录。不要复用陈旧缓存。
- `train_heads=true` 训练 mel 和 text 两个 head，adapter 导出没有 text head。必须通过导出/重载一致性 gate 后才能发布。[受训参数实现](https://github.com/oberkayb/index-tts-2.5-ft/blob/abdcbda0f6624d87134da8d1f4d482edac4a2fb9/itts25ft/modeling.py)
- adapter 不兼容 `use_accel`/DeepSpeed。不同语言 adapter 用独立进程/实例；未验证 vLLM 或 ComfyUI。

官方 [LICENSE](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/LICENSE) 明确包含微调与 LoRA 等衍生物，要求保留原始声明和协议，并规定下游分发条件。请同时查看 [中文版本](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/LICENSE_ZH.txt) 与所用模型 snapshot 的模型卡。这里记录授权边界，不把本仓库 MIT 扩展到它们。

早期source-only接口核查时尚未下载模型或启动训练环境，因此当时没有锁定模型snapshot、权重哈希和实际环境版本。后续已固定官方snapshot、核验完整资产并实际运行训练/导出/stock推理；当前范围、环境偏离和质量未通过的结果见 [VALIDATION.md](../VALIDATION.md)，不能将早期核查时点当作当前状态。部署方仍须保留完整revision、文件哈希、实际解析版本、`pip freeze --all`与CUDA/驱动记录，并检查checkpoint config确实为2.5。部分官方模块可首次联网取辅助模型；只在本地资产齐备后离线加载，缺失应停止，不把source-only检查当成任意环境兼容性证明。

## 可合法实现的训练路径

对上述固定官方 commit 的完整树和入口定义做了只读核查：没有发现可直接启动2.5 GPT微调的官方CLI/流水线。pyproject注册的是推理入口，TRT目录用于转换/推理；maskgct中的codec训练组件不能当作2.5 GPT微调入口。参见 [官方入口](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/pyproject.toml#L105)。因此当前不能以替换一个启动命令来消除外部trainer阻塞。

获得社区明确授权后接入其后端是较短路径；否则可以在遵守官方许可的前提下原创实现。官方 [UnifiedVoice.forward](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/gpt/model_v2.py#L596) 返回latents，没有现成训练loss或language-ID参数；[get_logits](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/gpt/model_v2.py#L528) 也没有attention_mask参数。原创后端必须正确处理语言conditioning、teacher-forcing对齐与padding/CE mask、Conv1D LoRA、冻结与行梯度/weight-decay隔离、BF16累积、评测、导出重载及完整恢复。可先用batch1加累积缩小范围，再验证padded batching。

原创batch1原生实现`hi_h200_native`直接调用固定官方模块；真实官方微型GPT的CPU测试使用随机小权重/合成输入，未用mock模型冒充真实模型。后续完整预训练模型已完成真实Hindi特征提取、v1的100次更新与跨进程恢复、完整GPT合并、单feature stock parity和真实WAV生成；独立fresh v2已在200次更新处保存状态。**Hindi质量仍未通过验收**，不能据机制通过或训练loss下降宣称成品可用。当前结果以 [VALIDATION.md](../VALIDATION.md) 为准，操作流程见 [NATIVE.md](NATIVE.md)。
