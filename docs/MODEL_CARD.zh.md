---
license: other
license_name: bilibili-model-license
license_link: https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/c39ce5ba981572cb187443877ff559dfb246ce63/LICENSE
language:
  - hi
base_model: IndexTeam/IndexTTS-2.5
base_model_relation: finetune
pipeline_tag: text-to-speech
datasets:
  - ai4bharat/indicvoices_r
tags:
  - indextts
  - hindi
  - lora
  - experimental
---

# IndexTTS 2.5 Hindi 实验候选：best100

**这是小数据实验性Hindi微调权重，不是官方Hindi模型，也不是生产推荐。** 两条冻结test生成的严格字符错误率仍为50%；人工Hindi听评未完成，旧语言有未排除的退化风险。实验权重公开不代表质量、自然度、说话人同意或生产验收已经通过，不能承诺全局替换原模型后旧语言零退化。

权重文件、不可变revision及逐文件哈希以[本模型仓库的Files](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental/tree/main)和发布manifest为准。下载时固定实际commit并核对清单；模型卡本身不证明权重已经上传或通过加载验收。

## 模型和固定来源

这是官方IndexTTS 2.5的GPT完整合并候选，采用本项目原创原生LoRA后端；没有复制或导入缺许可证的社区训练器。原始声学后端保持冻结，训练GPT attention LoRA、Hindi语言行、受限文字行及text/mel两个输出head；导出保留全部可训练状态。

- 原始模型：`IndexTeam/IndexTTS-2.5`，revision `c39ce5ba981572cb187443877ff559dfb246ce63`。
- 官方源码：`index-tts/index-tts`，commit `d9e41aac89fd00b3d71497fddb287b7f24613712`。
- 原创工具与非敏感验证记录：[allenflux/indextts25-hindi-h200](https://github.com/allenflux/indextts25-hindi-h200)。模型文件与审查后元数据的完整SHA256以发布文件清单为准。

官方2.5[声明支持](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/README.md)中文、英语、日语、西班牙语和阿拉伯语。Tokenizer的`hi`槽位17与`<|hi|>`只代表接口存在，不能当作官方Hindi训练或质量声明。

## 数据与训练

实际数据为[AI4Bharat IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r)，固定revision `5f4495c91d500742a58d1be2ab07d77f73c0acf8`，仅Hindi；Rasa未参与本次训练。只取得两份官方TRAIN分片，语料payload共946,963,686字节，未下载全库或官方TEST。以下三组均从已取得的官方TRAIN重建，并按speaker隔离：

| 本项目split | Clips | Speakers | 实际音频秒数 |
|---|---:|---:|---:|
| train | 48 | 12 | 491.0290416667 |
| val | 8 | 2 | 82.9501666667 |
| test | 8 | 2 | 69.9240833333 |

完整utterance上限20秒，不截断目标音频；保留源文与NFC/空白规范文本，谨慎保留nukta、virama及joiner。训练特征的24k→16k直接重采样与stock推理的librosa分支数值不完全等价；GPT单feature parity不等于全前端或语音质量等价。

单H200、BF16、batch1加4次梯度累积，LoRA rank8/alpha16/dropout0，AdamW起始学习率1e-4、warmup25、text loss权重0.2。原计划最大1000次optimizer更新保持不变；实际先停在20步，再由新进程恢复至500步停止。每100步验证8例，五个验证点中100步loss最低，随后升高，因此选择并冻结**第100步**，而非last500。候选只按VAL选择；后续test/ASR不用于重选checkpoint。三个历史run不能相加为本模型的训练步数。

真实完整checkpoint已转换为标准adapter，再完整合并GPT。H200上1条feature的native adapter与fresh stock模型FP32数值对照通过固定`rtol=atol=1e-4`，targets和argmax一致；它结合相对与绝对容差，不表示全部绝对差小于1e-4，也不是全数据集或音质证明。

## 冻结后的有限验收

预先固定两位test speaker各一例，原模型/候选各执行一次，全部保留，无retry、重选case或重新选模。下表为NFC和空白规范后的**strict**微平均，包含空格；与早期去标点的VAL分数不可混算。

| Hindi范围（2例） | 严格CER | 严格WER |
|---|---:|---:|
| 真人reference的ASR校准 | 31/166 = 0.186747 | 14/32 = 0.4375 |
| 原始模型生成 | 396/166 = 2.385542 | 146/32 = 4.5625 |
| best100生成 | 83/166 = **0.500000** | 27/32 = **0.84375** |

CER/WER可因插入错误超过1。候选有方向性改善，但样本量仅两例、真人reference的识别误差也较高，不能认证Hindi可用性、自然度或整体泛化。ASR为固定官方Whisper small，非人工Hindi听评。

另对官方五种旧语言各固定一例，均用Hindi参考音频做跨语言prompt stress；每个原/候选pair输入一致，缺乏各语言native-prompt及真人校准：

| 语言（每种1例） | 原模型严格CER | best100严格CER |
|---|---:|---:|
| zh | 1/17 | 3/17 |
| en | 1/46 | 1/46 |
| ja | 0/18 | 2/18 |
| es | 1/60 | 1/60 |
| ar | 0/41 | 0/41 |

中文、日文单例指标升高，需进一步听评与native-prompt参照；其他三个单例相同也不证明无退化。原模型7/7、候选7/7真实stock生成及16/16 GPU ASR完成，全部样本保留。详细方法、误差视图和范围见[有限验收记录](https://github.com/allenflux/indextts25-hindi-h200/blob/main/docs/FINITE_ACCEPTANCE.md)。

## 加载与已知限制

本候选是GPT权重，**不是完整推理包**。须使用固定官方2.5的配置、tokenizer、声学/声码器资产及正确Hindi语言参数，按[安装与隔离验证说明](https://github.com/allenflux/indextts25-hindi-h200/blob/main/docs/INSTALL_EXPERIMENTAL.md)在独立目录加载。普通stock推理通过不等于任意ComfyUI节点、vLLM、DeepSpeed或加速路径可直接替换；旧语言继续保留原始模型。

当前实际使用的[`T8mars/comfyui-indextts25-t8`](https://github.com/T8mars/comfyui-indextts25-t8/tree/1ff425ece89ef9824f58f0e4dbfb057991814e88)固定版本`1ff425ece89ef9824f58f0e4dbfb057991814e88`有两项独立阻塞：生成UI和运行时白名单拒绝`HI`；内置原模型清单的固定GPT文件大小及SHA要求也会拒绝本候选，关闭SHA检查仍不绕过大小检查。因此，**不能直接用本文件替换该节点的原始GPT**。

该实际节点类的CPU meta结构检查已通过：456/456个tensor的key及shape一致，missing/unexpected/shape mismatch均为0，strict加载通过且未初始化CUDA。这没有运行GPU合成或提交ComfyUI queue，**不解除上述语言入口和权重清单阻塞**，也不代表节点端到端兼容。具体证据及后续集成边界见[安装说明](https://github.com/allenflux/indextts25-hindi-h200/blob/main/docs/INSTALL_EXPERIMENTAL.md)。

当前不推荐生产部署或全局覆盖原模型。尚缺Hindi及各语言母语听评、充分的本语言prompt回归、自然度/相似度/故障率和更大独立测试集。测试环境、依赖差异和已知告警见项目[VALIDATION](https://github.com/allenflux/indextts25-hindi-h200/blob/main/VALIDATION.md)与[SECURITY](https://github.com/allenflux/indextts25-hindi-h200/blob/main/SECURITY.md)。不要据单一CER/WER或文件可加载宣称质量达标。

## 许可、归属与用途

权重适用bilibili Model Use License Agreement，不适用原创工具的MIT。完整`LICENSE`、`LICENSE_ZH.txt`、`MODEL_LICENSE`、`NON_ENDORSEMENT.txt`须随发布包保留；非官方说明不替代其中指定的声明。原模型权利人不为本微调背书或提供担保。

协议中文优先：若本人或关联方上月MAU超过1亿，或上一年度收入超过人民币1亿，须先取得单独书面许可；完整下游、用途与第三方授权义务仍适用。[固定英文协议](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/LICENSE)、[固定中文协议](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/LICENSE_ZH.txt)

数据归属：AI4Bharat；Ashwin Sankar等，2024，*IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS*，NeurIPS Datasets and Benchmarks。[论文](https://arxiv.org/abs/2409.05356)、[数据集](https://huggingface.co/datasets/ai4bharat/indicvoices_r)、[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)。作者声明取得参与者同意，这不等于本项目逐人独立核验全部人格权或未来克隆用途；访问同意不自动授予任意分发权。使用者须核实自身数据、提示音频和用途所需权利。

发布包不附带数据分片、原始音频、训练prompt、私人转写、个人访问材料、optimizer状态或凭据。实验公开允许复核已知不足，不代表生产推荐或对任何声音用途的额外授权。
