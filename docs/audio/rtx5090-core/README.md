# RTX 5090 四核心节点生成样本

两份 WAV 来自 RTX 5090 的两个独立进程，真实调用 ModelLoader、EmotionControl、SamplingConfig、Generate。不是完整 ComfyUI queue 或同进程原版→Hindi→原版测试；`quality_gate_passed=false`，不推荐据此全局替换生产模型。用户明确授权将这两份生成样本公开，并在其 Confluence 页面链接。

| 文件 | 输入文字 / 语言 | 时长 | 文件 SHA256 |
|---|---|---|---|
| [hindi-5090-core.wav](hindi-5090-core.wav) | नमस्ते। यह हिन्दी भाषा का एक छोटा परीक्षण है。 / HI | 3.506213 s | `4c0e9a404ef63f0a1ddc629bb6d6c20e6370b4f5c00eb5c50bc614cb77747dcb` |
| [base-zh-5090-core.wav](base-zh-5090-core.wav) | 你好，这是原始模型的切换测试。 / ZH | 3.192744 s | `76e7a6850db5d5df80f390871cb3a475a0f6e3231d683fb113ec174d41ef0b0c` |

均为 mono PCM16 / 22,050 Hz。文件只从原 `audio.wav` 复制并重命名，音频字节未修改。二者输入文字和语言不同，仅作功能演示，不能作为同文本音质 A/B、Hindi 母语听评或旧语言无退化证明。WAV 存在也不证明句尾完整。

共同参考来源：[IndexTeam/IndexTTS-2-Demo 固定公开示例](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo/blob/b01840e8e4fd9753743a6d0466cd73ae1d634a68/examples/voice_01.wav)。本目录不分发参考原音频；功能检查仅将原 48 kHz FLOAT 单声道转换为 PCM16，没有重采样或裁剪，转换后 SHA256 为 `fd23ddac68f2db3710481c7cbfc1a50a117d9c38bc5c3ee51c33444f0a9b019a`。示例来源的 [Apache LICENSE](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo/blob/b01840e8e4fd9753743a6d0466cd73ae1d634a68/LICENSE) 标识与 Bilibili 模型条款分别适用；公开示例不构成来源人物或原权利人对本实验的背书。

Hindi 模型：[冻结 best100](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental/tree/117c7922e4dcedc3f9885f3f6efa5be8f54c7919)，已合并完整 GPT；原版为 `IndexTeam/IndexTTS-2.5@c39ce5ba981572cb187443877ff559dfb246ce63`。训练实际使用 IndicVoices-R Hindi，归属及变更见 [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md)。本目录没有训练语料、转写、说话人标识、优化器状态、认证材料或私人运行日志。

完整 Bilibili 中英文条款见 [LICENSE](LICENSE) / [LICENSE_ZH.txt](LICENSE_ZH.txt)，另见 [MODEL_LICENSE](MODEL_LICENSE)、[NON_ENDORSEMENT.txt](NON_ENDORSEMENT.txt) 与 [校验清单](audio-manifest.json)。项目工具代码的 MIT 不代替模型或数据条款。本实验的任何修改不受原模型权利人背书、担保或保证，原权利人不承担与衍生实验相关的责任。其他可验证结果与切换边界见 [RTX 5090 手册](../../RTX5090_SWITCHING.md)。
