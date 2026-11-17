# 两张原理图的来源与保真账本

HTML是单文件源，图内source keys对应下列不可变源码。模型固定`IndexTeam/IndexTTS-2.5@c39ce5ba981572cb187443877ff559dfb246ce63`，官方源码固定`d9e41aac89fd00b3d71497fddb287b7f24613712`；原创训练实现引用本次已验证的`7c950e32597b09ac51405f16d9e1dfc87aded98b`。内容已与原理文档作者交叉核对，未从无许可社区代码取图。

## 实际推理：indextts25-inference

| key | 固定来源 | 图中事实 |
|---|---|---|
| S1 | [参考条件提取](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/infer_v2_5.py#L621) | 参考音频生成不同类型的声纹、情绪、语义与Mel条件 |
| S2 | [GPT生成](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/infer_v2_5.py#L740) | 文字/语言及参考条件驱动GPT，输出离散semantic codes；mel_head输出code logits |
| S3 | [codec / LR / CFM / BigVGAN](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/infer_v2_5.py#L824) | codec解码后长度调节，经声学生成与vocoder输出22.05kHz波形 |
| S4 | [CFM实例化DiT](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/s2mel/modules/flow_matching.py#L163) | 本图按锁定实现写CFM/DiT，不用报告的Zipformer标签替代固定代码 |

保留完整主链：文本/语言→GPT→离散codes→EnhancedCodec.decode→长度调节→CFM/DiT→BigVGAN→WAV。参考框是叶子簇压缩：CAMPPlus 192维声纹及其投影、W2V-BERT层17标准化特征与GPT情绪分支、参考语义/参考Mel集中为一个节点；三条出线**不是同一个向量**。默认`use_gpt_latent=False`，没有画不存在的GPT hidden latent直送CFM。

合并：文字前端与tokenizer（规范化由配置控制）放入“文本与语言”；参考条件各编码器放入一个参考框。省略：可选Qwen情绪文本、流式切块、缓存/加速、采样器细节和各tensor shape；它们不改变本图的主路径。未将报告与另一配置的描述差异判定为报告错误。

## Hindi LoRA：hindi-lora-training

| key | 固定来源 | 图中事实 |
|---|---|---|
| T1 | [原生数据](https://github.com/allenflux/indextts25-hindi-h200/blob/7c950e32597b09ac51405f16d9e1dfc87aded98b/src/hi_h200_native/data.py) / [冻结特征提取](https://github.com/allenflux/indextts25-hindi-h200/blob/7c950e32597b09ac51405f16d9e1dfc87aded98b/src/hi_h200_native/extract.py) | speaker隔离、真实特征与身份/配对校验 |
| T2 | [原生模型](https://github.com/allenflux/indextts25-hindi-h200/blob/7c950e32597b09ac51405f16d9e1dfc87aded98b/src/hi_h200_native/model.py) | 冻结其余参数，训练attention LoRA、hi行、所见text行、双head；code CE + 0.2 text CE |
| T3 | [候选选择](https://github.com/allenflux/indextts25-hindi-h200/blob/7c950e32597b09ac51405f16d9e1dfc87aded98b/docs/CHECKPOINT_SELECTION.md) | 五个固定VAL点选best100，不用后续test/ASR重选 |
| T4 | [完整合并](https://github.com/allenflux/indextts25-hindi-h200/blob/7c950e32597b09ac51405f16d9e1dfc87aded98b/src/hi_h200_native/export.py) | LoRA增量、可训练行、text/mel head合并到完整GPT；base和候选独立目录 |

为控制图面，训练配置、teacher-forcing CE合并在“受限微调”步骤；验证点与候选冻结合并在“仅按VAL选择”；完整合并与原/候选目录交付合并在末步骤。未画成只训练LoRA，也未把mel_head画成Mel谱预测器。受限text行含共用token，旧语言无损仍未验收；目录/实例隔离不等于已证明同进程回切或完整workflow。优化器/RNG/断点、存储、资源控制及发布许可细节留在正文。

## 校验与交付

- 两图均为`doc-wide` 1280×720，目标Confluence PNG@2为2560×1440。
- 技能`self_check.py --offline`：两图PASS；`verify_geometry.py`：2文件、0 finding。
- 干净纸面、无阴影；ink/soft/accent与实际浅蓝底均通过相应对比度。
- 已完成Mac本地浏览器实际截图视觉QA：两图可读，图内可横向滚到右端，正文不溢出，样式符合Confluence参考及字体fallback；这不是Hindi音频听评。
- HTML源未变；用户批准后，项目内Playwright1.65.2 / Chromium headless shell1255已使用技能原样export_png.py输出两张2560×1440 PNG，本地view_image及实际BBox/窄屏/网络QA通过；没有手绘SVG或别的图片流程替代。
- 安装、字体、尺寸、摘要和校验范围见[RENDER_RECEIPT](RENDER_RECEIPT.md)与[render-manifest.json](render-manifest.json)。
- 私人Confluence页面、Pod、音频、文本样本、凭据及实际工作目录未写入图或账本。
