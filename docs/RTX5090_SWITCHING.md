# RTX 5090：Hindi 模型切换的实测与待验收项

记录日期：2026-11-17。**四个核心 T8 节点的真实 GPU 推理已通过；完整工作流、同一进程回切和质量无损尚未验收。** 迁移后终端已恢复；下列四节点数字来自此前保存的真实回执。后续完整图控制器已完成CPU准备，但首次真实执行请求被共享GPU资源门槛拦住，尚未加载模型或运行整图。

## 产物到底是什么

训练使用 LoRA；公开 `gpt.pth` 已将 LoRA、Hindi/文字行和输出 heads 合并到完整 GPT 中，官方格式为 `{"model": state_dict}`，共 456 tensors。推理时不需要额外挂 LoRA。它替换的是 IndexTTS 2.5 的 **GPT 子模型**，仍依赖匹配的 tokenizer、codec、声学后端和声码器，不是可覆盖整个 2.5 目录的一份完整声学模型。

[公开实验模型](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental/tree/117c7922e4dcedc3f9885f3f6efa5be8f54c7919)已发布；13 文件完整匿名下载、全部 payload 校验和 GPT 完整流式 SHA 均已实际通过。下载公开权重不需要 HF token。

| 身份 | 固定值 |
|---|---|
| 本轮工具源码 | `7c950e32597b09ac51405f16d9e1dfc87aded98b` |
| 公开 Hindi revision | `117c7922e4dcedc3f9885f3f6efa5be8f54c7919` |
| 官方模型 base revision | `IndexTeam/IndexTTS-2.5@c39ce5ba981572cb187443877ff559dfb246ce63` |
| T8 插件基线 | `T8mars/comfyui-indextts25-t8@1ff425ece89ef9824f58f0e4dbfb057991814e88` |
| 实测 ComfyUI HEAD | `8ff6dc384ba5c410266b40e137799e049459d4f2`；宿主有既有局部改动，不能当作干净 checkout |
| 原 GPT SHA256 | `43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85` |
| Hindi GPT SHA256 | `423a8491b8856c7c3075254b12defc3ae47dbd9e0de285772523ddd4e9a97021` |
| Hindi GPT 大小 | 3,259,538,991 bytes |

## 5090 实际做了什么

RTX 5090，32,607 MiB 显存；Python 3.12、Torch 2.12.1+cu130，实际支持 sm120 和 BF16。开始前观测显存已用 654 MiB、利用率 0%、原 ComfyUI 队列为空，持久输出空间充足。此环境记录不是官方通用依赖锁。

使用全新隔离 T8 兼容副本，逐项核验两个完整 26 文件模型目录。在独立子进程中分别真实调用 `T8_IndexTTS25_ModelLoader`、`T8_IndexTTS25_EmotionControl`、`T8_IndexTTS25_SamplingConfig`、`T8_IndexTTS25_Generate` 的执行方法；不是 mock trainer，也不是只检查 checkpoint key。两例均返回 `four_real_nodes_generated_audio_quality_unverified` 并实际产生 PCM WAV。

| 实测项 | Hindi 候选 | 原版对照 |
|---|---|---|
| 语言/文字 | HI：नमस्ते। यह हिन्दी भाषा का एक छोटा परीक्षण है। | ZH：你好，这是原始模型的切换测试。 |
| 含加载的四节点耗时 | 31.6294 s | 31.1629 s |
| 节点报告推理耗时 | 2.77 s | 2.65 s |
| Torch peak allocated | 6,056,838,656 bytes | 6,056,875,008 bytes |
| WAV | mono PCM16 / 22,050 Hz | mono PCM16 / 22,050 Hz |
| WAV 时长 / 帧数 | 3.506213 s / 77,312 | 3.192744 s / 70,400 |
| WAV 文件大小 | 154,668 bytes | 140,844 bytes |
| WAV SHA256 | `4c0e9a404ef63f0a1ddc629bb6d6c20e6370b4f5c00eb5c50bc614cb77747dcb` | `76e7a6850db5d5df80f390871cb3a475a0f6e3231d683fb113ec174d41ef0b0c` |

这是两种不同文字/语言的功能检查，**不是同文本的音质 A/B**，不能从时长或速度推断发音正确、整句无截断或旧语言没有退化。它们使用两个独立进程，不构成同一 ComfyUI 进程内的 A→B→A 热切换证明。

保留的结果相对路径：

```text
WORK/outputs/hindi-node-probe-003/result.json
WORK/outputs/hindi-node-probe-003/audio.wav
WORK/outputs/base-node-probe-003/result.json
WORK/outputs/base-node-probe-003/audio.wav
```

两份 WAV 仍在持久 Pod 工作区，尚未下载为本轮本地交付，也未附入 Confluence；迁移完成后按 SHA 取得原文件再补可播放附件，不使用旧 H200 试听包冒充本次结果。公开 Git/HF 不包含这些 WAV 或原始参考音频。

本轮都使用同一合法公开示例参考：`IndexTeam/IndexTTS-2-Demo@b01840e8e4fd9753743a6d0466cd73ae1d634a68/examples/voice_01.wav`，原 SHA `e33e6ee0107a1dd58e1d66dd90c13df3d55a8683047cc3d7ea206dad84ed3fc8`。原 48 kHz FLOAT 单声道只转换为 PCM16，不重采样/裁剪；转换后 SHA `fd23ddac68f2db3710481c7cbfc1a50a117d9c38bc5c3ee51c33444f0a9b019a`。来源 Space 的 Apache 及 Bilibili 协议、转换 provenance 已保留；这不是 Hindi 母语参考回归，也未新增训练数据。

## 为什么需要一次兼容更新

未修改的 T8 基线仍会在加载前拒绝 `HI`，且原模型大小/哈希清单不接受 Hindi GPT。关闭 `verify_hashes` 不会关闭大小检查。不能仅把候选重命名为原 `gpt.pth`，或填充文件、伪造原模型身份。

[`check_t8_compat.py`](../scripts/check_t8_compat.py)准备的是独立兼容代码副本：保留所有原节点 ID、原模型清单和中英文许可，增加 HI 的 UI/运行时/分段入口及单独 Hindi 清单；选择原模型时仍按原身份校验，选择候选时按冻结候选身份校验，全部 26 个资产 SHA 强制验证。没有复制未许可社区训练代码，没有修改正在服务的插件或重启 ComfyUI。

原宿主目录的 26 项只读审计发现 25 项完全匹配；唯一配置差异为 `version: 2.0`，隔离模型目录使用固定官方 `version: 2.5` 配置，原目录保持不动。不能凭目录名认定是 2.5。

## 迁移完成后的重复验证步骤

本节是已实现 CLI 的复用步骤，不表示迁移后已执行。保留原模型和完整配置，使用真实持久 `WORK`、已存在的固定 T8 源目录和全新目标目录。不要覆盖生产目录或全局升级 Torch。

```bash
export HF_REV=117c7922e4dcedc3f9885f3f6efa5be8f54c7919
python scripts/check_t8_compat.py inspect --node-root "$T8_ORIGINAL"
python scripts/check_t8_compat.py prepare \
  --node-root "$T8_ORIGINAL" --destination "$WORK/t8-hindi-compat-new" \
  --candidate-revision "$HF_REV"
python scripts/check_t8_compat.py verify-copy \
  --node-root "$WORK/t8-hindi-compat-new" --candidate-revision "$HF_REV"
```

`prepare`只准备代码，不准备模型，不下载权重，不自动部署服务。公开 13 文件下载/校验见 [安装说明](INSTALL_EXPERIMENTAL.md) 和 [`download_public_release.py`](../scripts/download_public_release.py)。之后在独立目录准备清单要求的全部 26 资产：原版与 Hindi 只替换已声明的 GPT，使用匹配的官方 2.5 配置/词表，保留全部适用许可，不把仅 GPT 的 13 文件目录直接作为 T8 的完整模型目录。

已有实测目录相对 WORK 为 `models/base-official-001`、`models/hindi-best100-001`、`t8-hindi-compat-001`。迁移后需确认它们仍在真实持久存储、hash 不变，不能使用此前失败准备留下的部分 `models/original`。

先运行不加载 GPU 的 preflight；以下变量须指向实际路径，输出目录必须全新：

```bash
PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1 \
python scripts/probe_t8_nodes.py \
  --workspace-root "$WORK" --node-root "$T8_COMPAT" \
  --model-dir "$HINDI_MODEL_DIR" --comfy-root "$COMFY_ROOT" \
  --prompt "$WORK/reference/voice_01-pcm16.wav" \
  --prompt-sha256 fd23ddac68f2db3710481c7cbfc1a50a117d9c38bc5c3ee51c33444f0a9b019a \
  --candidate-revision "$HF_REV" --model-variant hindi --language HI \
  --text 'नमस्ते। यह हिन्दी भाषा का एक छोटा परीक्षण है।' \
  --output-dir "$WORK/outputs/hindi-preflight-new"
```

默认只查本地身份与输入，不加载模型。实际 GPU 运行必须另设全新输出目录，追加 `--execute --allow-cuda --gpu-id 0 --min-free-gpu-mib 24576 --min-free-ram-mib 16384 --max-utilization 10`，并重新观察共享资源；本轮控制器另有 600 秒 deadline、原 ComfyUI 队列监视与仅终止自建子进程的限制，这些**不是裸 probe CLI 自带的全部行为**。迁移完成前不要运行。

## 用户工作流如何切换与回滚

目标是保留现有 1–8 节点的类型、ID 与连线。一次兼容更新及完整工作流验收完成后，Hindi 模式选择独立完整 Hindi 模型目录 (`custom_model_path`) 和 `language=HI`；回滚时选择独立原模型目录和原语言。不是覆盖原文件或整个 IndexTTS 2.5 目录，也不是新增 LoRA 节点。

本轮四节点实际显式使用 CUDA 0、BF16、`acceleration_mode=off`、CUDA kernel off、`verify_hashes=true`、`persistent_reference_cache=false`、`quality_retry_count=0`、speaker 情绪模式。采样使用用户给定的 do_sample=true、temperature/top_p=0.8、top_k=30、beams=3、repetition_penalty=10、max_mel_tokens=1504、diffusion_steps=25、seed=1051829406；严格确定性/TF32 off 与离线文本前端适配也在回执中声明。其余参数以 `result.json` 为准。HI 自动分段上限 60 是实验起点，原语言原上限保留。

用户原参数中的 auto 设备/精度、`verify_hashes=false`、`persistent_reference_cache=true` 尚未按完整工作流逐项验收，不能将本次显式参数测试写成原 JSON 零差异通过。未更改现有 ComfyUI 节点部署或线上模型。

## 尚未通过的验收项

- 原 8 节点图的实际 ComfyUI queue 执行、最终 AudioEncrypt 输出及音频 URL/cache 链路。当前只直接调用四个核心节点，没有提交 queue。
- 同一实例内原版→Hindi→原版切换，模型/参考缓存身份、显存上限和释放、失败回滚；原 `release_after_run=false` 可能同时保留模型，不能保证零额外显存。
- 长 Hindi 分段、整句完整性与 max_mel 截断检查；节点内部 warning 捕获可能吞掉相关提示，WAV 存在不能证明无截断。
- Hindi 母语人工听评及足量原支持语言 native-prompt 回归。此前冻结 test 仅两条 Hindi，best100 严格 CER 50%；中日各一例 Hindi-prompt stress 变差，其他单例不变也不足以证明无损，详见[有限验收](FINITE_ACCEPTANCE.md)。
- 本轮两份音频的受控下载、哈希复核与 Confluence 可播放附件。

目前不需要追加训练来完成上述接口验收。保留 best100 不再按 test 挑选；功能通过不等于正式支持 Hindi 或建议生产全局替换。最新本地完整 CPU suite **278/278 通过、0 跳过、34.841 秒**，验证工具机制而不是语音质量。迁移前 Hindi 推理结束后真实快照回到 654 MiB/0%；连接断开后的当前进程/服务健康没有重新验证，不作推断。

## 数据与许可

实际训练只使用 `ai4bharat/indicvoices_r` 的 Hindi，revision `5f4495c91d500742a58d1be2ab07d77f73c0acf8`，两个 TRAIN 分片共 946,963,686 bytes；pilot 64 clips / 16 speakers，train/val/test 按 speaker 隔离。不是 Rasa；本项目 held-out test 不是发布者官方 TEST。来源 CC-BY-4.0 归属及变更说明、官方 Bilibili 中英文完整许可和非背书声明随公开包保留，工具 MIT 不替代模型许可。参考音频、训练语料及私有回执不进公开 Git/HF。

### 参考音频哈希记录更正

后续整图metadata预检因参考SHA不匹配而正确停止，尚未启动GPU。经原始PCM16转换记录、迁移前已成功四节点result.json和当前文件全SHA三向核对，正确值为`fd23ddac68f2db3710481c7cbfc1a50a117d9c38bc5c3ee51c33444f0a9b019a`，与既有音频一致。前版文字摘要/命令中的SHA有转录错误；本次更正固定值，未更换参考音频、模型或放宽校验。失败回执保留。

## 后续完整图控制器（仍待真实三轮通过）

[`stage_comfy_validation.py`](../scripts/stage_comfy_validation.py)已在真实持久models工作区完成CPU inspect/stage：固定ComfyUI完整Git树与原有精确三行ops修改受审计，使用独立代码、input/output/temp/user/SQLite及显式插件引用。源码引用不是操作系统强制只读，不宣称容器隔离。

[`run_comfy_validation.py`](../scripts/run_comfy_validation.py)与[`comfy_validation_server.py`](../scripts/comfy_validation_server.py)保留原8节点、5连接，实际依赖闭包为1/2/3/4/5/8；使用自建loopback新端口、关闭整图缓存，计划base ZH→Hindi HI→base ZH各一次。原loader的auto设备/精度、persistent_reference_cache=true、release_after_run=false均保留；实际CUDA/BF16、模型路径/身份、6节点执行事件、最终AudioEncrypt WAV与同源view字节必须分别验证。公开零占位仅用于encrypt=false，用户原始key和私人元数据不进入图或日志。

完整CPU suite在`38b417d28338ba965f514b358a9d204d20ab6fdf`运行 **311/311通过、0跳过、36.874秒**。参考SHA两处字面量更正后，25项相关CPU合同/负向/状态机回归通过，独立增量审查PASS；更正源码为`5cd341d631c00fdb6494ef50845a75ff5f67513e`。这些CPU测试不是实际ComfyUI服务或GPU推理。

更正后的真实CLI metadata已返回`metadata_verified_no_server_started`，三轮状态都是not_executed。首次执行请求在resource_admission退出，当时GPU已用22946MiB、可用9165MiB、利用率100%；控制器未创建服务或加载模型，自身已退出，错误回执保留。未清理生产模型缓存，未降低或取消保护以强行运行。

已观察到不同Pod共享models工作区，但原output项目目录并非处处可见；跨Pod复用前须重新核对真实依赖、源码和资产，不能按同一字符串路径假定文件相同。后续实际完整图结果与资源参数另行更新；本页仍不把控制器实现完成当作功能验收完成。
