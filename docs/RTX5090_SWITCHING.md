# RTX 5090：Hindi 模型切换的实测与待验收项

记录日期：2026-11-17。**已在同一自建 ComfyUI 服务内真实完成原版 ZH→Hindi HI→原版 ZH 三轮 queue 执行，三轮均成功；原 8 个节点 ID/类型与 5 条连线保持，首尾 WAV 文件和 PCM 完全相同。** 通过的是受控隔离环境的工作流与切换接口；Hindi 音质、整句完整性、旧语言无损及生产运行时等价性仍未通过，`quality_verified=false`、`speech_completeness_verified=false`、`framework_runtime_equivalence_verified=false`。没有将兼容副本部署到生产插件或替换原模型。

## 最新真实三轮：同进程原版→Hindi→原版

真实源码为 [`792305841240072e5a9f2616eb9e387126cae5a2`](https://github.com/allenflux/indextts25-hindi-h200/tree/792305841240072e5a9f2616eb9e387126cae5a2)。[公开脱敏验收报告](acceptance/rtx5090-switch-cycle-20261117.json)由已核实的原摘要脱敏导出：原摘要经 Kite 正常文件 Preview 逐字段取得，再在 Pod 计算 canonical JSON SHA256，与本地原摘要完全一致，值为 `ad6e7af9de23763d9ea903444fe0c714b68abfef4de3546400a8542bcb6ae0ad`。公开报告泛化内部目标名并增加脱敏说明，因此**不是声称公开文件本身的 canonical SHA 仍等于原摘要**；音频、性能和状态字段不变。这是实际模型与真实 ComfyUI queue 的结果，不是 CPU mock 状态机输出。下文 6-1/6-5 仅为操作者所选实例的简短标签，不公开内部完整名称或路径。

状态为 `three_run_same_process_cycle_completed_quality_unverified`；总 wall **81.038167 秒**，不是训练耗时或纯模型推理 benchmark。各轮保持同一服务进程，实际依赖闭包 `1/2/3/4/5/8` 六个节点每轮均执行；另外两个未连接节点保留 ID/类型，不冒称八个节点都执行。整图缓存关闭；T8 模型与参考缓存仍启用。

| 实测项 | 原版 ZH：切换前 | Hindi HI | 原版 ZH：切换后 |
|---|---|---|---|
| queue / history | completed / success | completed / success | completed / success |
| 实际设备 / 精度 | CUDA 0 / BF16 | CUDA 0 / BF16 | CUDA 0 / BF16 |
| 实际节点 | 1/2/3/4/5/8 | 1/2/3/4/5/8 | 1/2/3/4/5/8 |
| WAV | mono PCM16 / 22,050 Hz | mono PCM16 / 22,050 Hz | mono PCM16 / 22,050 Hz |
| 时长 / 帧数 | 3.192744 s / 70,400 | 3.506213 s / 77,312 | 3.192744 s / 70,400 |
| 文件大小 | 140,878 bytes | 154,702 bytes | 140,878 bytes |
| WAV SHA256 | `860898e8a837eb4dbb1b2be2781ed8bfa17974d40980fe513965bbbb4a500f6c` | `e0174b4e63ec533bd4281f9b7125551f6e8d254223738e1c228f63c5d6e198fe` | `860898e8a837eb4dbb1b2be2781ed8bfa17974d40980fe513965bbbb4a500f6c` |
| PCM SHA256 | `4ec3ca1595119da8a680cc76a1b177bd5018cf50637f97cc8586df62697c7b7d` | `277e11de06baf1f7f8216517fdc920969fc7d68dbe19a49a640f1e4b45dfddd4` | `4ec3ca1595119da8a680cc76a1b177bd5018cf50637f97cc8586df62697c7b7d` |
| 该轮结束观测显存 used | 6,711 MiB | 11,799 MiB | 11,799 MiB |

三处显存数字只是采样点，**不是峰值、进程独占用量或硬配额**。原模型 cache object 在 Hindi 中间轮仍被观察到，首尾重用同一个 object，原模型 `completed_runs` 从 1 到 2；因此第三轮不是图缓存直接返回。首尾 WAV 文件与 PCM 相同只证明这条固定 ZH 用例的回切一致性，不是全部语言或全部输入无损。

最终三个输出均来自节点 5 AudioEncrypt（`encrypt=false`），文件字节 SHA 与同源 `/view` 读取一致；另独立重新读取完整文件和 PCM 核对。三份新 WAV 仍保留在受控 Pod，**未获得额外公开这三份样本的授权，不自动加入公开 Git、HF 或附件**；此前获授权的两份四节点样本另见下节。

三轮完成后，自建服务按控制器清理以 SIGTERM 退出，return code `-15`，`owned_process_cleanup_verified=true`，`/proc` 检查无本项目活动验证进程。**这是完成后的正常清理，不是三轮任务失败。** 原服务 HTTP 200、队列 0 running / 0 pending；GPU used 654 MiB、free 31,457 MiB、utilization 0%。这些是结束时点观测，不能保证后续共享占用；已告知用户可恢复 6-5 派单。

### 本轮保持的参数与非等价边界

原 loader 的 `device=auto`、`precision=auto`、`verify_hashes=false`、`persistent_reference_cache=true`、`release_after_run=false` 已在这次完整图中真实执行，并观测解析为 CUDA 0 / BF16。兼容副本仍在加载前强制核对全部 26 项资产，因此 UI 的 `verify_hashes=false` 不表示取消本项目资产校验。节点 5 使用公开零占位 key 且关闭加密；不读取用户原 key 或私人 `extra_pnginfo`。

三轮沿用固定公开短句、同一合法参考、seed 1051829406；首尾原版 ZH 除输出文件前缀外图参数相同。采样保留 do_sample=true、temperature/top_p=0.8、top_k=30、beams=3、repetition_penalty=10、max_mel_tokens=1504、diffusion_steps=25，speaker 情绪模式，`quality_retry_count=0`；未用质量重试筛选输出。

隔离服务、独立状态目录与 SQLite、仅 loopback 的离线网络策略、cache-only 真实 normalizer 适配、只读执行事件与真实 loader cache 观测、严格确定性及 TF32 off，均属于本次基础设施差异。源码引用不是操作系统强制只读，也不宣称容器隔离。**不能写成原生产 JSON 零差异、生产框架运行时完全等价或零额外显存。**

## 产物到底是什么

训练使用 LoRA；公开 `gpt.pth` 已将 LoRA、Hindi/文字行和输出 heads 合并到完整 GPT 中，官方格式为 `{"model": state_dict}`，共 456 tensors。推理时不需要额外挂 LoRA。它替换的是 IndexTTS 2.5 的 **GPT 子模型**，仍依赖匹配的 tokenizer、codec、声学后端和声码器，不是可覆盖整个 2.5 目录的一份完整声学模型。

[公开实验模型](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental/tree/117c7922e4dcedc3f9885f3f6efa5be8f54c7919)已发布；13 文件完整匿名下载、全部 payload 校验和 GPT 完整流式 SHA 均已实际通过。下载公开权重不需要 HF token。

| 身份 | 固定值 |
|---|---|
| 完整三轮工具源码 | `792305841240072e5a9f2616eb9e387126cae5a2` |
| 历史四核心节点源码 | `7c950e32597b09ac51405f16d9e1dfc87aded98b` |
| 公开 Hindi revision | `117c7922e4dcedc3f9885f3f6efa5be8f54c7919` |
| 官方模型 base revision | `IndexTeam/IndexTTS-2.5@c39ce5ba981572cb187443877ff559dfb246ce63` |
| T8 插件基线 | `T8mars/comfyui-indextts25-t8@1ff425ece89ef9824f58f0e4dbfb057991814e88` |
| 实测 ComfyUI HEAD | `8ff6dc384ba5c410266b40e137799e049459d4f2`；宿主有既有局部改动，不能当作干净 checkout |
| 原 GPT SHA256 | `43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85` |
| Hindi GPT SHA256 | `423a8491b8856c7c3075254b12defc3ae47dbd9e0de285772523ddd4e9a97021` |
| Hindi GPT 大小 | 3,259,538,991 bytes |

## 历史四核心节点实测与已公开试听

以下是此前 6-1 的两个独立进程四节点检查，须与上节同进程整图区分。RTX 5090，32,607 MiB 显存；Python 3.12、Torch 2.12.1+cu130，实际支持 sm120 和 BF16。开始前观测显存已用 654 MiB、利用率 0%、原 ComfyUI 队列为空，持久输出空间充足。此环境记录不是官方通用依赖锁。

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

两份 WAV 已从持久 Pod 工作区下载，并在受控 Mac 项目目录逐文件复核上述完整 SHA、大小和 PCM 格式。用户另行明确授权公开这两份生成样本；固定 Git commit `2cfc5785d4438544e1d87afe907e4ce4b7da4984` 已发布两份 WAV、README、校验清单及完整许可/归属/非背书声明共九文件，独立审查通过。它们是此前 RTX 5090 四节点结果，不是旧 H200 试听包，也不是本轮新三轮的音频。

- [Hindi 固定 WAV 下载直链](https://raw.githubusercontent.com/allenflux/indextts25-hindi-h200/2cfc5785d4438544e1d87afe907e4ce4b7da4984/docs/audio/rtx5090-core/hindi-5090-core.wav)
- [原版 ZH 固定 WAV 下载直链](https://raw.githubusercontent.com/allenflux/indextts25-hindi-h200/2cfc5785d4438544e1d87afe907e4ce4b7da4984/docs/audio/rtx5090-core/base-zh-5090-core.wav)
- [试听目录、完整许可与归属](https://github.com/allenflux/indextts25-hindi-h200/tree/2cfc5785d4438544e1d87afe907e4ce4b7da4984/docs/audio/rtx5090-core)

两份 raw 文件已实际匿名读取并核对完整 SHA、大小与 `audio/wav` MIME。本机浏览器直接打开曾被客户端策略拦截；没有改变设置或绕过，**不能声称本机浏览器实际播放已验证**。可按浏览器策略下载后本地播放；Confluence 尚未上传音频附件。原参考音频未随试听目录再次分发；本轮新三份整图 WAV 也未自动公开。

本轮都使用同一合法公开示例参考：`IndexTeam/IndexTTS-2-Demo@b01840e8e4fd9753743a6d0466cd73ae1d634a68/examples/voice_01.wav`，原 SHA `e33e6ee0107a1dd58e1d66dd90c13df3d55a8683047cc3d7ea206dad84ed3fc8`。原 48 kHz FLOAT 单声道只转换为 PCM16，不重采样/裁剪；转换后 SHA `fd23ddac68f2db3710481c7cbfc1a50a117d9c38bc5c3ee51c33444f0a9b019a`。来源 Space 的 Apache 及 Bilibili 协议、转换 provenance 已保留；这不是 Hindi 母语参考回归，也未新增训练数据。

## 为什么需要一次兼容更新

未修改的 T8 基线仍会在加载前拒绝 `HI`，且原模型大小/哈希清单不接受 Hindi GPT。关闭 `verify_hashes` 不会关闭大小检查。不能仅把候选重命名为原 `gpt.pth`，或填充文件、伪造原模型身份。

[`check_t8_compat.py`](../scripts/check_t8_compat.py)准备的是独立兼容代码副本：保留所有原节点 ID、原模型清单和中英文许可，增加 HI 的 UI/运行时/分段入口及单独 Hindi 清单；选择原模型时仍按原身份校验，选择候选时按冻结候选身份校验，全部 26 个资产 SHA 强制验证。没有复制未许可社区训练代码，没有修改正在服务的插件或重启 ComfyUI。

原宿主目录的 26 项只读审计发现 25 项完全匹配；唯一配置差异为 `version: 2.0`，隔离模型目录使用固定官方 `version: 2.5` 配置，原目录保持不动。不能凭目录名认定是 2.5。

## 可复用的独立资产与 stage 流程

使用固定源码 `792305841240072e5a9f2616eb9e387126cae5a2`，不覆盖原模型、生产插件或全局升级 Torch。以下变量由操作者填写为真实规范路径；模型、参考和 stage 不能仅凭路径字符串假定跨 Pod 相同。stage 工具当前要求 `WORK` 正好是持久 models 下的专用项目目录：

```bash
export HF_REV=117c7922e4dcedc3f9885f3f6efa5be8f54c7919
export WORK="$COMFY_ROOT/models/indextts25-hindi-acceptance-20261117"
export T8_COMPAT="$WORK/t8-hindi-compat-new"
export STAGE="$WORK/comfy-validation-stage-new"
```

### 先准备完整普通文件模型目录

公开 13 文件下载与校验使用 [`download_public_release.py`](../scripts/download_public_release.py)，见[安装说明](INSTALL_EXPERIMENTAL.md)。它不准备完整 T8 模型，也不接受数据访问条款。`check_t8_compat.py prepare` 只复制兼容代码，不复制模型；没有模型准备子命令。

从已授权且按固定身份核验的本地资产，按原版/候选清单的相对路径分别复制完整 **26 项普通文件 payload**，保留所需 `hf_cache` 等子目录；不得借用缺少目标文件的符号链接、遗漏声学资产、把 13 文件直接当作完整目录或改原目录。`BASE_MODEL` 与 `HINDI_MODEL` 要独立，候选仅采用冻结 GPT 及匹配官方 2.5 配置。6-5 本轮两个目录共 **16,581,886,642 bytes**，已全量 SHA 复验；这是本轮固定资产总量，不是任意模型下载大小。额外目录项也须单独审核，不能因清单通过就公开整个目录。

模型和参考最好置于 `WORK`；若在外部已审阅依赖根，控制器须显式增加 `--readonly-dependency-root`，不能放宽到无关私人目录。完成代码副本准备：

```bash
python scripts/check_t8_compat.py inspect --node-root "$T8_ORIGINAL"
python scripts/check_t8_compat.py prepare \
  --node-root "$T8_ORIGINAL" --destination "$T8_COMPAT" \
  --candidate-revision "$HF_REV"
python scripts/check_t8_compat.py verify-copy \
  --node-root "$T8_COMPAT" --candidate-revision "$HF_REV"
```

### CPU inspect、stage 与 metadata

使用已存在且精确核对的 ComfyUI 和两个插件路径。以下示例选用新的高位 loopback 端口 `18190`，不使用生产端口；stage 不预留端口，真实执行前仍会检查。`--preserve-reviewed-ops` 仅适用于宿主恰好是已审核三行 ops 变化的场景，SHA 必须为 `68611fba7c1d8750f6bd7266d5c6a40deb5c74ac8930f68c2572f4d314d299c8`；没有这个精确变化时不能假装校验成功。

```bash
python scripts/stage_comfy_validation.py inspect \
  --workspace-root "$WORK" --comfy-root "$COMFY_ROOT" \
  --t8-root "$T8_COMPAT" \
  --audio-cache-root "$COMFY_ROOT/custom_nodes/comfyui-audio-cache" \
  --audio-crypt-root "$COMFY_ROOT/custom_nodes/comfyui-audiocrypt" \
  --destination "$STAGE" --candidate-revision "$HF_REV" \
  --port 18190 --preserve-reviewed-ops
```

inspect 只验证计划，不写 stage；把同一命令的 `inspect` 换成 `stage` 才创建全新独立代码/状态目录，然后执行：

```bash
python scripts/stage_comfy_validation.py verify --stage-root "$STAGE"
PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1 \
python scripts/run_comfy_validation.py \
  --workspace-root "$WORK" --stage-root "$STAGE" \
  --base-model "$BASE_MODEL" --hindi-model "$HINDI_MODEL" \
  --prompt "$PROMPT_PCM16" --report-dir "$FRESH_REPORT" \
  --candidate-revision "$HF_REV" \
  --prompt-sha256 fd23ddac68f2db3710481c7cbfc1a50a117d9c38bc5c3ee51c33444f0a9b019a
```

默认返回 metadata 状态，核对两套全部资产、真实 prompt 与固定 source，不导入模型或创建服务。`FRESH_REPORT` 必须全新；`PROMPT_PCM16` 是前述已获使用权的固定公开参考转换文件，不能以试听生成样本替代。真实依赖在 `WORK` 外时，给这一控制器命令增加对应规范路径的 `--readonly-dependency-root`。

### 显式有界执行

先重新观察原服务队列、GPU 与 RAM，获得合适共享窗口；仅在资源保护满足时，在同一控制器命令追加以下参数并使用全新报告目录：

```bash
--execute --allow-cuda --gpu-id 0 \
--min-free-gpu-mib 24576 --min-free-ram-mib 16384 \
--max-utilization 10 --max-used-gpu-mib 24576 \
--startup-timeout 120 --run-timeout 600
```

这些是可审阅的保守示例门槛，不是通用资源保证，也不冒称本次私有完整 CLI 的全部原样值。执行需 Linux、充足独立存储、固定依赖与明确 CUDA 开关；控制器不自动接受许可、安装依赖或下载模型。POST 超时不自动重发；部分输出和失败 report 保留，不能当三轮成功。仅允许终止自建子进程，原 ComfyUI 队列只读监视；用户另行授权的定向取消/空闲维护见历史记录。

## 用户工作流如何切换与回滚

本次已证明在隔离兼容副本中，保留 1–8 节点类型/ID、5 条连线及原 loader auto/cache 设置，选择独立原版目录和 `language=ZH`，再选择独立 Hindi 目录和 `language=HI`，最后回到同一原版 cache object，能够真实完成三轮。`custom_model_path` 选择完整目录；没有运行时外挂 LoRA，也没有覆盖整个 IndexTTS 2.5 目录。

可复用步骤是：先全量核对原版与候选，记录回退目录与固定 hash；在隔离副本改 `custom_model_path` 和对应语言做有界验收；若失败，保留失败记录并恢复选择独立原版目录及原语言，而不是重命名/覆盖权重或清空生产缓存。原版 ZH→Hindi HI→原版 ZH 的这条短句循环已通过；任意原语言、长输入、加密开启或其他参数仍需各自验证。

`release_after_run=false` 会保留两套模型，因此不能承诺零额外显存或失败时自动恢复全部运行状态。本次自建进程退出后观察显存回到空闲基线，但这不是长期显存回收保证。生产插件尚未部署此次兼容改动；要推广到生产，仍需明确部署动作、审查实际配置和回滚对象，并保留原模型。

## 尚未通过的验收项

- Hindi 母语人工听评、充分原支持语言的 native-prompt 回归与旧语言无损。冻结 test 仅两条 Hindi，best100 严格 CER 50%；中文/日文各一例 Hindi-prompt stress 指标变差，详见[有限验收](FINITE_ACCEPTANCE.md)。
- 长 Hindi 分段、整句完整性、max_mel 截断与异常声音检查；`speech_completeness_verified=false`，短 WAV 存在不证明句尾完整。
- 任意输入、加密开启、不同精度/加速/缓存策略、其他节点或生产插件部署。原图八节点/五连接、三轮 queue 和此固定用例回切已通过，不再列作全部未执行。
- 生产运行时完全等价、长期缓存/显存上限和所有失败路径自动回滚。当前 `framework_runtime_equivalence_verified=false`，隔离前端缓存适配与确定性设置明确保留。
- 本轮三份整图 WAV 的新增分发授权与人工听评。它们仍私有，不因旧两样本已获公开授权而自动公开。

目前不需要追加训练来完成这些功能接口记录。保留 best100，不按 test 或本轮生成重选；工程链路通过不等于正式 Hindi 支持或生产全局替换建议。

## 初始化修复与 CPU 验证范围

6-5 第一次提交在任何节点执行前约 0.01 秒失败：bootstrap 过早导入 server/TTS，使 `comfy_aimdo.model_vbar` 的 `lib=control.lib` 在原 main 初始化前绑定到 None。已通过恢复真实 `main.py` 的自然初始化顺序及延迟安装观测 hook 修复；没有禁用 dynamic、改写官方 execution 或伪造 vbar library。第二次真实运行才取得上节三轮成功，第一次失败与清理记录保留。

CPU 证据按实际快照区分：此前完整 suite **311/311、0 跳过、36.874 秒**；参考 SHA 更正后相关 **25 项**通过；`object_info` 授权修复后相关 **26 项**通过。初始化 hook 的测试快照（bootstrap SHA 前缀 `af3c`）完整 suite **316/316、38.586 秒**；随后恢复原 loader 引用的最小 cleanup 改动，bootstrap SHA256 为 `b340744917b058a5826353bcc8cdf29e00b9e41fce5e1561ef1ac0b223237f9b`，相关 **30/30、0.057 秒**通过并经独立安全审查。**没有把最后增量的定向通过写成该最终源码又跑了一遍完整 316 项。** 最终真实 GPU 证据对应公开源码 `792305841240072e5a9f2616eb9e387126cae5a2`。

## 数据与许可

实际训练只使用 `ai4bharat/indicvoices_r` 的 Hindi，revision `5f4495c91d500742a58d1be2ab07d77f73c0acf8`，两个 TRAIN 分片共 946,963,686 bytes；pilot 64 clips / 16 speakers，train/val/test 按 speaker 隔离。不是 Rasa；本项目 held-out test 不是发布者官方 TEST。来源 CC-BY-4.0 归属及变更说明、官方 Bilibili 中英文完整许可和非背书声明随公开包保留，工具 MIT 不替代模型许可。参考音频、训练语料及私有回执不进公开 Git/HF。

### 参考音频哈希记录更正

历史整图 metadata 预检曾因参考 SHA 不匹配而正确停止；该次预检未启动 GPU。经原始PCM16转换记录、迁移前已成功四节点result.json和当前文件全SHA三向核对，正确值为`fd23ddac68f2db3710481c7cbfc1a50a117d9c38bc5c3ee51c33444f0a9b019a`，与既有音频一致。前版文字摘要/命令中的SHA有转录错误；本次更正固定值，未更换参考音频、模型或放宽校验。失败回执保留。

## 6-1 历史定向取消与资源准入记录

用户明确允许队列空闲后释放模型缓存，并进一步授权取消阻塞任务。实际执行使用原 ComfyUI 的定向 `interrupt` 接口，绑定当时唯一运行任务的 ID，只发送一次；请求未采用无目标 ID 的全局中断方式，未杀死或重启原服务，也没有修改生产模型、插件或调度配置。私有任务 ID 与完整图未写入公开记录。

- 第一次空闲维护：被观察任务已自然结束，没有发出取消请求。连续空队列后释放缓存，返回 HTTP 200；观察到约 950 MiB 显存占用。后续第 2 次控制器执行仍被新派入任务触发的资源门槛拒绝。
- 第二次维护：定向取消返回 HTTP 200 后继续检查；最终目标任务历史包含 `execution_interrupted`，队列为 0 running / 0 pending，才确认取消生效。连续三次空队列后调用缓存释放，返回 HTTP 200；观察到 used 834 MiB、free 31,277 MiB、utilization 0%。
- 第 3 次控制器执行：重新核对固定源码和完整资产，资源准入时却已观察到 used 31,456 MiB、free 565 MiB、utilization 92%。控制器返回 `Shared GPU/RAM admission failed; no model is loaded`，未启动自建验证服务或模型推理；原服务继续接收任务。

因此没有反复取消新派入任务，也没有把取消成功写成整图验收成功。当时曾请求约 15 分钟不向 6-1 派入新任务的维护窗口，原服务保持运行；后来本项目改在 6-5 的可用窗口完成上述真实三轮，不再把这个历史等待写作当前阻塞。采样中止阈值不是 GPU 硬配额，不能保证与持续新派入的生产任务互不争抢。

第 3 次准入退出后，只读检查确认本项目 controller/bootstrap 没有活动进程；原 ComfyUI 队列接口仍为 HTTP 200，当时有 1 个运行任务、0 个等待任务。该健康快照只证明接口响应与验证进程退出，不证明生产任务或模型质量验收成功。


## 现有 5090 环境启用：首次更新与日常切换

补充日期：2026-11-18（Asia/Saigon）。**本节是尚未执行的生产推广步骤；本轮只更新文档。** 已通过的是独立兼容副本的三轮短样本，不是现网插件已部署。以下按本项目最后核验的原 T8 基线 `1ff425ece89ef9824f58f0e4dbfb057991814e88` 说明；部署前仍须核对目标实例实际版本。

### 现在能否只覆盖模型

**现在不能只把新 `gpt.pth` 覆盖到原目录。** 原插件会在加载前拒绝 `HI`，原始模型大小/哈希清单也不接受本候选；`verify_hashes=false` 仍不能绕过大小检查。5090 的 CUDA/BF16 硬件路径已验证，不需要因此重新训练或全局升级 Torch/ComfyUI；缺的是一次同插件兼容部署、完整模型资产和现网验收。

| 阶段 | 需要做什么 | 是否每次都要做 |
|---|---|---|
| 第一次启用 | 部署同一个 T8 插件的兼容版，保留节点类型/ID/连线；准备两套完整目录；维护重启加载新代码并验收 | 一次性，版本变化时重新核对 |
| 日常在已登记 base/best100 间切换 | 改节点1的 `custom_model_path` 和节点4的 `language` | 每次切换这两个字段；不因切目录而重启 |
| 换成新的训练 checkpoint | 登记新权重的真实身份、清单和适用语言，重新校验与验收 | 当前仅锁定 base/best100，不能任意替换未知 GPT |

不增加 LoRA 节点：公开 GPT 已合并训练增量。原模型目录保留，Hindi 另用独立目录，避免修改原文件和已缓存对象。

### 首次启用前准备什么

1. **插件与备份。** 保留当前 T8 源码、所有本地改动、工作流 JSON、原模型引用与哈希。备份放到 `custom_nodes` 扫描路径之外；不要让原版和兼容副本同时被自动加载，否则同名节点注册可能冲突。
2. **两套完整模型。** `BASE_MODEL` 是本次验证使用的官方 2.5 完整目录，`HINDI_MODEL` 是另一个完整目录；各核验清单所需的 26 项资产及 GPT 身份。HF 13文件包不能直接充当 T8 的完整 `model_dir`，还缺清单中的辅助资产。此前原宿主目录的 `config.yaml` 曾显示 `version: 2.0`；不能仅凭文件夹叫 IndexTTS-2.5 就把它当成本次已验的 base。新目录使用匹配的官方 2.5 配置，旧目录保留不覆盖。
3. **运行环境。** 保留已核的 Python/Torch/CUDA、ComfyUI 基线及既有受审 ops 修正；不是复制整个验证 stage/core，也不是在生产里安装验证 bootstrap/import hook。正常生产入口仍按现有官方 `main.py` 启动流程；不要额外提前 import server/TTS、重复初始化 control 或关闭 DynamicVRAM 来套用本次修复。
4. **工作负载。** 暂停向目标实例派新任务，等待 running/pending 都为0，再核对实际 GPU、RAM 和持久存储。`release_after_run=false` 可能同时保留两模型；本次11,799 MiB（约11.52 GiB）只是三轮结束点采样，不是峰值或与图像任务并发的显存保证。

兼容代码的已实现准备入口如下；`T8_SOURCE` 须是已核的固定原始 Git 源目录，`STAGED_T8` 须是扫描路径外的全新暂存目录，父目录须已存在且使用规范路径。**这些命令只准备/校验代码副本，不部署生产、不准备模型，也不重启服务。**

```bash
python scripts/check_t8_compat.py inspect --node-root "$T8_SOURCE"
python scripts/check_t8_compat.py prepare \
  --node-root "$T8_SOURCE" --destination "$STAGED_T8" \
  --candidate-revision 117c7922e4dcedc3f9885f3f6efa5be8f54c7919
python scripts/check_t8_compat.py verify-copy \
  --node-root "$STAGED_T8" \
  --candidate-revision 117c7922e4dcedc3f9885f3f6efa5be8f54c7919
```

### 一次性上线顺序

1. 完成上述备份、双目录完整校验和代码暂存，保留完整模型许可。
2. 在目标实例无新派单、队列清空的维护窗口，按现有发布流程把兼容代码部署为**唯一活动 T8 插件**；节点类型、ID、连线仍沿用原工作流。本项目没有自动部署子命令，不把 `prepare` 成功当成上线完成。
3. 用当前环境既有启动器做一次受控维护重启，使后端加载新代码，再刷新 ComfyUI 前端节点 schema，确认语言选项包含 `HI`、实际节点来自预期兼容版。具体服务管理命令须先核对实际启动器，本说明不猜测 `kill`、Pod重建或集群重启命令。
4. 在真实生产启动方式下，先原版短句→Hindi短句→原版短句，核对实际模型路径/哈希、最终音频、旧语言输出、缓存和显存，再恢复派单。原有外部音频 URL、正常文本前端缓存、并发和加密策略应按实际使用验收；本次只验证了 speaker 情绪模式、加速/kernel关闭等已列配置。

本次验证使用过隔离 loopback 服务/数据库/缓存、真实 normalizer 的缓存目录适配、确定性设置和只读观测；这些没有随兼容副本自动部署到现网，所以不能把隔离成功写成生产运行时完全等价。

### 日常切换具体改哪里

| 原工作流字段 | 使用 Hindi | 回到本次已验原版 |
|---|---|---|
| `"1".inputs.custom_model_path` | `HINDI_MODEL` 对应的容器内实际绝对目录 | `BASE_MODEL` 对应的容器内实际绝对目录 |
| `"4".inputs.language` | `HI` | 原语言，如 `ZH` |

`custom_model_path` 填**目录**，不是 `gpt.pth` 文件、ZIP、HF网址或Mac路径。以下仅为覆盖字段示例，须合并进原完整工作流；占位路径不能原样提交，JSON不会自动展开 `$HINDI_MODEL` 等变量。

```json
{
  "1": {"inputs": {"custom_model_path": "<HINDI_MODEL 的实际绝对目录>"}},
  "4": {"inputs": {"language": "HI"}}
}
```

节点类型、ID和5条连线不变；参考音频、文本仍按业务输入提供。无需外挂adapter，日常切换已登记的这两套目录无需因为切换而重启；首次加载仍会做完整资产哈希检查。原语言继续选择原版目录，不把 Hindi 候选设成所有语言的全局默认。

### 回滚分两层

- **模型级回切：** 把上述两个字段恢复到已验 `BASE_MODEL` 和原语言，不覆盖或删除任何权重。本次固定用例已确认同进程复用原对象，首尾WAV及PCM哈希一致；这不是所有文本的质量保证。
- **插件级回滚：** 如果新代码无法启动或现网验收失败，保持停止派单，恢复扫描路径外保存的原插件及原工作流/原模型引用，再通过既有启动器维护重启并复核服务。原工作流曾使用空 `custom_model_path` 时，代码级回滚应恢复其原始引用；它与切到本次新建官方2.5 base目录是两种不同操作。

完成接口接入仍不等于 Hindi 质量通过：当前 best100 是实验候选，母语人工听评、充分旧语言回归及长句完整性仍未通过本次验收。
