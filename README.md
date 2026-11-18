# IndexTTS 2.5 Hindi / H200

原创 Hindi 数据工具与实验性原生训练实现，直接调用固定官方模块，不导入或复制未许可社区代码。**当前方向已改为重新训练 Hindi 专用的纯 attention LoRA：原模型文件不改，全部 head/embedding 冻结；HI 使用独立 adapter 实例，旧语言使用从未加载该 adapter 的原模型实例。** 新 profile 与独立推理接口正在实现/验证，尚未完成新一轮 GPU 训练，也没有新的纯 LoRA 权重或质量结论。见 [Hindi 纯 LoRA 方案与进度](docs/HINDI_LORA.md)。

**历史 best100 的真实训练、完整 GPT 导出与 stock 生成已实现，正式质量 gate 仍未通过。** 最新64条/16人pilot按固定验证loss选出的best100已完成标准adapter转换、完整GPT合并、H200 FP32 stock parity和两句真实WAV。相同ASR口径下，新4词句CER从0.7222降至0.2222，未参与训练的VAL句从0.7449降至0.3469；这是小样本改善，ASR对真人参考也有明显误差，不能据此认证整体Hindi质量。人工Hindi试听与完整旧语言回归尚未完成，保留实验标识；本次公开包的许可复核和真实加载结果见[发布记录](docs/PUBLIC_RELEASE.md)。原模型未替换；已合并的完整GPT实验权重已[发布至HF](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental/tree/117c7922e4dcedc3f9885f3f6efa5be8f54c7919)，它不是整套声学模型。

完整范围见 [VALIDATION.md](VALIDATION.md)。初始化顺序修复的完整 CPU suite 为316/316通过、0跳过、38.586秒（af3c57be文件版本）；最终loader清理增量后另有30/30相关CPU检查和独立安全审查通过。旧源码`7c950e3`的278项与H200宿主189项记录保留各自版本范围。CPU检查与随机模型gate用于机制验证，实际微调在GPU上运行；不能据loss下降或WAV存在宣称音质达标。官方tokenizer的hi槽位17和`<|hi|>`不代表官方训练过Hindi。

最新冻结test与旧语言Hindi-prompt stress结果见 [有限验收记录](docs/FINITE_ACCEPTANCE.md)：两条Hindi test的严格CER仍为50%，中日各一例出现识别差异；人工听评与完整旧语言回归未完成，质量发布gate未通过。

你下载的两份试听ZIP都没有模型权重。真实GPT候选的独立加载与原始ComfyUI节点限制见 [INSTALL_EXPERIMENTAL](docs/INSTALL_EXPERIMENTAL.md)；公开13文件包使用新脚本入口，不能套用私有原生bundle步骤。

**历史 best100 的 RTX 5090 隔离完整工作流与同进程原版→Hindi→原版功能验收已通过。** 保留原8节点类型/ID及5条连线，实际依赖的6节点每轮真实执行；auto实际选择CUDA/BF16，原模型缓存保留并复用，首尾原版WAV及PCM完整哈希相同。见[5090切换手册](docs/RTX5090_SWITCHING.md)和[实际回执](docs/acceptance/rtx5090-switch-cycle-20261117.json)。这是固定短样本、隔离兼容路径的接口验证；生产插件尚未部署更新，不保证所有文本/旧语言质量无退化。未修改的T8基线仍拒绝HI及候选原始清单，先需同一插件的一次兼容更新；无需新增LoRA节点，旧模型保持不动。这不是新纯 LoRA 的 T8 接入或 GPU 验证。此前两份四节点[生成试听与完整许可](docs/audio/rtx5090-core/README.md)已按用户授权公开，它们与本次三轮回执分开标识。

**直接替换/单候选全局默认方案已按最新要求撤回**，仅保留 [历史方案与参数审计](docs/SINGLE_MODEL_UPGRADE.md)。旧 best100 同时更新了 attention LoRA、Hindi/文字行及两个完整 head；它不是纯 LoRA，不能改名复用，关闭或删除 A/B 也不能还原这些 head/embedding 变化。新训练从已核的官方原始 GPT 开始，使用 `attention_lora_only_v1`；GitHub/HF 后续同步新的源码、adapter 身份与验证范围，不覆盖或重命名旧 best100 发布物。

## 模型原理与微调依据

[原理与方法说明](docs/MODEL_RATIONALE.md)按固定源码解释文字到语义 codes、CFM/DiT 和声码器的实际接口；其中双 head 更新和完整 GPT 合并描述的是旧 best100 路线。新纯 LoRA 的全部 head/embedding 冻结范围见 [HINDI_LORA](docs/HINDI_LORA.md)。设计假设、论文与实现差异及质量证据分开列明。两张[推理结构图](docs/diagrams/indextts25-inference.html)和[训练/合并图](docs/diagrams/hindi-lora-training.html)为自含HTML，使用diagram-design制作；训练/合并图对应旧行/head方案，不是新纯 LoRA 的冻结范围；来源和压缩范围见[图示账本](docs/diagrams/SOURCES_AND_FIDELITY.md)。

## 原生训练路线

当前纯 LoRA 使用 [独立配置](configs/native-lora-only.example.json) 和 [新方案的训练前 gate](docs/HINDI_LORA.md)；示例默认 CPU/fp32，不会隐式启动 GPU。旧 profile、历史配置和恢复记录继续保留，不得混用 checkpoint 或把旧数据结果当作新训练结果。

从 [原生指南](docs/NATIVE.md) 开始，并先阅读 [安全与依赖范围](SECURITY.md)；完整原模型/合并模型的离线波形验收入口见 [合成指南](docs/SYNTHESIS.md)，验证点转adapter流程见 [Checkpoint选择](docs/CHECKPOINT_SELECTION.md)。配置示例为`configs/native-dataset.example.json`、`native-train.example.json`和`native-h200.example.json`。

1. 在独立持久目录准备Python3.11环境与官方固定源码，保留许可证。核心数据工具零第三方运行依赖；训练依赖通过本项目`[native]`extra或`requirements/native.in`安装。
2. 先运行资产助手的元数据模式，检查具体文件、体积和许可；显式下载只选已锁定资产，不拉全模型仓库、不覆盖现有模型。
3. 只读取已授权Hindi数据的小批次，保留来源/revision/许可记录，运行NFC、音频、去重与speaker隔离校验。
4. 用冻结的官方编码器生成原生特征并逐条校验，CPU检查与随机tiny BF16 gate用于机制验证，完整编码器和实际微调仍按共享GPU/宿主内存实时余量分阶段执行。
5. 先小样本过拟合，再真实STOP/恢复、100次optimizer更新smoke，最后做 [Hindi/旧语言和发布评测](docs/EVALUATION.md)。参数只是实验起点，不承诺GPU小时或音质。

原生实现使用无 padding 的 batch1 加梯度累积。新 `attention_lora_only_v1` 只允许 attention Conv1D 的 A/B 参与优化，语言/文字 embedding（包括 Hindi 行17）、所有 head、其余 GPT 与声学模块冻结；旧 `legacy_rows_heads_v1` 保留原来的行/head 更新语义，历史 best100 仅属于旧路线。旧 best100 的 fresh stock、H200 parity 和 T8 隔离三轮证据不能转记为新纯 LoRA 验证。新 adapter 及其独立推理接入需单独验收，不保证普通 ComfyUI LoRA、社区 adapter、vLLM 或 DeepSpeed 兼容。

## CPU 数据工具快速检查

从本仓库根目录、真实数据盘运行：

```bash
mkdir -p .runtime/tmp .runtime/cache
export TMPDIR="$PWD/.runtime/tmp"
export XDG_CACHE_HOME="$PWD/.runtime/cache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src"
python3 -m unittest discover -s tests -p test_data.py -v
python3 -m hi_h200 --help
python3 scripts/make_demo.py .runtime/demo
python3 -m hi_h200 prepare .runtime/demo/input.jsonl --audio-root .runtime/demo --output .runtime/prepared
python3 -m hi_h200 validate .runtime/prepared --audio-root .runtime/demo
```

目标目录须全新。六个demo文件是正弦音，与文字不对齐，只用于CPU管线检查，不能训练或评测语音。完整模型/特征/恢复测试需要真实Torch依赖和固定官方checkout，见原生指南；依赖缺失时跳过的测试不算通过。

raw UTF-8 JSONL每行必填：

```json
{"id":"hi_spk_a_clip_a","speaker_id":"spk_a","audio":"spk_a/clip_a.wav","text":"कृपया दरवाज़ा खोलिए।","language":"hi","source":"your-authorized-corpus","license":"your-reviewed-terms","consent":true}
```

[数据规范](docs/DATA.md) 说明保守NFC、原文保留、nukta/virama/joiner、数字/货币双确认、PCM WAV长度、去重与同speaker不同clip配对。最低三个speaker、每人两条；正式泛化评测要求更多独立speaker。许可/consent字段是审核记录，不是权利证明。

## 数据与许可

[Rasa/IndicVoices-R访问说明](docs/DATA_SOURCES.md)：gated条件必须用户本人处理，随后还要核实实际认证和明确配置`Hindi`。不要整库下载。Rasa缺可信speaker映射时不可用来保证隔离；正式test音频不得参与训练prompt或调参。

本仓库原创代码MIT；官方模型、官方源码、辅助权重和数据各循其许可，见 [NOTICE](NOTICE.md) 与 [上游记录](docs/UPSTREAM.md)。下载助手保留模型许可和模型卡；本项目不发布语料、原始权重或私人特征。HF认证只放独立本地目录，禁止把token发到聊天、命令或仓库。

外部社区兼容检查仍保留在`scripts/check_backend.py`等工具中，其训练器缺许可证，必须先取得明确许可。该路线的 [H200参考指南](docs/H200.md) 与 [恢复说明](docs/INTERRUPTION.md) 不适用于原生格式；原生路线本身不依赖它。

历史 best100 pilot 离线复用两份合法已有 Parquet，64条/16人、完整utterance max20秒；保留旧split的`prepared-frozen`，train48/val8/test8特征全部实际提取。从原始base独立完成BF16第20步到新进程第500步恢复训练，恢复段wall吞吐约1.3167 updates/s（含加载、验证和checkpoint），本进程Torch CUDA allocated峰值约5.77 GiB。8例验证loss在第100步最低后持续上升，已停止扩大该run，不选last500；best100的转换、7文件完整合并、parity、两句stock WAV、配对审计与六cases GPU ASR均实际完成。checkpoint按5个固定验证点的loss选择，未用ASR挑选；VAL与train按speaker隔离，但参与选模，最终test尚未质量验收。

v1、v2和pilot均从原始base独立开始，不能合计为一个800步模型。早期v1新句未改善、v2只在已知TRAIN句改善及noKV/conditioning诊断保留于验证记录；默认cache保持True。历史原始review ZIP保留未改。私下听评应使用已补入完整双语协议和数据归属说明的`WORK/outputs/review/pilot-best100-licensed-review.zip`（1,041,300 bytes，16条目，原5 WAV哈希不变）；全部有限验收另存为`WORK/outputs/review/frozen-best100-finite-private.zip`（4,707,736 bytes，76条目，19 WAV，含全部14生成）。两包已由操作者自行下载，并在本机完成清单和许可核验；未上传Library、未公开，交付和许可边界见 [DISTRIBUTION](docs/DISTRIBUTION.md)。

HF公开实验权重已发布：[固定revision `117c7922e4dcedc3f9885f3f6efa5be8f54c7919`](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental/tree/117c7922e4dcedc3f9885f3f6efa5be8f54c7919)；后续已在5090独立目录匿名完整下载13文件，GPT完整流式SHA与冻结值匹配，详见[发布记录](docs/PUBLIC_RELEASE.md)；[导出说明](docs/EXPORT.md) 保留真实评测与权利审核门槛。先前训练与stock合成任务已结束，5090隔离四节点已有上述两份WAV，后续集群操作按迁移安排暂停；`RUN_STATE.json`记录selected100、active training=false、quality_gate_passed=false及来源版本；此前两台已用共享pod的ComfyUI均未被我们停止、重启或修改；HTTP200仅在训练所在Pod的localhost直接实测，原Pod仅由Kite UI确认Ready1/1、restarts0。输出保留在独立持久目录；访问异常的未定原因与最终资源快照见验证记录。
