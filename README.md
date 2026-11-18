# IndexTTS 2.5 Hindi / H200

[产品交付实测与放行结论](docs/PRODUCT_DELIVERY.zh.md)记录真实完整 GPT、5090 三轮切换和 Hindi P0 质量阻塞。

[Hindi 独立验收与旧语言回归](docs/HINDI_INDEPENDENT_ACCEPTANCE.zh.md)记录新说话人测试、长段停止码风险和五语对照。

[Hindi 优化阶段记录](docs/HINDI_OPTIMIZATION.zh.md)包含 H200 分段与 ASR 对照，以及独立纯 LoRA 训练的冻结证据。

[换模验收方案 v1.0](docs/REPLACEMENT_ACCEPTANCE.md)覆盖 HI 独立切换、完整 GPT 全量替换、5090 原图测试、质量门槛和回滚。

原创 Hindi 数据工具与实验性原生训练实现，直接调用固定官方模块，不导入或复制未许可社区代码。**当前方向已改为重新训练 Hindi 专用的纯 attention LoRA：原模型文件不改，全部 head/embedding 冻结；HI 使用独立 adapter 实例，旧语言使用从未加载该 adapter 的原模型实例。** 新纯 LoRA 的 H200/BF16 run002 已真实从第2步停止后跨进程恢复至100 updates，并产生 step100 adapter；完整文件 SHA 与受限 CPU tensor 实物审计已核对；固定4例 TRAIN 的真实机制 gate 已通过；冻结pilot候选2/2句native未合并真实WAV及独立file/PCM回读已完成，质量验收仍未完成。另一个独立原始 base 的 pilot48 已真实完成2→100恢复，按预声明VAL8均值最低选出并冻结step100，候选SHA为 `15c080715b3b50bf38b33a70f196a9c9e987446e0a6bb87b2e0c8b7835f57dbd`；2/2句未合并native真实合成已完成，纯LoRA ZIP已实际上传并匿名核验；语音质量验收尚未完成。见 [Hindi 纯 LoRA 方案与进度](docs/HINDI_LORA.md)。

**历史 best100 的真实训练、完整 GPT 导出与 stock 生成已实现，正式质量 gate 仍未通过。** 最新64条/16人pilot按固定验证loss选出的best100已完成标准adapter转换、完整GPT合并、H200 FP32 stock parity和两句真实WAV。相同ASR口径下，新4词句CER从0.7222降至0.2222，未参与训练的VAL句从0.7449降至0.3469；这是小样本改善，ASR对真人参考也有明显误差，不能据此认证整体Hindi质量。人工Hindi试听与完整旧语言回归尚未完成，保留实验标识；本次公开包的许可复核和真实加载结果见[发布记录](docs/PUBLIC_RELEASE.md)。原模型未替换；已合并的完整GPT实验权重已[发布至HF](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental/tree/117c7922e4dcedc3f9885f3f6efa5be8f54c7919)，它不是整套声学模型。

完整范围见 [VALIDATION.md](VALIDATION.md)。初始化顺序修复的完整 CPU suite 为316/316通过、0跳过、38.586秒（af3c57be文件版本）；最终loader清理增量后另有30/30相关CPU检查和独立安全审查通过。旧源码`7c950e3`的278项与H200宿主189项记录保留各自版本范围。CPU检查与随机模型gate用于机制验证，实际微调在GPU上运行；不能据loss下降或WAV存在宣称音质达标。官方tokenizer的hi槽位17和`<|hi|>`不代表官方训练过Hindi。

最新冻结test与旧语言Hindi-prompt stress结果见 [有限验收记录](docs/FINITE_ACCEPTANCE.md)：两条Hindi test的严格CER仍为50%，中日各一例出现识别差异；人工听评与完整旧语言回归未完成，质量发布gate未通过。

你下载的两份试听ZIP都没有模型权重。真实GPT候选的独立加载与原始ComfyUI节点限制见 [INSTALL_EXPERIMENTAL](docs/INSTALL_EXPERIMENTAL.md)；公开13文件包使用新脚本入口，不能套用私有原生bundle步骤。

**历史 best100 的 RTX 5090 隔离完整工作流与同进程原版→Hindi→原版功能验收已通过。** 保留原8节点类型/ID及5条连线，实际依赖的6节点每轮真实执行；auto实际选择CUDA/BF16，原模型缓存保留并复用，首尾原版WAV及PCM完整哈希相同。见[5090切换手册](docs/RTX5090_SWITCHING.md)和[实际回执](docs/acceptance/rtx5090-switch-cycle-20261117.json)。这是固定短样本、隔离兼容路径的接口验证；生产插件尚未部署更新，不保证所有文本/旧语言质量无退化。未修改的T8基线仍拒绝HI及候选原始清单，先需同一插件的一次兼容更新；无需新增LoRA节点，旧模型保持不动。这不是新纯 LoRA 的 T8 接入或 GPU 验证。此前两份四节点[生成试听与完整许可](docs/audio/rtx5090-core/README.md)已按用户授权公开，它们与本次三轮回执分开标识。

**直接替换/单候选全局默认方案已按最新要求撤回**，仅保留 [历史方案与参数审计](docs/SINGLE_MODEL_UPGRADE.md)。旧 best100 同时更新了 attention LoRA、Hindi/文字行及两个完整 head；它不是纯 LoRA，不能改名复用，关闭或删除 A/B 也不能还原这些 head/embedding 变化。新训练从已核的官方原始 GPT 开始，使用 `attention_lora_only_v1`；GitHub/HF 后续同步新的源码、adapter 身份与验证范围，不覆盖或重命名旧 best100 发布物。

## 新纯 LoRA 实际进度

**pilot48 已真实完成并冻结step100。** 从原始 base 独立开始，非 run002 adapter 继续；旧合法 pilot64 为 TRAIN48/12人（491.029s）、VAL8/2人（82.950s）、TEST8/2人（69.924s）。H200/BF16、累积4、lr1e-4、warmup10、max100、seed260618、每25步保存/验证。fresh2和新进程resume100均exit0、verified2/100、自有进程退出，wall分别40.20832155458629和110.89273961819708秒；严格完成1..100，不与run002累加，wall不是纯GPU时间。

预声明25/50/75/100的VAL8均值依次为8.110908448696136、7.306136906147003、7.053911030292511、6.989858150482178，按最低值选100并写FROZEN_SELECTION，TEST/ASR不参与或事后改选。发布候选5,931,907 bytes / SHA256 `15c080715b3b50bf38b33a70f196a9c9e987446e0a6bb87b2e0c8b7835f57dbd`已完整读取核对；进程Torch peak allocated/reserved为4,043,693,056 / 4,339,007,488 bytes。该候选与下述run002小集机制adapter不同；2/2句native unmerged真实WAV已执行成功并独立回读；质量/独立TEST/5090/T8 pure未通过，独立HF仓库的ZIP权重包已在固定revision `88e6297852699eb6dfbf6d4677446d37afcacf5f` 上传，匿名完整ZIP/manifest/adapter SHA核对通过。实际恢复没有与连续运行逐位对照，不宣称bitwise连续等价。

训练源码 `ed19c4a511be4baafe87b22f74adf6b192a0a55a`；合法小集为 TRAIN16/4人、VAL4/1人、TEST4/1人，未扩集。本轮 rank8/alpha16/dropout0、lr3e-4、warmup5、累积1、max100、seed260618，所有 head/embedding 与基础参数冻结。fresh2 与新进程 resume100 均 exit0、verified step 正确且自有进程退出；共100次更新。step100 adapter 为5,931,907 bytes，SHA256 `902d0729e9b6ac340bfd6e0282294865b184dcc73a8a90dd711df310cdb6cff1`，受限 CPU 实读已核精确96个 FP32/finite A/B tensor、1,474,560参数、step100及纯 profile，前后文件SHA一致、CUDA未初始化；文件仍在受控 Pod，未公开权重。

失败001因 reentrant checkpoint / 全冻输入导致梯度链断，0 updates / 0 checkpoints，记录保留；pure-only non-reentrant 修复不解冻基础参数。最新修复后完整CPU **369/369、43.927秒** 通过（绝对官方源码路径、CUDA隐藏、Hub离线），覆盖non-reentrant修复、pure_overfit和15项eval guard；评估源码已发布 `31fd9fe11d84d5f328afa36c49f363dbe6f9cf24`，训练来源仍是ed19c4a。353项旧快照保留为历史，不替代这次回归。末TRAIN单条loss与VAL均值不用于证明下降。另一次固定4例TRAIN GPU机制比较得到before `9.192850112915039`、after `6.462177276611328`、ratio `0.7029568846698167≤0.9`，模块gate与冻结状态检查通过，控制器exit0/自有进程退出、wall48.250421秒；独立operator-binding核对数据身份一致，模块内部绑定标志仍false。该TRAIN机制gate不是质量；两句独立真实合成另有实际证据，语音质量与T8兼容仍未通过，详见 [HINDI_LORA](docs/HINDI_LORA.md)。独立[pure HF仓库固定权重revision](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-LoRA-experimental/tree/88e6297852699eb6dfbf6d4677446d37afcacf5f)现已公开提供 `Hindi-LoRA-step100-public.zip`：6,941,415 bytes，ZIP SHA256 `1ede2c2ebc0a196d7cdb3013941487500d21f02269b790eab8cca8da6e9c29fd`；匿名完整下载、26项CRC、manifest与adapter SHA均核对通过，见[发布包回执](docs/acceptance/pure-lora-public-package-20261118.json)。仓库该revision只有ZIP、README和.gitattributes，26文件在ZIP内；网页封面后于09bc09ae更新，包内冻结模型卡a53保持不变。[纯LoRA模型卡](docs/HF_PURE_LORA_CARD.zh.md)不等于质量或T8通过。

冻结pilot48候选已由source31fd在H200通过精确training base identity/96A/B后，构造48模块真实wrapper完成第一句未合并WAV；原GPT磁盘文件未换，KV=true，old-instance touched=false。第一例169,038 bytes、84,480帧、3.8312925170068026秒，file SHA `643b7e5f032039b353e8c935de85f7ce830e2bd29e6195dfb0d3bfa5ebd78810`、PCM SHA `eaaccc3a8af22e0c48ce15c778162380999416739468538dfeeb804c3240a243`独立回读一致；guard exit0/自有进程退出。第二例也已成功：84,558 bytes、42,240帧、1.9156462585034013秒，file SHA `562d2a8632777cc001b0d629ec56877a10abad35b4e418d35cd5b57be05542cd`、PCM SHA `0d84b07822247419df5947ba7851064281207f3877b380c11f6424c863b591d0`独立回读一致，guard exit0/自有子进程退出。因此只记2/2 native工程句功能成功，不当作独立TEST或质量。原GPT最终完整SHA仍43a8…，原ComfyHTTP200/队列0/0；新WAV未授权公开。greedy/beams1/penalty1/textnorm=false/maxmel256与原T8图采样参数不同；两句功能成功不等于原工作流、新5090、T8 pure或质量通过，依赖版本差异见[HINDI_LORA](docs/HINDI_LORA.md)。

## 下载与校验纯 LoRA 发布包

使用[固定ZIP下载链接](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-LoRA-experimental/resolve/88e6297852699eb6dfbf6d4677446d37afcacf5f/Hindi-LoRA-step100-public.zip)，不按变化中的main猜测权重身份。以下目录须全新；先验证压缩包完整SHA，再解压并验证包内清单：

```bash
set -euo pipefail
mkdir hindi-lora-step100-88e62978
cd hindi-lora-step100-88e62978
curl --fail --location --output Hindi-LoRA-step100-public.zip \
  "https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-LoRA-experimental/resolve/88e6297852699eb6dfbf6d4677446d37afcacf5f/Hindi-LoRA-step100-public.zip"
printf '%s  %s\n' \
  "1ede2c2ebc0a196d7cdb3013941487500d21f02269b790eab8cca8da6e9c29fd" \
  "Hindi-LoRA-step100-public.zip" | shasum -a 256 -c -
mkdir package
unzip Hindi-LoRA-step100-public.zip -d package
cd package
shasum -a 256 -c SHA256SUMS
```

包内 `adapter.pt` 是原生纯A/B，不是完整GPT或通用ComfyUI LoRA。原始base及声学权重未包含，仍须匹配的原模型、精确training manifest与完整合成依赖；不要将ZIP或adapter直接覆盖原GPT。许可、base-metadata和详细前置见[HINDI_LORA](docs/HINDI_LORA.md)。上传核验不代表独立TEST/ASR/母语质量、新5090或T8 pure已通过。

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

历史 best100 发布及迁移时点记录：HF公开实验权重已发布：[固定revision `117c7922e4dcedc3f9885f3f6efa5be8f54c7919`](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-best100-experimental/tree/117c7922e4dcedc3f9885f3f6efa5be8f54c7919)；后续已在5090独立目录匿名完整下载13文件，GPT完整流式SHA与冻结值匹配，详见[发布记录](docs/PUBLIC_RELEASE.md)；[导出说明](docs/EXPORT.md) 保留真实评测与权利审核门槛。先前训练与stock合成任务已结束，5090隔离四节点已有上述两份WAV，当时后续集群操作按迁移安排暂停；`RUN_STATE.json`记录selected100、active training=false、quality_gate_passed=false及来源版本；此前两台已用共享pod的ComfyUI均未被我们停止、重启或修改；HTTP200仅在训练所在Pod的localhost直接实测，原Pod仅由Kite UI确认Ready1/1、restarts0。输出保留在独立持久目录；访问异常的未定原因与最终资源快照见验证记录。
