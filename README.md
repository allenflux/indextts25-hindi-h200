# IndexTTS 2.5 Hindi / H200 训练准备

这是原创的 Hindi 数据准备、训练前检查与实验操作项目，目标是在单张 H200 上探索 IndexTTS 2.5 的 Hindi LoRA。**不是已训练好的 Hindi 模型，也不是官方 Hindi 支持声明。** 没有捆绑 trainer、模型、语料或第三方源码。当前只验证 CPU 工具；没有进行 GPU 训练，没有测量 GPU 性能或质量。

官方 2.5 tokenizer 的 `hi` 是第 17 个零基语言槽，并存在 `<|hi|>` token；这仅说明接口预留。社区 Turkish 微调路线可作为外部后端参考：冻结声学后端，训练 GPT LoRA、目标语言/文本行与输出 heads。Hindi 可行性仍需实验。锁定版本与许可边界见 [NOTICE](NOTICE.md)、[上游核查](docs/UPSTREAM.md) 和 `upstream.lock.json`。

## 先运行 CPU 工具

Python 3.11–3.13，CPU 核心工具零第三方运行依赖。直接使用源码即可，不必 pip 安装。所有工作与缓存放在真实挂载下；示例不自动下载任何东西：

```bash
# 从数据盘上的本仓库根目录运行
mkdir -p .runtime/tmp .runtime/cache
export TMPDIR="$PWD/.runtime/tmp"
export XDG_CACHE_HOME="$PWD/.runtime/cache"
export HF_HOME="$PWD/.runtime/cache/huggingface"
export TORCH_HOME="$PWD/.runtime/cache/torch"
export PIP_CACHE_DIR="$PWD/.runtime/cache/pip"
export UV_CACHE_DIR="$PWD/.runtime/cache/uv"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src"
python3 -m unittest discover -s tests -v
python3 -m hi_h200 --help
python3 scripts/make_demo.py .runtime/demo
python3 -m hi_h200 prepare .runtime/demo/input.jsonl --audio-root .runtime/demo --output .runtime/prepared
python3 -m hi_h200 validate .runtime/prepared --audio-root .runtime/demo
```

演示文件是合成正弦音，只有管线测试用途，**与示例文字不对齐，禁止拿去训练或评测语音质量**。示例输出目录若已存在，换一个新目录；工具拒绝覆盖。

## 自己的 Hindi 语料

输入为 UTF-8 JSONL，每行一条 clip，例如（示例路径并不存在）：

```json
{"id":"hi_spk_a_clip_a","speaker_id":"spk_a","audio":"spk_a/clip_a.wav","text":"कृपया दरवाज़ा खोलिए।","language":"hi","source":"your-authorized-corpus","license":"your-verified-terms","consent":true}
```

至少三个 speaker，每个至少两条独立 clip，才能产生非空 train/val/test 与同说话人不同 clip 提示配对。生产验证集应远大于这个最低值。完整 schema、Unicode、数字策略、去重与拆分规则见 [数据规范](docs/DATA.md)；Rasa/IndicVoices-R 的 gated 访问与说话人限制见 [候选数据来源](docs/DATA_SOURCES.md)。只接受受限 PCM WAV；其他音频先离线转码并人工核对。按 speaker 拆分后再做特征提取，验证/测试音频或其特征不得用于训练提示。

## H200 操作顺序

1. 阅读 [许可与版本核查](docs/UPSTREAM.md)。社区训练器没有 LICENSE；先取得许可或更换兼容后端，本项目不会替你授予权利。
2. 运行 CPU 数据检查和源代码预检。准备自己的授权记录、完整模型文件及哈希清单；实际依赖和 CUDA 环境按锁定源码校验。
3. 按 [H200 runbook](docs/H200.md) 分别预处理三个 split，生成去重的提示配对，并渲染经过字段检查的 BF16 LoRA 配置。
4. 先运行小样本过拟合 gate，再做 100-step 吞吐/显存 smoke test；两个阶段通过后才扩大数据。示例参数是实验起点，不承诺 GPU 小时或音质。
5. 使用 [评测与发布门槛](docs/EVALUATION.md) 做 Hindi 盲测与旧语言回归，记录导出前后差异。原始模型只读保存，adapter 按语言单独保存。

`configs/hindi-h200.example.json` 使用真实后端参数名，JSON 是 YAML 的子集。生成器只写配置并打印实际 CLI，不执行训练。`scripts/check_backend.py --help` 显示源检查与本地前置条件检查接口；失败是明确阻塞，不会生成训练成功记录。脚本不支持 DeepSpeed/use_accel；不要据此假定 vLLM、ComfyUI 或并发动态 adapter 切换可用。

## 验证范围

最终 CPU 检查记录见 [VALIDATION.md](VALIDATION.md)。GPU、特征提取、训练、推理、模型下载、真实 Hindi 数据对齐、端到端兼容性及质量均未验证。上游模型的文本前端可能进一步替换标点；务必检查实际 token、音素覆盖与试听，不能把 NFC 处理等同于完整 Hindi 文本前端。

本仓库不自动上传数据，也不创建 GPU 作业。`.gitignore` 排除数据、权重、adapter、缓存与常见凭据；提交前仍须检查 staged 文件。原创代码 MIT，外部模型与训练后端另循各自权利条件。

本次 GitHub 代码仓库按用户授权公开。未来 Hugging Face owner 为 `fluxa0223`；目前没有现成模型、未创建 HF 仓库或上传权重，见 [未来导出说明](docs/EXPORT.md)。

共享 pod 运行前必须先完成 [中断/恢复与共享资源 gate](docs/INTERRUPTION.md)。代码发布不代表实际 trainer 的断点恢复或GPU验收已通过。
