# IndexTTS 2.5 Hindi / H200

原创 Hindi 数据工具与实验性原生 LoRA 训练实现。`hi_h200_native` 直接调用固定官方模块，已在CPU真实微型GPT上验证前向/反向、优化更新、完整adapter和停止/恢复；没有导入或复制未许可社区代码。

**真实预训练2.5权重、真实Hindi音频全链、GPU与语音质量的验证范围以 [VALIDATION.md](VALIDATION.md) 为准。** 目前不能把微型模型测试当作Hindi训练或音质成功。官方tokenizer预留hi槽位17与`<|hi|>`不代表官方训练过Hindi。

## 原生训练路线

从 [原生指南](docs/NATIVE.md) 开始，配置示例为`configs/native-dataset.example.json`、`native-train.example.json`和`native-h200.example.json`。

1. 在独立持久目录准备Python3.11环境与官方固定源码，保留许可证。核心数据工具零第三方运行依赖；训练依赖通过本项目`[native]`extra或`requirements/native.in`安装。
2. 先运行资产助手的元数据模式，检查具体文件、体积和许可；显式下载只选已锁定资产，不拉全模型仓库、不覆盖现有模型。
3. 只读取已授权Hindi数据的小批次，保留来源/revision/许可记录，运行NFC、音频、去重与speaker隔离校验。
4. 用冻结的官方编码器生成原生特征并逐条校验，先CPU小样本预检，再按共享GPU实际余量选择是否进行BF16验证。
5. 先小样本过拟合，再真实STOP/恢复、100次optimizer更新smoke，最后做 [Hindi/旧语言和发布评测](docs/EVALUATION.md)。参数只是实验起点，不承诺GPU小时或音质。

原生实现当前只支持无padding的batch1加梯度累积；冻结声学模块/GPT基础参数，训练attention Conv1D LoRA、Hindi语言行、受限文字行和两个输出head。保存完整可训练状态，原始权重只读。不同语言adapter各自版本管理；不宣称社区adapter、vLLM、ComfyUI或DeepSpeed兼容。

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

未来HF owner为`fluxa0223`，目前未创建模型仓库或上传权重；[导出说明](docs/EXPORT.md) 保留真实评测与权利审核门槛。共享pod不停止、重启或改动ComfyUI，训练输出使用独立持久目录。
