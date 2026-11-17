# best100 有限验收记录

冻结候选的本轮有限验收已实际完成：原模型 **7/7**、best100 **7/7** 真实stock生成，**16/16** GPU ASR完成（14条生成及2条Hindi test真人reference），三个子进程均退出0、无停止原因、无覆盖缺口。**Hindi诊断有方向性改善，但best100严格CER仍为50%；正式质量gate未通过。** 全部样本保留，没有retry、重选case或重新选checkpoint；该轮未生产替换、未HF发布，人工试听未完成；后续HF发布见[发布记录](PUBLIC_RELEASE.md)。

工具源码固定于 `d3bb1f575c68e227cd1e442fc6586339a6459d83`，公开远端SHA已核对。Mac完整CPU suite为 **222/222通过、0跳过、34.351秒**；批量生成与有限ASR已独立审查。best100此前按固定验证loss选定，本轮test与ASR不用于选模。历史训练、合并和VAL证据见 [VALIDATION](../VALIDATION.md) 与 [EVALUATION](EVALUATION.md)。

## 固定方案与输入审计

Hindi test两位speaker各取一例，与train按speaker隔离，各保留一条真人reference。zh/en/ja/es/ar各一例，均使用Hindi prompt，是跨语言prompt stress；没有各语言native-prompt与真人校准，不能称完整旧语言回归。

7个case每个原模型/best100各执行一次。两个variant分进程串行，每个进程只加载一个真实stock模型；seed1234，Hindi max mel512、其余256。14条生成均`frontend_verified`，7对输入的canonical text SHA、prompt SHA、seed、settings、stock token IDs及language slot一致，14 WAV哈希审计通过。原模型和best100 GPT最终哈希均与冻结记录相同。

ASR主指标以冻结authored/test canonical text为参考：NFC、空白归一后按codepoint计算严格CER（含空格），不以stock内部tagger/casing文本替换。下表均为 **strict** 结果，与此前去标点评分不能直接混算；去标点及en/es casefold仅是预先固定的辅助视图。CER/WER可因插入错误超过1；zh/ja的空白WER不作可解释降级指标。

## Hindi test结果

| 范围 | 严格CER：编辑数 / 参考字符数 | 严格WER：编辑数 / 参考词数 |
|---|---|---|
| 两条真人reference微平均 | 31/166 = 0.186747 | 14/32 = 0.4375 |
| 原模型两条生成微平均 | 396/166 = 2.385542 | 146/32 = 4.5625 |
| best100两条生成微平均 | 83/166 = 0.500000 | 27/32 = 0.84375 |

| 匿名test speaker | 真人reference严格CER | 原模型严格CER | best100严格CER |
|---|---|---|---|
| A | 12/79 = 0.151899 | 198/79 = 2.506329 | 35/79 = 0.443038 |
| B | 19/87 = 0.218391 | 198/87 = 2.275862 | 48/87 = 0.551724 |

两位speaker方向一致，但样本量仅两例，真人reference的ASR错误也明显；这些数字不能认证Hindi可用性、自然度、发音或整体泛化，更不能以单一WER阈值判定质量通过。

## 旧语言Hindi-prompt stress

| 语言（每种1例） | 原模型严格CER | best100严格CER |
|---|---|---|
| zh | 1/17 = 0.058824 | 3/17 = 0.176471 |
| en | 1/46 = 0.021739 | 1/46 = 0.021739 |
| ja | 0/18 = 0 | 2/18 = 0.111111 |
| es | 1/60 = 0.016667 | 1/60 = 0.016667 |
| ar | 0/41 = 0 | 0/41 = 0 |

中文、日文的本轮单例指标升高，需母语人工试听及native-prompt参照进一步检查，不能宣称保证无退化。其余三个单例指标相同，也不证明普遍无退化。Hindi真人reference不能校准这些语言，ASR误识别不必然等于TTS错误。

## 执行边界与结束状态

每个variant上限300秒，ASR上限180秒；每秒核对设备身份、GPU监控total-used至少32000 MiB和宿主可用内存至少64000 MiB的配置门槛，仅允许终止控制器自己创建的子进程。本轮实际时长：原模型 **84.81秒**、best100 **79.06秒**、ASR **20.21秒**，三个退出码均0、无停止原因；没有训练，没有停止、重启或修改ComfyUI。

整设备观测峰值为原模型阶段 **9350 MiB（76样本）**、best100阶段 **9486 MiB（71样本）**，监控的total-used最小值分别 **134421 / 134285 MiB**，未扣NVML reserved，不是实际`memory.free`；包含ComfyUI，不能冒充本进程显存峰值。结束时nvidia-smi实际`memory.free`为 **140404 MiB**；结束快照为 **2753 MiB已用、0%利用率**，本项目Python进程为NONE，训练所在Pod localhost的ComfyUI健康检查为 **HTTP200**。两份GPT最终哈希未变，`RUN_STATE.json`为`active_training=false`、`quality_gate_passed=false`。 后续接近门槛时应采用NVML `memory.free`；本轮真实余量充足，不因此重跑或重选。

人工Hindi与各语言母语试听、充分native-prompt回归、数据/模型发布权利审核仍未完成，该轮未替换生产模型、未创建HF模型仓库或上传权重；后续实验发布另见[发布记录](PUBLIC_RELEASE.md)。原始音频、转写、prompt及完整私有报告留在受控目录，本文仅列非敏感统计；本地许可/评审包的交付状态另按实际回执记录。
