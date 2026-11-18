# Hindi 优化阶段记录

产品决策保持 **HOLD**。本记录覆盖H200 上固定 VAL 的配对合成、ASR 辅助比较和独立纯 LoRA 训练。分段对照使用[源码 205ce4e](https://github.com/allenflux/indextts25-hindi-h200/tree/205ce4eff670a58caf3d910e3816923135f65e15)。官方旧语言路由和[已冻结HF adapter](https://huggingface.co/fluxa0223/IndexTTS-2.5-Hindi-LoRA-experimental/tree/88e6297852699eb6dfbf6d4677446d37afcacf5f)继续保留。新候选的独立 Hindi 验收、母语者听评与生产验收仍待完成。

固定版本先由语义GPT预测离散语音codes，再经EnhancedCodec、长度适配、CFM/DiT声学阶段及BigVGAN生成波形。[实际模型链路](https://github.com/allenflux/indextts25-hindi-h200/blob/205ce4eff670a58caf3d910e3816923135f65e15/docs/MODEL_RATIONALE.md)说明了这些接口。当前纯attention方案只更新24层attention投影的低秩A/B，head、embedding及声学资产冻结。共享attention仍可能影响语言行为；参数保护与EOS完成都不能证明发音、自然度或旧语言质量通过。

## 固定输入与配对结果

复用[IndicVoices-R Hindi固定revision](https://huggingface.co/datasets/ai4bharat/indicvoices_r/tree/5f4495c91d500742a58d1be2ab07d77f73c0acf8)中既有的同一VAL8、2位speaker，没有使用TEST。文字长度70—217，3例超过旧入口160字符门槛。3份prompt超过15秒，仅这3份确定性取开头12秒；原转写和目标真实音频均未缩短。

启用`stock-token-v1`后，按官方token前端分段，文字 token 预算参数为 120、全文最多8段、每段语义code上限512。固定greedy、beam1、repetition penalty 1、seed1234和KV cache；每例只尝试一次。提高code上限只影响生成阶段，不能解决模型加载前的文字长度、前端或资产身份检查。两组均使用完整官方stock推理路径、H200/BF16及相同8例输入。

| 条件 | 完成WAV | 出现EOS的段 | 报告耗时 |
|---|---:|---:|---:|
| 官方原模型 | 8/8 | 12/12 | 94.089秒 |
| 纯attention step100合并GPT，e87候选 | 7/8 | 11/12 | 103.048秒 |

已独立确认全部已生成WAV的哈希。候选`hi-val-03`生成512个code仍未出现EOS，保留为失败。另一份候选WAV完整保留，长20.517秒，超过ASR的20秒门槛。两组同一处分段边界实际落在组合标记之前；字符覆盖核对不能保证grapheme边界与读音连续，也没有据此证明字符丢失。耗时包含本轮加载和处理范围，不能当作纯GPU吞吐。

Whisper small `20250625`在H200完成24条预定记录中的22条识别；上述无完整WAV及超长候选分别保留失败、排除原因。8份真实参考的严格CER为458/1037，即44.17%；参与配对比较的同6份参考为406/762，即53.28%。同6例中，原模型为1095/762，即143.70%，e87为545/762，即71.52%；候选在5/6例的编辑数较低。CER计入插入错误，可以超过100%。ASR报告耗时39.584秒。

固定同六条的最短编辑路径，平局按替换、删除、插入顺序处理。误差构成随识别器和对齐口径变化，仅用于定位后续听评重点。

| 条件 | 替换 | 删除 | 插入 |
|---|---:|---:|---:|
| 真人参考 | 240 | 45 | 121 |
| 官方原模型 | 446 | 127 | 522 |
| step100 合并 GPT | 280 | 245 | 20 |

候选的总编辑数降低，同时删除增加。需要听评区分漏读、发音难以识别和 ASR 自身错误。

这个观察只支持同6例、同识别器条件下的相对编辑数变化。真实参考本身的高错误率说明识别器校准仍不足，不能据此判定Hindi质量通过，也不能与此前不同TEST桶直接合并。prompt与目标音频的emotion条件可能存在训练/推理分布差异，目前只是待检假设；后续应固定同例、speaker、文字、seed和生成上限，单独改变emotion来源，不把多个改动混为原因。

## 复现与证据范围

使用固定源码、已有离线资产和同一cases清单。父控制器的参数白名单及`batch_synth`子模块均须支持`stock-token-v1`；该子模块参数位于控制器`--`之后。分别将`VARIANT`设为`original`、`merged`，每次指定新的输出和控制器目录，资源门槛取已核准记录。

```bash
"$PYTHON" -B "$SRC/scripts/run_pure_eval_guarded.py" \
  --module batch_synth --workspace-root "$WORK" --report-dir "$FRESH_GUARD" \
  --expected-merged-gpt-sha256 "$E87_SHA256" --gpu-id "$AUTHORIZED_GPU" \
  --min-free-gpu-mib "$MIN_GPU_MIB" --min-free-ram-mib "$MIN_RAM_MIB" \
  --max-utilization "$MAX_UTIL" --execute -- \
  --workspace-root "$WORK" --official-repo "$OFFICIAL" \
  --model-dir "$MODEL" --model-manifest "$SYNTH_MANIFEST" \
  --merged-bundle "$E87_BUNDLE" --cases "$FIXED_CASES" --variant "$VARIANT" \
  --output-dir "$FRESH_OUTPUT" --segmentation-mode stock-token-v1
```

执行前后核对两组仍为相同8个case、逐例文字与prompt哈希、3份prompt的裁剪规则，以及每段完整文字覆盖、code数、EOS、失败和未执行项。ASR只比较双方都符合门槛的同6例；保留全部8例台账。父控制器退出、模块结果、WAV哈希分别检查。此处两次合成和一次识别的自有子进程均已通过`/proc`确认退出，原有服务健康、队列为0/0；这不代表已实施生产切换。

下列SHA256固定本轮证据。原始音频、私人转写、数据清单、日志及集群路径不随本文分发。

```text
cases = 70ae0e2d35313571e4bd2dd95917d2c8d06df00cc45903b1989a583ba42c47e1
experiment = d9999ce83293d0d520b0f15f27836c55a7b8cf6985e73f5253120f2665d31738
original_report = b83a23f0715464a755275f3510dd472d7554a837ed1cada8e11805210fac7969
candidate_report = c1efb4caec9998411a1785bba76bec78a7a131812dfe5ca5f038f07adeefb9b0
asr_report = 22dcfc0bbdc0c65e6c844d771d6441e46578b41c51742edee9720e595943fd25
candidate_gpt = e87ed7e2f8ea632af261969ee76ab428fdbfe38acaa4e76c785994be63237943
```

## 独立纯 LoRA 300 步计划

从相同官方 base 新建训练。沿用 TRAIN48、VAL8 和注意力 A/B 训练范围。启用 BF16，rank 8、alpha 16、dropout 0、累积 4、learning rate 0.0001、warmup 10、权重衰减 0.01 和 seed 260618。300 步和旧 100 步有不同的 cosine 调度总长，前者从原版开始训练。源码与配置见[固定提交 443999f](https://github.com/allenflux/indextts25-hindi-h200/tree/443999f329bf9e79693d9514288bb48e310026c1)。

H200 首次资源准入触发保护，未加载模型。空闲后真实执行 `fresh2` 并结束进程，CPU 校验 96 个 FP32 A/B 张量、优化器、采样器及实际文件 SHA。新进程 `resume300` 仅在 step2 两个文件与此前回执一致后开始。从第 2 步恢复并实际达到第 300 步，两个阶段分别耗时 41.726 和 178.034 秒，均包含启动、训练、检查点写入和收口检查。进程退出后复核了全部 300 个更新记录、六个 VAL 点与检查点 SHA。

| 步数 | 完整 VAL8 平均 loss |
|---:|---:|
| 50 | 7.195542 |
| 100 | 6.720754 |
| 150 | 6.589467 |
| 200 | 6.532398 |
| 250 | 6.508521 |
| 300 | 6.497442 |

预声明规则取完整 VAL8 的最低平均 loss，平局取更早步数。已冻结 step300。旧独立 step100 计划的 VAL loss 为 6.989858。两个计划的调度总长不同，这项比较仅反映已有数据和教师强制前向，语音验收仍依据真实合成、ASR 校准和人工听评。

在受控环境设置 CUDA 设备掩码为 `0` 并离线运行，从固定源码先用 `--stage fresh2`，核验实际回执后再用 `--stage resume300`。两阶段使用同样的 `--plan pilot48-300-v1`、`--workspace-root`、`--config` 与 `--config-sha256`。训练中只探测原服务队列及资源，达到阈值会停止自有进程并保留已有检查点。

```text
config_sha256 = 09fbfa6bf395fbb2f8b631a5fb7623676f7f4af6bc73819a74348d3ca318d573
plan_sha256 = d5dcc345c9ebe11681c0b43ce04702aff749f05e1110c889ca6d0aba53fec888
fresh2_controller_sha256 = d1790c03b2181b9983f217ca3f16026e0e65752c4024c52bc8d95860b33c7561
resume300_controller_sha256 = c1a90a184038d7481f476d36f8825dfd4c2c54215725e86228f5f62773c2f8b1
selection_sha256 = b8d4b0d29328726dae4348f63c3e0533fd10ab4458bfb5584cd7a2f4a4c11027
checkpoint300_sha256 = 44992333810bca1db188b09ef2614cd35f3465921a97f769d25fc106bd7b10e9
adapter300_sha256 = 035c586d265d3159d851ba59b3a6519d06e140b3894d50819c7ad93dc07ab782
```

## step300 独立导出与真实生成

CPU 从冻结适配器构建官方 2.5 格式完整 GPT，文件为 3,259,538,991 字节。固定 VAL 第 0 条特征在 CPU FP32 独立重载，比对未合并 A/B 路径与合并后的官方 GPT。logits、loss 与 argmax 对照通过，两路径的目标张量一致，固定容差 `rtol=atol=1e-4`。该证据覆盖一条真实前向特征。

H200 使用同一份 8 条 VAL 清单与同样的 prompt、seed、分段及生成参数。新候选成功生成 **8/8** 个完整 WAV、**12/12** 段返回 EOS，所有 WAV 均以 SHA256 复读核对。旧候选达到 512 码仍无 EOS 的同一例 `hi-val-03`，这次返回 235 个码（含 EOS），生成 9.334 秒 WAV。一次组合标记切点仍然存在。候选批次墙钟耗时 96.757 秒，退出码 0，自有生成进程已退出。

新候选用 `run_pure_eval_guarded.py` 的独立本地登记。父参数显式提供 `--candidate-manifest`、`--candidate-manifest-sha256`、`--expected-merged-gpt-sha256`，子参数仍为 `--segmentation-mode stock-token-v1`。加载前绑定 adapter、完整 GPT、COMPLETE 和 provenance，继续执行实际权重全量哈希检查。保留原版和已冻结 step100 作为回滚基线。

```text
step300_gpt_sha256 = e2c526972292286b55829ab3ec0ba7d11259c172704d7ed871b4787abf4e00db
step300_complete_sha256 = 48fe36e14057b6f0cddc42611ab8d7fea477d6ca0943eb2fd35090c661cb529c
step300_provenance_sha256 = 4f98fcc004a4fdabb094b3ee6fae6dee92b248a8811d0fce693fe77cdc7e3d7b
step300_registration_sha256 = a84870771b2d2e1891db2d1f4b11b171901c88d7a90ab63003346d6dc55d0759
step300_cpu_fp32_report_sha256 = e1677b489483561361088638f41febb1344c44129a14786244cb86983570fbc3
step300_h200_batch_report_sha256 = b0d1c0a6e4fb80b718a06d88b1991d1abc737a69e48e789353c4b28c03682b67
```

## step300 同设备识别复测

复用同一版 Whisper small、H200、固定语言和解码配置。原版与真人参考的输入 WAV 哈希及 8 条识别结果与上轮逐条一致。新批次完成 24/24 条识别，有 8 个完整原版/step300 配对。三候选严格横比使用共同可识别的六条，避免用不同难度和分母的样本比较。

| 范围 | 同一组真人参考 | 原始模型 | step100 | step300 |
|---|---:|---:|---:|---:|
| 三路共同 6 条 | 406/762，53.28% | 1095/762，143.70% | 545/762，71.52% | **293/762，38.45%** |
| 原版与 step300 完整 8 条 | 458/1037，44.17% | 1448/1037，139.63% | 部分样本不入 ASR | **376/1037，36.26%** |

同六条上 step300 的严格字符编辑数均低于 step100。此前达到 cap 的例子现已进入识别，其严格 CER 为 34/98，即 34.69%。真人参考仍出现较高误差，合成端的较低 CER 无法推导出真人听感、韵律或语义保真水平。step300 的挑选和本次语音对照均使用现有 VAL。需要独立语料和母语者听评确认泛化。

GPU ASR 成功退出，耗时 41.920 秒，公开汇总 SHA256 为 `d44ed8297a639257d42dcec0b29a43bc9cdecb5b8a0f92f757f359fba3215e47`。单段五例对照在资源准入时止步，本记录依照两轮同配置多段结果比较权重。首次 GPU 识别准入也被共享利用率拦截，恢复空闲后完成同设备对照。备用 CPU 识别在取得 GPU 结果后退出，未纳入统计。

## 交付结论

**保留 step300 为隔离实验候选，生产放行状态仍为 HOLD。** 生成覆盖与固定配对识别指标有实测改善。该候选的 5090 T8 原工作流、旧语言回归、并发与回滚演练仍需各自验收。当前生产使用官方原版路由，公开的 step100 LoRA 保持固定版本。这次同步的是代码和验收记录。各阶段的放行标准见[产品交付与换模验收](https://github.com/allenflux/indextts25-hindi-h200/blob/d02afde4f10d1dce012d83adab23180243e264db/docs/PRODUCT_DELIVERY.zh.md)。

最终源码的离线 CPU 全套检查通过 434/434 项。真实 H200 合成、训练、前向、ASR 分开记录各自证据。
