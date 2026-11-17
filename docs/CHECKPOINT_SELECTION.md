# 从完整检查点选择候选 Adapter

选择一个较早的完整 checkpoint，不需要恢复训练或改写原计划。本次 pilot 的原计划始终为 `max_updates=1000`：先在20次更新处停止，再由新进程恢复并在500次更新处停止。已记录的验证曲线在 step100 最低，随后升高，因此选择100作为候选。真实 step100 完整 checkpoint 已在 CPU 上成功转换，状态为 `selected_checkpoint_adapter_complete`，checkpoint 张量验证与标准 adapter 加载验证均通过；没有恢复训练、optimizer 或 RNG。随后已完成7文件完整 GPT 合并、H200完整 FP32 单条 feature 的原生 adapter 与 fresh stock parity、两句 stock WAV、配对审计及6 cases真实 GPU ASR，整体流水线退出码0。小样本 CER 观察与具体实绩见 [pilot 实际记录](../VALIDATION.md#64条pilot的500步训练与候选选择)。人工审听与旧语言回归尚未验收，不能据这些结果宣称成品、泛化或发布验收通过。

`hi_h200_native.checkpoint_adapter` 默认仅做 metadata/hash 预检：不导入 Torch、不加载模型或 checkpoint 张量、不生成 adapter。它会读取完整 checkpoint 字节计算 SHA，核对原配置、独立留存的 run-manifest SHA、绑定源码/数据路径、sidecar 与显式 step。只有加 `--execute` 才进入真实 CPU 验证。

先阅读 [SECURITY](../SECURITY.md)、[实际验证范围](../VALIDATION.md) 与 [评测门槛](EVALUATION.md)。仅处理本次自己训练、来源和身份已确认的完整 checkpoint；陌生 checkpoint、其旁边的 sidecar 和对它们现算的 SHA 不能互相认证。`weights_only=True` 也不是 sandbox。执行环境遵循 SECURITY 中的可信文件与已验证依赖边界，不为消除报错启用不安全 pickle 回退。

## 预检，再显式执行

以下在同一 Bash 会话运行。`PY` 指向本项目已验证、适合可信 checkpoint 任务的 Python；路径只是通用布局。`RUN_MANIFEST_SHA256` 必须取自独立留存的可信运行记录，不把一组未知文件的自算摘要当作来源证明。

```bash
export WORK=/your/persistent/isolated/workspace
export REPO="$WORK/repo"
export PYTHONPATH="$REPO/src"
export PYTHONDONTWRITEBYTECODE=1

RUN="$WORK/runs/native-pilot1000"
CONFIG="$WORK/configs/native-pilot1000.json"
SELECTED="$WORK/selections/step100"
mkdir -p "$WORK/selections"

select_cmd=(
  "$PY" -B -m hi_h200_native.checkpoint_adapter
  --workspace-root "$WORK"
  --config "$CONFIG"
  --run-manifest "${RUN}.native-run.json"
  --run-manifest-sha256 "${RUN_MANIFEST_SHA256:?Use the independently recorded run-manifest SHA256}"
  --checkpoint "$RUN/checkpoints/step00000100.pt"
  --expected-step 100
  --selection "Observed validation loss lowest at step100 before later increase; quality acceptance pending"
  --output-dir "$SELECTED"
)
"${select_cmd[@]}"
```

配置、run manifest、checkpoint 和数据/官方源码路径须在 `WORK` 内；原始模型及其 manifest 可位于明确授权的只读共享目录。输出父目录须存在，`SELECTED` 须为全新目录，并与原 run、源码、模型和数据分离。预检通过不等于张量验证通过，也不重新计算哪个 step 最优。

确认预检结果、可信来源及 CPU 内存余量后：

```bash
"${select_cmd[@]}" --execute
```

执行使用受限 CPU 加载，复核完整 checkpoint 的实际 step、配置/源码/base/data/adapter 身份、优化器边界与状态结构、张量及受保护参数；不恢复 optimizer、scheduler 或 RNG，不做 forward/backward 或任何训练更新，也不改变原始 `max_updates=1000` 计划；在500次更新处停止不等于将计划改为500次。完整模型与 checkpoint 可能占用数 GiB CPU 内存并产生磁盘读取，不能当作零资源操作。

成功目录应包含：

- `adapter.pt`：标准原生 adapter，覆盖全部可训练参数，包括两种 head。
- `adapter.pt.sha256.json`：adapter 摘要 sidecar。
- `selection.json`：选择依据、来源 checkpoint/sidecar/run 身份与验证记录。
- `COMPLETE.json`：完成标记及所列产物的摘要。

只有实际成功回执与完整标记才能记录“已转换”。失败后可能留下不完整新目录；先检查，不能覆盖重试、改 source hash、改原计划或手工补造 COMPLETE。原始完整 checkpoint 保留不动。

## 接入已有 merged GPT 导出

转换成功后，从其可信 `selection.json` 读取 adapter 摘要，再调用现有 export；不要将未知 adapter 的自算摘要用于认证。

```bash
OFFICIAL="$WORK/external/index-tts"
MODEL_DIR="$WORK/assets/official-extract"
MODEL_MANIFEST="$MODEL_DIR/model-manifest.json"
ADAPTER_SHA256="$("$PY" -c 'import json,sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text())["adapter_sha256"])' "$SELECTED/selection.json")"
mkdir -p "$WORK/exports"

CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 "$PY" -B -m hi_h200_native.export \
  --official-repo "$OFFICIAL" --model-dir "$MODEL_DIR" \
  --model-manifest "$MODEL_MANIFEST" \
  --adapter "$SELECTED/adapter.pt" --adapter-sha256 "$ADAPTER_SHA256" \
  --protect-root "$RUN" --protect-root "$SELECTED" \
  --output "$WORK/exports/step100-gpt"
```

export 目标也须不存在；按其真实回执和 COMPLETE 核验 `gpt.pth`、许可与身份清单。转换/导出不是恢复训练，不证明 stock 波形合成、ASR、Hindi音质或旧语言回归通过。继续按 [评测门槛](EVALUATION.md) 和 [合成指南](SYNTHESIS.md) 验收，不覆盖原模型或现有服务。

本次对应完整 CPU 回归记录为 **200/200通过，32.050秒**，其中新增 converter 3项、candidate cap 4项。测试通过与真实 step100 转换的实际结果分别记录；不得混用。
