# 中断、恢复与共享 pod 验收

本方案在源码层审计过，但**尚未跑真实 trainer 的停止/恢复往返**。原创 checkpoint 校验器的 CPU 合成测试只验证检查逻辑，不能写成“GPU 断点恢复已验证”。真实训练仍受后端许可、数据访问权、模型/依赖与资源 gate 阻塞。

## 持久化与共享资源

只在用户明确授权的现有 pod 内评估，具体名单保留在本地运行记录，不发布内部资源标识。running/ready 不是空闲证明，pending/未 ready 不是可分配资源。优先单张 H200 小实验，不能据名称判断型号/容量、访问权或可用显存。

启动前只读核实：pod 的准确 namespace/UID、GPU 型号与实际占用、GPU共享方式/MIG、当前进程与持久卷挂载、可用磁盘、配额、驱逐/重启记录和 termination grace。不要停止、重启或改变 ComfyUI；不要改集群、调度、驱动或其他 pod。不创建新计费资源，不暴露 kubeconfig/token。

把源码、配置、数据清单、模型哈希、日志和 checkpoint 放在已验证的持久卷中；容器层/tmp不算可恢复存储。检查保存临时文件与目标 checkpoint 同一文件系统。一次只启一个训练进程，限定 `CUDA_VISIBLE_DEVICES`；不要同时铺满所有列出的 GPU。

收集负载观察窗口后，需有足够余量覆盖训练模型加载+batch峰值，以及现有 ComfyUI 的突发需求。空闲时 nvidia-smi 采样不能预测 ComfyUI 峰值；不能证明安全余量时不启动训练。运行中监控新进程、显存增长、OOM、非有限 loss 和持久盘余量，达到预先记录的停止阈值就请求**自己的训练**协作停止，不操作现有服务。

## 真实停止语义

锁定社区 trainer 只识别 `OUTPUT_DIR/STOP` 文件：启动时会先删除旧 STOP，运行时在 optimizer 更新边界检查，保存检查点再返回。请求停止应在确认训练已启动后：

```bash
test -n "${RUN:-}" && test -d "$RUN" && touch "$RUN/STOP"
```

不要为 STOP 退出设置几秒钟保证；当前 batch、验证和保存可能需要时间。退出后核对新 checkpoint 的 step/epoch/data_batches 与日志，再对文件做哈希和本地备份。

该版本没有 SIGTERM/SIGINT 或 KeyboardInterrupt 保存处理。kill、pod驱逐或OOM都可能丢失自最近一次完整周期 checkpoint 以来的工作；STOP 不是驱逐保护。周期 save_interval=25 是实验起点，基于实测保存开销与可接受损失调整。原子保存使用 `.pt.tmp` 后同目录 replace，可避免把半写文件当最终文件；没有 fsync，原子可见不等于故障持久性，不能保证存储故障或强制退出时一定有新 checkpoint。

## 恢复预检与演练

上游 checkpoint 包含 trainable/optimizer/scheduler/scaler/step/epoch/data_batches/baselines/spec。BF16 的 scaler 可以是 None。恢复读取可训练参数及 optimizer/scheduler/scaler 和数据 cursor；**未保存 Python/NumPy/PyTorch/CUDA RNG 状态**，不能承诺逐位相同的轨迹。样本 shuffle、worker状态与随机算子差异需在验收中记录。

另一个实质限制：正常 max_steps 完成后保存的最终 checkpoint 把 epoch 写成配置 epochs、data_batches写0。它不是用于延长训练的精确 cursor；不能直接拿 smoke的最终100步文件继续跑。用于恢复演练的应是运行中周期保存或协作 STOP 保存的 checkpoint，校验器会拒绝 epoch>=epochs 的完成态。

恢复演练必须先在已授权小样本上完成，再扩大：

1. 固定同一 source/model/env/config/prepared/features 审计与持久 run目录，保证每epoch足够完整的累积batch。以有限 max_steps 的全新 run 启动，先记录 step与loss。
2. 在至少一次成功optimizer更新和周期保存之后，创建 STOP。等待自己的训练进程正常退出；不要用最终max_steps完成文件替代。保留日志与实际退出码，确认checkpoint可读且step>0、epoch<epochs、cursor合理。
3. 对已完成checkpoint做本地只读副本；用 `scripts/check_checkpoint.py --help` 的真实检查接口核验文件SHA、checkpoint结构与绑定的原始config/preflight。它只用显式安全的 CPU `torch.load(weights_only=True)`；不回退到任意pickle加载，不能加载无可信来源的checkpoint。
4. 仅在全部前置检查通过后，用相同完整配置和已存在的checkpoint恢复。实际训练CLI是 `train.py --config FILE --output-dir SAME_RUN --resume EXISTING_CHECKPOINT`。上游缺失resume路径可能静默从头开始，因此禁止跳过文件存在和SHA预检。
5. 确认恢复日志加载了期望step，下一次optimizer更新严格大于该step且不是从零开始；optimizer/scheduler与LR进度一致，loss有限，继续生成checkpoint，停止/再恢复一次仍可用。对相同固定prompt保留恢复前后试听；允许随机轨迹差异，不能允许加载参数/数据版本差异。

恢复验收表：

| 项目 | 当前状态 |
|---|---|
| 校验器的合成CPU结构/哈希测试 | 见 VALIDATION.md |
| 真实trainer协作STOP保存与退出 | 未执行 |
| 重启后step/cursor/optimizer/scheduler加载 | 未执行 |
| 新step递增、loss有限、再次保存 | 未执行 |
| pod重建后持久卷文件仍可读 | 未验证；不能为测试擅自重启ComfyUI pod |
| 强制驱逐/SIGTERM自动保存 | 不支持，不做生产pod破坏性测试 |
| RNG逐位复现 | 后端未提供，不作承诺 |

只有真实恢复演练、100-step资源检查以及 Hindi/旧语言/导出一致性验收都通过后，才扩大数据或发布 adapter。所有中断日志、sidecar、checkpoint与私人数据均留在本地被忽略目录，不进入公开代码仓库。

上游恢复不会验证保存时的 spec/config/base/data，且会在加载checkpoint前覆盖 train_args.json、重建 text_rows.json。恢复前必须在独立持久审计目录保存这些文件与原始config/preflight/hash；所有 source、base、数据顺序/权重、长度、batch/bucket、seed参数保持不变。scheduler 按当前参数重建再载入状态，改max_steps/epochs等会改变后续调度，不能作为同一恢复验收。

不要用 adapter/merged 权重恢复训练，它们没有 optimizer/cursor；不要依赖 auto 选择最高编号checkpoint，因为不会自动回退损坏文件，必须使用已验证的明确路径。轮转只跟踪当前进程新文件，恢复后旧文件可能继续占盘，需要监控磁盘但不要自动删除唯一恢复点。


## 原创 checkpoint 工具命令

以下操作只处理自己的可信本地训练产物，不下载、不启动训练。bind-run 必须在全新训练启动前执行；配置/绑定放在run输出目录外，避免混淆空目录检查。环境锁只是已保存环境清单的哈希，仍须独立证明当前安装与它匹配。

```bash
# config和preflight由render_backend_config生成，使用同一PROFILE。
: "${HI_H200_WORK:?Set persistent work directory}"
: "${BACKEND:?Set approved pinned backend directory}"
PROFILE=overfit
python3 scripts/check_checkpoint.py bind-run \
  --config "$HI_H200_WORK/runs/hi-${PROFILE}.json" \
  --preflight "$HI_H200_WORK/runs/hi-${PROFILE}.json.preflight.json" \
  --backend-repo "$BACKEND" --rights-record "$HI_H200_WORK/rights.json" \
  --weight-manifest "$HI_H200_WORK/weights.json" \
  --environment-lock "$HI_H200_WORK/reports/gpu-requirements-resolved.txt" \
  --train-feature-report "$HI_H200_WORK/reports/train.feature-audit.json" \
  --val-feature-report "$HI_H200_WORK/reports/val.feature-audit.json" \
  --output "$HI_H200_WORK/reports/hi-${PROFILE}.run-binding.json"

# 完成协作STOP后，手工将真实完整checkpoint路径赋给CKPT，不能指向.tmp或adapter。
: "${GPU_PY:?Set approved backend environment Python}"
: "${CKPT:?Set explicit authenticated checkpoint path}"
"$GPU_PY" scripts/check_checkpoint.py seal \
  --binding "$HI_H200_WORK/reports/hi-${PROFILE}.run-binding.json" \
  --checkpoint "$CKPT" --trusted-own-checkpoint \
  --output "$HI_H200_WORK/reports/hi-${PROFILE}.resume.json"

# 每次恢复前重新核对当前文件和全部绑定，不跳过错误、不回退到不安全pickle。
"$GPU_PY" scripts/check_checkpoint.py verify \
  --sidecar "$HI_H200_WORK/reports/hi-${PROFILE}.resume.json" \
  --checkpoint "$CKPT" --trusted-own-checkpoint
```

seal/verify 会CPU加载训练checkpoint，可能占用较大内存；不要与生产CPU/内存负载争用。实际训练resume命令只有在verify成功、资源与全部许可gate都通过后才可运行。工具明确保留 `runtime_roundtrip_verified: false`：结构/哈希正确不能代替真正重启并继续训练的证据。
