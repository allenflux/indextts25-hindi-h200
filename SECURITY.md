# 安全范围与已知依赖告警

本项目不宣称依赖无漏洞。2026-11-17 对本仓库的只读 Dependabot 查询显示 **52 项 open 告警：23 high、23 moderate、6 low**，对应 **14 个不同 advisory**，在 `pyproject.toml` 和三份 native/overlay 依赖清单中重复计数。没有 dismiss 告警；宿主运行库不同于清单时，也不能据此写成仓库告警已修复。

核心数据工具没有第三方运行依赖；下列风险来自可选原生环境。`requirements/native-cpu-macos.lock.txt` 是 CPython3.11/Torch2.8 的历史测试复现记录，`requirements/native.in` 和 `[native]` 目前也保留该兼容性版本。**这些旧 pin 不是推荐的生产环境或未知 checkpoint 加载基线。**

## 与当前受控任务有关的边界

| 依赖 | 已核实的影响 | 当前处理 |
|---|---|---|
| Torch2.8 | CVE-2026-24747 可绕过 `weights_only=True`；修复版本2.10.0。`weights_only` 不是 sandbox，也不能证明文件可信。 | 不读取未知 `.pt/.pth`。真实权重工作优先使用经过本项目实际兼容性 gate 的 Torch>=2.10 与匹配 torchaudio；不得为升级改动现有服务的全局环境。 |
| 已有 Pod 的 Torch2.12.1 | 已超过上述 RCE、`unpack_sequence` 与 `lstm_cell` 告警的修复版本；仍匹配低危 `jit.script` 告警（修复2.13.0）。 | 只使用固定官方代码和本项目代码，不接收用户脚本或模型定义。已有库只读借用；不据此宣称整个环境安全或自动升级全局 Torch。 |
| Transformers4.52.1 | 当前清单匹配9个 advisory，涉及自定义生成代码下载、Auto/嵌套配置、chat-template 保存、HF Trainer 和特定正则处理路径。 | 固定本地类与配置、先校验哈希、离线执行；不启动网络推理服务，不接受任意 Hub 模型/config、remote code、custom generation、远程 attention kernels 或外来 chat template。 |
| PyArrow20 | 当前包版本匹配 Arrow C++ IPC 预缓冲漏洞；公告明确 Python bindings 不受该漏洞影响，当前导入器使用 Python `ParquetFile`。 | 保留告警及路径判断，不把它解释为任意 Parquet 都安全。只处理固定 revision、限定大小并核验哈希的训练分片；若升级23.0.1，应先在独立环境验证导入器。 |

来源：[PyTorch checkpoint 公告](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)、[JIT 问题](https://github.com/pytorch/pytorch/issues/149623)、[Transformers 自定义代码下载](https://github.com/advisories/GHSA-x9r9-c232-4q39)、[配置加载](https://github.com/advisories/GHSA-29pf-2h5f-8g72)、[chat-template 保存](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w)、[Arrow 公告](https://github.com/advisories/GHSA-rgxp-2hwp-jwgg)。这些是受控任务的可达路径判断，不是漏洞修复或告警关闭证明。

## 文件、加载与进程约束

- 只加载固定官方 revision、通过完整文件哈希清单核验的可信资产，或本次自己训练且与 run/config/data/base 身份绑定的 checkpoint。SHA 验证字节一致性，不能单独证明来源、许可或安全性。
- 保持 `torch.load(..., weights_only=True)`，不回退到 `weights_only=False`，不为未知文件添加 `safe_globals`。官方 helper 的间接加载也在本项目进程设置 `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`；若受限加载失败，停止并检查，不绕过。
- 在资产已齐备后，计算进程设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。`trust_remote_code=False` 本身也不是完整安全边界；必须结合固定配置、允许列表和本地加载。原生路线不使用 HF `Trainer`、TensorFlow 权重转换、MarianTokenizer 或任意正则优化器配置。
- 缓存、输出与依赖 overlay 留在项目隔离目录；HF认证单独放在ComfyUI所有实际served的input/output/temp根之外。`WORK`若位于output持久卷，认证不能随之放在`WORK/hf-home`；0700/0600不能隔离同UID的ComfyUI文件读取。专属临时目录也须先确认不在served根内，不保留旧served路径副本或符号链接回链。缓存可继续留在`WORK`。
- 临时认证随容器替换/临时盘丢失后，需要用户本人重新登录；不读取或输出token内容，不修改全局credential store。持久模型、特征和checkpoint齐备的离线训练/恢复不需要token。认证迁移不构成对既往访问的无泄漏证明。
- 不修改现有 ComfyUI、全局包或服务环境。任何新环境都先通过实际 CPU、tiny BF16、停止/恢复与完整资产加载检查，再推进对应真实任务。

不盲目升级 Transformers 到5.x来消除计数：这会越过固定官方4.52.1接口，必须独立做兼容性验证。可评估新的 Torch>=2.10 安装组合，但在验证前不改写已测试的旧 lock、不把未知组合标成可用。当前受控 GPU gate 和有界数据准备可以按上述边界继续；这些限制不等于准许接收不可信 checkpoint 或开放推理服务。
