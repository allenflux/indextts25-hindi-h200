# CPU 验证记录

验证日期：2026-11-17。工作目录与临时文件均位于已核实的 CodexData APFS 挂载；没有修改其他项目、安装依赖或下载模型/语料。测试只读借用现有解释器，禁用 pycache，临时文件写入本项目 `.runtime`。

## 已实际执行

- Python **3.13.5**、NumPy **2.5.4**：完整测试 **58/58通过**（32数据测试、17后端/特征测试、9检查点预检测试）。检查点测试使用合成状态/mock Torch，未实跑真实Torch文件加载或trainer恢复。
- 系统Python无NumPy时：核心和静态后端测试通过，特征测试明确跳过；不把跳过当作验证通过。GPU环境仍应按官方锁定依赖使用Python3.11与NumPy2.2.6等约束。
- 六个合成PCM tone的 `make_demo → prepare → validate` CLI通过，每个split两条clip、一位synthetic speaker；没有真实语音对齐或质量含义。
- 11个 CLI help 检查通过，14个Python文件AST解析、JSON/TOML解析和环境脚本 `sh -n` 通过。
- 只读在内存解析固定社区commit的实际argparse：示例配置38个key，没有未知字段；确认 `--output-dir` 为required CLI参数。未保存上游源码到项目。
- 缺少源目录时后端预检以exit2、`status: blocked`明确失败，未伪装成训练准备成功。
- 独立输入/泄漏审查：路径穿越、符号链接、控制/方向字符、特殊token、数字/货币、重复JSON key/NaN、PCM/NFC去重、speaker隔离、prompt不同clip、backend清单篡改、特征遗漏/陈旧文字、NPZ恶意shape/no-pickle、未知配置和旧审计哈希均有检查或测试覆盖。

## 这些测试证明什么

数据与特征测试生成自己的合成WAV/NPZ，不读取私人数据。NumPy数组是真实加载/检查；frontend对象与checkpoint内容的单元测试使用合成对象，**没有执行实际IndexTTS/backend TextFrontend编码、模型加载或GPU训练**。源接口及许可核查属于只读源码审计，不是端到端运行验证。

## 尚未验证 / 实际训练阻塞

1. 社区后端缺少LICENSE，使用者需取得明确授权或换用重新审计的合适实现；本仓库没有复制其代码。
2. Rasa/IndicVoices-R均须用户本人登录处理gated信息共享与访问条件；未接受、未读行、未下载。Rasa缺可信speaker映射时不能进入speaker隔离训练。
3. 没有实际模型snapshot/权重/辅助资产哈希、GPU依赖安装、离线加载、全部真实frontend/token序列或真实特征提取验证。
4. 没有执行真实trainer STOP/恢复往返、小样本过拟合、100-step H200吞吐/显存、Hindi/旧语言评测或adapter导出重载一致性。
5. 共享pod的持久存储、占用、OOM/驱逐风险需在单独本地运行记录中只读评估，不把running/ready当空闲。

没有现成Hindi权重，没有HF仓库创建或上传。当前所有质量/资源数值保持未测，不承诺GPU小时。
