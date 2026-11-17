# Hindi 数据来源与实际访问范围

早期阶段只核查公开模型卡/配置，当时尚未登录、读取语料或下载音频。此后用户本人处理了访问条件，本项目已实际使用**IndicVoices-R的Hindi子集**；Rasa未下载或参与本次训练。真实范围见[VALIDATION](../VALIDATION.md)，分发边界见[DISTRIBUTION](DISTRIBUTION.md)。

## 本次实际取得的数据

- 数据集：`ai4bharat/indicvoices_r`，固定revision `5f4495c91d500742a58d1be2ab07d77f73c0acf8`，配置`Hindi`。
- 已记录操作者本人处理访问条件、认证与数据读取检查成功；此处不记录用户名或凭据。后续离线复用这些已取得文件不等于重新认证成功，也不承诺新环境仍有认证。
- 固定版本Hindi清单为101份Parquet、47,225,888,374字节，**这是元数据清单总量，不是实际下载量**。
- 只实际下载两份官方TRAIN分片：`Hindi/train-00000-of-00099.parquet`（479,633,889字节）及`Hindi/train-00001-of-00099.parquet`（467,329,797字节），语料payload合计**946,963,686字节**。该数不含模型、README等小型审计文件、缓存副本、解码音频或特征；未拉全Hindi子集，未使用官方TEST分片。
- 当前pilot从这些官方TRAIN分片构建按speaker隔离的train48/val8/test8，共64 clips/16 speakers。这里的本项目test不等于发布方官方TEST；实际处理与哈希留在受控工作目录，仓库不打包真实音频或私人转写。

固定数据卡标为CC-BY-4.0；缓存frontmatter未含`extra_gated_*`字段，不足以证明平台没有其他条件。完整来源、许可及接受条件的证据范围见[DISTRIBUTION](DISTRIBUTION.md)，不能把可访问直接当成所有后续用途授权。

## 最初的候选预检

下表保留最初候选判断，不能替代上述实际取得与使用记录。

| 来源 | 人工访问前置条件 | 公开元数据中的Hindi配置 | 最初预检限制 |
|---|---|---|---|
| [AI4Bharat Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) | 用户本人登录并阅读/处理共享账户信息及访问条件，工具不能代接受 | config名`Hindi`，列有train/test shards | 当时可见schema缺speaker_id；未取得可信映射，不用gender、style或文件名猜身份。本次未使用 |
| [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r) | 同样由用户本人处理gated条件；新环境还须核实实际认证 | config名`Hindi`；schema有speaker_id，audio标注48kHz | 后续已取得上述有限批次并按真实音频校验/转码；文本对齐及音质不能仅靠schema或许可字段认证 |

Rasa全库标称约**388GB**，不是本项目测量值；其Hindi子集大小和存储/缓存架构未实测。本次没有下载Rasa。以后处理任何来源仍须显式只选Hindi、限定分片/行数和本地配额；不能把streaming当成零下载保证。

## 继续读取前的流程

1. 页面显示gated时，由用户本人核对并处理条件；不绕过、不自动接受。保留适用repo/revision、许可及访问审核记录。
2. 记录允许用途与说话人同意依据，区分发布方声明、操作者审核和独立核验。数据可读不代表所有后续用途均获授权。
3. 核查真实schema、原始/规范文本、音频采样与完整长度、speaker键、PCM/文本去重，再构建项目JSONL。不得用猜测身份补齐分组。
4. 按speaker隔离训练、验证和test；test不得参与训练或选模prompt、调参和checkpoint选择。预先固定的test推理可使用该评测内同speaker另一clip作参考，须记录角色且不事后挑好例。
5. 在本地保存source/revision/clip映射、授权依据、文件哈希及转换参数，运行prepare/validate，再按已记录质量门槛决定下一步。私人记录不随公开代码提交。

原生限额导入器已实现，接口与预算见[NATIVE](NATIVE.md)；CLI复用缓存时仍先核实认证与Hub元数据，不是离线入口。直接读取已合法取得且重新校验SHA的本地分片须保留既有授权/下载审计，不得伪称进行了新访问检查。

需要在线访问时，可按HF官方[auth_check API](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.auth_check)只验证读取权限；遇GatedRepoError停止，不自动接受条款。Rasa默认配置可能跨语言，因此不能依赖默认config。
