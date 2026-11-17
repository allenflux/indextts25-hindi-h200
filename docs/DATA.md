# 数据格式与泄漏控制

输入 UTF-8 JSONL 必填 `id`, `speaker_id`, `audio`, `text`, `language`, `source`, `license`, `consent`。`language` 必须是 `hi`；`consent` 必须为布尔 true。ID 只允许安全 ASCII 标识符，不将未经验证的 ID 直接用于路径。`source` 记录可追溯来源，`license` 记录已人工审核的条款，不能用它代替许可文件或说话人同意证据。证据放在本地受控目录；不要提交私人证据。

`audio` 必须是相对显式 `--audio-root` 的本地路径，解析符号链接后也必须留在该根内。拒绝绝对路径、URL、目录穿越、控制字符；不执行文件名或文本。音频仅支持未压缩单声道 16-bit PCM WAV，默认 24 kHz，可显式选择其他采样率。默认时长 1–20 秒，文本字符长度也检查；这些仅是粗筛，不能替代强制对齐或人工听检。参考音频不应超过后端的 `max-ref-seconds`；本项目建议固定到 15 秒并保留足够冗余。

文本输出同时保留 `text_raw` 与 `text_norm`。只做 Unicode NFC 与空白归一；不做 NFKC、去音标、ASCII 转写或删除 Devanagari combining marks。保留 nukta、virama、ZWJ/ZWNJ 和 danda。NFC 可能将预组合 nukta 字母变为语义等价组合形式，原文始终保留。不自动统一不同标点、口音、拼写或 code-switching。拒绝双向控制字符、危险控制符与模型特殊 token 的分隔符，避免数据文本注入语言 token。

默认拒绝 Unicode 数字和货币符号，需要人工写成实际朗读形式，比如金额要明确币种及口语读法，日期/电话号码要按录音口径展开。若确实需要保留数字或货币原样，传 `--allow-as-spoken-numbers` 并逐条加 `numbers_as_spoken: true`，表示已人工确认文字与音频完全匹配。它不会调用 TTS normalizer 或自动推导 Hindi 读法。

去重默认全量严格检查规范文本和解码后的 PCM 哈希，避免不同 WAV 容器掩盖重复录音。重复文本也拒绝，宁可人工处理常见短句，避免 train/test 泄漏；近重复、切片重叠、同一个人多个 ID 和同一录制 session 的相关性仍须人工或额外声纹/录制元数据审核。若同一人多 ID，必须先合并成同一 `speaker_id`。

稳定 seed 加 speaker ID 决定拆分，生成 `manifest.json` 审计参数/统计及 train/val/test；同一 speaker 只出现在一个 split，所有 prompt 从目标自己的 split 内选择。至少三人、每人至少两条不同 clip。生产 gate 要求每个 held-out split 至少十位 speaker、每位若干 clip；这属于人工实验验收，不由最低格式检查替代。

后端原始清单使用单独的 `backend_train.jsonl`、`backend_val.jsonl`、`backend_test.jsonl`，映射 `speaker_id→speaker`, `text_norm→text`，保留 id 和 audio。每个 split 独立预处理并禁止后端重新分割；预处理生成的 `utterances_train.jsonl` 名称不代表它属于训练集。将每份文件放入不同的 feature 目录，保留对应源 split 元数据。完成后再按 runbook 生成同说话人不同 clip 的特征配对；test 特征清单永不传入训练器。

生成清单包含绝对本地路径或原文，可能有隐私信息，全部写入被忽略的目录。不要手动 `git add -f` 这些文件。发布模型时也不要附带训练 manifest、原始 prompt 音频或可逆的私人特征。

特征配对命令需显式设置 `--max-text-tokens`、`--max-code-tokens` 与 `--max-prompt-code-tokens`，并重新传 `--audio-root` 以复查 prepared 数据。核心配对仅核查清单；真正训练前还必须运行可选 NumPy 特征审计，对每个数组、词表范围和实际前端编码做一致性检查。若后端标点映射导致 feature text 与 `text_norm` 不同，会失败；人工审查并重建源文字/特征，不要改检查器绕过。
