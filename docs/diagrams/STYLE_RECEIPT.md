# Confluence参考风格回执

两图采用diagram-design的`architecture`与`process`语法，`doc-wide` 1280×720，混合技术读者、静态、离线系统字体模式；目标wiki格式为PNG@2（2560×1440），HTML为源文件。用户已选择参考Confluence网站样式，颜色来自本次对已加载、已授权Confluence文档DOM的实际采样；本次没有重新请求该网站。私人文档URL/页面ID不写入公开仓库，采样上下文保留在会话中。

## 检测与项目token

| 角色 | DOM检测 / 本项目采用 | 置信度与处理 |
|---|---|---|
| paper | 检测`#ffffff`；采用`#fafaf7` | 高；按技能将纯白调为暖中性纸面 |
| ink | `#292a2e` | 高；保留采样值 |
| accent / link | `#1868db` | 高；保留采样值，仅突出1–2个焦点 |
| muted | `#5e6570` | 派生；未声称来自DOM |
| soft | `#646b75` | 派生；保留小字对比度 |
| essential border | `#7a7f87` | 派生；节点边界对纸面3.85:1 |
| rule | `#d8dce2` | 派生装饰线，不单独承载语义 |
| accent tint | `#eef3fb` | 派生；蓝色文字在其上4.66:1 |
| surface | 节点白色；输入/数据轻灰 | 按语义类型区分，无阴影、无点阵纸 |

纸面对比度：ink **13.71:1**，muted **5.62:1**，soft **5.15:1**，accent/link **4.97:1**。文字角色均超过4.5:1；必要边界超过3:1。焦点浅蓝底没有改变主文字的ink色。

## 字体：检测值不等于输出值

| 角色 | 实际DOM检测 | 输出栈与状态 |
|---|---|---|
| 页面标题 | Atlassian Sans，28px，400 | `'Instrument Serif', 'Noto Serif', 'Songti SC', Georgia, serif`，28px/400；**fallback**，保留技能serif标题角色 |
| 正文 / 节点名 | Atlassian Sans，400 | `'Geist', 'PingFang SC', 'Microsoft YaHei', system-ui, sans-serif`；节点名12px/600，中文说明12px；**fallback** |
| code / 技术标签 | Atlassian Mono，400 | `'Geist Mono', 'SFMono-Regular', Consolas, ui-monospace, monospace`；9px技术副标/8px标签；**fallback** |
| callout | 未检测独立角色 | 标题serif栈，14px/400 italic；**fallback** |

没有复制、嵌入或外连Atlassian专有字体，也没有Google Fonts请求；未声称精确复现该字体。每个字体栈均能退回平台字体，中文显式给出SC回退。系统字体的实际字形会随机器变化，PNG用于固定像素呈现。已在Mac本地浏览器实际打开两HTML并截图检查：图文可读，图内横向scroller可达右端WAV，正文无横向溢出，Confluence参考与fallback外观符合预期。这是原理图视觉QA，不是语音质量人工听评；获批的项目内purpose renderer现已完成PNG及computed-style检查：标题28px/400 serif栈、节点12px/600 sans栈、技术9px/400 mono栈、callout14px/400 italic serif栈。两PNG又经本地view_image检查通过；这仍不声称专有字体精确匹配，见[渲染回执](RENDER_RECEIPT.md)。

## 配置范围

只在本项目HTML/CSS及此回执中实施，未修改已安装技能的`style-guide.md`、全局profile或用户目录。未新增`.diagram-design`全局/项目profile绑定。架构图9节点/10箭头，过程图4节点/3箭头，均不超9节点/12连接；所有连接正交、先连线后节点。过程图按4步骤×2责任lane的参数网格作doc-wide归一，字体保持标准角色阶及中文12px下限。

技能版本：diagram-design **2.6.164**。当前`self_check.py --offline`与`verify_geometry.py`两图均PASS；初版外部source链接改成了纯文本固定URL，HTML不依赖远程资源。图内S/T来源键及压缩范围见[SOURCES_AND_FIDELITY](SOURCES_AND_FIDELITY.md)。PNG导出必须使用技能内置`export_png.py`，不另画SVG模拟导出；最初基础python3无Playwright、python命令不存在；取得用户明确安装许可后才创建独立项目venv/cache，实际Playwright1.65.2与Chromium1255已完成导出。没有修改原有训练env或用户Chrome，详见[渲染回执](RENDER_RECEIPT.md)。
