# PNG@2 渲染回执

用户明确批准项目内渲染环境后执行。没有改全局Python、已有训练env、已安装技能、用户Chrome/扩展或任何GPU模型环境；两份冻结HTML的SHA在导出前后相同。

## 实际锁定环境

- 基础解释器：项目既有Python **3.11.15**，仅用于`-m venv`创建全新`.runtime/diagram-export-env`。
- 实际安装[Playwright **1.65.2**](https://pypi.org/project/playwright/1.65.2/)，官方PyPI元数据要求Python >=3.10，本机解释器兼容。安装使用官方PyPI、仅binary wheels。
- 按官方[浏览器安装说明](https://playwright.dev/python/docs/browsers)只安装匹配的Chromium headless shell **159.0.8154.6 / revision1255**，未借用用户Chrome。
- 完整venv freeze：`greenlet==3.5.6`、`playwright==1.65.2`、`pip==24.0`、`setuptools==79.0.1`。这是图表专用环境，不改模型依赖锁。
- TMPDIR、PIP_CACHE_DIR、PLAYWRIGHT_BROWSERS_PATH、XDG_CACHE_HOME均在本项目`.runtime/diagram-export-cache/`，配置的临时文件、包缓存、浏览器、状态与日志均定位到本项目；设置`DO_NOT_TRACK=1`，不加载外部字体/资源。

原始安装元数据、pip wheel哈希报告、freeze、Chromium安装日志、导出日志和布局/字体QA JSON保存在`.runtime/diagram-export-receipts/`（不提交运行时缓存）。PNG由diagram-design **2.6.164**的内置`export_png.py`原样执行，绝对路径指向已解析技能目录；helper SHA256为`430ddcc2d39441808bd4217fbdeb944dbb6de0be281782705a81747986d7d348`，运行前后未变。

## 产物

| PNG | 尺寸 | Bytes | SHA256 |
|---|---|---:|---|
| [indextts25-inference.png](indextts25-inference.png) | 2560×1440 | 195232 | `cce92b9b0fd9d5f3581db226919d7a326821de5ae5d62fbf7a673a58964b6e78` |
| [hindi-lora-training.png](hindi-lora-training.png) | 2560×1440 | 222241 | `d5ee07eac9bacf7f2bcbfbd24385c9b4e3fe73aa1ce28d460125058534adf8ee` |

[render-manifest.json](render-manifest.json)另存HTML与PNG身份和实际渲染版本。源文件分别为[indextts25-inference.html](indextts25-inference.html)与[hindi-lora-training.html](hindi-lora-training.html)，没有手绘SVG转图或重写导出器。

## 实际QA

- 内置`self_check.py --offline`两图PASS；`verify_geometry.py`两图、0 finding。
- Chromium实际文本BBox：viewBox外0项、节点内溢出0项；390px视口正文无横向溢出，图容器可滚至最右端，SVG min-width1280px。
- 两页渲染的HTTP/HTTPS请求0，QA context还对外部请求设为拒绝；安装/导出没有字体超时或fallback告警。字体本来就声明为离线fallback，未声称精确Atlassian字体。
- 实际computed CSS：标题28px/400 serif栈；人类可读节点名12px/600 sans栈；技术副标9px/400 mono栈；callout14px/400 italic serif栈。这里记录CSS栈，不把首选字体声明当成已安装的专有字体。
- 两张最终PNG已通过本地`view_image`打开检查：内容、右端WAV/合并框、文字与正交连线完整，没有观察到裁切、重叠或缺字。此前本地浏览器截图QA亦通过。

这些是原理图的安全、几何、字体和视觉检查，不是Hindi语音人工听评、模型质量或GPU训练验证。HTML、PNG及回执已冻结；Confluence附件采用上述对应PNG产物。
