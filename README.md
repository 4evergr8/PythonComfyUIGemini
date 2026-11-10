# PythonComfyUIGemini

ComfyUI 自定义节点：调用 Gemini / Gemma 模型优化图像生成提示词。

输入一段 Danbooru 风格 tag 或自然语言描述，节点将其送入 LLM，按你设定的系统提示词重写后输出，直接接给 Anima、Illustrious、NoobAI 等图像模型。

---

## 功能

- **零依赖**：不需要 `pip install`，放入 `custom_nodes` 即可用
- **单节点闭环**：输入文本 → LLM 优化 → 输出文本，无需额外脚本
- **提示词完全可编辑**：默认提示词只是起点，节点里直接改
- **支持前端中断**：流式读取，点击取消后约 0.2 秒内断开请求
- **多 Key 轮换**：`apikey.txt` 里写多个 key，失败自动换下一个
- **思考模式开关**：`thinking` 控制推理深度
- **种子可复现**：固定 seed + 固定参数 = 固定输出

---

## 安装

### 1. 放到 custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/4evergr8/PythonComfyUIGemini.git
```

或直接下载 ZIP 解压到 `ComfyUI/custom_nodes/PythonComfyUIGemini/`。

### 2. 依赖：无需额外安装

**本节点不需要安装任何第三方依赖。** 它只用到 `google-genai`，而 ComfyUI 环境已内置（便携版自带 3.0.0），放入 `custom_nodes` 后直接可用。

不要执行 `pip install`，也不需要装 requirements。

仅当你启动时看到 `ModuleNotFoundError: No module named 'google.genai'`，才说明环境确实缺失，此时再补装：

```bash
# 便携版
cd ComfyUI/custom_nodes/PythonComfyUIGemini
../../python_embeded/python.exe -m pip install google-genai

# 手动部署 / Linux
pip install google-genai
```

### 3. 配置 API Key

在节点目录下新建 `apikey.txt`，一行一个 key：

```
AIzaSy...key1
AIzaSy...key2
AIzaSy...key3
```

- 支持多个 key，运行时随机打乱顺序，某个 key 失败会自动尝试下一个
- 文件与 `__init__.py` 同级
- 空行会被忽略

### 4. 重启 ComfyUI

Python 模块不会热更新，改完文件必须完全退出进程再启动。

---

## 使用方法

节点位于 **Gemini → Gemini 提示词增强**。

```
[文本输入 / 手工填写] ──► [Gemini 提示词增强] ──► [CLIP Text Encode] ──► ...
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | 下拉 | `gemini-3.5-flash-lite` | 使用的模型，见下方列表 |
| `seed` | INT | `42` | 0 ~ 2147483647，控制输出可复现性 |
| `thinking` | BOOLEAN | `True` | 开启为 `high`，关闭为 `minimal`。指令复杂时建议开启 |
| `temperature` | FLOAT | `1.0` | 0 ~ 2，越高越发散 |
| `top_p` | FLOAT | `0.95` | 0 ~ 1 |
| `top_k` | INT | `40` | 0 ~ 100 |
| `max_output_tokens` | INT | `50000` | 1 ~ 65536，需留出思考 token 的空间 |
| `placeholder` | STRING | `{TEXT}` | 系统提示词中的占位符 |
| `prompt` | STRING | 见下 | 系统提示词，可任意修改 |
| `text` | STRING | `""` | 待优化的原始提示词 |

**输出**：`text`（STRING），即 LLM 返回的优化结果。

### 占位符机制

`prompt` 里的 `{TEXT}` 会被 `text` 参数的内容替换，然后整段发给模型。

例如 `prompt` 是：

```
优化这段提示词：
{TEXT}
只输出结果。
```

`text` 是 `1girl, smile`，实际发送的内容就是：

```
优化这段提示词：
1girl, smile
只输出结果。
```

如果你把 `placeholder` 改成 `{{INPUT}}`，那 `prompt` 里就要写 `{{INPUT}}`。

---

## 默认提示词

节点内置的默认提示词面向 **Anima 模型的纯 Tag 优化**，完整内容如下：

```
请优化以下 Anima 模型使用的纯 Tag 提示词：

{TEXT}

严格遵守以下规则：

1. 以原始 Tag 为基础进行优化，保留原有画面主题、人物设定、服装设计、动作、表情、构图和装饰特征。不要改变原始画面的核心内容。

2. 采用双层排序结构：
   - 第一层：主体信息，按照重要程度排列人物数量、核心构图、镜头视角、人物特征、动作姿势、服装、表情等核心 Tag。
   - 第二层：细节补充，继续沿着原有 Tag 丰富人物、身体、服装、动作、装饰、画面表现及光影细节。

3. 不要机械地按照分类平均分配 Tag。将更多细节集中在人物表现、服装设计、身体细节、姿势、画面质感和光影上，优先丰富已有内容。

4. 积极扩展原始 Tag 的表现力，使最终画面朝向精致、漂亮、具有视觉吸引力的高质量二次元插画方向优化。可以补充能够体现人物美感、服装精致度、姿势表现力、画面层次感和光影效果的相关 Tag。

5. 不要凭空添加原始提示词没有体现的场景、背景物品或环境设定。没有背景信息就不要添加背景 Tag。不要为了凑数量强行扩展无关内容。

6. 删除人物独有的身份识别信息，例如发色、瞳色，以及 halo、光环等光环相关信息。

7. 删除重复、矛盾、冗余或没有实际意义的 Tag。允许将原有 Tag 替换为更准确、更规范的 Danbooru 风格 Tag。

8. 只能使用具有实际画面描述意义的内容 Tag。例如人物数量、人物特征、服装、动作、表情、构图、环境、光影等。禁止添加任何质量评价、画质评价、风格评价、画师、模型或生成效果类标签，包括但不限于 masterpiece、best quality、high quality、absurdres、highres、official art 等。优化方向应通过具体的画面内容 Tag 实现，而不是通过质量标签实现。

9. 不要局限于原始 Tag 的数量，可以根据已有内容合理丰富，但所有新增 Tag 都必须与原始画面内容具有明确关联，不得无依据地改变人物设计或画面设定。

10. 最终输出必须包含 50～60 个 Tag，按照逗号分隔计算，每个 Tag 计为一个词。不得通过重复或无关 Tag 凑数。

11. 输出前自行检查 Tag 数量、重复项、排除项和排列顺序。

最终只输出优化后的纯 Tag 提示词。不要输出标题、分类说明、数量统计、Markdown 代码块或任何解释。
```

### 自定义提示词的建议

默认提示词是「纯 Tag 优化」。如果你想改成 **Anima hybrid 格式**（tag 行 + 空行 + 自然语言），把 `prompt` 换成下面这份：

```
Convert this tag list into an Anima prompt. Follow the format exactly.
{TEXT}

# Output: two layers, separated by exactly one blank line
Print only the two blocks themselves. Never print the layer headings, labels, or any bullet markers.

## Layer 1 — hard tags, single line
- Comma plus space separated, lowercase.
- Include: subject count, appearance, outfit, pose, expression, props, scene anchors.
- Remove character identity: hair colour, eye colour, halo, character names, series names.
- Group tags by their final noun and keep at most two per group.
- Emoji-style expression tags must survive verbatim.
- Hard tags must match the shot size: for a close-up drop legs and footwear; for full body keep them.
- Tag count by complexity: simple 16–30, standard 22–38, complex 30–48.

## Layer 2 — natural language
- Sentence one: shot size, first visual subject, composition mode, overall layout.
- Sentence two: composition map — left, right, centre, foreground, middle ground, background.
- Sentence three: subject scale and position, action, gaze, contact with props.
- Final sentence: main light source and direction, what it lights, what stays in shadow; then one dominant hue plus at most two accents bound to named objects.
- Never add foreground occlusion unless the input already contains it.
- No quality words, artist names, model names. No "the image shows".
```

区别在于：默认版输出单一 tag 列表；hybrid 版额外生成一段自然语言，用来控制构图、主体占比、光照和色彩层次。

---

## 支持的模型

```
gemini-4-argon
gemini-3.8-flash
gemini-3.7-flash
gemini-3.6-flash
gemini-3.5-flash
gemini-3.5-flash-lite        ← 默认
gemini-3.1-pro-preview
gemini-3.1-flash-lite
gemini-3-flash-preview
gemini-2.5-pro
gemini-2.5-flash
gemini-2.5-flash-lite
gemini-flash-latest
gemini-flash-lite-latest
gemini-pro-latest
gemma-4-31b-it
gemma-4-26b-a4b-it
gemini-omni-1.2-flash
gemini-omni-1.1-flash
deep-research-preview-04-2026
deep-research-max-preview-04-2026
```

模型列表写在 `__init__.py` 的 `MODELS` 常量里，可自行增删。

---

## 常见问题

**Q：报错 `Gemini 返回空内容，finish_reason=MAX_TOKENS`**

思考 token 吃光了输出预算。把 `max_output_tokens` 调大（建议 16384 以上），或把 `thinking` 关掉。

**Q：报错 `400 API key not valid`**

`apikey.txt` 里的 key 有误，或文件不在 `__init__.py` 同级目录。

**Q：报错 `500 INTERNAL` / `503 UNAVAILABLE`**

上游服务波动。节点会自动换 key 重试，若持续失败请换模型或稍后再试。Gemma 系列在高峰期失败率明显高于 Gemini 系列。

**Q：点了取消，但还是要等一会儿才停**

节点用流式读取，主线程每 200ms 检查一次中断信号。如果模型在思考阶段长时间不吐数据块，最坏情况下要等一个数据块的间隔。正常情况下响应时间在 0.3 秒以内。

**Q：改完提示词没生效**

Python 模块常驻内存，必须完全退出 ComfyUI 进程再启动，只关窗口无效。启动后浏览器 `Ctrl + Shift + R` 硬刷新。

**Q：需要代理**

本机环境变量或系统代理需要覆盖 `generativelanguage.googleapis.com`。节点内部没有硬编码代理配置。

---

## 目录结构

```
PythonComfyUIGemini/
├── __init__.py       节点实现
├── apikey.txt        API Key（自行创建，不要提交到仓库）
├── gemini_chat.py    独立对话脚本
├── gemini_common.py  公共函数
└── gemini_models.py  模型探测脚本
```

---

## 许可

未指定。使用前请自行确认所调用模型的服务条款。
