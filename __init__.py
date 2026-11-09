import os
import queue
import random
import threading

from google import genai
from google.genai import types

API_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apikey.txt")
DEFAULT_MODEL = "gemini-3.5-flash-lite"

SAFETY_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
    "HARM_CATEGORY_JAILBREAK",
]

HTTP_OPTIONS = types.HttpOptions(timeout=300_000)

DEFAULT_PROMPT = """
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

8. **只能使用具有实际画面描述意义的内容 Tag。** 例如人物数量、人物特征、服装、动作、表情、构图、环境、光影等。禁止添加任何质量评价、画质评价、风格评价、画师、模型或生成效果类标签，包括但不限于 masterpiece、best quality、high quality、absurdres、highres、official art 等。优化方向应通过具体的画面内容 Tag 实现，而不是通过质量标签实现。

9. 不要局限于原始 Tag 的数量，可以根据已有内容合理丰富，但所有新增 Tag 都必须与原始画面内容具有明确关联，不得无依据地改变人物设计或画面设定。

10. 最终输出必须包含 50～60 个 Tag，按照逗号分隔计算，每个 Tag 计为一个词。不得通过重复或无关 Tag 凑数。

11. 输出前自行检查 Tag 数量、重复项、排除项和排列顺序。

最终只输出优化后的纯 Tag 提示词。不要输出标题、分类说明、数量统计、Markdown 代码块或任何解释。



"""

MODELS = [
    "gemini-4-argon",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
    "gemini-omni-1.2-flash",
    "gemini-omni-1.1-flash",
    "deep-research-preview-04-2026",
    "deep-research-max-preview-04-2026",
]

_clients = {}


def load_keys():
    with open(API_KEY_FILE, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_client(key):
    if key not in _clients:
        _clients[key] = genai.Client(api_key=key, http_options=HTTP_OPTIONS)
    return _clients[key]


def generate(model, contents, config):
    """后台线程读流，主线程每 200ms 检查一次 ComfyUI 中断，取消时立即退出。"""
    from comfy.model_management import throw_exception_if_processing_interrupted

    keys = load_keys()
    random.shuffle(keys)
    error = None
    for key in keys:
        try:
            return consume(key, model, contents, config,
                           throw_exception_if_processing_interrupted)
        except Exception as e:
            error = e
    raise error


def consume(key, model, contents, config, check_interrupt):
    q = queue.Queue()
    stop = threading.Event()

    def worker():
        try:
            for chunk in get_client(key).models.generate_content_stream(
                    model=model, contents=contents, config=config):
                if stop.is_set():
                    break
                q.put(("text", chunk))
        except BaseException as e:
            q.put(("error", e))
        finally:
            q.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()

    parts = []
    last = None
    while True:
        try:
            kind, payload = q.get(timeout=0.2)
        except queue.Empty:
            try:
                check_interrupt()
            except BaseException:
                stop.set()
                raise
            continue
        if kind == "done":
            return "".join(parts), last
        if kind == "error":
            raise payload
        parts.append(payload.text or "")
        last = payload


class GeminiPromptEnhance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODELS, {"default": DEFAULT_MODEL}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2 ** 31 - 1}),
                "thinking": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": 50000, "min": 1, "max": 65536}),
                "placeholder": ("STRING", {"default": "{TEXT}", "multiline": False}),
                "prompt": ("STRING", {"default": DEFAULT_PROMPT, "multiline": True}),
                "text": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Gemini"

    def run(self, text, model, seed, thinking, temperature, top_p, top_k,
            max_output_tokens, placeholder, prompt):
        contents = prompt.replace(placeholder, text)
        config = types.GenerateContentConfig(
            seed=seed % (2 ** 31),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            safety_settings=[types.SafetySetting(category=c, threshold="OFF")
                             for c in SAFETY_CATEGORIES],
            thinking_config=types.ThinkingConfig(
                thinking_level="high" if thinking else "minimal"),
        )
        out, last = generate(model, contents, config)
        if not out.strip():
            reason = last.candidates[0].finish_reason if last and last.candidates else "无响应"
            raise RuntimeError(f"Gemini 返回空内容，finish_reason={reason}")
        return (out,)


NODE_CLASS_MAPPINGS = {"GeminiPromptEnhance": GeminiPromptEnhance}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiPromptEnhance": "Gemini 提示词增强"}
