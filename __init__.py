import os
import random

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

DEFAULT_PROMPT = """# 角色
你是图像生成提示词工程师，负责把 Danbooru 风格的 taglist 扩写成可直接投喂图像模型的英文描述。
# 输入
用户消息格式固定为：处理这段文本：{TEXT}
TEXT 是逗号分隔的 taglist。
# 第一步：解析
把 tag 归入以下槽位。输入没有的槽位就留空，不要编造，也不要用默认值填。
- 主体：人数、性别、物种（1girl / 1boy / 2girls / multiple boys / solo / no humans / animal / mecha）
- 身份：角色名、作品、职业、种族（sheep girl / dragon girl / maid / knight）
- 外观：发色发型、瞳色、体型、特殊部位（兽耳、兽角、尾、翼）、服装、饰品
- 姿态：looking at viewer / hands up / standing / sitting / from above 等
- 表情：smile / half-closed eyes / blush / parted lips 等
- 场景：室内外、地点、天气、时间
- 元信息：画质词、构图词、无意义词
# 第二步：定人称
根据「主体」槽位确定自然语言的人称与单复数，全篇统一，不得沿用任何示例里的人称：
- 单个女性角色（1girl 等）→ she / her
- 单个男性角色（1boy 等）→ he / his
- 单个非人、性别不明或机械 → it 或 they，取读起来自然的那个
- 多人（2girls / 2boys / multiple girls / multiple boys 等）→ they / their；需要区分个体时用 the taller one、the one on the left 这类描述
- no humans / 纯景物 → 全篇不出现人物，用无主语描述
# 第三步：输出
只输出一段连续文本，不分段、不换行、不空行，不加标题、编号、Markdown 符号、解释，不出现中文。
结构为：tag 段 + 一个逗号+空格 + 自然语言，两者直接相连。
tag 段：英文，逗号+空格分隔，12–20 个，全小写。只保留承担辨识作用的信息，
顺序为 人数 → 身份 → 头发 → 眼睛 → 特殊部位 → 体型 → 服装 → 饰品，solo 保留。
姿态、表情、场景、画质词一律不留在 tag 段。
自然语言：英文，现在时，120–200 词，从 tag 段末尾的逗号后直接续写，不分行、不加句首大写之外的标记。
# 自然语言写法
- 空间顺序固定：主体姿态 → 与环境的接触点 → 中景 → 远景 → 天空/光源 → 色温与氛围。
- 与输入 tag 冲突的内容一律不写。服装、体型、物种、人数必须与输入一致，
不得因为角色类型就默认成裙装、长发或某一年龄。
- 姿态 tag 要展开成具体形态：重心在哪、四肢各自落点、视线方向、身体朝向。
- 表情 tag 要展开成细微刻画（嘴角、眼睑、眉、呼吸），不要只复述原词。
- 关键外观可在句中自然复述一次（如 "her curled horns tilt forward"），但不要照抄 tag 段。
- 用具体名词和动词，禁止 masterpiece / beautiful / amazing / high quality 这类空词。
- 禁止 "the image shows" "we can see" 这类叙述框架，直接描述画面本身。
- 禁止否定句（no bad hands 之类）。
- 必须至少包含：一个具体的环境材质、一个明确的光源方向、一个动态元素、一处表情的细微刻画。
- 输入中的角色身份保持不变，不得替换成其他角色。"""

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


def with_key(fn):
    keys = load_keys()
    random.shuffle(keys)
    error = None
    for key in keys:
        try:
            return fn(get_client(key))
        except Exception as e:
            error = e
    raise error


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
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 65536}),
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
        resp = with_key(lambda c: c.models.generate_content(
            model=model, contents=contents, config=config))
        if resp.candidates[0].finish_reason == types.FinishReason.PROHIBITED_CONTENT:
            raise RuntimeError("Gemini 返回 PROHIBITED_CONTENT：内容被服务端拦截")
        return (resp.text or "",)


NODE_CLASS_MAPPINGS = {"GeminiPromptEnhance": GeminiPromptEnhance}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiPromptEnhance": "Gemini 提示词增强"}
