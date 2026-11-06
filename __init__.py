import os
import random

from google import genai
from google.genai import types

API_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apikey.txt")
DEFAULT_MODEL = "gemini-3.1-flash-lite"

SAFETY_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
    "HARM_CATEGORY_JAILBREAK",
]

HTTP_OPTIONS = types.HttpOptions(timeout=300_000)

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
                "text": ("STRING", {"forceInput": True}),
                "model": (MODELS, {"default": DEFAULT_MODEL}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2 ** 31 - 1}),
                "thinking": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": 4096, "min": 1, "max": 65536}),
                "placeholder": ("STRING", {"default": "{TEXT}", "multiline": False}),
                "prompt": ("STRING", {"default": "处理这段文本：{TEXT}", "multiline": True}),
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
            seed=seed,
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
        return (resp.text,)


NODE_CLASS_MAPPINGS = {"GeminiPromptEnhance": GeminiPromptEnhance}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiPromptEnhance": "Gemini 提示词增强"}
