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

DEFAULT_PROMPT = """Convert this tag list into an Anima prompt. Follow the format exactly.
{TEXT}

# Decide before writing (internal only, never output)
1. What is the first visual subject — a character, a building, an environment, or a prop?
2. Which composition mode fits it: character-dominant, character-and-environment balanced, environment-dominant, or small-figure scale anchor?
3. Where do the main visual anchors sit?
4. What role does the character serve — main subject, scale reference, colour focus, or narrative entry?
5. How are foreground, middle ground, and background organised?
6. How do key light and colour establish hierarchy?

# Output: two layers, separated by exactly one blank line
Print only the two blocks themselves. Never print the layer headings, labels, or any bullet markers.

## Layer 1 — hard tags, single line
- Comma plus space separated, lowercase, underscores replaced by spaces.
- Include: subject count, per-character appearance and outfit, pose, expression, props, and concise scene anchors.
- Exclude: full sentences, subject scale, composition layout, light direction, colour hierarchy, quality tags, artist tags, and safety tags. Those are appended externally.
- Never invent a tag the input does not support. If unsure whether a tag exists, express the idea in the natural language layer instead.
- Remove character identity: hair colour, eye colour, halo, character names, and series names.
- Group tags by their final noun and keep at most two per group, covering different dimensions. When a general and a specific form share the same noun, keep only the specific one.
- Emoji-style expression tags must survive verbatim.
- Hard tags must match the shot size you will describe: for a close-up drop legs, feet, footwear, and lower garments; for an upper-body shot drop skirt, trousers, and leg tags; for a full-body shot keep them because the body is visible.
- Tag count by complexity: simple 16–30, standard 22–38, complex environment or multiple characters 30–48.

## Layer 2 — natural language
- 2–3 sentences for a simple character-dominant scene, 3–5 for a complex environment, 3–6 for multiple characters.
- Sentence one: shot size, first visual subject, composition mode, main visual anchors, overall layout.
- Sentence two: the composition map, using left, right, centre, upper, lower, foreground, middle ground, and background — main zones, framing elements, leading lines, and breathing space.
- Sentence three: the subject's scale in the frame, its position, its function, its action, its gaze, and how it contacts existing props. For multiple characters, give each character its own anchoring sentence first, then a separate sentence for interaction that names who acts on whom.
- Final sentence: the main light source, its direction, which part of the subject it lights, what the background light does, and what remains in shadow; then colour roles — one dominant hue and at most two accents, each bound to a named object or area.
- Never add foreground occlusion unless the input already contains a foreground object.
- Never state hair colour, eye colour, halo, character names, or series names.
- Never write "the image shows" or "we can see". No quality words, artist names, or model names.
- Concrete nouns and verbs, present tense, active voice.

# Final check
- Do the hard tags match the shot size named in the natural language?
- Any contradictory pair — close-up versus full body, looking at viewer versus facing away, open mouth versus closed mouth?
- Did every emoji-style expression tag survive?
- Is any tag a duplicate, or a general form of another tag?

Output only the two layers."""

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
        out, last = generate(model, contents, config)
        if not out.strip():
            reason = last.candidates[0].finish_reason if last and last.candidates else "无响应"
            raise RuntimeError(f"Gemini 返回空内容，finish_reason={reason}")
        return (out,)


NODE_CLASS_MAPPINGS = {"GeminiPromptEnhance": GeminiPromptEnhance}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiPromptEnhance": "Gemini 提示词增强"}
