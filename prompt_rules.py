"""Image-authoring rules adapted from the user's standalone master prompts."""

import hashlib
import re
from pathlib import Path

from config import MASTER_PROMPT_DIR


MASTER_FILES = {"flux2": "flux2prompt.txt", "anima": "anima_prompt.txt"}
QWEN_MASTER_FILES = {
    "generation": "qwen_image_2.1_system_prompt_t2i.txt",
    "editing": "qwen_image_2.1_system_prompt_edit.txt",
    "outpaint": "qwen_image_2.1_system_prompt_edit.txt",
}

COMMON_RULES = """Preserve the user's explicit subjects/count, identity, appearance, pose, wardrobe,
setting, colors, lighting, composition, style and exact visible lettering. Bind each
character's attributes to that character. Do not invent extra subjects or conflicting
lighting/style. Quote visible lettering exactly, including its case and punctuation.
Use only references actually supplied and inspected; never invent reference details.
A sufficient brief or YOLO means proceed with compatible choices. Ask one focused
question only for material ambiguity or requested guidance, not merely short input.
Pass only image-prompt text to prompt; keep settings, explanations and Markdown outside
it. When the selected model supports negative_prompt, keep negatives there; otherwise keep essential
constraints in the positive text. Negatives must not exclude requested effects.
Word counts below are editorial targets, never truncation limits. Later explicit user
instructions take precedence over creative defaults. Inspect returned images or use
preview_canvas after generation; do not claim to have inspected an unseen result."""

MODEL_RULES = {
    "flux2": """FLUX master (flux2prompt.txt): connected visual prose, usually 30–80 words;
front-load the main priority, then setting/details, lighting and atmosphere. Expand only
for meaningful requirements. No keyword dump, redundant quality adjectives or appended
tag suffix. Preserve requested lighting; 'normal' depends on the actual setting.
The installed backend is FLUX.2 Dev NVFP4 (32B), not Klein. Use the subject, action,
style and context in priority order; name each reference image's role for edits and
describe the requested change and preserved identity explicitly. Describe desired
results positively: Dev does not use negative_prompt. Quote lettering and specify its
placement and typography; bind exact requested colors to their objects. Native sampling
uses Mistral Small, flux2 VAE, Euler and Flux2Scheduler, 50 steps with embedded guidance
4 (28 steps is a faster trade-off). The API's cfg means embedded FluxGuidance for this
backend, not a second negative-conditioning pass. NVFP4 is weight quantization, not a
four-step distilled recipe. The local master still has a historical Klein heading;
these Dev-specific task and sampling rules adapt it to the installed model.""",
    "anima": """Anima master (anima_prompt.txt): use a compact mix of lowercase visual tags
and natural-language sentences; ordinary tags use spaces, not underscores. Keep each
character's identity/appearance/clothes/action together. Use anime/illustration language
unless the user specifies otherwise. No random artists or forced cinematic lighting.
For the installed Anima-Aesthetic, omit score_* quality tags from BOTH prompts; base
checkpoints may use them. Preserve quoted visible lettering even when it contains tag-like
text. Quality tags are optional. Usually 30–60 words for simple images, 60–120 for moderate,
100–180 for complex scenes; no padding or truncation to hit counts. Put compatible negatives
in negative_prompt. Aesthetic starting point: worst quality, low quality, artist name,
blurry, jpeg artifacts, chromatic aberration. Remove any negative conflicting with the brief.
Do not append negatives mechanically or put a negative-prompt section inside prompt.""",
    "qwen21": """Qwen 2.1 has separate generation and edit masters. Adapt their JSON output
to MCP arguments: pass only rewritten_prompt's text as prompt, never the JSON object or
the master itself. Keep wh_ratio/ratio_follow and resolution out of descriptive prose
(preserve literal lettering requested by the user). Use actual tool sizing arguments;
the master does not add new MCP fields or change sampling presets. Do not apply Flux
word targets, Anima tags or H3/video syntax to Qwen.""",
}

QWEN_TASK_RULES = {
    "generation": """Qwen generation master (qwen_image_2.1_system_prompt_t2i.txt): describe
the finished frame in English, present tense, third person, as an observer rather than
instructions to a renderer. Preserve every fixed object, count, color, position and exact
text in its original script. Fill open scene details coherently without contradicting
the brief. Aim for roughly 400–500 words/about twenty sentences, expanding a short brief;
quiet subjects may be shorter and dense layouts longer. This is an authoring target.
Open with medium, style, subject and background/palette; inventory elements and readable
text, then walk the background, top, left/center/right and bottom, or a close subject's
pose, face, clothing and edges. Use about 8–14 positional phrases across the frame.
State lighting, direction and surface effects explicitly; finish with one overall
composition sentence. Usually one paragraph; stacked regions may have one per region.
Quote exact visible text in straight double quotes with position and typography, including
chart/table labels and values. Do not invent signage for a text-free scene. Describe
unreadable marks as unreadable rather than inventing letters. Use specific colors,
materials and counts; unnamed objects by class rather than brand. Hedge only genuinely
open ambiguities, never fixed user details. No quality boosters or commands in the prose.
Map wh_ratio to width/height (or per-job dimensions in batch_generate), using the model
grid. Honor explicit dimensions/ratio; otherwise normally 3:2 horizontal, 2:3 vertical,
1:1 for square icons/covers, 16:9 for wide presentations, 1:2 or 9:16 for tall screens.
Set dimensions explicitly when choosing a ratio; the tool's default is still 1024x1024.""",
    "editing": """Qwen edit master (qwen_image_2.1_system_prompt_edit.txt): lead with the
operation in a decisive, affirmative, single-paragraph directive grounded in inspected
inputs. Change only named attributes, at the requested strength, clearly enough to avoid
an unchanged-looking result; preservation must not weaken the edit. Keep untargeted
identity, accessories, product design/count and rendering medium at input fidelity.
Use one preservation clause; do not redescribe unchanged appearances and cause drift.
Point to actual identity references rather than inventing facial features. For a new
scene built from references, develop the requested scene/light/layout; for a local edit,
stay restrained. No unrequested cleanup. Keep revealed regions physically coherent.
Description language: Chinese for a Chinese instruction; English for English or any
other instruction language. Rendered text is a separate decision: explicit exact text
or target language first, otherwise the input's dominant text language, otherwise the
user's instruction language. Quote each readable string exactly in double quotes;
no unsolicited bilingual text or genre-driven switch to English. Preserve user-supplied
proper nouns and units. Do not add text whose exact content cannot be determined.
Single image without references or mask: refer naturally to 'the image', without tags.
Multiple inputs: use <image1>, <image2>, etc., naming each role individually. In this MCP,
the canvas is <image1>, reference_paths follow in order, and an optional edit mask is
last. Open the intended composition/base image as the canvas before editing; the tool
always has a canvas, even when references are only identity sources.
Map ratio_follow to the intended canvas, not a prompt field. edit_image returns that
canvas's dimensions; max_side controls working detail, not aspect ratio. Use outpaint
or an explicit canvas operation for requested framing changes. For outpaint, name the
operation and use direction/amount for the extension; without an explicit amount the
guide suggests roughly 30–50% on the requested axis. The guide's 2K preference is not
an automatic resolution change: edit_image can use max_side=2048 for detail, while
outpaint keeps its existing internal resolution. Do not put ratios or size metadata
in the descriptive prompt, or pass wh_ratio/ratio_follow as unsupported tool fields.""",
}

TASK_RULES = {
    "generation": "Create one still-image description; map Anima positive_prompt to prompt and negative_prompt to its separate tool argument. Do not send code-block labels or the master instructions to the image model.",
    "editing": "Describe the smallest requested change and what must stay unchanged. Keep the source as the base; assign reference roles explicitly. Use region/mask_path for local edits. Undo an unsuccessful edit before retrying. The master file's standalone response format is not the MCP argument format.",
    "outpaint": "Describe the content that should continue into the new margin, matching the source's style, light, scale and perspective. Preserve source content; do not introduce an unrelated scene. Direction/amount remain tool arguments, not invented camera motion or H3 shots.",
}


def with_prompt_rules(task, models):
    """Put rules in tools/list, including for clients that ignore initialize instructions."""
    def decorate(function):
        rules = "\n\n".join([COMMON_RULES, TASK_RULES[task]] + [model_prompt_rules(m, task) for m in models])
        function.__doc__ = (function.__doc__ or "") + "\n\nMASTER PROMPT RULES:\n" + rules + (
            "\nUse get_prompt_guidance(model, task) for the full current master when needed. "
            "These rules already apply; that extra tool call is not a prerequisite for generation.")
        return function
    return decorate


def model_prompt_rules(model, task):
    rules = MODEL_RULES[model]
    if model == "qwen21":
        rules += "\n" + QWEN_TASK_RULES["generation" if task == "generation" else "editing"]
    return rules


def get_prompt_guidance(model, task):
    if model not in MODEL_RULES or task not in TASK_RULES:
        raise ValueError("Choose model flux2/qwen21/anima and task generation/editing/outpaint")
    if model == "anima" and task != "generation":
        raise ValueError("Anima supports generation only; use qwen21 or flux2 for this task")
    filename = QWEN_MASTER_FILES[task] if model == "qwen21" else MASTER_FILES[model]
    path = Path(MASTER_PROMPT_DIR) / filename
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read master prompt {path}. Check MASTER_PROMPT_DIR: {error}") from error
    source = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "text": raw.decode("utf-8-sig")}
    return {"model": model, "task": task, "rules": [COMMON_RULES, TASK_RULES[task], model_prompt_rules(model, task)],
            "sources": [source],
            "scope": "Image authoring only. H3 director/formatter, music/audio masters, backups and historical reviews do not define Photoshop still-image prompts. Sampling uses the active MCP presets, not unrelated master-file recipes."}


def unwrap_prompt(text):
    """Remove a single copy/paste code fence, retaining all text inside it."""
    match = re.fullmatch(r"\s*```(?:text|plaintext)?[ \t]*\r?\n(.*?)\r?\n```\s*", text, re.DOTALL)
    if match and "```" not in match[1]:
        return match[1]
    return text


def _without_score_tags(text):
    # Split only unquoted comma-separated tags; lettering is protected verbatim.
    parts = []
    start = 0
    quote = None
    escaped = False
    closers = {'"': '"', "'": "'", "“": "”", "‘": "’", "`": "`"}
    for i, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in closers and not (char in ("'", "‘") and i > 0 and text[i - 1].isalnum()):
            quote = closers[char]
        elif char == ",":
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    kept = [part for part in parts if not re.fullmatch(r"\s*score_\d+\s*", part)]
    return ",".join(kept).strip() if len(kept) != len(parts) else text


def prepare_prompts(model, checkpoint, prompt, negative_prompt):
    positive, negative = unwrap_prompt(prompt), unwrap_prompt(negative_prompt)
    adjustments = []
    if (positive, negative) != (prompt, negative_prompt):
        adjustments.append("Removed a surrounding prompt code fence")
    if model == "flux2" and negative:
        negative = ""
        adjustments.append("FLUX.2 Dev does not use negative_prompt; express desired constraints in prompt")
    if model == "anima" and not Path(checkpoint.replace("\\", "/")).name.startswith("anima-base-"):
        cleaned = _without_score_tags(positive), _without_score_tags(negative)
        if cleaned != (positive, negative):
            adjustments.append("Removed standalone score_* tags for Anima-Aesthetic/unknown checkpoint; quoted text preserved")
        positive, negative = cleaned
    return positive, negative, adjustments
