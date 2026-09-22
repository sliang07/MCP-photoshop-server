"""
MCP Photoshop Server - main entrypoint.
Registers all tools: canvas management, AI generation, editing, layers, history.
"""

import asyncio
import io
import json
import logging
import os
import secrets
import subprocess
import uuid
from pathlib import Path
from typing import Literal, Optional

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, ImageChops

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import TextContent as _TextContent

def TextContent(text="", **kwargs):
    """Wrapper that adds type='text' automatically for MCP SDK compatibility."""
    return _TextContent(type="text", text=text, **kwargs)

from config import (
    COMFYUI_URL, DEFAULT_WIDTH, DEFAULT_HEIGHT, WEBSOCKET_TIMEOUT,
    MODEL_UPSCALE_FACE, MODEL_UPSCALE_ANIME, MODEL_UPSCALE_GENERAL,
    COMFYUI_AUTO_KILL, VRAM_PRESSURE_THRESHOLD_MB,
)
from comfy_client import ComfyUIClient
from canvas import Canvas, BLEND_MODES
from session import SessionManager
from editing import (register_editing_tools, model_profiles, build_generation_workflow,
                     build_edit_workflow, prepare_image, png_bytes)
from prompt_rules import get_prompt_guidance, prepare_prompts, unwrap_prompt, with_prompt_rules
from project import load_project, save_project
from jobs import GenerationJobs

# ----------------------------------------------------------------------- #
#  Globals
# ----------------------------------------------------------------------- #

logger = logging.getLogger(__name__)

# Some clients ignore initialize instructions, so tool descriptions also state
# the automatic-startup behavior.
GPU_BATCH_RULE = """Editing: open_image first, then edit_image (Qwen Image 2.1 by default). Use the same session_id throughout. Describe the requested change and what must stay the same. Reference images follow the canvas in order. For multiple Qwen inputs use numbered tags (<image1> is the canvas); for a lone canvas use natural wording without a tag. For local edits pass region or a white-to-edit mask_path; layer visibility masks are separate. Inspect the image returned by edit_image. Undo an unsuccessful attempt before retrying so errors do not accumulate. Qwen/qwen2511 are retired; flux2 (FLUX.2 Dev NVFP4, 50 steps/guidance 4) remains an explicit option. Do not ask the user to choose a backend, seed, or steps for ordinary edits.

Layer editing: edit_image(layer_index=...) edits only that layer's pixels and preserves its settings. output_mode=extract uses Qwen 2.1 to add an extracted subject as a new layer; inspect the preview and has_transparency. Use set_layer_visibility to compare layers, export_mask to reuse a selection as an independent edit mask, and save_project/open_project for layered documents. Project opening starts a fresh undo history.

Model selection: generate_image, batch_generate and submit_generation_job accept model=flux2 (photorealistic), qwen21 (detail, typography, alpha), or anima (anime/illustration). edit_image and outpaint accept backend=qwen21 or flux2. Anima is generation-only. Choose for the user's task and omit steps/cfg to use model-specific defaults. get_editing_capabilities reports availability per task. Never claim a missing model ran, or substitute one silently. Upscaling and semantic selection use their dedicated models.

Background generation: submit_generation_job returns an ID, retains the full input list, and exports each completed image before starting the next. Poll get_job_status or list_jobs on the same server. cancel_job finishes the current image and skips remaining images without interrupting other jobs. Jobs survive an HTTP client disconnect only while the MCP server process runs; closing a stdio server or restarting loses workers/status, but exported files remain. batch_generate still waits for the whole batch before exporting.

Prompt authoring: follow the master-prompt rules in each tool's description. get_prompt_guidance exposes the current local Flux, Anima and Qwen master text without starting ComfyUI. Qwen generation uses its t2i master; editing and outpaint use its edit master. Pass rewritten_prompt text as prompt; map size metadata to supported tool arguments instead of sending the master JSON to the image model. H3/video and audio masters do not define Photoshop image prompts. Keep exact user details and lettering, avoid unnecessary interviews, and keep positive/negative prompts separate from settings.

ComfyUI starts automatically when a dependent tool is called, including get_editing_capabilities and get_comfyui_status. Call the requested tool directly; do not ask the user to start ComfyUI or open its browser UI. ComfyUI being stopped between operations is expected with idle shutdown enabled. If automatic startup actually fails, report the returned error.

GPU batching rule: this host's single 32GB GPU may be shared with the `qwen38` docker (the LLM backend itself - Ollama-compatible API on :11434) and Open WebUI (:3000). Check actual contention before proposing container changes. If the GPU is available, run the requested edit directly.
- Single generation with sufficient free GPU memory: run it directly. FLUX.2 Dev defaults to 50 steps, not the retired Klein four-step recipe.
- For batches that need GPU memory currently occupied by qwen38: first hand the complete input list to submit_generation_job on an independently running host MCP server, or to a detached host runner. Verify that the worker's process will remain running when the LLM connection ends. Sequential awaited edit_image calls are not a detached batch.
- Only then ask the user for explicit approval to run `docker stop qwen38` (and optionally `open-webui`) so ComfyUI gets the full GPU. Stop only after approval and verified background ownership of the complete batch.
- Warning: `qwen38` serves the LLM itself, so stopping it ends this session. That is acceptable only because the batch keeps running on the host; say so to the user and remind them to run `docker start qwen38 open-webui` afterwards.
- `searxng` is CPU-only; never stop it for GPU speed.
Full procedure: MEMORY.md, section "GPU Contention & Batching Rule"."""

# Transport security (DNS-rebinding protection): FastMCP auto-enables it when
# bound to loopback, but only allows 127.0.0.1 / localhost / ::1. Open WebUI
# (Docker) reaches this server via host.docker.internal, so allow that host
# explicitly. Protection stays enabled for all other hosts. (Harmless in
# stdio mode - the checks only apply to the HTTP endpoints.)
_TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", "host.docker.internal:*"],
    allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*", "http://host.docker.internal:*"],
)

app = FastMCP("mcp-photoshop-server", instructions=GPU_BATCH_RULE, transport_security=_TRANSPORT_SECURITY)
comfy = ComfyUIClient(COMFYUI_URL)
sessions = SessionManager()


async def free_or_kill_based_on_pressure(comfy_client: ComfyUIClient) -> None:
    """Decide whether to free memory or kill ComfyUI based on VRAM pressure.

    If free VRAM is below VRAM_PRESSURE_THRESHOLD_MB, kill ComfyUI entirely to
    release all VRAM (CUDA context, allocator pools, model weights) so vLLM
    gets full access. Otherwise, just free_memory() to unload models while
    keeping the process warm for fast subsequent calls.

    Falls back to nvidia-smi if ComfyUI's /system_stats doesn't provide VRAM info.
    """
    threshold_bytes = VRAM_PRESSURE_THRESHOLD_MB * 1024 * 1024

    # Try ComfyUI system_stats first
    free_vram_bytes = None
    try:
        stats = await comfy_client.get_system_stats()
        # ComfyUI may report vram_free in different key paths depending on version
        system = stats.get("system", {})
        free_vram_bytes = system.get("vram_free") or system.get("gpu_free")
    except Exception:
        pass

    # Fallback: read from nvidia-smi
    if free_vram_bytes is None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            # Take the first GPU's free memory
            free_mb = float(result.stdout.strip().split("\n")[0].strip())
            free_vram_bytes = free_mb * 1024 * 1024
        except Exception:
            # Can't determine VRAM — default to free_memory (safe fallback)
            logger.debug("Could not read VRAM stats, falling back to free_memory()...")
            await comfy_client.free_memory()
            return

    free_mb_display = free_vram_bytes / 1024 / 1024
    if free_vram_bytes < threshold_bytes:
        logger.warning("Low VRAM (%.0f MB free < %d MB threshold), killing ComfyUI...", free_mb_display, VRAM_PRESSURE_THRESHOLD_MB)
        await comfy_client.kill_comfyui()
    else:
        logger.info("Sufficient VRAM (%.0f MB free >= %d MB threshold), calling free_memory()...", free_mb_display, VRAM_PRESSURE_THRESHOLD_MB)
        await comfy_client.free_memory()


# ======================================================================= #
#  Workflow builders
# ======================================================================= #

def _make_node_id() -> str:
    return str(uuid.uuid4())[:8]


def build_upscale_workflow(
    image_filename: str, upscale_model: str = "4x_NMKD-Siax_200k.pth", scale: int = 2,
) -> dict:
    """Build an upscale workflow using ImageScaleBy + UpscaleModel."""
    nid = _make_node_id
    n1 = nid()  # LoadImage
    n2 = nid()  # UpscaleModelLoader
    n3 = nid()  # ImageUpscaleWithModel
    n4 = nid()  # SaveImage

    return {
        n1: {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        n2: {"class_type": "UpscaleModelLoader", "inputs": {"model_name": upscale_model}},
        n3: {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": [n2, 0], "image": [n1, 0]}},
        n4: {"class_type": "SaveImage", "inputs": {"images": [n3, 0], "filename_prefix": "mcp_upscale"}},
    }


# ======================================================================= #
#  Helper: run workflow and return result image bytes
# ======================================================================= #

async def run_workflow(workflow: dict, output_type: str = "images", timeout: Optional[int] = None) -> Optional[bytes]:
    """Submit workflow, wait for completion, return result bytes.

    Uses VRAM pressure management: after each generation, calls
    free_or_kill_based_on_pressure() to either free memory or kill ComfyUI
    depending on VRAM pressure from vLLM.

    comfy.run_workflow_and_wait() pre-fetches output bytes into
    history["_cached_file_bytes"] BEFORE auto-killing ComfyUI (when
    COMFYUI_AUTO_KILL is on), because /view is unreachable once the
    process is dead. Prefer that cache; only fetch/free manually if
    auto-kill is off and ComfyUI is still alive.

    timeout: optional override for WEBSOCKET_TIMEOUT.
    """
    history = await comfy.run_workflow_and_wait(workflow, timeout=timeout)
    if not history:
        return None

    cached = history.get("_cached_file_bytes")
    if cached is not None:
        # Auto-kill path: ComfyUI is already killed, nothing to clean up
        return cached

    # Fallback path: auto-kill disabled, ComfyUI still running
    files = ComfyUIClient.get_output_files_from_history(history, file_type=output_type)
    if not files:
        return None
    first = files[0]
    try:
        return await comfy.get_output_file(
            first.get("filename", ""), subfolder=first.get("subfolder", ""), output_dir=first.get("type", "output"),
        )
    finally:
        # VRAM pressure management: only free memory, never kill during generation
        if not COMFYUI_AUTO_KILL:
            try:
                await comfy.free_memory()
            except Exception as e:
                logger.warning("Cleanup failed: %s", e)


# ======================================================================= #
#  MCP Tool Registrations
# ======================================================================= #

@app.tool("new_canvas")
async def new_canvas(width: int = 1024, height: int = 1024, bg_color: str = "white", session_id: str = "default"):
    """Create a new blank canvas."""
    try:
        img = Image.new("RGB", (1, 1), bg_color)
        color = img.getpixel((0, 0))
        canvas = sessions.create(session_id, width, height, color)
        return [TextContent(text=json.dumps(canvas.get_info()))]
    except Exception as e:
        return [TextContent(text=f"Error creating canvas: {str(e)}")]


@app.tool("open_image")
async def open_image(path: str, session_id: str = "default"):
    """Load an existing image as the active document."""
    try:
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        canvas = sessions.create(session_id, w, h)
        canvas.layers.clear()
        canvas.undo_stack.clear()
        canvas.redo_stack.clear()
        canvas.add_layer(name="Layer 0", image=img)
        return [TextContent(text=json.dumps(canvas.get_info()))]
    except Exception as e:
        return [TextContent(text=f"Error opening image: {str(e)}")]


@app.tool("save_project")
async def save_project_tool(path: str, session_id: str = "default"):
    """Save the current document as a ZIP-based layered project (.mcpproj). Preserves layers/masks/settings. Saving replaces the supplied path atomically. This is not PSD."""
    try:
        canvas = sessions.get(session_id)
        if canvas is None:
            raise ValueError(f"Session '{session_id}' does not exist.")
        save_project(canvas, path)
        resolved_path = os.path.abspath(path)
        info = {
            "path": resolved_path,
            "width": canvas.width,
            "height": canvas.height,
            "layer_count": len(canvas.layers)
        }
        return [TextContent(text=json.dumps(info))]
    except Exception as e:
        return [TextContent(text=f"Error saving project: {str(e)}")]


@app.tool("open_project")
async def open_project_tool(path: str, session_id: str = "default"):
    """Open a ZIP-based layered project (.mcpproj). Replaces the chosen document and starts fresh history. This is not PSD."""
    try:
        canvas = load_project(path)
        sessions.replace(session_id, canvas)
        return [TextContent(text=json.dumps(canvas.get_info()))]
    except Exception as e:
        return [TextContent(text=f"Error opening project: {str(e)}")]


@app.tool("export")
async def export(path: Optional[str] = None, format: str = "PNG", quality: int = 95, session_id: str = "default"):
    """Export the current canvas to a file."""
    try:
        canvas = sessions.get_or_create(session_id)
        fmt = "JPEG" if format.upper() == "JPG" else format.upper()
        img = canvas.composite_rgb() if fmt == "JPEG" else canvas.composite()
        if not path:
            path = f"export_{uuid.uuid4().hex[:8]}.{format.lower()}"
        img.save(path, fmt, quality=quality)
        return [TextContent(text=f"Exported to {path} ({fmt})")]
    except Exception as e:
        return [TextContent(text=f"Error exporting: {str(e)}")]


@app.tool("get_info")
async def get_info(session_id: str = "default"):
    """Get current canvas dimensions, layer count, and session info."""
    try:
        canvas = sessions.get_or_create(session_id)
        return [TextContent(text=json.dumps(canvas.get_info(), indent=2))]
    except Exception as e:
        return [TextContent(text=f"Error: {str(e)}")]


@app.tool("get_prompt_guidance")
async def get_prompt_guidance_tool(model: Literal["flux2", "qwen21", "anima"] = "flux2",
                                   task: Literal["generation", "editing", "outpaint"] = "generation"):
    """Read the applicable local master prompt and its Photoshop tool adaptation.

    No ComfyUI startup or GPU use. Returns the current full master with path and hash.
    Qwen generation selects the t2i master; editing/outpaint select the edit master.
    Standalone code-block/JSON output instructions become raw MCP prompt arguments;
    size metadata maps to supported tool arguments, not text sent to the image model.
    Read when needed; the essential rules are already included in the image tools.
    """
    return [TextContent(text=json.dumps(get_prompt_guidance(model, task), ensure_ascii=False, indent=2))]


@app.tool("generate_image")
@with_prompt_rules("generation", ("flux2", "qwen21", "anima"))
async def generate_image(prompt: str, model: Literal["flux2", "qwen21", "anima"] = "flux2", width: int = 1024, height: int = 1024,
                         steps: Optional[int] = None, cfg: Optional[float] = None, seed: Optional[int] = None, negative_prompt: str = "",
                         session_id: str = "default", timeout: Optional[int] = None):
    """Generate an image from text. Automatically starts ComfyUI; no manual startup is required.

    Choose model='flux2' for generation (FLUX.2 Dev NVFP4, 50 steps/guidance 4, guidance-distilled),
    'qwen21' for detail, typography or transparent output (custom 30 steps/CFG 3),
    or 'anima' for anime/illustration (30 steps/CFG 4). Omit steps/cfg for these defaults.
    For flux2, cfg controls embedded FluxGuidance; negative_prompt is unused and reported
    if supplied. Express desired constraints positively in prompt.
    get_editing_capabilities reports installed choices. For changes to an existing
    image use edit_image instead. A missing model is reported, never silently substituted.
    """
    try:
        if model not in ("flux2", "qwen21", "anima"):
            raise ValueError("model must be flux2, qwen21 or anima")
        await comfy.start_comfyui()
        profile = model_profiles(await comfy.get_object_info(), "generation")[model]
        if not profile["available"]:
            raise ValueError(f"{model} is unavailable: {profile['missing']}. Call get_editing_capabilities.")
        prompt, negative_prompt, adjustments = prepare_prompts(model, profile["model"], prompt, negative_prompt)
        workflow = build_generation_workflow(model, profile, prompt, negative_prompt,
                                             width, height, steps, cfg, seed)
        result = await run_workflow(workflow, timeout=timeout or (1800 if model in ("flux2", "qwen21") else None))
        if not result:
            return [TextContent(text="Generation failed.")]
        img = Image.open(io.BytesIO(result)).convert("RGBA")
        canvas = sessions.get(session_id)
        if canvas is None or canvas.width != img.width or canvas.height != img.height:
            canvas = sessions.create(session_id, img.width, img.height, (0, 0, 0, 0))
        idx = canvas.add_layer(name=f"Generated: {prompt[:30]}", image=img)
        content = [TextContent(text=f"Generated image ({img.width}x{img.height}) as layer {idx}. Model: {model}")]
        if adjustments:
            content.append(TextContent(text=json.dumps({"prompt_adjustments": adjustments,
                           "effective_prompt": prompt, "negative_prompt": negative_prompt}, ensure_ascii=False)))
        return content
    except Exception as e:
        return [TextContent(text=f"Generation error: {str(e)}")]


@app.tool("crop")
async def crop_tool(x: int, y: int, width: int, height: int, session_id: str = "default"):
    """Crop the entire document (all images and masks) to the specified region."""
    try:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        canvas = sessions.get_or_create(session_id)
        box = (x, y, x + width, y + height)
        results = []
        for layer in canvas.layers:
            new_image = layer.image.crop(box)
            new_mask = layer.mask.crop(box) if layer.mask else None
            results.append((layer, new_image, new_mask))
        for layer, new_image, new_mask in results:
            layer.image = new_image
            layer.mask = new_mask
        canvas.width = width
        canvas.height = height
        canvas._save_state()
        return [TextContent(text=f"Cropped to {width}x{height} at ({x},{y})")]
    except Exception as e:
        return [TextContent(text=f"Crop error: {str(e)}")]


@app.tool("resize")
async def resize_tool(width: int, height: int, maintain_aspect: bool = False, session_id: str = "default"):
    """Resize the entire document (all images and masks) to the requested canvas size.
    When maintain_aspect=True, scales all layers using one scale based on original canvas
    dimensions and centers fitted content in a transparent target image and black target mask."""
    try:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        canvas = sessions.get_or_create(session_id)
        orig_w, orig_h = canvas.width, canvas.height
        if maintain_aspect:
            scale = min(width / orig_w, height / orig_h)
            fitted_w = max(1, int(orig_w * scale))
            fitted_h = max(1, int(orig_h * scale))
            offset_x = (width - fitted_w) // 2
            offset_y = (height - fitted_h) // 2
        else:
            fitted_w, fitted_h = width, height
            offset_x, offset_y = 0, 0
        results = []
        for layer in canvas.layers:
            new_image = layer.image.resize((fitted_w, fitted_h), Image.LANCZOS)
            if maintain_aspect:
                target = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                target.paste(new_image, (offset_x, offset_y))
                new_image = target
            new_mask = None
            if layer.mask:
                mask_resized = layer.mask.resize((fitted_w, fitted_h), Image.LANCZOS)
                if maintain_aspect:
                    target_mask = Image.new("L", (width, height), 0)
                    target_mask.paste(mask_resized, (offset_x, offset_y))
                    new_mask = target_mask
                else:
                    new_mask = mask_resized
            results.append((layer, new_image, new_mask))
        for layer, new_image, new_mask in results:
            layer.image = new_image
            layer.mask = new_mask
        canvas.width = width
        canvas.height = height
        canvas._save_state()
        return [TextContent(text=f"Resized to {width}x{height}")]
    except Exception as e:
        return [TextContent(text=f"Resize error: {str(e)}")]


@app.tool("rotate")
async def rotate_tool(degrees: float, expand: bool = True, bg_color: str = "transparent", session_id: str = "default"):
    """Rotate the active layer counter-clockwise.
    expand=False keeps original dimensions. expand=True expands canvas to fit the rotated
    active layer, centering it and padding other layers by the original canvas offset."""
    try:
        canvas = sessions.get_or_create(session_id)
        active_idx = canvas.active_layer_index
        active_layer = canvas.layers[active_idx]
        fill_color = (0, 0, 0, 0) if bg_color == "transparent" else bg_color
        rotated_image = active_layer.image.rotate(degrees, expand=expand, fillcolor=fill_color, resample=Image.BICUBIC)
        rotated_mask = None
        if active_layer.mask:
            rotated_mask = active_layer.mask.rotate(degrees, expand=expand, fillcolor=0, resample=Image.BICUBIC)
        if expand:
            new_w = max(canvas.width, rotated_image.width)
            new_h = max(canvas.height, rotated_image.height)
            rot_off_x = (new_w - rotated_image.width) // 2
            rot_off_y = (new_h - rotated_image.height) // 2
            orig_off_x = (new_w - canvas.width) // 2
            orig_off_y = (new_h - canvas.height) // 2
        else:
            new_w, new_h = canvas.width, canvas.height
            rot_off_x = rot_off_y = orig_off_x = orig_off_y = 0
        results = []
        for i, layer in enumerate(canvas.layers):
            if i == active_idx:
                img, mask = rotated_image, rotated_mask
                off_x, off_y = rot_off_x, rot_off_y
            else:
                img, mask = layer.image, layer.mask
                off_x, off_y = orig_off_x, orig_off_y
            if expand:
                new_image = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
                new_image.paste(img, (off_x, off_y))
                new_mask = None
                if mask:
                    new_mask = Image.new("L", (new_w, new_h), 0)
                    new_mask.paste(mask, (off_x, off_y))
            else:
                new_image = img
                new_mask = mask
            results.append((layer, new_image, new_mask))
        for layer, new_image, new_mask in results:
            layer.image = new_image
            layer.mask = new_mask
        canvas.width = new_w
        canvas.height = new_h
        canvas._save_state()
        return [TextContent(text=f"Rotated {degrees} degrees")]
    except Exception as e:
        return [TextContent(text=f"Rotate error: {str(e)}")]


@app.tool("flip")
async def flip_tool(axis: str = "horizontal", session_id: str = "default"):
    """Flip the active layer and its mask. axis: 'horizontal' or 'vertical'."""
    try:
        canvas = sessions.get_or_create(session_id)
        layer = canvas.layers[canvas.active_layer_index]
        if axis == "horizontal":
            new_image = ImageOps.mirror(layer.image)
            new_mask = ImageOps.mirror(layer.mask) if layer.mask else None
        else:
            new_image = ImageOps.flip(layer.image)
            new_mask = ImageOps.flip(layer.mask) if layer.mask else None
        layer.image = new_image
        layer.mask = new_mask
        canvas._save_state()
        return [TextContent(text=f"Flipped {axis}")]
    except Exception as e:
        return [TextContent(text=f"Flip error: {str(e)}")]


@app.tool("adjust")
async def adjust_tool(brightness: Optional[float] = None, contrast: Optional[float] = None,
                      saturation: Optional[float] = None, hue: Optional[float] = None, sharpness: Optional[float] = None,
                      session_id: str = "default"):
    """Adjust colors. Each parameter is a multiplier (1.0=no change). hue is in degrees (0-360)."""
    try:
        canvas = sessions.get_or_create(session_id)
        layer = canvas.layers[canvas.active_layer_index]
        img = layer.image
        if brightness is not None:
            img = ImageEnhance.Brightness(img).enhance(brightness)
        if contrast is not None:
            img = ImageEnhance.Contrast(img).enhance(contrast)
        if saturation is not None:
            img = ImageEnhance.Color(img).enhance(saturation)
        if sharpness is not None:
            img = ImageEnhance.Sharpness(img).enhance(sharpness)
        if hue is not None and hue % 360 != 0:
            alpha = img.getchannel("A")
            h, s, v = img.convert("HSV").split()
            offset = round((hue % 360) / 360.0 * 255) % 256
            h = h.point(lambda p: (p + offset) % 256)
            img = Image.merge("HSV", (h, s, v)).convert("RGBA")
            img.putalpha(alpha)
        layer.image = img
        canvas._save_state()
        return [TextContent(text="Adjustments applied.")]
    except Exception as e:
        return [TextContent(text=f"Adjust error: {str(e)}")]


def _pixelate(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    tw = max(1, w // size)
    th = max(1, h // size)
    tiny = img.resize((tw, th), Image.LANCZOS)
    return tiny.resize((w, h), Image.NEAREST)


def _apply_sepia(img: Image.Image) -> Image.Image:
    """Apply sepia tone while preserving alpha channel."""
    if img.mode == "RGBA":
        r, g, b, a = img.split()
        rgb_img = Image.merge("RGB", (r, g, b))
        alpha = a
    else:
        rgb_img = img.convert("RGB")
        alpha = None
    data = list(rgb_img.getdata())
    new_data = []
    for r, g, b in data:
        tr = min(255, int(0.393 * r + 0.769 * g + 0.189 * b))
        tg = min(255, int(0.349 * r + 0.686 * g + 0.168 * b))
        tb = min(255, int(0.272 * r + 0.534 * g + 0.131 * b))
        new_data.append((tr, tg, tb))
    rgb_img.putdata(new_data)
    if alpha:
        return Image.merge("RGBA", (rgb_img.split() + (alpha,)))
    return rgb_img.convert("RGBA")


@app.tool("apply_filter")
async def apply_filter(name: str, session_id: str = "default", *, radius: float = 2.0, size: int = 8, levels: int = 4):
    """Apply a filter: blur, gaussian_blur, sharpen, contour, detail, edge_enhance, edge_enhance_more, find_edges, smooth, smooth_more, emboss, pixelate, posterize, solarize, invert, grayscale, sepia. Params: radius (gaussian_blur), size (pixelate), levels (posterize)."""
    try:
        canvas = sessions.get_or_create(session_id)
        layer = canvas.layers[canvas.active_layer_index]
        img = layer.image
        alpha = img.getchannel("A")
        filters = {
            "blur": lambda: img.filter(ImageFilter.BLUR),
            "gaussian_blur": lambda: img.filter(ImageFilter.GaussianBlur(radius)),
            "sharpen": lambda: img.filter(ImageFilter.SHARPEN),
            "contour": lambda: img.filter(ImageFilter.CONTOUR),
            "detail": lambda: img.filter(ImageFilter.DETAIL),
            "edge_enhance": lambda: img.filter(ImageFilter.EDGE_ENHANCE),
            "edge_enhance_more": lambda: img.filter(ImageFilter.EDGE_ENHANCE_MORE),
            "find_edges": lambda: img.filter(ImageFilter.FIND_EDGES),
            "smooth": lambda: img.filter(ImageFilter.SMOOTH),
            "smooth_more": lambda: img.filter(ImageFilter.SMOOTH_MORE),
            "emboss": lambda: img.filter(ImageFilter.EMBOSS),
            "pixelate": lambda: _pixelate(img, size),
            "posterize": lambda: ImageOps.posterize(img.convert("RGB"), levels).convert("RGBA"),
            "solarize": lambda: ImageOps.solarize(img.convert("RGB")).convert("RGBA"),
            "invert": lambda: ImageOps.invert(img.convert("RGB")).convert("RGBA"),
            "grayscale": lambda: img.convert("L").convert("RGBA"),
            "sepia": lambda: _apply_sepia(img),
        }
        if name not in filters:
            return [TextContent(text=f"Unknown filter: {name}. Available: {', '.join(filters.keys())}")]
        result = filters[name]()
        if name in ("invert", "grayscale", "posterize", "solarize"):
            result.putalpha(alpha)
        layer.image = result
        canvas._save_state()
        return [TextContent(text=f"Filter '{name}' applied.")]
    except Exception as e:
        return [TextContent(text=f"Filter error: {str(e)}")]


@app.tool("add_text")
async def add_text(text: str, x: int = 50, y: int = 50, font_size: int = 48, color: str = "white",
                   font: Optional[str] = None, stroke_width: int = 0, stroke_color: str = "black",
                   session_id: str = "default"):
    """Add text overlay to the active layer. Preserve the user's lettering exactly,
    including case, punctuation and language; do not paraphrase or add prompt tags.
    The default font has limited glyph coverage; supply a font path for other scripts.
    """
    try:
        canvas = sessions.get_or_create(session_id)
        layer = canvas.layers[canvas.active_layer_index]
        img = layer.image.copy()
        draw = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype(font, font_size) if font and os.path.exists(font) else ImageFont.load_default(size=font_size)
        except OSError:
            f = ImageFont.load_default(size=font_size)
        if stroke_width > 0:
            draw.text((x, y), text, font=f, fill=stroke_color, stroke_width=stroke_width, stroke_fill=stroke_color)
        draw.text((x, y), text, font=f, fill=color)
        layer.image = img
        canvas._save_state()
        return [TextContent(text=f"Text added at ({x},{y})")]
    except Exception as e:
        return [TextContent(text=f"Text error: {str(e)}")]


@app.tool("add_layer")
async def add_layer_tool(name: Optional[str] = None, source_path: Optional[str] = None,
                         opacity: float = 1.0, blend_mode: str = "normal",
                         session_id: str = "default"):
    """Add a new layer. Optionally load from source_path."""
    try:
        canvas = sessions.get_or_create(session_id)
        img = Image.open(source_path).convert("RGBA") if source_path else None
        layer_name = name or f"Layer {len(canvas.layers)}"
        idx = canvas.add_layer(name=layer_name, image=img, opacity=opacity, blend_mode=blend_mode)
        return [TextContent(text=f"Added '{layer_name}' as layer {idx}")]
    except Exception as e:
        return [TextContent(text=f"Add layer error: {str(e)}")]


@app.tool("select_layer")
async def select_layer_tool(index: int, session_id: str = "default"):
    """Select a layer by index (0 = bottom)."""
    try:
        canvas = sessions.get_or_create(session_id)
        if canvas.select_layer(index):
            return [TextContent(text=f"Selected layer {index}: {canvas.layers[index].name}")]
        return [TextContent(text=f"Invalid layer index: {index}")]
    except Exception as e:
        return [TextContent(text=f"Select layer error: {str(e)}")]


@app.tool("set_blend_mode")
async def set_blend_mode_tool(mode: str, index: Optional[int] = None, session_id: str = "default"):
    """Set blend mode: normal, multiply, screen, overlay, darken, lighten, color_dodge, color_burn, hard_light, soft_light, difference, exclusion."""
    try:
        canvas = sessions.get_or_create(session_id)
        if canvas.set_blend_mode(mode, index):
            return [TextContent(text=f"Blend mode set to '{mode}'")]
        return [TextContent(text=f"Invalid blend mode: {mode}")]
    except Exception as e:
        return [TextContent(text=f"Blend mode error: {str(e)}")]


@app.tool("set_layer_opacity")
async def set_layer_opacity_tool(opacity: float, index: Optional[int] = None, session_id: str = "default"):
    """Set layer opacity (0.0=transparent, 1.0=opaque)."""
    try:
        canvas = sessions.get_or_create(session_id)
        if canvas.set_layer_opacity(index, opacity):
            return [TextContent(text=f"Opacity set to {opacity}")]
        return [TextContent(text="Invalid layer index")]
    except Exception as e:
        return [TextContent(text=f"Opacity error: {str(e)}")]


@app.tool("set_layer_visibility")
async def set_layer_visibility_tool(visible: bool, index: Optional[int] = None, session_id: str = "default"):
    """Show or hide a layer, including originals hidden by an AI edit."""
    try:
        canvas = sessions.get_or_create(session_id)
        if canvas.set_layer_visibility(visible, index):
            return [TextContent(text=f"Layer visibility set to {visible}")]
        return [TextContent(text="Invalid layer index")]
    except Exception as e:
        return [TextContent(text=f"Visibility error: {str(e)}")]


@app.tool("rename_layer")
async def rename_layer_tool(name: str, index: Optional[int] = None, session_id: str = "default"):
    """Rename a layer."""
    try:
        canvas = sessions.get_or_create(session_id)
        if canvas.rename_layer(name, index):
            return [TextContent(text=f"Layer renamed to '{name}'")]
        return [TextContent(text="Invalid layer index")]
    except Exception as e:
        return [TextContent(text=f"Rename error: {str(e)}")]


@app.tool("duplicate_layer")
async def duplicate_layer_tool(index: Optional[int] = None, name: Optional[str] = None, session_id: str = "default"):
    """Insert an independent copy above a layer, preserve its settings, and select the copy."""
    try:
        canvas = sessions.get_or_create(session_id)
        idx = canvas.duplicate_layer(index, name)
        if idx is not None:
            return [TextContent(text=f"Duplicated layer at index {idx}")]
        return [TextContent(text="Invalid layer index")]
    except Exception as e:
        return [TextContent(text=f"Duplicate error: {str(e)}")]


@app.tool("translate_layer")
async def translate_layer_tool(dx: int, dy: int, index: Optional[int] = None, session_id: str = "default"):
    """Move layer pixels and mask by relative (dx, dy). Clips overflow; undo restores it."""
    try:
        canvas = sessions.get_or_create(session_id)
        if canvas.translate_layer(dx, dy, index):
            return [TextContent(text=f"Layer translated by ({dx}, {dy})")]
        return [TextContent(text="Invalid layer index")]
    except Exception as e:
        return [TextContent(text=f"Translate error: {str(e)}")]


@app.tool("merge_down")
async def merge_down_tool(session_id: str = "default"):
    """Merge active layer into the layer below."""
    try:
        canvas = sessions.get_or_create(session_id)
        if canvas.merge_down():
            return [TextContent(text="Merged down.")]
        return [TextContent(text="Cannot merge: already on bottom layer.")]
    except Exception as e:
        return [TextContent(text=f"Merge error: {str(e)}")]


@app.tool("delete_layer")
async def delete_layer_tool(index: Optional[int] = None, session_id: str = "default"):
    """Delete a layer by index or the active layer."""
    try:
        canvas = sessions.get_or_create(session_id)
        if canvas.delete_layer(index):
            return [TextContent(text="Layer deleted.")]
        return [TextContent(text="Cannot delete layer.")]
    except Exception as e:
        return [TextContent(text=f"Delete error: {str(e)}")]


@app.tool("reorder_layer")
async def reorder_layer_tool(index: Optional[int] = None, direction: str = "up", session_id: str = "default"):
    """Move layer up or down in the stack."""
    try:
        canvas = sessions.get_or_create(session_id)
        idx = index if index is not None else canvas.active_layer_index
        if canvas.reorder_layer(idx, direction):
            return [TextContent(text=f"Layer moved {direction}.")]
        return [TextContent(text="Cannot move: already at edge.")]
    except Exception as e:
        return [TextContent(text=f"Reorder error: {str(e)}")]


@app.tool("select_rect")
async def select_rect(x: int, y: int, width: int, height: int, session_id: str = "default"):
    """Create a rectangular selection mask on the active layer."""
    try:
        canvas = sessions.get_or_create(session_id)
        layer = canvas.layers[canvas.active_layer_index]
        mask = Image.new("L", layer.image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x, y, x + width, y + height], fill=255)
        layer.mask = mask
        canvas._save_state()
        return [TextContent(text=f"Rectangular mask at ({x},{y}) {width}x{height}")]
    except Exception as e:
        return [TextContent(text=f"Select error: {str(e)}")]


@app.tool("select_ellipse")
async def select_ellipse(x: int, y: int, rx: int, ry: int, session_id: str = "default"):
    """Create an elliptical selection mask centered at (x,y)."""
    try:
        canvas = sessions.get_or_create(session_id)
        layer = canvas.layers[canvas.active_layer_index]
        mask = Image.new("L", layer.image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=255)
        layer.mask = mask
        canvas._save_state()
        return [TextContent(text=f"Elliptical mask at ({x},{y}) radii ({rx},{ry})")]
    except Exception as e:
        return [TextContent(text=f"Select error: {str(e)}")]


@app.tool("clear_mask")
async def clear_mask_tool(index: Optional[int] = None, session_id: str = "default"):
    """Remove the mask from the active layer."""
    try:
        canvas = sessions.get_or_create(session_id)
        idx = index if index is not None else canvas.active_layer_index
        if 0 <= idx < len(canvas.layers):
            canvas.layers[idx].mask = None
            canvas._save_state()
            return [TextContent(text="Mask cleared.")]
        return [TextContent(text="Invalid layer index.")]
    except Exception as e:
        return [TextContent(text=f"Clear mask error: {str(e)}")]


@app.tool("undo")
async def undo_tool(session_id: str = "default"):
    """Undo the last operation."""
    try:
        canvas = sessions.get_or_create(session_id)
        if canvas.undo():
            return [TextContent(text="Undone.")]
        return [TextContent(text="Nothing to undo.")]
    except Exception as e:
        return [TextContent(text=f"Undo error: {str(e)}")]


@app.tool("redo")
async def redo_tool(session_id: str = "default"):
    """Redo the last undone operation."""
    try:
        canvas = sessions.get_or_create(session_id)
        if canvas.redo():
            return [TextContent(text="Redone.")]
        return [TextContent(text="Nothing to redo.")]
    except Exception as e:
        return [TextContent(text=f"Redo error: {str(e)}")]


@app.tool("get_comfyui_status")
async def get_comfyui_status_tool(start_if_needed: bool = True):
    """Check ComfyUI connection and system info. Automatically starts ComfyUI when needed.

    No manual startup is required. Set start_if_needed=False for a passive check.
    A stopped ComfyUI is normal between operations; dependent tools start it automatically.
    """
    try:
        if start_if_needed:
            await comfy.start_comfyui()
        stats = await comfy.get_system_stats()
        return [TextContent(text=json.dumps(stats, indent=2))]
    except Exception as e:
        message = ("Automatic startup or status check failed; inspect the error and ComfyUI startup configuration."
                   if start_if_needed else
                   "ComfyUI is stopped or unreachable. Call this tool with start_if_needed=True, or call the requested editing/generation tool; it automatically starts ComfyUI.")
        return [TextContent(text=json.dumps({"connected": False, "auto_start_attempted": start_if_needed,
                                           "error": str(e), "message": message}))]


@app.tool("clear_vram")
async def clear_vram_tool():
    """Free GPU VRAM by unloading cached models."""
    try:
        await comfy.free_memory()
        return [TextContent(text="VRAM cleared.")]
    except Exception as e:
        return [TextContent(text=f"Clear VRAM error: {str(e)}")]


# ---- Outpaint ----


@app.tool("outpaint")
@with_prompt_rules("outpaint", ("flux2", "qwen21"))
async def outpaint_tool(prompt: str, direction: str = "right", amount: int = 256,
                        steps: Optional[int] = None, seed: Optional[int] = None,
                        session_id: str = "default", backend: Literal["flux2", "qwen21"] = "flux2", timeout: Optional[int] = None):
    """AI outpaint: extend the canvas in a direction and fill the new area. Automatically starts ComfyUI; no manual startup is required.

    direction: left, right, top, bottom. backend: flux2 (FLUX.2 Dev NVFP4, 50 steps/guidance 4)
    or qwen21 (Qwen Image 2.1, custom 30 steps/CFG 3). Anima does not support instruction editing.
    Omit steps for the selected model's default. Extends the active layer using
    reference conditioning, then restores its original pixels exactly.
    Qwen works at about 1 megapixel internally, then returns the requested canvas size.
    """
    try:
        prompt = unwrap_prompt(prompt)
        if backend not in ("flux2", "qwen21"):
            raise ValueError("backend must be flux2 or qwen21; Anima supports generation only")
        if amount <= 0:
            raise ValueError("amount must be positive")
        canvas = sessions.get_or_create(session_id)
        original_state = canvas.undo_stack[-1]
        layer = canvas.layers[canvas.active_layer_index]
        w, h = layer.image.size
        directions = {"left": (w + amount, h), "right": (w + amount, h),
                      "top": (w, h + amount), "bottom": (w, h + amount)}
        if direction not in directions:
            return [TextContent(text=f"Invalid direction: {direction}. Use: left, right, top, bottom")]
        target_w, target_h = directions[direction]
        await comfy.start_comfyui()
        profile = model_profiles(await comfy.get_object_info(), "outpaint")[backend]
        if not profile["available"]:
            raise ValueError(f"{backend} is unavailable: {profile['missing']}. Call get_editing_capabilities.")
        steps = profile["default_steps"] if steps is None else steps
        seed = secrets.randbits(63) if seed is None else seed
        # Create padded canvas
        padded = Image.new("RGBA", (target_w, target_h), "white" if backend == "qwen21" else (0, 0, 0, 0))
        if direction == "left":
            padded.paste(layer.image, (amount, 0))
        elif direction == "right":
            padded.paste(layer.image, (0, 0))
        elif direction == "top":
            padded.paste(layer.image, (0, amount))
        else:
            padded.paste(layer.image, (0, 0))
        # Upload padded image
        pad_name = f"outpaint_{uuid.uuid4().hex[:8]}.png"
        working = prepare_image(padded, max(padded.size), 32 if backend == "qwen21" else 16)
        filename = await comfy.upload_image(png_bytes(working), pad_name)
        if backend == "qwen21":
            instruction = (f"Outpaint the image: replace the white strip on the {direction} with "
                           f"a seamless continuation of the scene: {prompt}. "
                           "Preserve the original image's placement, scale, content and style.")
        else:
            instruction = (f"Extend the scene in image 1 into the transparent {direction} margin. "
                           f"Fill that margin with opaque image content: {prompt}. "
                           "Keep the original image's placement, scale, content and style unchanged. "
                           "Continue the scene seamlessly across the boundary.")
        workflow = build_edit_workflow(backend, profile, instruction, [filename],
                                       *working.size, steps, seed, resolution=1024 if backend == "qwen21" else 0)
        result = await run_workflow(workflow, timeout=timeout or 1800)
        if not result:
            return [TextContent(text="Outpaint failed.")]
        img = Image.open(io.BytesIO(result)).convert("RGBA")
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        if sessions.get(session_id) is not canvas or canvas.undo_stack[-1] is not original_state:
            raise RuntimeError("Canvas changed during generation; outpaint result was not applied")
        # Extend every other layer (and its mask) to the new canvas size so the
        # composite stays consistent; their new area is transparent.
        if direction == "left":
            offset = (amount, 0)
        elif direction == "top":
            offset = (0, amount)
        else:
            offset = (0, 0)
        img.paste(layer.image, offset)
        for existing in canvas.layers:
            if existing is layer:
                continue
            if existing.image.size != (target_w, target_h):
                extended = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
                extended.paste(existing.image, offset)
                existing.image = extended
            if existing.mask is not None and existing.mask.size != (target_w, target_h):
                extended_mask = Image.new("L", (target_w, target_h), 0)
                extended_mask.paste(existing.mask, offset)
                existing.mask = extended_mask
        if layer.mask is not None and layer.mask.size != (target_w, target_h):
            extended_mask = Image.new("L", (target_w, target_h), 255)
            extended_mask.paste(layer.mask, offset)
            layer.mask = extended_mask
        layer.image = img
        canvas.width = target_w
        canvas.height = target_h
        canvas._save_state()
        return [TextContent(text=f"Outpaint {direction} complete ({target_w}x{target_h}). Model: {backend}; steps: {steps}; seed: {seed}.")]
    except Exception as e:
        return [TextContent(text=f"Outpaint error: {str(e)}")]


# ---- Levels / Curves ----

@app.tool("levels")
async def levels_tool(black_point: int = 0, mid_point: float = 1.0, white_point: int = 255, session_id: str = "default"):
    """Adjust levels. black_point: 0-255 (input black), mid_point: 0.1-2.0 (gamma), white_point: 0-255 (input white)."""
    try:
        canvas = sessions.get_or_create(session_id)
        layer = canvas.layers[canvas.active_layer_index]
        img = layer.image.convert("RGB")
        # Build 256-point lookup table
        range_wp = max(1, white_point - black_point)
        gamma = 1.0 / max(0.01, mid_point)
        table = [max(0, min(255, int(255.0 * (max(0.0, (i - black_point) / range_wp) ** gamma)))) for i in range(256)]
        r, g, b = img.split()
        r = r.point(table)
        g = g.point(table)
        b = b.point(table)
        layer.image = Image.merge("RGB", (r, g, b)).convert("RGBA")
        canvas._save_state()
        return [TextContent(text="Levels adjusted.")]
    except Exception as e:
        return [TextContent(text=f"Levels error: {str(e)}")]


@app.tool("curves")
async def curves_tool(red: str = "", green: str = "", blue: str = "", session_id: str = "default"):
    """Adjust curves. Each channel is a comma-separated list of (input,output) pairs, e.g. '0,0 128,140 255,255'."""
    def _parse_curve(s: str) -> list:
        if not s:
            return [(i, i) for i in range(256)]
        pairs = [tuple(int(x.strip()) for x in p.split(",")) for p in s.split()]
        # Interpolate to 256-point LUT
        lut = [0] * 256
        for i in range(256):
            for j in range(len(pairs) - 1):
                if pairs[j][0] <= i <= pairs[j + 1][0]:
                    t = (i - pairs[j][0]) / max(1, pairs[j + 1][0] - pairs[j][0])
                    lut[i] = max(0, min(255, int(pairs[j][1] + t * (pairs[j + 1][1] - pairs[j][1]))))
                    break
        return lut

    try:
        canvas = sessions.get_or_create(session_id)
        layer = canvas.layers[canvas.active_layer_index]
        img = layer.image.convert("RGB")
        r, g, b = img.split()
        r_lut = _parse_curve(red) if red else list(range(256))
        g_lut = _parse_curve(green) if green else list(range(256))
        b_lut = _parse_curve(blue) if blue else list(range(256))
        r = r.point(r_lut)
        g = g.point(g_lut)
        b = b.point(b_lut)
        layer.image = Image.merge("RGB", (r, g, b)).convert("RGBA")
        canvas._save_state()
        return [TextContent(text="Curves adjusted.")]
    except Exception as e:
        return [TextContent(text=f"Curves error: {str(e)}")]


# ---- Upscale ----

@app.tool("upscale")
async def upscale_tool(factor: int = 2, model: str = "anime", session_id: str = "default"):
    """Upscale by factor (1-4), preserving layer alignment. Automatically starts ComfyUI; no manual startup is required.

    model: anime, face, general, or an installed model filename.
    """
    try:
        if factor not in (1, 2, 3, 4):
            raise ValueError("factor must be 1, 2, 3, or 4")
        model_map = {"anime": MODEL_UPSCALE_ANIME, "face": MODEL_UPSCALE_FACE, "general": MODEL_UPSCALE_GENERAL}
        upscale_model = model_map.get(model, model)
        canvas = sessions.get_or_create(session_id)
        original_state = canvas.undo_stack[-1]
        layer = canvas.layers[canvas.active_layer_index]
        target_size = (canvas.width * factor, canvas.height * factor)
        img_name = f"upscale_{uuid.uuid4().hex[:8]}.png"
        buf = io.BytesIO()
        layer.image.save(buf, format="PNG")
        img_name = await comfy.upload_image(buf.getvalue(), img_name)
        workflow = build_upscale_workflow(image_filename=img_name, upscale_model=upscale_model)
        result = await run_workflow(workflow)
        if not result:
            return [TextContent(text="Upscale failed.")]
        with Image.open(io.BytesIO(result)) as opened:
            img = opened.convert("RGBA").resize(target_size, Image.Resampling.LANCZOS)
        # ComfyUI's RGB upscaler cannot retain source transparency itself.
        img.putalpha(layer.image.convert("RGBA").getchannel("A").resize(target_size, Image.Resampling.LANCZOS))
        if sessions.get_or_create(session_id) is not canvas or canvas.undo_stack[-1] is not original_state:
            raise RuntimeError("Canvas changed during upscaling; result was not applied")
        resized = []
        for existing in canvas.layers:
            new_image = img if existing is layer else existing.image.resize(target_size, Image.Resampling.LANCZOS)
            new_mask = existing.mask.resize(target_size, Image.Resampling.LANCZOS) if existing.mask is not None else None
            resized.append((existing, new_image, new_mask))
        for existing, new_image, new_mask in resized:
            existing.image, existing.mask = new_image, new_mask
        canvas.width, canvas.height = target_size
        canvas._save_state()
        return [TextContent(text=f"Upscaled to {img.width}x{img.height} ({factor}x) with model '{upscale_model}'.")]
    except Exception as e:
        return [TextContent(text=f"Upscale error: {str(e)}")]


# ---- Semantic Selection (simple threshold-based, SAM-ready) ----

@app.tool("select_object")
async def select_object_tool(description: str, threshold: int = 128, session_id: str = "default"):
    """
    Create a selection mask using simple color/region heuristics.
    description: color name or keyword (red, blue, green, sky, dark, light, white, black).
    Uses PIL getdata/putdata for batch processing (no numpy, no pixel-by-pixel loop).
    """
    try:
        canvas = sessions.get_or_create(session_id)
        layer = canvas.layers[canvas.active_layer_index]
        img = layer.image.convert("RGB")
        w, h = img.size
        desc = description.lower().strip()

        # Split into channels for fast batch processing
        r_ch, g_ch, b_ch = img.split()
        r_data = r_ch.getdata()
        g_data = g_ch.getdata()
        b_data = b_ch.getdata()

        # Build mask data as list
        mask_data = []
        color_map = {
            "yellow": (255, 255, 0), "purple": (128, 0, 128),
            "orange": (255, 165, 0), "cyan": (0, 255, 255),
            "pink": (255, 192, 203), "brown": (139, 69, 19),
        }

        if desc in color_map:
            tr, tg, tb = color_map[desc]
            max_dist = threshold * 1.5
            for r, g, b in zip(r_data, g_data, b_data):
                dist = ((r - tr) ** 2 + (g - tg) ** 2 + (b - tb) ** 2) ** 0.5
                mask_data.append(255 if dist < max_dist else 0)
        else:
            for r, g, b in zip(r_data, g_data, b_data):
                brightness = (r + g + b) / 3
                select = False
                if desc in ("sky", "light", "white"):
                    select = brightness > threshold
                elif desc in ("dark", "shadow", "black"):
                    select = brightness <= threshold
                elif desc == "red":
                    select = r > threshold and g < threshold and b < threshold
                elif desc == "blue":
                    select = b > threshold and r < threshold and g < threshold
                elif desc == "green":
                    select = g > threshold and r < threshold and b < threshold
                elif desc in color_map:
                    tr, tg, tb = color_map[desc]
                    dist = ((r - tr) ** 2 + (g - tg) ** 2 + (b - tb) ** 2) ** 0.5
                    select = dist < (threshold * 1.5)
                else:
                    return [TextContent(text=f"Unknown object: '{description}'. Supported: red, blue, green, sky, dark, light, white, black, yellow, purple, orange, cyan, pink, brown")]
                mask_data.append(255 if select else 0)

        # Create mask from data
        mask = Image.new("L", (w, h))
        mask.putdata(mask_data)
        layer.mask = mask
        canvas._save_state()
        return [TextContent(text=f"Semantic mask created for '{description}' (threshold={threshold}).")]
    except Exception as e:
        return [TextContent(text=f"Select object error: {str(e)}")]


# ---- Sessions (multi-document) ----

@app.tool("list_sessions")
async def list_sessions_tool():
    """List all open document sessions: id, canvas size, layer count, active layer, and undo depth."""
    try:
        report = {}
        for sid in sorted(sessions.list_sessions()):
            c = sessions.get(sid)
            report[sid] = {"size": [c.width, c.height], "layers": len(c.layers),
                           "active_layer": c.active_layer_index,
                           "undo_depth": max(0, len(c.undo_stack) - 1)}
        return [TextContent(text=json.dumps(report, indent=2))]
    except Exception as e:
        return [TextContent(text=f"List sessions error: {str(e)}")]


@app.tool("close_session")
async def close_session_tool(session_id: str = "default"):
    """Close a document session, freeing its canvas. Every canvas tool takes a session_id (default 'default')."""
    try:
        if sessions.delete(session_id):
            return [TextContent(text=f"Session '{session_id}' closed.")]
        return [TextContent(text=f"No open session named '{session_id}'. See list_sessions.")]
    except Exception as e:
        return [TextContent(text=f"Close session error: {str(e)}")]


# ---- Batch Generation ----

DEFAULT_BATCH_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "batch_output"


@app.tool("batch_generate")
@with_prompt_rules("generation", ("flux2", "qwen21", "anima"))
async def batch_generate_tool(jobs: list[dict], export_dir: Optional[str] = None,
                              export_format: str = "PNG", timeout: Optional[int] = None):
    """Batch text-to-image generation: queue a whole job list on ComfyUI in one pass, await completion, and export all results.

    Automatically starts ComfyUI when needed; no manual startup is required.

    jobs: list of job objects, each with 'prompt' (required) and optional
    'model' ('flux2' photorealistic / 'qwen21' detail, typography, alpha / 'anima' anime), 'width', 'height', 'steps', 'cfg',
    'seed', 'negative_prompt', and 'filename' (output basename without extension).
    Models may differ per job. Omitted steps/cfg use model defaults:
    flux2=50/4, qwen21=30/3 (custom preset), anima=30/4. Explicit values are preserved.
    For flux2 jobs, cfg is embedded FluxGuidance and negative_prompt is unused/reported.
    This tool waits for the batch results before exporting. Use submit_generation_job for a returned job ID
    and per-image exports; its MCP server process must remain running. For work that must continue after
    the LLM stops, verify independent server lifetime or use a detached host runner (see MEMORY.md).
    export_dir defaults to <server>/batch_output; export_format: PNG, JPG, or JPEG.
    Existing files are preserved by adding numeric filename suffixes.
    timeout: seconds for the whole batch wait. Returns a JSON summary with per-job
    status, output file path, and seed.
    """
    try:
        if not jobs:
            return [TextContent(text="jobs must contain at least one job")]
        fmt = export_format.upper()
        if fmt not in ("PNG", "JPG", "JPEG"):
            return [TextContent(text="export_format must be PNG, JPG, or JPEG")]
        for i, job in enumerate(jobs):
            if not isinstance(job, dict) or not str(job.get("prompt", "")).strip():
                return [TextContent(text=f"Job {i} is invalid: each job needs a non-empty 'prompt' string")]
            model = job.get("model", "flux2")
            if model not in ("flux2", "qwen21", "anima"):
                return [TextContent(text=f"Job {i}: 'model' must be 'flux2', 'qwen21' or 'anima'")]
        await comfy.start_comfyui()
        profiles = model_profiles(await comfy.get_object_info(), "generation")
        workflows = []
        seeds = []
        settings = []
        for i, job in enumerate(jobs):
            model = job.get("model", "flux2")
            profile = profiles[model]
            if not profile["available"]:
                raise ValueError(f"Job {i}: {model} is unavailable: {profile['missing']}. Call get_editing_capabilities.")
            steps = profile["default_steps"] if job.get("steps") is None else int(job["steps"])
            cfg = profile["default_cfg"] if job.get("cfg") is None else float(job["cfg"])
            seed = job.get("seed")
            seed = int(seed) if seed is not None else secrets.randbits(63)
            prompt, negative_prompt, adjustments = prepare_prompts(
                model, profile["model"], str(job["prompt"]), str(job.get("negative_prompt", "")))
            workflows.append(build_generation_workflow(model, profile,
                prompt=prompt, negative_prompt=negative_prompt,
                width=int(job.get("width", DEFAULT_WIDTH)), height=int(job.get("height", DEFAULT_HEIGHT)),
                steps=steps, cfg=cfg, seed=seed))
            seeds.append(seed)
            setting = {"steps": steps, "cfg": cfg, "checkpoint": profile["model"]}
            if adjustments:
                setting.update(prompt_adjustments=adjustments, effective_prompt=prompt, negative_prompt=negative_prompt)
            settings.append(setting)
        if timeout is None:
            timeout = sum(1800 if job.get("model", "flux2") in ("flux2", "qwen21") else WEBSOCKET_TIMEOUT for job in jobs)
        batch = await comfy.batch_run_workflows(workflows, timeout=timeout)
        batch = batch[:len(jobs)]  # defensive: results must align 1:1 with jobs
        out_dir = Path(export_dir) if export_dir else DEFAULT_BATCH_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = "jpg" if fmt in ("JPG", "JPEG") else fmt.lower()
        results = []
        for i, entry in enumerate(batch):
            job = jobs[i]
            record = {"index": i, "prompt": job.get("prompt"), "model": job.get("model", "flux2"),
                      "seed": seeds[i], "status": "ok", "file": None, **settings[i]}
            history = entry.get("history")
            data = history.get("_cached_file_bytes") if history else None
            if data is None and history:
                files = ComfyUIClient.get_output_files_from_history(history, file_type="images")
                if files:
                    first = files[0]
                    data = await comfy.get_output_file(
                        first.get("filename", ""), subfolder=first.get("subfolder", ""),
                        output_dir=first.get("type", "output"))
            if not data:
                record["status"] = "failed"
                record["error"] = entry.get("error") or "no output image"
                results.append(record)
                continue
            base = str(job.get("filename") or f"batch_{i:03d}")
            base = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)[:60]
            with Image.open(io.BytesIO(data)) as img:
                img.load()
                if fmt in ("JPG", "JPEG"):
                    img = img.convert("RGB")
                candidate = base
                suffix = 0
                while True:
                    candidate_path = out_dir / f"{candidate}.{ext}"
                    try:
                        output_file = candidate_path.open("xb")
                        break
                    except FileExistsError:
                        suffix += 1
                        candidate = f"{base}_{suffix}"
                try:
                    with output_file:
                        if fmt in ("JPG", "JPEG"):
                            img.save(output_file, "JPEG", quality=95)
                        else:
                            img.save(output_file, "PNG")
                    record["size"] = list(img.size)
                except Exception:
                    candidate_path.unlink(missing_ok=True)
                    raise
            record["file"] = str(candidate_path)
            results.append(record)
        summary = {"jobs": len(results), "ok": sum(1 for r in results if r["status"] == "ok"),
                   "failed": sum(1 for r in results if r["status"] != "ok"),
                   "export_dir": str(out_dir), "format": ext}
        return [TextContent(text=json.dumps({"summary": summary, "results": results}, indent=2))]
    except Exception as e:
        return [TextContent(text=f"Batch generation error: {str(e)}")]


async def _generate_job_item(item: dict, index: int, export_dir: Optional[str], export_format: str, timeout_per_image: Optional[int]):
    entry = item.copy()
    if not entry.get("filename"):
        entry["filename"] = f"batch_{index:03d}"
    response = await batch_generate_tool(jobs=[entry], export_dir=export_dir, export_format=export_format, timeout=timeout_per_image)
    try:
        payload = json.loads(response[0].text)
    except json.JSONDecodeError as e:
        raise RuntimeError(response[0].text) from e
    return payload["results"][0]


generation_jobs = GenerationJobs(_generate_job_item)


@app.tool("submit_generation_job")
@with_prompt_rules("generation", ("flux2", "qwen21", "anima"))
async def submit_generation_job(jobs: list[dict], export_dir: Optional[str] = None,
                                export_format: str = "PNG", timeout_per_image: Optional[int] = None):
    """Submit a generation job for asynchronous processing.

    Jobs are processed one image at a time; each image is exported before the next begins.
    Lifecycle: process-local; survives HTTP client disconnect only while the MCP server keeps running;
    restart/closed stdio process loses job state/worker; exported files persist.
    Cancellation is graceful: current image finishes and exports; remaining images skipped, no Comfy interrupt.

    Supported job fields: prompt (required), model ('flux2' photorealistic / 'qwen21' detail, typography, alpha / 'anima' anime), width, height, steps, cfg, seed, negative_prompt, filename.
    Default steps/guidance: flux2=50/4, qwen21=30/3, anima=30/4.
    timeout_per_image: optional seconds per image; omitted uses the selected model's timeout.
    """
    try:
        if not jobs:
            return [TextContent(text="jobs must contain at least one job")]
        fmt = export_format.upper()
        if fmt not in ("PNG", "JPG", "JPEG"):
            return [TextContent(text="export_format must be PNG, JPG, or JPEG")]
        if timeout_per_image is not None and timeout_per_image <= 0:
            return [TextContent(text="timeout_per_image must be positive when provided")]
        for i, job in enumerate(jobs):
            if not isinstance(job, dict):
                return [TextContent(text=f"Job at index {i} must be a dict")]
            if not str(job.get("prompt", "")).strip():
                return [TextContent(text=f"Job at index {i} must have a non-empty prompt")]
            model = job.get("model", "flux2")
            if model not in ("flux2", "qwen21", "anima"):
                return [TextContent(text=f"Job at index {i} has invalid model '{model}'. Must be flux2, qwen21, or anima.")]
        report = generation_jobs.submit(jobs, export_dir=export_dir, export_format=fmt, timeout_per_image=timeout_per_image)
        return [TextContent(text=json.dumps(report, indent=2))]
    except Exception as e:
        return [TextContent(text=f"Submit generation job error: {str(e)}")]


@app.tool("get_job_status")
async def get_job_status(job_id: str):
    """Get the status of a generation job."""
    try:
        report = generation_jobs.status(job_id)
        return [TextContent(text=json.dumps(report, indent=2))]
    except Exception as e:
        return [TextContent(text=f"Get job status error: {str(e)}")]


@app.tool("list_jobs")
async def list_jobs():
    """List all generation jobs."""
    try:
        summaries = generation_jobs.list_jobs()
        return [TextContent(text=json.dumps(summaries, indent=2))]
    except Exception as e:
        return [TextContent(text=f"List jobs error: {str(e)}")]


@app.tool("cancel_job")
async def cancel_job(job_id: str):
    """Cancel a generation job.

    Cancellation is graceful: current image finishes and exports; remaining images skipped, no Comfy interrupt.
    """
    try:
        report = generation_jobs.cancel(job_id)
        return [TextContent(text=json.dumps(report, indent=2))]
    except Exception as e:
        return [TextContent(text=f"Cancel job error: {str(e)}")]


# ======================================================================= #
#  Main
# ======================================================================= #

register_editing_tools(app, comfy, sessions, run_workflow)

if __name__ == "__main__":
    # Make app logging visible: in streamable-HTTP mode (run_openwebui.bat)
    # output lands in mcp_http.log; in stdio mode logs go to stderr, which
    # Cline's MCP transport safely ignores (protocol uses stdin/stdout).
    logging.basicConfig(level=os.getenv("MCP_LOG_LEVEL", "INFO"))
    # stdio (default - e.g. Cline) or streamable-http (Open WebUI):
    # run_openwebui.bat sets MCP_TRANSPORT=streamable-http.
    app.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))
