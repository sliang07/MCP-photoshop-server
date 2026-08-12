"""
MCP Photoshop Server - main entrypoint.
Registers all tools: canvas management, AI generation, editing, layers, history.
"""

import asyncio
import io
import json
import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, ImageChops

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent as _TextContent

def TextContent(text="", **kwargs):
    """Wrapper that adds type='text' automatically for MCP SDK compatibility."""
    return _TextContent(type="text", text=text, **kwargs)

from config import (
    COMFYUI_URL, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_STEPS, DEFAULT_CFG,
    MODEL_FLUX2, MODEL_FLUX2_TEXT_ENCODER, MODEL_FLUX2_VAE, MODEL_KONTEXT,
    MODEL_ANIMA, MODEL_ANIMA_TEXT_ENCODER, MODEL_ANIMA_VAE,
    MODEL_UPSCALE_FACE, MODEL_UPSCALE_ANIME,
    COMFYUI_AUTO_KILL, VRAM_PRESSURE_THRESHOLD_MB,
)
from comfy_client import ComfyUIClient
from canvas import Canvas, BLEND_MODES
from session import SessionManager

# ----------------------------------------------------------------------- #
#  Globals
# ----------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
app = FastMCP("mcp-photoshop-server")
comfy = ComfyUIClient(COMFYUI_URL)
sessions = SessionManager()

DEFAULT_SESSION = "default"


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


def build_txt2img_workflow(
    prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024,
    steps: int = 20, cfg: float = 1.5, seed: Optional[int] = None, model: str = "flux2",
) -> dict:
    """Build a txt2img workflow for Flux2 Klein or ANIMA.

    Flux2 Klein now uses built-in ComfyUI nodes (CLIPTextEncodeFlux + SamplerCustomAdvanced)
    instead of the custom ComfyUI-Flux2Klein-Enhancer nodes which had architecture mismatches.
    """
    nid = _make_node_id

    if model == "anima":
        return build_anima_workflow(prompt=prompt, negative_prompt=negative_prompt,
                                    width=width, height=height, steps=steps, cfg=cfg, seed=seed)

    # Flux2 Klein workflow
    # UNETLoader -> CLIPLoader -> VAELoader -> Flux2KleinSectionedEncoder -> EmptyFlux2LatentImage ->
    # BasicGuider -> RandomNoise -> BasicScheduler -> KSamplerSelect -> SamplerCustomAdvanced ->
    # VAEDecode -> SaveImage
    #
    # NOTE: CLIPTextEncodeFlux is for Flux.1 (CLIP-L + T5-XXL), NOT Flux2.
    # Flux2 uses Flux2KleinSectionedEncoder for text encoding.
    # NOTE: Flux2KleinKSamplerExperimental has architecture mismatches - use SamplerCustomAdvanced instead.
    #
    # Flux2KleinSectionedEncoder inputs (matching img2img workflow):
    #   clip, front_text, mid_text, end_text, separator
    n1  = nid()  # UNETLoader (outputs MODEL:0)
    n2  = nid()  # CLIPLoader (outputs CLIP:0) - type: "flux2"
    n3  = nid()  # VAELoader (outputs VAE:0)
    n4  = nid()  # Flux2KleinSectionedEncoder (outputs CONDITIONING:0)
    n5  = nid()  # EmptyFlux2LatentImage (outputs LATENT:0)
    n6  = nid()  # BasicGuider
    n7  = nid()  # RandomNoise
    n8  = nid()  # BasicScheduler
    n9  = nid()  # KSamplerSelect
    n10 = nid()  # SamplerCustomAdvanced
    n11 = nid()  # VAEDecode
    n12 = nid()  # SaveImage

    seed_value = seed if seed is not None else 42

    # Split prompt into sections for Flux2KleinSectionedEncoder
    parts = [p.strip() for p in prompt.split(",")]
    front_text = parts[0] if parts else prompt
    mid_text = ", ".join(parts[1:]) if len(parts) > 1 else ""
    end_text = ""

    return {
        n1:  {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL_FLUX2, "weight_dtype": "default"}},
        n2:  {"class_type": "CLIPLoader", "inputs": {"clip_name": MODEL_FLUX2_TEXT_ENCODER, "type": "flux2"}},
        n3:  {"class_type": "VAELoader", "inputs": {"vae_name": MODEL_FLUX2_VAE}},
        n4:  {"class_type": "Flux2KleinSectionedEncoder", "inputs": {
            "clip": [n2, 0],
            "front_text": front_text,
            "mid_text": mid_text,
            "end_text": end_text,
            "separator": "comma",
        }},
        n5:  {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        n6:  {"class_type": "BasicGuider", "inputs": {"model": [n1, 0], "conditioning": [n4, 0]}},
        n7:  {"class_type": "RandomNoise", "inputs": {"noise_seed": seed_value}},
        n8:  {"class_type": "BasicScheduler", "inputs": {"model": [n1, 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        n9:  {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        n10: {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [n7, 0],
            "guider": [n6, 0],
            "sampler": [n9, 0],
            "sigmas": [n8, 0],
            "latent_image": [n5, 0],
        }},
        n11: {"class_type": "VAEDecode", "inputs": {"samples": [n10, 0], "vae": [n3, 0]}},
        n12: {"class_type": "SaveImage", "inputs": {"images": [n11, 0], "filename_prefix": "mcp_flux2"}},
    }


def build_anima_workflow(
    prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024,
    steps: int = 20, cfg: float = 1.5, seed: Optional[int] = None,
) -> dict:
    """Build a txt2img workflow for ANIMA (anime-style generation).

    Uses separate UNETLoader, CLIPLoader, and VAELoader (not CheckpointLoaderSimple).
    Sampler: er_sde, Scheduler: simple.
    """
    nid = _make_node_id

    n1  = nid()  # UNETLoader (diffusion model)
    n2  = nid()  # CLIPLoader (text encoder)
    n3  = nid()  # VAELoader
    n4  = nid()  # CLIPTextEncode (positive)
    n5  = nid()  # CLIPTextEncode (negative)
    n6  = nid()  # EmptyLatentImage
    n7  = nid()  # KSampler
    n8  = nid()  # VAEDecode
    n9  = nid()  # SaveImage

    return {
        n1: {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL_ANIMA, "weight_dtype": "default"}},
        n2: {"class_type": "CLIPLoader", "inputs": {"clip_name": MODEL_ANIMA_TEXT_ENCODER, "type": "stable_diffusion"}},
        n3: {"class_type": "VAELoader", "inputs": {"vae_name": MODEL_ANIMA_VAE}},
        n4: {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": [n2, 0]}},
        n5: {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": [n2, 0]}},
        n6: {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        n7: {"class_type": "KSampler", "inputs": {
            "model": [n1, 0],
            "positive": [n4, 0],
            "negative": [n5, 0],
            "latent_image": [n6, 0],
            "seed": seed if seed is not None else 42,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "er_sde",
            "scheduler": "simple",
            "denoise": 1.0,
        }},
        n8: {"class_type": "VAEDecode", "inputs": {"samples": [n7, 0], "vae": [n3, 0]}},
        n9: {"class_type": "SaveImage", "inputs": {"images": [n8, 0], "filename_prefix": "mcp_anima"}},
    }


def build_img2img_kontext_workflow(
    prompt: str, image_filename: str, strength: float = 0.7,
    seed: Optional[int] = None, guidance: float = 4.0,
) -> dict:
    """Build img2img workflow using Flux2 Klein CLIP pattern (CLIPLoader + Flux2KleinSectionedEncoder).

    Uses the same text encoder pipeline as build_txt2img_workflow since both use the
    same Flux2 Klein UNET. The img2img path encodes the source image to latent,
    then uses BasicGuider + RandomNoise + BasicScheduler + KSamplerSelect + SamplerCustomAdvanced
    for the denoising step (instead of Flux2KleinKSamplerExperimental which only
    supports full denoise).
    """
    nid = _make_node_id
    n1  = nid()  # LoadImage
    n2  = nid()  # UNETLoader
    n3  = nid()  # CLIPLoader (single, type: "flux2")
    n4  = nid()  # VAELoader
    n5  = nid()  # VAEEncode
    n6  = nid()  # Flux2KleinSectionedEncoder
    n7  = nid()  # BasicGuider
    n8  = nid()  # RandomNoise
    n9  = nid()  # BasicScheduler
    n10 = nid()  # KSamplerSelect (provides sampler object to SamplerCustomAdvanced)
    n11 = nid()  # SamplerCustomAdvanced
    n12 = nid()  # VAEDecode
    n13 = nid()  # SaveImage

    # Split prompt into sections for Flux2KleinSectionedEncoder
    parts = [p.strip() for p in prompt.split(",")]
    front_text = parts[0] if parts else prompt
    mid_text = ", ".join(parts[1:]) if len(parts) > 1 else ""
    end_text = ""

    return {
        n1:  {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        n2:  {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL_FLUX2, "weight_dtype": "default"}},
        n3:  {"class_type": "CLIPLoader", "inputs": {"clip_name": MODEL_FLUX2_TEXT_ENCODER, "type": "flux2"}},
        n4:  {"class_type": "VAELoader", "inputs": {"vae_name": MODEL_FLUX2_VAE}},
        n5:  {"class_type": "VAEEncode", "inputs": {"pixels": [n1, 0], "vae": [n4, 0]}},
        n6:  {"class_type": "Flux2KleinSectionedEncoder", "inputs": {
            "clip": [n3, 0],
            "front_text": front_text,
            "mid_text": mid_text,
            "end_text": end_text,
            "separator": "comma",
        }},
        n7:  {"class_type": "BasicGuider", "inputs": {"model": [n2, 0], "conditioning": [n6, 0]}},
        n8:  {"class_type": "RandomNoise", "inputs": {"noise_seed": seed if seed is not None else 42}},
        n9:  {"class_type": "BasicScheduler", "inputs": {"model": [n2, 0], "scheduler": "simple", "steps": 30, "denoise": strength}},
        n10: {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        n11: {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [n8, 0], "guider": [n7, 0], "sampler": [n10, 0], "sigmas": [n9, 0], "latent_image": [n5, 0]}},
        n12: {"class_type": "VAEDecode", "inputs": {"samples": [n11, 0], "vae": [n4, 0]}},
        n13: {"class_type": "SaveImage", "inputs": {"images": [n12, 0], "filename_prefix": "mcp_img2img"}},
    }


def build_inpaint_workflow(
    prompt: str, image_filename: str, mask_filename: str,
    seed: Optional[int] = None, guidance: float = 4.0, steps: int = 30,
) -> dict:
    """Build inpaint workflow using Kontext (Flux.1) loader chain.

    Uses DualCLIPLoader (clip_l + t5xxl) + ae.safetensors VAE to match
    the flux1-dev-kontext UNET architecture — NOT the Flux2 Klein loaders.
    """
    nid = _make_node_id
    n1  = nid()  # LoadImage (base)
    n2  = nid()  # LoadImage (mask)
    n3  = nid()  # ImageToMask (red channel = mask)
    n4  = nid()  # UNETLoader (Kontext - Flux.1 architecture)
    n5  = nid()  # DualCLIPLoader (clip_l + t5xxl, type="flux")
    n6  = nid()  # VAELoader (ae.safetensors - Flux.1 VAE)
    n7  = nid()  # VAEEncode
    n8  = nid()  # CLIPTextEncode
    n9  = nid()  # FluxGuidance
    n10 = nid()  # BasicGuider
    n11 = nid()  # RandomNoise
    n12 = nid()  # BasicScheduler
    n13 = nid()  # KSamplerSelect
    n14 = nid()  # SetLatentNoiseMask
    n15 = nid()  # SamplerCustomAdvanced
    n16 = nid()  # VAEDecode
    n17 = nid()  # SaveImage

    return {
        n1:  {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        n2:  {"class_type": "LoadImage", "inputs": {"image": mask_filename}},
        n3:  {"class_type": "ImageToMask", "inputs": {"image": [n2, 0], "channel": "red"}},
        n4:  {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL_KONTEXT, "weight_dtype": "default"}},
        n5:  {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "clip_l.safetensors",
            "clip_name2": "t5\\t5xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "flux",
        }},
        n6:  {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        n7:  {"class_type": "VAEEncode", "inputs": {"pixels": [n1, 0], "vae": [n6, 0]}},
        n8:  {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": [n5, 0]}},
        n9:  {"class_type": "FluxGuidance", "inputs": {"conditioning": [n8, 0], "guidance": guidance}},
        n10: {"class_type": "BasicGuider", "inputs": {"model": [n4, 0], "conditioning": [n9, 0]}},
        n11: {"class_type": "RandomNoise", "inputs": {"noise_seed": seed if seed is not None else 42}},
        n12: {"class_type": "BasicScheduler", "inputs": {"model": [n4, 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        n13: {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        n14: {"class_type": "SetLatentNoiseMask", "inputs": {"samples": [n7, 0], "mask": [n3, 0]}},
        n15: {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [n11, 0], "guider": [n10, 0], "sampler": [n13, 0], "sigmas": [n12, 0], "latent_image": [n14, 0]}},
        n16: {"class_type": "VAEDecode", "inputs": {"samples": [n15, 0], "vae": [n6, 0]}},
        n17: {"class_type": "SaveImage", "inputs": {"images": [n16, 0], "filename_prefix": "mcp_inpaint"}},
    }


def build_outpaint_workflow(
    prompt: str, image_filename: str, target_width: int, target_height: int,
    seed: Optional[int] = None, guidance: float = 4.0, steps: int = 30,
) -> dict:
    """Build outpaint workflow using Kontext (Flux.1) loader chain.

    Uses DualCLIPLoader (clip_l + t5xxl) + ae.safetensors VAE to match
    the flux1-dev-kontext UNET architecture — NOT the Flux2 Klein loaders.

    Alpha mask: uses LoadImage's native MASK output (index 1) instead of
    ImageToMask with channel="alpha" because ComfyUI's IMAGE output is
    always 3-channel RGB (no alpha channel).
    """
    nid = _make_node_id
    n1  = nid()  # LoadImage (outputs IMAGE:0, MASK:1)
    n2  = nid()  # ImageScale (pad to target size)
    n3  = nid()  # UNETLoader (Kontext - Flux.1 architecture)
    n4  = nid()  # DualCLIPLoader (clip_l + t5xxl, type="flux")
    n5  = nid()  # VAELoader (ae.safetensors - Flux.1 VAE)
    n6  = nid()  # VAEEncode
    n7  = nid()  # CLIPTextEncode
    n8  = nid()  # FluxGuidance
    n9  = nid()  # BasicGuider
    n10 = nid()  # RandomNoise
    n11 = nid()  # BasicScheduler
    n12 = nid()  # KSamplerSelect
    n13 = nid()  # SetLatentNoiseMask
    n14 = nid()  # SamplerCustomAdvanced
    n15 = nid()  # VAEDecode
    n16 = nid()  # SaveImage

    return {
        n1:  {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        n2:  {"class_type": "ImageScale", "inputs": {"image": [n1, 0], "width": target_width, "height": target_height, "upscale_method": "lanczos", "crop": "disabled"}},
        n3:  {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL_KONTEXT, "weight_dtype": "default"}},
        n4:  {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "clip_l.safetensors",
            "clip_name2": "t5\\t5xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "flux",
        }},
        n5:  {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        n6:  {"class_type": "VAEEncode", "inputs": {"pixels": [n2, 0], "vae": [n5, 0]}},
        n7:  {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": [n4, 0]}},
        n8:  {"class_type": "FluxGuidance", "inputs": {"conditioning": [n7, 0], "guidance": guidance}},
        n9:  {"class_type": "BasicGuider", "inputs": {"model": [n3, 0], "conditioning": [n8, 0]}},
        n10: {"class_type": "RandomNoise", "inputs": {"noise_seed": seed if seed is not None else 42}},
        n11: {"class_type": "BasicScheduler", "inputs": {"model": [n3, 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        n12: {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        # Use LoadImage's native MASK output (index 1) instead of ImageToMask channel="alpha"
        n13: {"class_type": "SetLatentNoiseMask", "inputs": {"samples": [n6, 0], "mask": [n1, 1]}},
        n14: {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [n10, 0], "guider": [n9, 0], "sampler": [n12, 0], "sigmas": [n11, 0], "latent_image": [n13, 0]}},
        n15: {"class_type": "VAEDecode", "inputs": {"samples": [n14, 0], "vae": [n5, 0]}},
        n16: {"class_type": "SaveImage", "inputs": {"images": [n15, 0], "filename_prefix": "mcp_outpaint"}},
    }


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


def build_controlnet_workflow(
    prompt: str, image_filename: str, controlnet_name: str = "control_v11f1p_sd15_depth_fp16.safetensors",
    control_strength: float = 0.8, width: int = 1024, height: int = 1024,
    steps: int = 20, cfg: float = 1.5, seed: Optional[int] = None,
) -> dict:
    """Build a ControlNet-guided generation workflow using built-in ComfyUI nodes.
    Flux2 uses separate UNETLoader + CLIPLoader + VAELoader (not CheckpointLoaderSimple)."""
    nid = _make_node_id
    n1   = nid()  # LoadImage (control image)
    n2   = nid()  # ImageScale (resize control image to match target dimensions)
    n_un = nid()  # UNETLoader
    n_cl = nid()  # CLIPLoader
    n_va = nid()  # VAELoader
    n3   = nid()  # ControlNetLoader
    n4   = nid()  # CLIPTextEncode
    n5   = nid()  # FluxGuidance
    n6   = nid()  # BasicGuider
    n7   = nid()  # ControlNetApply (apply control to conditioning)
    n8   = nid()  # EmptyLatentImage
    n9   = nid()  # RandomNoise
    n10  = nid()  # BasicScheduler
    n11  = nid()  # KSamplerSelect
    n12  = nid()  # SamplerCustomAdvanced
    n13  = nid()  # VAEDecode
    n14  = nid()  # SaveImage

    return {
        n1:   {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        n2:   {"class_type": "ImageScale", "inputs": {"image": [n1, 0], "width": width, "height": height, "upscale_method": "lanczos", "crop": "disabled"}},
        n_un: {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL_FLUX2, "weight_dtype": "default"}},
        n_cl: {"class_type": "CLIPLoader", "inputs": {"clip_name": MODEL_FLUX2_TEXT_ENCODER, "type": "flux2"}},
        n_va: {"class_type": "VAELoader", "inputs": {"vae_name": MODEL_FLUX2_VAE}},
        n3:   {"class_type": "ControlNetLoader", "inputs": {"control_net_name": controlnet_name}},
        n4:   {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": [n_cl, 0]}},
        n5:   {"class_type": "FluxGuidance", "inputs": {"conditioning": [n4, 0], "guidance": cfg}},
        n6:   {"class_type": "BasicGuider", "inputs": {"model": [n_un, 0], "conditioning": [n5, 0]}},
        n7:   {"class_type": "ControlNetApply", "inputs": {"positive": [n5, 0], "control_net": [n3, 0], "image": [n2, 0], "strength": control_strength}},
        n8:   {"class_type": "EmptyLatentImage", "inputs": {"width": width // 8, "height": height // 8, "batch_size": 1}},
        n9:   {"class_type": "RandomNoise", "inputs": {"noise_seed": seed if seed is not None else 42}},
        n10:  {"class_type": "BasicScheduler", "inputs": {"model": [n_un, 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        n11:  {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        n12:  {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [n9, 0], "guider": [n6, 0], "sampler": [n11, 0], "sigmas": [n10, 0], "latent_image": [n8, 0]}},
        n13:  {"class_type": "VAEDecode", "inputs": {"samples": [n12, 0], "vae": [n_va, 0]}},
        n14:  {"class_type": "SaveImage", "inputs": {"images": [n13, 0], "filename_prefix": "mcp_controlnet"}},
    }


def build_style_transfer_workflow(
    prompt: str, content_filename: str, style_filename: str,
    style_strength: float = 0.8, width: int = 1024, height: int = 1024,
    steps: int = 20, seed: Optional[int] = None,
) -> dict:
    """Build a style transfer workflow using CLIPVision + StyleModel (Redux) with built-in nodes.
    Flux2 uses separate UNETLoader + CLIPLoader + VAELoader (not CheckpointLoaderSimple)."""
    nid = _make_node_id
    n1   = nid()  # LoadImage (content)
    n2   = nid()  # LoadImage (style reference)
    n_un = nid()  # UNETLoader
    n_cl = nid()  # CLIPLoader
    n_va = nid()  # VAELoader
    n4   = nid()  # CLIPVisionEncode (style image)
    n5   = nid()  # StyleModelLoader
    n6   = nid()  # CLIPTextEncode
    n7   = nid()  # FluxGuidance
    n8   = nid()  # BasicGuider
    n9   = nid()  # StyleModelApply
    n10  = nid()  # EmptyLatentImage
    n11  = nid()  # RandomNoise
    n12  = nid()  # BasicScheduler
    n13  = nid()  # KSamplerSelect
    n14  = nid()  # SamplerCustomAdvanced
    n15  = nid()  # VAEDecode
    n16  = nid()  # SaveImage

    return {
        n1:   {"class_type": "LoadImage", "inputs": {"image": content_filename}},
        n2:   {"class_type": "LoadImage", "inputs": {"image": style_filename}},
        n_un: {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL_FLUX2, "weight_dtype": "default"}},
        n_cl: {"class_type": "CLIPLoader", "inputs": {"clip_name": MODEL_FLUX2_TEXT_ENCODER, "type": "flux2"}},
        n_va: {"class_type": "VAELoader", "inputs": {"vae_name": MODEL_FLUX2_VAE}},
        n4:   {"class_type": "CLIPVisionEncode", "inputs": {"image": [n2, 0]}},
        n5:   {"class_type": "StyleModelLoader", "inputs": {"style_model_name": "flux1-redux-dev.safetensors"}},
        n6:   {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": [n_cl, 0]}},
        n7:   {"class_type": "FluxGuidance", "inputs": {"conditioning": [n6, 0], "guidance": 1.5}},
        n8:   {"class_type": "BasicGuider", "inputs": {"model": [n_un, 0], "conditioning": [n7, 0]}},
        n9:   {"class_type": "StyleModelApply", "inputs": {"conditioning": [n7, 0], "style_model": [n5, 0], "clip_vision_output": [n4, 0], "strength": style_strength}},
        n10:  {"class_type": "EmptyLatentImage", "inputs": {"width": width // 8, "height": height // 8, "batch_size": 1}},
        n11:  {"class_type": "RandomNoise", "inputs": {"noise_seed": seed if seed is not None else 42}},
        n12:  {"class_type": "BasicScheduler", "inputs": {"model": [n_un, 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        n13:  {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        n14:  {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [n11, 0], "guider": [n8, 0], "sampler": [n13, 0], "sigmas": [n12, 0], "latent_image": [n10, 0]}},
        n15:  {"class_type": "VAEDecode", "inputs": {"samples": [n14, 0], "vae": [n_va, 0]}},
        n16:  {"class_type": "SaveImage", "inputs": {"images": [n15, 0], "filename_prefix": "mcp_style_transfer"}},
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
async def new_canvas(width: int = 1024, height: int = 1024, bg_color: str = "white"):
    """Create a new blank canvas."""
    try:
        img = Image.new("RGB", (1, 1), bg_color)
        color = img.getpixel((0, 0))
        canvas = sessions.create(DEFAULT_SESSION, width, height, color)
        return [TextContent(text=json.dumps(canvas.get_info()))]
    except Exception as e:
        return [TextContent(text=f"Error creating canvas: {str(e)}")]


@app.tool("open_image")
async def open_image(path: str):
    """Load an existing image as the active document."""
    try:
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        canvas = sessions.create(DEFAULT_SESSION, w, h)
        canvas.layers.clear()
        canvas.undo_stack.clear()
        canvas.redo_stack.clear()
        canvas.add_layer(name="Layer 0", image=img)
        return [TextContent(text=json.dumps(canvas.get_info()))]
    except Exception as e:
        return [TextContent(text=f"Error opening image: {str(e)}")]


@app.tool("export")
async def export(path: Optional[str] = None, format: str = "PNG", quality: int = 95):
    """Export the current canvas to a file."""
    try:
        canvas = sessions.get_default_session()
        fmt = "JPEG" if format.upper() == "JPG" else format.upper()
        img = canvas.composite_rgb() if fmt == "JPEG" else canvas.composite()
        if not path:
            path = f"export_{uuid.uuid4().hex[:8]}.{format.lower()}"
        img.save(path, fmt, quality=quality)
        return [TextContent(text=f"Exported to {path} ({fmt})")]
    except Exception as e:
        return [TextContent(text=f"Error exporting: {str(e)}")]


@app.tool("get_info")
async def get_info():
    """Get current canvas dimensions, layer count, and session info."""
    try:
        canvas = sessions.get_default_session()
        return [TextContent(text=json.dumps(canvas.get_info(), indent=2))]
    except Exception as e:
        return [TextContent(text=f"Error: {str(e)}")]


@app.tool("generate_image")
async def generate_image(prompt: str, model: str = "flux2", width: int = 1024, height: int = 1024,
                         steps: int = 6, cfg: float = 1.5, seed: Optional[int] = None, negative_prompt: str = ""):
    """Generate an image from text. model: 'flux2' (photorealistic) or 'anima' (anime). Flux2 Klein requires 4-6 steps max."""
    try:
        # FLUX2 Klein 9B requires 4-6 steps max — cap to prevent failures
        if model == "flux2" and steps > 6:
            steps = 6
        workflow = build_txt2img_workflow(prompt=prompt, negative_prompt=negative_prompt,
                                          width=width, height=height, steps=steps, cfg=cfg, seed=seed, model=model)
        result = await run_workflow(workflow)
        if not result:
            return [TextContent(text="Generation failed.")]
        img = Image.open(io.BytesIO(result)).convert("RGBA")
        canvas = sessions.get_default_session()
        if canvas.width != img.width or canvas.height != img.height:
            canvas = sessions.create(DEFAULT_SESSION, img.width, img.height)
        idx = canvas.add_layer(name=f"Generated: {prompt[:30]}", image=img)
        return [TextContent(text=f"Generated image ({img.width}x{img.height}) as layer {idx}. Model: {model}")]
    except Exception as e:
        return [TextContent(text=f"Generation error: {str(e)}")]


@app.tool("img2img")
async def img2img_tool(prompt: str, strength: float = 0.7, guidance: float = 4.0, seed: Optional[int] = None):
    """Transform the current canvas using AI instructed editing (Flux Kontext)."""
    try:
        canvas = sessions.get_default_session()
        composite = canvas.composite()
        temp_name = f"img2img_{uuid.uuid4().hex[:8]}.png"
        buffer = io.BytesIO()
        composite.save(buffer, format="PNG")
        await comfy.upload_image(buffer.getvalue(), temp_name)
        workflow = build_img2img_kontext_workflow(prompt=prompt, image_filename=temp_name,
                                                  strength=strength, seed=seed, guidance=guidance)
        result = await run_workflow(workflow)
        if not result:
            return [TextContent(text="img2img failed.")]
        img = Image.open(io.BytesIO(result)).convert("RGBA")
        idx = canvas.add_layer(name=f"img2img: {prompt[:30]}", image=img)
        return [TextContent(text=f"img2img complete. Layer {idx}.")]
    except Exception as e:
        return [TextContent(text=f"img2img error: {str(e)}")]


@app.tool("character_transform")
async def character_transform(prompt: str, guidance: float = 4.0, seed: Optional[int] = None):
    """Transform a character (pose, expression, action) using Flux Kontext."""
    return await img2img_tool(prompt=prompt, strength=0.7, guidance=guidance, seed=seed)


@app.tool("crop")
async def crop_tool(x: int, y: int, width: int, height: int):
    """Crop the active layer to the specified region."""
    try:
        canvas = sessions.get_default_session()
        # Crop all layers to preserve alignment
        for layer in canvas.layers:
            layer.image = layer.image.crop((x, y, x + width, y + height))
        canvas.width = width
        canvas.height = height
        canvas._save_state()
        return [TextContent(text=f"Cropped to {width}x{height} at ({x},{y})")]
    except Exception as e:
        return [TextContent(text=f"Crop error: {str(e)}")]


@app.tool("resize")
async def resize_tool(width: int, height: int, maintain_aspect: bool = False):
    """Resize the active layer."""
    try:
        canvas = sessions.get_default_session()
        layer = canvas.layers[canvas.active_layer_index]
        img = layer.image
        if maintain_aspect:
            img = img.copy()
            img.thumbnail((width, height), Image.LANCZOS)
            new_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            new_img.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
            layer.image = new_img
        else:
            layer.image = layer.image.resize((width, height), Image.LANCZOS)
        canvas.width = width
        canvas.height = height
        canvas._save_state()
        return [TextContent(text=f"Resized to {width}x{height}")]
    except Exception as e:
        return [TextContent(text=f"Resize error: {str(e)}")]


@app.tool("rotate")
async def rotate_tool(degrees: float, expand: bool = True, bg_color: str = "transparent"):
    """Rotate the active layer (counter-clockwise)."""
    try:
        canvas = sessions.get_default_session()
        layer = canvas.layers[canvas.active_layer_index]
        fill_color = (0, 0, 0, 0) if bg_color == "transparent" else bg_color
        layer.image = layer.image.rotate(degrees, expand=expand, fillcolor=fill_color, resample=Image.BICUBIC)
        canvas._save_state()
        return [TextContent(text=f"Rotated {degrees} degrees")]
    except Exception as e:
        return [TextContent(text=f"Rotate error: {str(e)}")]


@app.tool("flip")
async def flip_tool(axis: str = "horizontal"):
    """Flip the active layer. axis: 'horizontal' or 'vertical'."""
    try:
        canvas = sessions.get_default_session()
        layer = canvas.layers[canvas.active_layer_index]
        layer.image = ImageOps.mirror(layer.image) if axis == "horizontal" else ImageOps.flip(layer.image)
        canvas._save_state()
        return [TextContent(text=f"Flipped {axis}")]
    except Exception as e:
        return [TextContent(text=f"Flip error: {str(e)}")]


@app.tool("adjust")
async def adjust_tool(brightness: Optional[float] = None, contrast: Optional[float] = None,
                      saturation: Optional[float] = None, hue: Optional[float] = None, sharpness: Optional[float] = None):
    """Adjust colors. Each parameter is a multiplier (1.0=no change). hue is in degrees (0-360)."""
    try:
        canvas = sessions.get_default_session()
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
        if hue is not None and hue != 0:
            h, s, v = img.convert("HSV").split()
            h_data = [(x + int(hue)) % 256 for x in h.getdata()]
            h = h.copy()
            h.putdata(h_data)
            img = Image.merge("HSV", (h, s, v)).convert("RGBA")
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
async def apply_filter(name: str, **params):
    """Apply a filter: blur, gaussian_blur, sharpen, contour, detail, edge_enhance, find_edges, emboss, pixelate, posterize, solarize, invert, grayscale, sepia."""
    try:
        canvas = sessions.get_default_session()
        layer = canvas.layers[canvas.active_layer_index]
        img = layer.image
        filters = {
            "blur": lambda **k: img.filter(ImageFilter.BLUR),
            "gaussian_blur": lambda **k: img.filter(ImageFilter.GaussianBlur(k.get("radius", 2))),
            "sharpen": lambda **k: img.filter(ImageFilter.SHARPEN),
            "contour": lambda **k: img.filter(ImageFilter.CONTOUR),
            "detail": lambda **k: img.filter(ImageFilter.DETAIL),
            "edge_enhance": lambda **k: img.filter(ImageFilter.EDGE_ENHANCE),
            "edge_enhance_more": lambda **k: img.filter(ImageFilter.EDGE_ENHANCE_MORE),
            "find_edges": lambda **k: img.filter(ImageFilter.FIND_EDGES),
            "smooth": lambda **k: img.filter(ImageFilter.SMOOTH),
            "smooth_more": lambda **k: img.filter(ImageFilter.SMOOTH_MORE),
            "emboss": lambda **k: img.filter(ImageFilter.EMBOSS),
            "pixelate": lambda **k: _pixelate(img, k.get("size", 8)),
            "posterize": lambda **k: ImageOps.posterize(img.convert("RGB"), k.get("levels", 4)).convert("RGBA"),
            "solarize": lambda **k: ImageOps.solarize(img.convert("RGB")).convert("RGBA"),
            "invert": lambda **k: ImageOps.invert(img.convert("RGB")).convert("RGBA"),
            "grayscale": lambda **k: img.convert("L").convert("RGBA"),
            "sepia": lambda **k: _apply_sepia(img),
        }
        if name not in filters:
            return [TextContent(text=f"Unknown filter: {name}. Available: {', '.join(filters.keys())}")]
        layer.image = filters[name](**params)
        canvas._save_state()
        return [TextContent(text=f"Filter '{name}' applied.")]
    except Exception as e:
        return [TextContent(text=f"Filter error: {str(e)}")]


@app.tool("add_text")
async def add_text(text: str, x: int = 50, y: int = 50, font_size: int = 48, color: str = "white",
                   font: Optional[str] = None, stroke_width: int = 0, stroke_color: str = "black"):
    """Add text overlay to the active layer."""
    try:
        canvas = sessions.get_default_session()
        layer = canvas.layers[canvas.active_layer_index]
        img = layer.image.copy()
        draw = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype(font, font_size) if font and os.path.exists(font) else ImageFont.load_default()
        except Exception:
            f = ImageFont.load_default()
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
                         opacity: float = 1.0, blend_mode: str = "normal"):
    """Add a new layer. Optionally load from source_path."""
    try:
        canvas = sessions.get_default_session()
        img = Image.open(source_path).convert("RGBA") if source_path else None
        layer_name = name or f"Layer {len(canvas.layers)}"
        idx = canvas.add_layer(name=layer_name, image=img, opacity=opacity, blend_mode=blend_mode)
        return [TextContent(text=f"Added '{layer_name}' as layer {idx}")]
    except Exception as e:
        return [TextContent(text=f"Add layer error: {str(e)}")]


@app.tool("select_layer")
async def select_layer_tool(index: int):
    """Select a layer by index (0 = bottom)."""
    try:
        canvas = sessions.get_default_session()
        if canvas.select_layer(index):
            return [TextContent(text=f"Selected layer {index}: {canvas.layers[index].name}")]
        return [TextContent(text=f"Invalid layer index: {index}")]
    except Exception as e:
        return [TextContent(text=f"Select layer error: {str(e)}")]


@app.tool("set_blend_mode")
async def set_blend_mode_tool(mode: str, index: Optional[int] = None):
    """Set blend mode: normal, multiply, screen, overlay, darken, lighten, color_dodge, color_burn, hard_light, soft_light, difference, exclusion."""
    try:
        canvas = sessions.get_default_session()
        if canvas.set_blend_mode(mode, index):
            return [TextContent(text=f"Blend mode set to '{mode}'")]
        return [TextContent(text=f"Invalid blend mode: {mode}")]
    except Exception as e:
        return [TextContent(text=f"Blend mode error: {str(e)}")]


@app.tool("set_layer_opacity")
async def set_layer_opacity_tool(opacity: float, index: Optional[int] = None):
    """Set layer opacity (0.0=transparent, 1.0=opaque)."""
    try:
        canvas = sessions.get_default_session()
        if canvas.set_layer_opacity(index, opacity):
            return [TextContent(text=f"Opacity set to {opacity}")]
        return [TextContent(text="Invalid layer index")]
    except Exception as e:
        return [TextContent(text=f"Opacity error: {str(e)}")]


@app.tool("merge_down")
async def merge_down_tool():
    """Merge active layer into the layer below."""
    try:
        canvas = sessions.get_default_session()
        if canvas.merge_down():
            return [TextContent(text="Merged down.")]
        return [TextContent(text="Cannot merge: already on bottom layer.")]
    except Exception as e:
        return [TextContent(text=f"Merge error: {str(e)}")]


@app.tool("delete_layer")
async def delete_layer_tool(index: Optional[int] = None):
    """Delete a layer by index or the active layer."""
    try:
        canvas = sessions.get_default_session()
        if canvas.delete_layer(index):
            return [TextContent(text="Layer deleted.")]
        return [TextContent(text="Cannot delete layer.")]
    except Exception as e:
        return [TextContent(text=f"Delete error: {str(e)}")]


@app.tool("reorder_layer")
async def reorder_layer_tool(index: Optional[int] = None, direction: str = "up"):
    """Move layer up or down in the stack."""
    try:
        canvas = sessions.get_default_session()
        idx = index if index is not None else canvas.active_layer_index
        if canvas.reorder_layer(idx, direction):
            return [TextContent(text=f"Layer moved {direction}.")]
        return [TextContent(text="Cannot move: already at edge.")]
    except Exception as e:
        return [TextContent(text=f"Reorder error: {str(e)}")]


@app.tool("select_rect")
async def select_rect(x: int, y: int, width: int, height: int):
    """Create a rectangular selection mask on the active layer."""
    try:
        canvas = sessions.get_default_session()
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
async def select_ellipse(x: int, y: int, rx: int, ry: int):
    """Create an elliptical selection mask centered at (x,y)."""
    try:
        canvas = sessions.get_default_session()
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
async def clear_mask_tool(index: Optional[int] = None):
    """Remove the mask from the active layer."""
    try:
        canvas = sessions.get_default_session()
        idx = index if index is not None else canvas.active_layer_index
        if 0 <= idx < len(canvas.layers):
            canvas.layers[idx].mask = None
            canvas._save_state()
            return [TextContent(text="Mask cleared.")]
        return [TextContent(text="Invalid layer index.")]
    except Exception as e:
        return [TextContent(text=f"Clear mask error: {str(e)}")]


@app.tool("undo")
async def undo_tool():
    """Undo the last operation."""
    try:
        canvas = sessions.get_default_session()
        if canvas.undo():
            return [TextContent(text="Undone.")]
        return [TextContent(text="Nothing to undo.")]
    except Exception as e:
        return [TextContent(text=f"Undo error: {str(e)}")]


@app.tool("redo")
async def redo_tool():
    """Redo the last undone operation."""
    try:
        canvas = sessions.get_default_session()
        if canvas.redo():
            return [TextContent(text="Redone.")]
        return [TextContent(text="Nothing to redo.")]
    except Exception as e:
        return [TextContent(text=f"Redo error: {str(e)}")]


@app.tool("get_comfyui_status")
async def get_comfyui_status_tool():
    """Check ComfyUI connection and system info."""
    try:
        stats = await comfy.get_system_stats()
        return [TextContent(text=json.dumps(stats, indent=2))]
    except Exception as e:
        return [TextContent(text=f"ComfyUI not reachable: {str(e)}")]


@app.tool("clear_vram")
async def clear_vram_tool():
    """Free GPU VRAM by unloading cached models."""
    try:
        await comfy.free_memory()
        return [TextContent(text="VRAM cleared.")]
    except Exception as e:
        return [TextContent(text=f"Clear VRAM error: {str(e)}")]


# ---- Inpaint / Outpaint ----

@app.tool("inpaint")
async def inpaint_tool(prompt: str, guidance: float = 4.0, steps: int = 30, seed: Optional[int] = None):
    """AI inpaint the masked region of the active layer. Requires a mask set via select_rect/select_ellipse first."""
    try:
        canvas = sessions.get_default_session()
        layer = canvas.layers[canvas.active_layer_index]
        if layer.mask is None:
            return [TextContent(text="No mask set. Use select_rect or select_ellipse first.")]
        # Upload base image
        img_name = f"inpaint_img_{uuid.uuid4().hex[:8]}.png"
        buf = io.BytesIO()
        layer.image.save(buf, format="PNG")
        await comfy.upload_image(buf.getvalue(), img_name)
        # Upload mask image
        mask_name = f"inpaint_mask_{uuid.uuid4().hex[:8]}.png"
        mask_buf = io.BytesIO()
        layer.mask.save(mask_buf, format="PNG")
        await comfy.upload_image(mask_buf.getvalue(), mask_name)
        workflow = build_inpaint_workflow(prompt=prompt, image_filename=img_name, mask_filename=mask_name,
                                          seed=seed, guidance=guidance, steps=steps)
        result = await run_workflow(workflow)
        if not result:
            return [TextContent(text="Inpaint failed.")]
        img = Image.open(io.BytesIO(result)).convert("RGBA")
        layer.image = img
        layer.mask = None
        canvas._save_state()
        return [TextContent(text=f"Inpaint complete ({img.width}x{img.height}).")]
    except Exception as e:
        return [TextContent(text=f"Inpaint error: {str(e)}")]


@app.tool("outpaint")
async def outpaint_tool(prompt: str, direction: str = "right", amount: int = 256,
                        guidance: float = 4.0, steps: int = 30, seed: Optional[int] = None):
    """AI outpaint: extend the canvas in a direction and fill the new area. direction: left, right, top, bottom."""
    try:
        canvas = sessions.get_default_session()
        layer = canvas.layers[canvas.active_layer_index]
        w, h = layer.image.size
        directions = {
            "left":    (w + amount, h, (-amount, 0)),
            "right":   (w + amount, h, (0, 0)),
            "top":     (w, h + amount, (0, 0)),
            "bottom":  (w, h + amount, (0, 0)),
        }
        if direction not in directions:
            return [TextContent(text=f"Invalid direction: {direction}. Use: left, right, top, bottom")]
        target_w, target_h, paste_pos = directions[direction]
        # Create padded canvas
        padded = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
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
        buf = io.BytesIO()
        padded.save(buf, format="PNG")
        await comfy.upload_image(buf.getvalue(), pad_name)
        workflow = build_outpaint_workflow(prompt=prompt, image_filename=pad_name,
                                           target_width=target_w, target_height=target_h,
                                           seed=seed, guidance=guidance, steps=steps)
        result = await run_workflow(workflow)
        if not result:
            return [TextContent(text="Outpaint failed.")]
        img = Image.open(io.BytesIO(result)).convert("RGBA")
        layer.image = img
        canvas.width = target_w
        canvas.height = target_h
        canvas._save_state()
        return [TextContent(text=f"Outpaint {direction} complete ({target_w}x{target_h}).")]
    except Exception as e:
        return [TextContent(text=f"Outpaint error: {str(e)}")]


# ---- Levels / Curves ----

@app.tool("levels")
async def levels_tool(black_point: int = 0, mid_point: float = 1.0, white_point: int = 255):
    """Adjust levels. black_point: 0-255 (input black), mid_point: 0.1-2.0 (gamma), white_point: 0-255 (input white)."""
    try:
        canvas = sessions.get_default_session()
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
async def curves_tool(red: str = "", green: str = "", blue: str = ""):
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
        canvas = sessions.get_default_session()
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
async def upscale_tool(factor: int = 2, model: str = "anime"):
    """Upscale the active layer using an AI upscaling model via ComfyUI. model: 'anime' (RealESRGAN_x4plus_anime_6B) or 'face' (4xFaceUpDAT)."""
    try:
        model_map = {
            "anime": MODEL_UPSCALE_ANIME,
            "face": MODEL_UPSCALE_FACE,
        }
        # Allow direct model name pass-through
        upscale_model = model_map.get(model, model)
        canvas = sessions.get_default_session()
        layer = canvas.layers[canvas.active_layer_index]
        img_name = f"upscale_{uuid.uuid4().hex[:8]}.png"
        buf = io.BytesIO()
        layer.image.save(buf, format="PNG")
        await comfy.upload_image(buf.getvalue(), img_name)
        workflow = build_upscale_workflow(image_filename=img_name, upscale_model=upscale_model)
        result = await run_workflow(workflow)
        if not result:
            return [TextContent(text="Upscale failed.")]
        img = Image.open(io.BytesIO(result)).convert("RGBA")
        layer.image = img
        canvas.width = img.width
        canvas.height = img.height
        canvas._save_state()
        return [TextContent(text=f"Upscaled to {img.width}x{img.height} with model '{upscale_model}'.")]
    except Exception as e:
        return [TextContent(text=f"Upscale error: {str(e)}")]


# ---- Semantic Selection (simple threshold-based, SAM-ready) ----

@app.tool("select_object")
async def select_object_tool(description: str, threshold: int = 128):
    """
    Create a selection mask using simple color/region heuristics.
    description: color name or keyword (red, blue, green, sky, dark, light, white, black).
    Uses PIL getdata/putdata for batch processing (no numpy, no pixel-by-pixel loop).
    """
    try:
        canvas = sessions.get_default_session()
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


# ---- ControlNet ----

@app.tool("controlnet_generate")
async def controlnet_generate(prompt: str, controlnet: str = "depth", strength: float = 0.8,
                               width: int = 1024, height: int = 1024, steps: int = 20,
                               cfg: float = 1.5, seed: Optional[int] = None):
    """Generate an image guided by the current canvas via ControlNet. controlnet: 'depth', 'canny', or 'pose'."""
    try:
        canvas = sessions.get_default_session()
        composite = canvas.composite()
        ctrl_name = f"ctrlnet_{uuid.uuid4().hex[:8]}.png"
        buf = io.BytesIO()
        composite.save(buf, format="PNG")
        await comfy.upload_image(buf.getvalue(), ctrl_name)
        controlnet_map = {
            "depth": "control_v11f1p_sd15_depth_fp16.safetensors",
            "canny": "control_v11p_sd15_canny_fp16.safetensors",
            "pose": "control_v11p_sd15_openpose_fp16.safetensors",
        }
        model_name = controlnet_map.get(controlnet, controlnet)
        workflow = build_controlnet_workflow(prompt=prompt, image_filename=ctrl_name,
                                             controlnet_name=model_name, control_strength=strength,
                                             width=width, height=height, steps=steps, cfg=cfg, seed=seed)
        result = await run_workflow(workflow)
        if not result:
            return [TextContent(text="ControlNet generation failed.")]
        img = Image.open(io.BytesIO(result)).convert("RGBA")
        idx = canvas.add_layer(name=f"ControlNet: {prompt[:30]}", image=img)
        return [TextContent(text=f"ControlNet generation complete. Layer {idx}.")]
    except Exception as e:
        return [TextContent(text=f"ControlNet error: {str(e)}")]


# ---- Style Transfer ----

@app.tool("style_transfer")
async def style_transfer_tool(prompt: str, style_path: str, strength: float = 0.8,
                               width: int = 1024, height: int = 1024, steps: int = 20,
                               seed: Optional[int] = None):
    """Generate an image with the style of a reference image. Uses Redux StyleModel for style transfer."""
    try:
        canvas = sessions.get_default_session()
        # Upload content (current canvas)
        content_name = f"style_content_{uuid.uuid4().hex[:8]}.png"
        composite = canvas.composite()
        buf = io.BytesIO()
        composite.save(buf, format="PNG")
        await comfy.upload_image(buf.getvalue(), content_name)
        # Upload style reference
        style_name = f"style_ref_{uuid.uuid4().hex[:8]}.png"
        style_img = Image.open(style_path).convert("RGB")
        buf2 = io.BytesIO()
        style_img.save(buf2, format="PNG")
        await comfy.upload_image(buf2.getvalue(), style_name)
        workflow = build_style_transfer_workflow(prompt=prompt, content_filename=content_name,
                                                  style_filename=style_name, style_strength=strength,
                                                  width=width, height=height, steps=steps, seed=seed)
        result = await run_workflow(workflow)
        if not result:
            return [TextContent(text="Style transfer failed.")]
        img = Image.open(io.BytesIO(result)).convert("RGBA")
        idx = canvas.add_layer(name=f"Style: {prompt[:30]}", image=img)
        return [TextContent(text=f"Style transfer complete. Layer {idx}.")]
    except Exception as e:
        return [TextContent(text=f"Style transfer error: {str(e)}")]


# ======================================================================= #
#  Main
# ======================================================================= #

if __name__ == "__main__":
    app.run()