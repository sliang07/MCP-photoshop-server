"""Reference-guided editing with ComfyUI core nodes and MCP image previews."""

import base64
import io
import json
import secrets
from pathlib import Path
from typing import Literal

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps
from mcp.types import ImageContent, TextContent

from config import (MODEL_ANIMA, MODEL_ANIMA_TEXT_ENCODER, MODEL_ANIMA_VAE,
                    MODEL_FLUX2, MODEL_FLUX2_TEXT_ENCODER, MODEL_FLUX2_VAE)


def model_options(info, loader, field):
    """Support both legacy combo lists and current ComfyUI COMBO schemas."""
    spec = info.get(loader, {}).get("input", {}).get("required", {}).get(field, [])
    if spec and isinstance(spec[0], list):
        return spec[0]
    if len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options", [])
    return []


def resolve_model(options, *basenames):
    for basename in basenames:
        for option in options:
            if option.replace("\\", "/").split("/")[-1] == basename:
                return option
    return None


def edit_profiles(info):
    return model_profiles(info, "editing")


def model_profiles(info, task):
    """Resolve installed weights and check the nodes needed for this task."""
    if task not in ("generation", "editing", "outpaint"):
        raise ValueError(f"Unknown model task: {task}")
    unets = model_options(info, "UNETLoader", "unet_name")
    clips = model_options(info, "CLIPLoader", "clip_name")
    vaes = model_options(info, "VAELoader", "vae_name")
    profiles = {}
    for name, model_names, clip_names, vae_name, refs, steps, cfg, use_for in (
        ("flux2", (MODEL_FLUX2, "flux-2-klein-9b-fp8.safetensors"),
         (MODEL_FLUX2_TEXT_ENCODER, "qwen_3_8b.safetensors"), MODEL_FLUX2_VAE, None, 4, 1.0,
         "Fast generation, instruction edits and reference-guided edits (Klein 9B distilled)."),
        ("qwen21", ("qwen_image_2.1_int8_convrot.safetensors", "qwen_image_2.1_bf16.safetensors"),
          ("qwen3vl_8b_int8_convrot.safetensors", "qwen3vl_8b_w4a8.safetensors", "qwen3vl_8b_bf16.safetensors"),
          "qwen_image_2.1_vae_bf16.safetensors", 15, 30, 3.0,
          "Detailed generation, typography, transparent output, instruction and reference-guided edits."),
        ("anima", (MODEL_ANIMA, "anima-base-v1.0.safetensors"),
         (MODEL_ANIMA_TEXT_ENCODER,), MODEL_ANIMA_VAE, 0, 30, 4.0,
         "Anime and illustration text-to-image only; not instruction editing or typography."),
    ):
        if name == "anima" and task != "generation":
            continue
        profile = {
            "model": resolve_model(unets, *model_names),
            "clip": resolve_model(clips, *clip_names),
            "vae": resolve_model(vaes, vae_name),
            "default_steps": steps, "default_cfg": cfg, "use_for": use_for,
            "sampler": "er_sde" if name == "anima" else "euler",
            "scheduler": "Flux2Scheduler" if name == "flux2" else "simple",
            "denoise": 1.0,
        }
        if task != "generation":
            profile["max_additional_references"] = refs
        missing = [key for key in ("model", "clip", "vae") if not profile[key]]
        if not missing:
            if task == "generation":
                graph = build_generation_workflow(name, profile, "check", width=512, height=512, seed=1)
            else:
                graph = build_edit_workflow(name, profile, "check", ["check.png"], 512, 512, steps, 1)
            missing = sorted({n["class_type"] for n in graph.values()} - info.keys())
        profile["missing"] = missing
        profile["available"] = not missing
        profiles[name] = profile
    return profiles


def build_generation_workflow(backend, profile, prompt, negative_prompt="", width=1024, height=1024,
                              steps=None, cfg=None, seed=None):
    """Native ComfyUI text-to-image graphs, with per-model sampling defaults."""
    if backend not in ("flux2", "qwen21", "anima"):
        raise ValueError("model must be flux2, qwen21 or anima")
    steps = profile["default_steps"] if steps is None else steps
    cfg = profile["default_cfg"] if cfg is None else cfg
    seed = secrets.randbits(63) if seed is None else seed
    graph = {}

    def node(kind, **inputs):
        key = str(len(graph) + 1)
        graph[key] = {"class_type": kind, "inputs": inputs}
        return [key, 0]

    model = node("UNETLoader", unet_name=profile["model"], weight_dtype="default")
    clip_type = {"flux2": "flux2", "qwen21": "qwen_image", "anima": "stable_diffusion"}[backend]
    clip = node("CLIPLoader", clip_name=profile["clip"], type=clip_type)
    vae = node("VAELoader", vae_name=profile["vae"])
    if backend == "qwen21":
        encoded = node("TextEncodeQwenImage21", clip=clip, prompt=prompt,
                       negative_prompt=negative_prompt, resolution=1024)
        positive, negative = encoded, [encoded[0], 1]
    else:
        positive = node("CLIPTextEncode", clip=clip, text=prompt)
        negative = node("CLIPTextEncode", clip=clip, text=negative_prompt)
    latent = node("EmptyFlux2LatentImage" if backend == "flux2" else "EmptyLatentImage",
                  width=width, height=height, batch_size=1)
    if backend == "flux2":
        guider = node("CFGGuider", model=model, positive=positive, negative=negative, cfg=cfg)
        noise = node("RandomNoise", noise_seed=seed)
        sigmas = node("Flux2Scheduler", steps=steps, width=width, height=height)
        sampler = node("KSamplerSelect", sampler_name=profile["sampler"])
        sampled = node("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler,
                       sigmas=sigmas, latent_image=latent)
    else:
        sampled = node("KSampler", model=model, positive=positive, negative=negative,
                       latent_image=latent, seed=seed, steps=steps, cfg=cfg,
                       sampler_name=profile["sampler"],
                       scheduler=profile["scheduler"], denoise=profile["denoise"])
    decoded = node("VAEDecode", samples=sampled, vae=vae)
    node("SaveImage", images=decoded, filename_prefix=f"mcp_{backend}")
    return graph


def build_edit_workflow(backend, profile, prompt, images, width, height, steps, seed, resolution=0):
    """Use reference conditioning, separate from the sampled output latent."""
    if backend not in ("flux2", "qwen21"):
        raise ValueError("Instruction editing requires qwen21 or flux2; Anima supports generation only")
    graph = {}

    def node(kind, **inputs):
        key = str(len(graph) + 1)
        graph[key] = {"class_type": kind, "inputs": inputs}
        return [key, 0]

    model = node("UNETLoader", unet_name=profile["model"], weight_dtype="default")
    clip = node("CLIPLoader", clip_name=profile["clip"], type="flux2" if backend == "flux2" else "qwen_image")
    vae = node("VAELoader", vae_name=profile["vae"])
    loaded = []
    for filename in images:
        image = node("LoadImage", image=filename)
        if backend == "qwen21":
            image = node("JoinImageWithAlpha", image=image, alpha=[image[0], 1])
        loaded.append(image)
    if backend == "flux2":
        positive = node("CLIPTextEncode", clip=clip, text=prompt)
        for image in loaded:
            latent = node("VAEEncode", pixels=image, vae=vae)
            positive = node("ReferenceLatent", conditioning=positive, latent=latent)
        guider = node("BasicGuider", model=model, conditioning=positive)
        noise = node("RandomNoise", noise_seed=seed)
        sigmas = node("Flux2Scheduler", steps=steps, width=width, height=height)
        sampler = node("KSamplerSelect", sampler_name=profile["sampler"])
        empty = node("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)
        sampled = node("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler,
                       sigmas=sigmas, latent_image=empty)
    else:
        model = node("QwenImage21Cache", model=model, device="auto", dtype="default")
        encoder = node("TextEncodeQwenImage21", clip=clip, vae=vae, prompt=prompt,
                       negative_prompt="", resolution=resolution,
                       **{f"images.image_{i + 1}": image for i, image in enumerate(loaded)})
        positive, negative, latent = encoder, [encoder[0], 1], [encoder[0], 2]
        sampled = node("KSampler", model=model, positive=positive, negative=negative,
                       latent_image=latent, seed=seed, steps=steps, cfg=profile["default_cfg"],
                       sampler_name=profile["sampler"], scheduler=profile["scheduler"], denoise=profile["denoise"])
    decoded = node("VAEDecode", samples=sampled, vae=vae)
    node("SaveImage", images=decoded, filename_prefix="mcp_instruction_edit")
    return graph


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def preview_content(image, max_size=1024):
    if not 64 <= max_size <= 2048:
        raise ValueError("max_size must be between 64 and 2048")
    image = image.copy()
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return ImageContent(type="image", mimeType="image/png",
                        data=base64.b64encode(png_bytes(image)).decode("ascii"))


def make_edit_mask(size, mask_path=None, region=None, feather=0):
    """A separate edit mask never changes layer visibility. White means edit."""
    if mask_path and region is not None:
        raise ValueError("Use either mask_path or region, not both")
    if not 0 <= feather <= 128:
        raise ValueError("feather must be between 0 and 128 pixels")
    if mask_path:
        with Image.open(mask_path) as opened:
            mask = ImageOps.exif_transpose(opened).convert("L")
        if mask.size != size:
            raise ValueError("Mask dimensions must match the canvas")
    elif region is not None:
        if len(region) != 4 or any(type(v) is not int for v in region):
            raise ValueError("region must be [x, y, width, height] in canvas pixels")
        x, y, w, h = region
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > size[0] or y + h > size[1]:
            raise ValueError("region must fit inside the canvas")
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rectangle((x, y, x + w - 1, y + h - 1), fill=255)
    else:
        if feather:
            raise ValueError("feather requires mask_path or region")
        return None
    if mask.getbbox() is None:
        raise ValueError("Edit mask is empty")
    if feather:
        # Inward feathering keeps all originally black pixels untouched.
        mask = ImageChops.multiply(mask, mask.filter(ImageFilter.GaussianBlur(feather)))
    return mask


def prepare_image(image, max_side, multiple=32):
    image = image.convert("RGBA")
    scale = min(1.0, max_side / max(image.size))
    size = tuple(max(multiple, round(v * scale / multiple) * multiple) for v in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


SAM3_CHECKPOINT = "sam3.pt"
SAM3_1_CHECKPOINT = "sam3.1_multiplex_fp16.safetensors"


def sam3_capabilities(info):
    """Report SAM 3 readiness from live node info without starting it or loading GPU models."""
    ckpt = resolve_model(model_options(info, "ImageOnlyCheckpointLoader", "ckpt_name"), SAM3_CHECKPOINT)
    ckpt_1 = resolve_model(model_options(info, "CheckpointLoaderSimple", "ckpt_name"), SAM3_1_CHECKPOINT)
    node_present = "SAM3_Detect" in info
    return {
        "available": bool(ckpt) and node_present,
        "checkpoint": ckpt,
        "sam3_1": ckpt_1,
        "text_prompt_available": bool(ckpt_1) and node_present,
        "node_present": node_present,
    }


def count_selection_regions(mask, max_side=256):
    """Approximate object count: connected components of a downsampled binary mask."""
    if mask.getbbox() is None:
        return 0
    small = mask.copy()
    small.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    w, h = small.size
    data = small.load()
    seen = bytearray(w * h)
    count = 0
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if seen[idx] or data[x, y] < 128:
                continue
            count += 1
            seen[idx] = 1
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h:
                        nidx = ny * w + nx
                        if not seen[nidx] and data[nx, ny] >= 128:
                            seen[nidx] = 1
                            stack.append((nx, ny))
    return count


def build_semantic_select_workflow(ckpt, image, point=None, box=None, prompt=None, threshold=0.5, refine=2):
    """SAM3 text/point/box detection -> union mask saved as an image (via MaskToImage).

    Without a text prompt the model loads through ImageOnlyCheckpointLoader
    (original sam3.pt). With a text prompt it loads through CheckpointLoaderSimple
    (SAM 3.1 checkpoint - its text-encoder shape is what the 0.33.0 clip wrapper
    expects) and the prompt is encoded via CLIPTextEncode into the node's optional
    ``conditioning`` input. Point/box prompts can combine with either path.
    """
    graph = {}

    def node(kind, **inputs):
        key = str(len(graph) + 1)
        graph[key] = {"class_type": kind, "inputs": inputs}
        return [key, 0]

    source = node("LoadImage", image=image)
    detect_inputs = {
        "image": source, "threshold": threshold,
        "refine_iterations": refine, "individual_masks": False,
    }
    if prompt is not None:
        loader = node("CheckpointLoaderSimple", ckpt_name=ckpt)
        text = node("CLIPTextEncode", clip=[loader[0], 1], text=prompt)
        detect_inputs["model"] = [loader[0], 0]
        detect_inputs["conditioning"] = text
    else:
        detect_inputs["model"] = node("ImageOnlyCheckpointLoader", ckpt_name=ckpt)
    if point is not None:
        detect_inputs["positive_coords"] = json.dumps([{"x": int(point[0]), "y": int(point[1])}])
    if box is not None:
        x, y, w, h = (int(v) for v in box)
        detect_inputs["bboxes"] = {"x": x, "y": y, "width": w, "height": h}
    mask = node("SAM3_Detect", **detect_inputs)
    visual = node("MaskToImage", mask=mask)
    node("SaveImage", images=visual, filename_prefix="mcp_semantic_select")
    return graph


def register_editing_tools(app, comfy, sessions, run_workflow):
    @app.tool("preview_canvas")
    async def preview_canvas(max_size: int = 1024, session_id: str = "default"):
        """See the current canvas as an MCP image. Use before editing and to inspect results."""
        canvas = sessions.get_or_create(session_id)
        return [TextContent(type="text", text=json.dumps(canvas.get_info())),
                preview_content(canvas.composite(), max_size)]

    @app.tool("get_editing_capabilities")
    async def get_editing_capabilities(start_if_needed: bool = True):
        """Check installed models for generation, editing and outpainting. Automatically starts ComfyUI when needed.

        No manual startup is required. This check does not load generation models.
        Set start_if_needed=False only for a passive check; offline does not mean editing is unavailable.
        Choose qwen21 for detail/text/alpha, flux2 for speed, anima for anime generation.
        Anima is generation-only; upscaling and semantic selection use their dedicated models.
        """
        try:
            if start_if_needed:
                await comfy.start_comfyui()
            info = await comfy.get_object_info()
            stats = await comfy.get_system_stats()
        except Exception as error:
            message = ("Automatic startup or capability check failed; inspect the error and ComfyUI startup configuration."
                       if start_if_needed else
                       "ComfyUI is stopped or unreachable. Call this tool with start_if_needed=True, or call the requested editing tool; it automatically starts ComfyUI.")
            return [TextContent(type="text", text=json.dumps({"connected": False,
                    "auto_start_attempted": start_if_needed, "error": str(error), "message": message}))]
        report = {
            "connected": True, "comfyui_version": stats.get("system", {}).get("comfyui_version"),
            "devices": stats.get("devices", []), "editing": edit_profiles(info),
            "generation": model_profiles(info, "generation"),
            "outpaint": model_profiles(info, "outpaint"),
            "default_edit_backend": "qwen21",
            "default_generation_model": "flux2", "default_outpaint_backend": "flux2",
            "models": {"diffusion_models": model_options(info, "UNETLoader", "unet_name"),
                       "text_encoders": model_options(info, "CLIPLoader", "clip_name"),
                       "vae": model_options(info, "VAELoader", "vae_name"),
                       "upscalers": model_options(info, "UpscaleModelLoader", "model_name")},
            "sam3": sam3_capabilities(info),
            "notes": ["Availability checks files and nodes; it does not benchmark generation or guarantee VRAM fit.",
                      "semantic_select runs SAM 3 text/point/box prompts on the canvas (point/box use sam3.pt; text prompts use sam3.1_multiplex_fp16.safetensors, both in models/checkpoints). select_object stays the fast color-heuristic fallback.",
                      "edit_image supports independent white-to-edit masks and returns a new undoable layer.",
                      "Legacy select_object uses color heuristics, not semantic object segmentation.",
                      "outpaint supports flux2 or qwen21 reference conditioning; original pixels are restored after generation. Style/identity guidance uses edit_image reference_paths.",
                      "batch_generate queues a whole job list on one WebSocket connection and exports each result to disk as it completes (see the GPU batching rule).",
                      "The GPU may be shared with qwen38. Never stop that LLM container without explicit approval.",
                      "Full GPU batching procedure: MEMORY.md, section 'GPU Contention & Batching Rule'."],
        }
        return [TextContent(type="text", text=json.dumps(report, indent=2))]

    @app.tool("edit_image")
    async def edit_image(prompt: str, backend: Literal["qwen21", "flux2"] = "qwen21", reference_paths: list[str] | None = None,
                         mask_path: str | None = None, region: list[int] | None = None,
                         feather: int = 0, steps: int | None = None, seed: int | None = None,
                         max_side: int = 1024, session_id: str = "default", timeout: int | None = None):
        """Instruction-edit the canvas: remove objects, replace backgrounds, restyle, or use identity references.

        Automatically starts ComfyUI when needed; no manual startup is required.

        backend: qwen21 (default, Qwen Image 2.1, custom 30 steps/CFG 3, RGBA)
        or flux2 (Klein 9B distilled, 4 steps/CFG 1). Both use Euler.
        Both support object removal, background replacement, restyling and references.
        Choose qwen21 for typography/alpha or flux2 for speed. Anima is generation-only.
        Qwen 2511 and the original Qwen editor have been retired. Open the source image first.
        Image 1 is the canvas; images 2 onward are reference_paths, in order. For Qwen use
        <image1>, <image2>, etc. Describe the change and the identity/features to preserve.
        Qwen recommends at most 10 total images; this ComfyUI node accepts 16 including the mask.
        max_side limits working resolution (1024 default; use 2048 for fine detail), rounded to the model grid.
        Optional mask_path is grayscale, white=edit, black=preserve; or region=[x,y,width,height].
        The mask is sent to the model as the last image and also enforces exact preservation outside it.
        Layer visibility masks/selections are not edit masks: pass mask_path or region explicitly.
        Returns one undoable replacement layer and a preview; original layers remain hidden.
        Inspect the returned preview before retrying; undo a failed attempt before another edit.
        timeout defaults to 1800 seconds for Qwen. Identity preservation is model-dependent.
        """
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if backend not in ("flux2", "qwen21"):
            raise ValueError("backend must be qwen21 or flux2; qwen and qwen2511 have been retired")
        if max_side < 32:
            raise ValueError("max_side must be at least 32")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        if seed is None:
            seed = secrets.randbits(63)
        if not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        canvas = sessions.get(session_id)
        if canvas is None:
            raise ValueError("Open an image or create a canvas in this session before calling edit_image")
        original_state = canvas.undo_stack[-1]
        source = canvas.composite()
        mask = make_edit_mask(source.size, mask_path, region, feather)
        references = reference_paths or []
        limit = (15 - int(mask is not None)) if backend == "qwen21" else None
        if limit is not None and len(references) > limit:
            raise ValueError(f"{backend} supports at most {limit} additional references")
        multiple = 32 if backend == "qwen21" else 16
        images = [prepare_image(source, max_side, multiple)]
        for path in references:
            with Image.open(Path(path)) as reference:
                images.append(prepare_image(ImageOps.exif_transpose(reference), max_side, multiple))
        image_name = (lambda i: f"<image{i}>") if backend == "qwen21" else (lambda i: f"image {i}")
        instruction = f"Edit {image_name(1)}. {prompt.strip()}\nPreserve content and identity not affected by the requested change."
        if references:
            instruction += " Use the reference images only for the features requested; keep the canvas as the base image."
        if mask is not None:
            images.append(mask.resize(images[0].size, Image.Resampling.NEAREST).convert("RGBA"))
            instruction += (f"\n{image_name(len(images))} is an edit mask for {image_name(1)}: "
                            "white marks the area to change, black marks the area to preserve. "
                            "Apply the requested change only in the white area. Do not render the mask in the result.")
        # Readiness checks happen before uploads and expensive inference.
        await comfy.start_comfyui()
        info = await comfy.get_object_info()
        profile = edit_profiles(info)[backend]
        if not profile["available"]:
            raise ValueError(f"{backend} is unavailable: {profile['missing']}. Call get_editing_capabilities.")
        steps = profile["default_steps"] if steps is None else steps
        if steps < 1:
            raise ValueError("steps must be positive")
        files = []
        for image in images:
            name = f"mcp_edit_{secrets.token_hex(12)}.png"
            files.append(await comfy.upload_image(png_bytes(image), name))
        width, height = images[0].size
        workflow = build_edit_workflow(backend, profile, instruction, files, width, height, steps, seed)
        result = await run_workflow(workflow, timeout=timeout or (1800 if backend == "qwen21" else None))
        if not result:
            raise RuntimeError("ComfyUI returned no image; the canvas was not modified")
        with Image.open(io.BytesIO(result)) as opened:
            edited = opened.convert("RGBA")
        if edited.size != source.size:
            edited = edited.resize(source.size, Image.Resampling.LANCZOS)
        if sessions.get_or_create(session_id) is not canvas or canvas.undo_stack[-1] is not original_state:
            raise RuntimeError("Canvas changed during generation; result was not applied. It remains in ComfyUI output.")
        if mask is not None:
            edited = Image.composite(edited, source, mask)
        for layer in canvas.layers:
            layer.visible = False
        index = canvas.add_layer(name=f"Edit ({backend}, seed {seed}): {prompt[:40]}", image=edited)
        report = {"layer": index, "backend": backend, "model": profile["model"], "seed": seed,
                  "steps": steps, "cfg": profile["default_cfg"],
                  "sampler": profile["sampler"], "scheduler": profile["scheduler"],
                  "reference_count": len(references), "masked": mask is not None,
                  "generation_size": [width, height], "canvas_size": list(source.size),
                  "effective_prompt": instruction}
        return [TextContent(type="text", text=json.dumps(report)), preview_content(canvas.composite())]

    @app.tool("semantic_select")
    async def semantic_select(prompt: str | None = None, point: list[int] | None = None, box: list[int] | None = None,
                              threshold: float = 0.5, refine: int = 2, timeout: int | None = None,
                              session_id: str = "default"):
        """Select a semantic object on the active layer using SAM 3 (text, point, and/or box prompts).

        Automatically starts ComfyUI when needed; no manual startup is required.

        prompt: text description of the object (e.g. "red circle") - runs the SAM 3.1
        checkpoint's text encoder (needs sam3.1_multiplex_fp16.safetensors in
        models/checkpoints). point: [x, y] pixel on the target object. box: [x, y,
        width, height] in canvas pixels. Provide at least one (any combination is
        allowed; point/box alone use sam3.pt). Runs SAM3_Detect on the current
        canvas, sets the resulting union mask as the active layer's selection, and
        returns a mask preview plus coverage/bbox/object-count info. Slower than
        select_object (loads a ~3.4GB model; CPU inference takes minutes).
        """
        if prompt is not None and not prompt.strip():
            raise ValueError("prompt must be a non-empty text description")
        if point is None and box is None and not prompt:
            raise ValueError("Provide a text prompt, a point [x, y], a box [x, y, width, height], or a combination")
        if point is not None and (len(point) != 2 or any(type(v) is not int for v in point)):
            raise ValueError("point must be [x, y] integers in canvas pixels")
        if box is not None and (len(box) != 4 or any(type(v) is not int for v in box)):
            raise ValueError("box must be [x, y, width, height] integers in canvas pixels")
        canvas = sessions.get_or_create(session_id)
        original_state = canvas.undo_stack[-1]
        source = canvas.composite()
        width, height = source.size
        if point is not None and not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError(f"point {point} is outside the {width}x{height} canvas")
        if box is not None:
            bx, by, bw, bh = box
            if min(bx, by) < 0 or min(bw, bh) <= 0 or bx + bw > width or by + bh > height:
                raise ValueError(f"box {box} must fit inside the {width}x{height} canvas")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if not 0 <= refine <= 5:
            raise ValueError("refine must be between 0 and 5")
        # Readiness checks happen before uploads and expensive inference.
        await comfy.start_comfyui()
        info = await comfy.get_object_info()
        caps = sam3_capabilities(info)
        if not caps["node_present"]:
            raise ValueError("SAM 3 is unavailable: missing the SAM3_Detect node. Call get_editing_capabilities.")
        if prompt:
            if not caps["sam3_1"]:
                raise ValueError(f"SAM 3.1 text prompting is unavailable: missing checkpoint {SAM3_1_CHECKPOINT} in "
                                 f"models/checkpoints. Call get_editing_capabilities.")
            ckpt = caps["sam3_1"]
        else:
            if not caps["checkpoint"]:
                raise ValueError(f"SAM 3 is unavailable: missing checkpoint {SAM3_CHECKPOINT} in models/checkpoints. "
                                 f"Call get_editing_capabilities.")
            ckpt = caps["checkpoint"]
        filename = await comfy.upload_image(png_bytes(source.convert("RGB")), f"mcp_semantic_{secrets.token_hex(12)}.png")
        workflow = build_semantic_select_workflow(ckpt, filename, point, box, prompt, threshold, refine)
        result = await run_workflow(workflow, timeout=timeout)
        if not result:
            raise RuntimeError("ComfyUI returned no mask; the canvas was not modified")
        with Image.open(io.BytesIO(result)) as opened:
            mask = opened.convert("L")
        if mask.size != source.size:
            mask = mask.resize(source.size, Image.Resampling.LANCZOS)
        if sessions.get_or_create(session_id) is not canvas or canvas.undo_stack[-1] is not original_state:
            raise RuntimeError("Canvas changed during detection; result was not applied.")
        bbox = mask.getbbox()
        if bbox is None:
            report = {"tool": "semantic_select", "selected": False, "checkpoint": ckpt,
                      "prompt": {"text": prompt, "point": point, "box": box},
                      "message": "No object detected for the given prompt; the active layer was not changed."}
            return [TextContent(type="text", text=json.dumps(report)), preview_content(mask)]
        layer = canvas.layers[canvas.active_layer_index]
        layer.mask = mask
        canvas._save_state()
        white = sum(mask.histogram()[128:])
        report = {"tool": "semantic_select", "selected": True, "checkpoint": ckpt,
                  "prompt": {"text": prompt, "point": point, "box": box, "threshold": threshold, "refine": refine},
                  "layer": canvas.active_layer_index, "mask_size": [width, height],
                  "bbox": list(bbox), "selected_pixels": white,
                  "coverage": round(white / (width * height), 4),
                  "objects": count_selection_regions(mask)}
        return [TextContent(type="text", text=json.dumps(report)), preview_content(mask)]
