# MCP Photoshop Server

A local MCP (Model Context Protocol) server that combines **Photoshop-style image editing** with **AI image generation** powered by ComfyUI.

> **Designed for use with [Cline](https://github.com/cline/cline)** — an AI-powered coding assistant. This server integrates as an MCP tool provider to enable image generation and editing directly from your Cline workflow.

## Instruction editing upgrade

See [EDITING_UPGRADE.md](EDITING_UPGRADE.md) for the verified local inventory, model downloads, known legacy limitations, and usage.

- `preview_canvas` returns an image the assistant can inspect.
- `get_editing_capabilities` reports live model/node availability.
- `edit_image` adds instruction and reference-guided edits through native FLUX.2 or Qwen workflows, independent edit masks, protected outside pixels, previews, and undoable layers.
- `upscale` now honors its size factor and preserves layer/mask alignment and transparency.

`img2img` is a legacy FLUX.2 denoising transform, despite its old Kontext naming. Existing ControlNet/Redux style-transfer workflows require further repair; use `edit_image` for reference style guidance.

Reconnect the MCP server after saving any in-memory work to load the new tools.

## Features

### Canvas Management
- `new_canvas` — Create a blank canvas with custom dimensions and background color
- `open_image` — Load an existing image file
- `export` — Save to PNG/JPG/WEBP or return base64 data
- `get_info` — View canvas dimensions, layers, undo/redo state

### AI Image Generation & Editing
- `generate_image` — Text-to-image via Flux2 (photorealistic) or ANIMA (anime)
- `img2img` — Legacy FLUX.2 denoising transform; prefer `edit_image` for instructions
- `character_transform` — Legacy pose/expression denoising transform; use `edit_image` references for identity guidance
- `inpaint` — AI fill masked region (requires mask from select_rect/select_ellipse)
- `outpaint` — AI extend canvas in a direction (left, right, top, bottom)

### Instruction Editing
- `edit_image` — Instruction and reference-guided editing: FLUX.2 (fast, ~4 steps) or Qwen Image Edit 2511 FP8 ("qwen2511", ~40 steps); up to 2 extra references, optional white-to-edit mask, new undoable layer
- `get_editing_capabilities` — Live ComfyUI model/node availability for every editing backend (notes also carry the GPU batching rule)
- `preview_canvas` — Render the current canvas so the assistant can inspect results

### AI-Guided Generation
- `controlnet_generate` — Generate image guided by current canvas (depth/canny/pose)
- `style_transfer` — Generate image with style of a reference image (Redux StyleModel)

### Deterministic Editing (Pillow — no GPU needed)
- `crop`, `resize`, `rotate`, `flip` — Transform operations
- `adjust` — Brightness, contrast, saturation, hue, sharpness
- `levels` — Black point, mid point (gamma), white point adjustment
- `curves` — Per-channel tone curves (R/G/B) with control points
- `apply_filter` — 17 filters: blur, gaussian_blur, sharpen, contour, detail, edge_enhance, edge_enhance_more, find_edges, smooth, smooth_more, emboss, pixelate, posterize, solarize, invert, grayscale, sepia
- `add_text` — Text overlay with font, color, stroke support

### Upscaling
- `upscale` — AI upscaling via ComfyUI (ESRGAN/SUPIR models)

### Layer System
- `add_layer`, `select_layer`, `delete_layer`, `merge_down`, `reorder_layer`
- `set_blend_mode` — 12 modes: normal, multiply, screen, overlay, darken, lighten, color_dodge, color_burn, hard_light, soft_light, difference, exclusion
- `set_layer_opacity` — Per-layer transparency

### Selections & Masks
- `select_rect` — Rectangular mask
- `select_ellipse` — Elliptical mask
- `select_object` — Heuristic color/region selection (red, blue, sky, dark, etc.)
- `clear_mask` — Remove layer mask

### History
- `undo` / `redo` — Full operation history (up to 20 steps)

### System
- `get_comfyui_status` — Check ComfyUI connection
- `clear_vram` — Free GPU memory

> **GPU batching rule:** the host GPU is shared with the `qwen38` LLM docker and Open WebUI. For multiple or long ComfyUI generations, queue the whole batch to run in the background, then ask the user for explicit approval to stop the `qwen38` container (stopping it ends the LLM session; the batch keeps running on the host). `searxng` is CPU-only and never needs stopping. Full procedure: `MEMORY.md` → "GPU Contention & Batching Rule".

## Installation

### Prerequisites
- **Python 3.10+**
- **ComfyUI** installed (auto-start supported — the server automatically launches ComfyUI when needed via `COMFYUI_PYTHON` + `COMFYUI_MAIN` env vars. You can also start ComfyUI manually on port 8188 if preferred.)
- **⚠️ VRAM Warning:** Running ComfyUI without auto-kill causes major OOM on shared GPUs (e.g., 32GB GPU with vLLM). Set `COMFYUI_AUTO_KILL=1` (recommended) with `COMFYUI_IDLE_TIMEOUT=5` to kill ComfyUI 5 seconds after each task, preventing resource exhaustion and Cline freezes.
- Required models installed in ComfyUI:

  **Flux2 Klein (photorealistic generation):**
  - `flux-2-klein-9b.safetensors` (diffusion_models) — **IMPORTANT: use 4-6 steps max**. Image quality actively diminishes after ~6 steps (artifacts, over-smoothing) and latency increases linearly with each additional step.
  - `qwen_3_8b_fp8mixed.safetensors` (text_encoders) — Flux2 Klein text encoder
  - `flux2-vae.safetensors` (vae) — Flux2 Klein VAE

  **ANIMA (anime generation):**
  - `anima-aesthetic-v1.1.safetensors` (diffusion_models)
  - `qwen_3_06b_base.safetensors` (text_encoders)
  - `qwen_image_vae.safetensors` (vae)

   **Flux Kontext (img2img, inpaint, outpaint):**
   - `flux1-dev-kontext_fp8_scaled.safetensors` (diffusion_models)

   **Upscaling:**
   - `RealESRGAN_x4plus_anime_6B.pth` (upscale_models) — Anime upscaling
   - `4xFaceUpDAT.pth` (upscale_models) — Face upscaling

   **ControlNet:**
   - `control_v11f1p_sd15_depth_fp16.safetensors` (controlnet) — Depth guidance
   - `control_v11p_sd15_canny_fp16.safetensors` (controlnet) — Canny guidance
   - `control_v11p_sd15_openpose_fp16.safetensors` (controlnet) — Pose guidance

   **Style Transfer:**
   - `flux1-redux-dev.safetensors` (style_models) — Redux style transfer

### Install Dependencies
```bash
cd mcp-photoshop-server
pip install -r requirements.txt
```

### Configure MCP Client
Add to your MCP client configuration (e.g., Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "photoshop": {
      "command": "python",
      "args": ["<path-to-server>/server.py"],
      "env": {
        "COMFYUI_URL": "http://127.0.0.1:8188",
        "COMFYUI_PYTHON": "<path-to-comfyui>/python_embeded/python.exe",
        "COMFYUI_MAIN": "<path-to-comfyui>/ComfyUI/main.py",
        "COMFYUI_AUTO_KILL": "1",
        "COMFYUI_IDLE_TIMEOUT": "5",
        "VRAM_PRESSURE_THRESHOLD_MB": "8192"
      }
    }
  }
}
```

### Environment Variables
See [`.env_example`](.env_example) for a complete reference. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI API endpoint |
| `COMFYUI_START_CMD` | *(optional)* | Path to ComfyUI start .bat (alternative to COMFYUI_PYTHON + COMFYUI_MAIN) |
| `COMFYUI_PYTHON` | *(required for auto-start)* | Path to ComfyUI's embedded `python.exe` |
| `COMFYUI_MAIN` | *(required for auto-start)* | Path to ComfyUI's `main.py` |
| `COMFYUI_ARGS` | `--windows-standalone-build` | Extra startup arguments |
| `COMFYUI_AUTO_KILL` | `1` (recommended) | Kill after generation (`0`=VRAM pressure mode, `1`=idle timeout mode) |
| `COMFYUI_IDLE_TIMEOUT` | `5` | Seconds of inactivity before auto-kill (when AUTO_KILL=1) |
| `COMFYUI_START_TIMEOUT` | `180` | Seconds to wait for ComfyUI to start |
| `VRAM_PRESSURE_THRESHOLD_MB` | `8192` | Kill ComfyUI if free VRAM drops below this (MB) |
| `WEBSOCKET_TIMEOUT` | `600` | Seconds to wait for workflow completion |
| `MAX_UNDO_STEPS` | `20` | Maximum undo history entries |
| `LOG_LEVEL` | `WARNING` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

## Architecture

```
mcp-photoshop-server/
├── server.py           # MCP server + 40 tool registrations (FastMCP initialize instructions carry the GPU batching rule)
◜─── editing.py          # Instruction editing backends (FLUX.2 / Qwen 2511) + live capabilities
├── config.py           # Configuration & model names
├── comfy_client.py     # ComfyUI API client (REST + WebSocket)
├── canvas.py           # Layered document with blend modes + undo/redo
├── session.py          # Per-session document management
├── requirements.txt    # Python dependencies
├── MEMORY.md           # Project memory bank
◜─── EDITING_UPGRADE.md  # Editing backend model guide + live verification evidence
◜─── tests/              # Regression suite (17 tests)
◜─── verification/       # Live GPU verification scripts + artifacts
├── .env_example        # Environment variable template
├── .gitignore          # Git ignore rules
└── README.md
```

### Design Philosophy
- **AI operations** (generation, inpainting, transforms, ControlNet, style) → routed through ComfyUI
- **Deterministic edits** (crop, resize, adjustments, filters) → executed locally with Pillow (fast, no GPU)
- **Layered document model** → mirrors Photoshop's mental model with blend modes, masks, and undo history

## Usage Examples

### Generate and Edit
```
1. generate_image(prompt="a cat wearing sunglasses on a beach")
2. adjust(brightness=1.2, saturation=1.3)
3. apply_filter(name="sharpen")
4. export(path="output/cat.png")
```

### Layer Compositing
```
1. open_image(path="background.jpg")
2. add_layer(name="overlay", source_path="overlay.png")
3. set_blend_mode(mode="multiply")
4. set_layer_opacity(opacity=0.7)
5. export(path="output/composite.png")
```

### AI Edit Workflow
```
1. open_image(path="portrait.jpg")
2. edit_image(prompt="change background to a forest at sunset", backend="flux2")
3. undo  # if not satisfied
4. edit_image(prompt="match the jacket color to the reference", backend="qwen2511", reference_paths=["ref.jpg"])
5. preview_canvas()
6. adjust(contrast=1.1)
7. export(path="output/edited.jpg", format="JPG")
# Several qwen2511 jobs: queue the whole batch first, then follow the GPU batching rule (System section above).
```

### Inpainting Workflow
```
1. open_image(path="photo.jpg")
2. select_rect(x=100, y=50, width=200, height=150)
3. inpaint(prompt="a red rose bouquet")
4. export(path="output/inpainted.png")
```

### Upscaling
```
1. open_image(path="small_photo.jpg")
2. upscale(model="4x_NMKD-Siax_200k.pth")
3. export(path="output/upscaled.png")
```

### ControlNet Guided Generation
```
1. open_image(path="sketch.png")
2. controlnet_generate(prompt="a detailed cityscape", controlnet="canny", strength=0.8)
3. export(path="output/cityscape.png")
```

### Style Transfer
```
1. open_image(path="photo.jpg")
2. style_transfer(prompt="the same scene", style_path="painting.jpg", strength=0.8)
3. export(path="output/styled.png")
```

## Future Work
- [x] SAM-based semantic selection — delivered via the `semantic_select` tool (SAM 3 text/point/box prompts, live-verified 2026-09-17)
- [ ] Batch processing / multi-session support
- [ ] Additional ComfyUI custom nodes integration