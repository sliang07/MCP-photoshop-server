# Memory Bank — MCP Photoshop Server

> Last updated: 2026-08-12
> Status: 36 tools — all confirmed working (image-only, no video)
> Location: `mcp-photoshop-server`

---

## 1. Project Overview

**What:** A local MCP (Model Context Protocol) server that combines Photoshop-style image editing with AI image generation powered by ComfyUI.

**Why:** Provides an AI model (e.g., Claude) with a comprehensive set of image manipulation tools through a single MCP interface, enabling multi-step image editing workflows.

**Core Architecture — Two-Track Design:**
- **AI Operations** (generation, inpainting, transforms, ControlNet, style transfer) → Routed through ComfyUI API
- **Deterministic Edits** (crop, resize, adjustments, filters) → Executed locally with Pillow (fast, no GPU)

**ComfyUI Lifecycle Management:**
- **Auto-start**: Automatically launches ComfyUI via embedded Python when needed (no manual startup required)
- **Idle timeout auto-kill**: When `COMFYUI_AUTO_KILL=1`, ComfyUI is killed after `COMFYUI_IDLE_TIMEOUT` (default 180s) of inactivity instead of immediately. This allows chaining multiple ComfyUI tools without restarting each time, while still freeing VRAM when idle.
- **Port management**: Handles Windows TIME_WAIT issues by waiting for port 8188 to be fully bindable
- **3-strategy kill**: Direct process handle → port-based (netstat) → command-line matching (wmic)
- **Pre-fetch output**: Downloads result bytes BEFORE killing ComfyUI when auto-kill is enabled
- **Retry logic**: If ComfyUI fails to start, kills stale processes, clears port, and retries once

**VRAM Pressure Management:**
- **Adaptive cleanup**: After each generation, checks free VRAM via ComfyUI system_stats or nvidia-smi
- **Kill threshold**: If free VRAM < 8192 MB, kills ComfyUI entirely to release all VRAM for vLLM
- **Free memory**: If sufficient VRAM, calls free_memory() to unload models while keeping process warm
- **Configured via**: `VRAM_PRESSURE_THRESHOLD_MB` (default: 8192)

**OOM Issue:** Running ComfyUI directly (without auto-kill) causes major OOM when shared with vLLM on a 32GB GPU. The idle timeout (180s) balances convenience (chaining tools) with VRAM management (killing when idle).

---

## 2. File Architecture

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `config.py` | Configuration constants, model names, ComfyUI lifecycle | `COMFYUI_URL`, `COMFYUI_PYTHON`, `COMFYUI_MAIN`, `COMFYUI_ARGS`, `COMFYUI_AUTO_KILL`, `COMFYUI_IDLE_TIMEOUT`, `COMFYUI_START_TIMEOUT`, `WEBSOCKET_TIMEOUT`, `VRAM_PRESSURE_THRESHOLD_MB` |
| `comfy_client.py` | ComfyUI API wrapper with auto-start/idle-kill lifecycle | `ComfyUIClient`: `start_comfyui()`, `kill_comfyui()`, `run_workflow_and_wait()`, `_schedule_idle_kill()`, `_cancel_idle_kill()`, `submit_workflow()`, `upload_image()`, `get_output_file()`, `free_memory()` |
| `canvas.py` | Layered document model with blend modes | `Canvas`: layers with 12 blend modes, masks, undo/redo stack (20 steps), `resize_canvas()`, `composite()`, `composite_rgb()`, `BLEND_MODES` dict |
| `session.py` | Per-session document management | `SessionManager`: `get_or_create()`, `get()`, `create()`, `delete()`, `get_default_session()` |
| `server.py` | MCP server + all tool registrations + workflow builders | 36 `@app.tool()` registrations, 12 workflow builder functions, `run_workflow()` helper, `free_or_kill_based_on_pressure()` |
| `requirements.txt` | Python dependencies | `mcp<2.0.0`, `Pillow>=10.0.0`, `httpx>=0.27.0`, `websockets>=12.0`, `numpy>=1.24.0` |
| `README.md` | User documentation | Installation, usage examples, architecture overview |
| `MEMORY.md` | This file — project memory bank |

---

## 3. Tool Inventory (36 Tools)

### Canvas Management (4)
| Tool | Signature | Description |
|------|-----------|-------------|
| `new_canvas` | `(width=1024, height=1024, bg_color="white")` | Create blank canvas |
| `open_image` | `(path: str)` | Load existing image file as active document |
| `export` | `(path=None, format="PNG", quality=95)` | Save to file or return base64 data |
| `get_info` | `()` | Canvas dimensions, layers, undo/redo state |

### AI Image Generation (3)
| Tool | Signature | Description |
|------|-----------|-------------|
| `generate_image` | `(prompt, model="flux2", width=1024, height=1024, steps=20, cfg=1.5, seed=None, negative_prompt="")` | txt2img via Flux2 (photorealistic) or ANIMA (anime) |
| `img2img` | `(prompt, strength=0.7, guidance=4.0, seed=None)` | AI instructed editing via Flux Kontext |
| `character_transform` | `(prompt, guidance=4.0, seed=None)` | Character pose/expression/action transforms |

### AI Editing (2)
| Tool | Signature | Description |
|------|-----------|-------------|
| `inpaint` | `(prompt, guidance=4.0, steps=30, seed=None)` | AI fill masked region (requires mask from select_rect/select_ellipse) |
| `outpaint` | `(prompt, direction="right", amount=256, guidance=4.0, steps=30, seed=None)` | Extend canvas + AI fill (direction: left/right/top/bottom) |

### AI-Guided Generation (2)
| Tool | Signature | Description |
|------|-----------|-------------|
| `controlnet_generate` | `(prompt, controlnet="depth", strength=0.8, width=1024, height=1024, steps=20, cfg=1.5, seed=None)` | Depth/canny/pose guided generation |
| `style_transfer` | `(prompt, style_path, strength=0.8, width=1024, height=1024, steps=20, seed=None)` | Style transfer via Redux StyleModel |

### Transforms (4)
| Tool | Signature | Description |
|------|-----------|-------------|
| `crop` | `(x, y, width, height)` | Crop active layer to region |
| `resize` | `(width, height, maintain_aspect=False)` | Resize active layer |
| `rotate` | `(degrees, expand=True, bg_color="transparent")` | Rotate counter-clockwise |
| `flip` | `(axis="horizontal")` | Flip horizontal or vertical |

### Color Adjustments (3)
| Tool | Signature | Description |
|------|-----------|-------------|
| `adjust` | `(brightness, contrast, saturation, hue, sharpness)` | Multi-parameter adjustment (multipliers, hue in degrees) |
| `levels` | `(black_point=0, mid_point=1.0, white_point=255)` | Levels/gamma adjustment via 256-point LUT |
| `curves` | `(red="", green="", blue="")` | Per-channel tone curves (format: "0,0 128,140 255,255") |

### Filters (1) — 17 filters available
| Tool | Signature | Description |
|------|-----------|-------------|
| `apply_filter` | `(name, **params)` | 17 filters: blur, gaussian_blur, sharpen, contour, detail, edge_enhance, edge_enhance_more, find_edges, smooth, smooth_more, emboss, pixelate, posterize, solarize, invert, grayscale, sepia |

### Text (1)
| Tool | Signature | Description |
|------|-----------|-------------|
| `add_text` | `(text, x=50, y=50, font_size=48, color="white", font=None, stroke_width=0, stroke_color="black")` | Text overlay with font, color, stroke |

### Upscaling (1)
| Tool | Signature | Description |
|------|-----------|-------------|
| `upscale` | `(factor=2, model="anime")` | AI upscaling via ComfyUI (model: "anime" or "face") |

### Layer System (7)
| Tool | Signature | Description |
|------|-----------|-------------|
| `add_layer` | `(name=None, source_path=None, opacity=1.0, blend_mode="normal")` | Add new layer, optionally from file |
| `select_layer` | `(index: int)` | Select layer by index (0 = bottom) |
| `set_blend_mode` | `(mode, index=None)` | Set blend mode (12 modes supported) |
| `set_layer_opacity` | `(opacity, index=None)` | Set layer transparency (0.0-1.0) |
| `merge_down` | `()` | Merge active layer into layer below |
| `delete_layer` | `(index=None)` | Delete layer by index or active |
| `reorder_layer` | `(index=None, direction="up")` | Move layer up/down in stack |

### Selections & Masks (4)
| Tool | Signature | Description |
|------|-----------|-------------|
| `select_rect` | `(x, y, width, height)` | Rectangular mask |
| `select_ellipse` | `(x, y, rx, ry)` | Elliptical mask centered at (x,y) |
| `select_object` | `(description, threshold=128)` | Heuristic color/region selection |
| `clear_mask` | `(index=None)` | Remove layer mask |

**select_object supported descriptions:** `red, blue, green, sky, dark, shadow, light, white, black, yellow, purple, orange, cyan, pink, brown`

### History (2)
| Tool | Signature | Description |
|------|-----------|-------------|
| `undo` | `()` | Undo last operation (up to 20 steps) |
| `redo` | `()` | Redo last undone operation |

### System (2)
| Tool | Signature | Description |
|------|-----------|-------------|
| `get_comfyui_status` | `()` | Check ComfyUI connection and system info |
| `clear_vram` | `()` | Free GPU VRAM by unloading cached models |

---

## 4. ComfyUI Lifecycle Management

### Auto-Start Flow
1. `run_workflow_and_wait()` calls `_cancel_idle_kill()` then `start_comfyui()` before each workflow
2. `start_comfyui()` checks if ComfyUI is already running via `/history` endpoint
3. If not running: ensures port 8188 is free (kills stale processes if needed)
4. Launches ComfyUI via embedded Python: `COMFYUI_PYTHON -s COMFYUI_MAIN COMFYUI_ARGS...`
5. Falls back to `.bat` if Python path doesn't exist
6. Polls `/history` every 2s until reachable (up to `COMFYUI_START_TIMEOUT`=180s)
7. On timeout: kills process, waits for port clear, retries once

### Idle Timeout Auto-Kill (when `COMFYUI_AUTO_KILL=1`)
1. After workflow completes successfully: schedules idle kill via `_schedule_idle_kill()`
2. Idle timer set to `COMFYUI_IDLE_TIMEOUT` (default 180s / 3 minutes)
3. If a new workflow starts within the timeout: `_cancel_idle_kill()` cancels pending timer
4. After timeout with no new workflow: `_do_idle_kill()` kills ComfyUI, freeing VRAM
5. On error (RuntimeError, TimeoutError): immediate kill (no idle delay)
6. Pre-fetches output file bytes via `/view` endpoint before any kill

### VRAM Pressure Management (when `COMFYUI_AUTO_KILL=0`, default)
- After each generation, `free_or_kill_based_on_pressure()` checks free VRAM
- Reads from ComfyUI `/system_stats` API first, falls back to `nvidia-smi`
- If free VRAM < `VRAM_PRESSURE_THRESHOLD_MB` (default 8192): kills ComfyUI to release all VRAM
- If free VRAM >= threshold: calls `free_memory()` to unload models, keeps process warm

### Key Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_PYTHON` | *(required via env var)* | Embedded Python path |
| `COMFYUI_MAIN` | *(required via env var)* | ComfyUI entry point |
| `COMFYUI_ARGS` | `--windows-standalone-build` | Launch args |
| `COMFYUI_AUTO_KILL` | `0` | Kill after generation (0=use free_memory, 1=idle timeout) |
| `COMFYUI_IDLE_TIMEOUT` | `180` | Seconds of inactivity before auto-killing (when AUTO_KILL=1) |
| `COMFYUI_START_TIMEOUT` | `180` | Seconds to wait for ComfyUI startup |
| `WEBSOCKET_TIMEOUT` | `600` | Seconds to wait for workflow completion (10 min) |
| `VRAM_PRESSURE_THRESHOLD_MB` | `8192` | Kill ComfyUI if free VRAM below this (MB) |

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI API endpoint |
| `COMFYUI_INPUT_DIR` | `None` (auto-detected) | ComfyUI input directory |
| `COMFYUI_OUTPUT_DIR` | `None` (auto-detected) | ComfyUI output directory |
| `MAX_UNDO_STEPS` | `20` | Maximum undo history entries |

### Windows-Specific Handling
- Redirects stdin from `/dev/null` to prevent console crashes
- Uses `DETACHED_PROCESS` flag for independent process lifecycle
- Sets `PYTHONIOENCODING=utf-8` to prevent UnicodeEncodeError from emoji logging
- Does NOT redirect stdout/stderr (embedded Python crashes on file redirect)

---

## 5. Workflow Builders (server.py)

| Builder | Description | Key Nodes |
|---------|-------------|-----------|
| `build_txt2img_workflow()` | Text-to-image dispatcher (Flux2 or ANIMA) | Dispatches to builder below based on model param |
| `build_flux2_workflow()` | Text-to-image for Flux2 Klein 9B | **UNETLoader**, **CLIPLoader** (type: "flux2"), **VAELoader**, **Flux2KleinSectionedEncoder**, **EmptyFlux2LatentImage**, **Flux2KleinKSamplerExperimental**, **VAEDecode**, **SaveImage** |
| `build_anima_workflow()` | Text-to-image for ANIMA (anime) | **UNETLoader**, **CLIPLoader** (stable_diffusion), **VAELoader**, **CLIPTextEncode**, **EmptyLatentImage** (pixel dims), **KSampler** (er_sde), **VAEDecode**, **SaveImage** |
| `build_img2img_kontext_workflow()` | AI instructed editing | **LoadImage**, **UNETLoader**, **CLIPLoader**, **VAELoader**, **VAEEncode**, **Flux2KleinSectionedEncoder**, **BasicGuider**, **RandomNoise**, **BasicScheduler**, **SamplerCustomAdvanced**, **VAEDecode**, **SaveImage** |
| `build_inpaint_workflow()` | Masked region inpainting | **LoadImage** (base+mask), **ImageToMask**, **UNETLoader** (Kontext), **DualCLIPLoader**, **VAELoader** (ae.safetensors), **VAEEncode**, **CLIPTextEncode**, **FluxGuidance**, **SetLatentNoiseMask**, **SamplerCustomAdvanced**, **VAEDecode**, **SaveImage** |
| `build_outpaint_workflow()` | Canvas extension + inpaint | Same Flux.1 chain as inpaint, uses LoadImage native MASK output |
| `build_upscale_workflow()` | AI upscaling | **LoadImage**, **UpscaleModelLoader**, **ImageUpscaleWithModel**, **SaveImage** |
| `build_controlnet_workflow()` | ControlNet-guided generation | **LoadImage**, **UNETLoader**, **CLIPLoader**, **VAELoader**, **ControlNetLoader**, **ControlNetApply**, **SamplerCustomAdvanced**, **VAEDecode**, **SaveImage** |
| `build_style_transfer_workflow()` | Style transfer via Redux | **LoadImage** (content+style), **CLIPVisionEncode**, **StyleModelLoader**, **StyleModelApply**, **SamplerCustomAdvanced**, **VAEDecode**, **SaveImage** |

---

## 6. Design Decisions

- **Pillow for deterministic edits** — Faster (no GPU round-trip), more precise, simpler
- **Layer Model** — Full non-destructive layers with 12 blend modes, masks, opacity
- **Mask Handling** — PIL "L" mode images (0=protected, 255=selected)
- **Undo/Redo** — Full state snapshots (deepcopy via Layer.to_dict/from_dict), limited to 20 steps
- **Session Management** — Canvas instances keyed by session ID, default session: "default"
- **ComfyUI Auto-Start** — Eliminates manual ComfyUI startup
- **Idle Timeout Kill** — 180s idle timeout balances tool chaining with VRAM management
- **Pre-Fetch Output** — Downloads result before killing ComfyUI
- **VRAM Pressure Management** — Adaptive cleanup based on VRAM pressure from vLLM
- **No External Test Scripts** — All testing done through MCP tool calls, not standalone Python scripts
- **No Video Tools** — Video generation removed due to OOM issues (H3 + LTX models too large for shared GPU with vLLM)
- **All Logging via Python logging** — No `print()` calls; use `logger.info/warning/debug/error` throughout

---

## 7. ComfyUI Models Required

### Image Generation
| Model | Directory | Purpose |
|-------|-----------|---------|
| `flux-2-klein-9b.safetensors` | diffusion_models | Text-to-image (photorealistic — Flux2 Klein 9B) — **CRITICAL: use 4-6 steps max** (quality diminishes after ~6 steps, artifacts/over-smoothing, increased latency) |
| `qwen_3_4b.safetensors` | text_encoders | Flux2 Klein text encoder |
| `flux2-vae.safetensors` | vae | Flux2 Klein VAE |
| `anima-aesthetic-v1.1.safetensors` | diffusion_models | Text-to-image (anime — ANIMA) |
| `qwen_3_06b_base.safetensors` | text_encoders | ANIMA text encoder |
| `qwen_image_vae.safetensors` | vae | ANIMA VAE |
| `flux1-dev-kontext_fp8_scaled.safetensors` | diffusion_models | img2img, inpaint, outpaint (Flux Kontext) |

### Upscaling & ControlNet
| Model | Directory | Purpose |
|-------|-----------|---------|
| `RealESRGAN_x4plus_anime_6B.pth` | upscale_models | Anime upscaling |
| `4xFaceUpDAT.pth` | upscale_models | Face upscaling |
| `control_v11f1p_sd15_depth_fp16.safetensors` | controlnet | Depth-guided ControlNet |
| `control_v11p_sd15_canny_fp16.safetensors` | controlnet | Canny-guided ControlNet |
| `control_v11p_sd15_openpose_fp16.safetensors` | controlnet | Pose-guided ControlNet |
| `flux1-redux-fp16.safetensors` | style_models | Style transfer (Redux) |

---

## 8. Run Configuration

### Start Server
```bash
cd <path-to-mcp-photoshop-server>
pip install -r requirements.txt
python server.py
```

### MCP Client Configuration
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
        "COMFYUI_IDLE_TIMEOUT": "180",
        "VRAM_PRESSURE_THRESHOLD_MB": "8192"
      }
    }
  }
}
```

---

## 9. Testing Status

### Verified Working (36 tools)
- Canvas Management: `new_canvas`, `export`, `get_info` ✅
- Transforms: `crop`, `resize`, `rotate`, `flip` ✅
- Color Adjustments: `adjust`, `levels`, `curves` ✅
- Filters: `apply_filter` (all 17 filters) ✅
- Text: `add_text` ✅
- Layer System: all 7 tools ✅
- Selections: `select_rect`, `select_ellipse`, `clear_mask` ✅
- History: `undo`, `redo` ✅
- System: `get_comfyui_status`, `clear_vram` ✅
- AI Generation: `generate_image` (Flux2 + ANIMA), `img2img`, `character_transform` ✅
- AI Editing: `inpaint`, `outpaint` ✅
- AI-Guided: `controlnet_generate`, `style_transfer` ✅
- Upscale: `upscale` (anime + face models) ✅

### Idle Timeout Chaining (tested 2026-08-12)
- `generate_image` → `img2img` chained successfully without ComfyUI restart
- Idle timer reset on second call, no restart needed

---

## 10. Known Limitations & Future Work

- `select_object` uses color/region heuristics (not SAM)
- Single ComfyUI instance (no concurrent workflows)
- Undo stores full snapshots (memory-intensive)
- No lasso/freehand selection
- No brush/paint tools
- No batch processing / multi-session support
- Running ComfyUI without auto-kill causes major OOM on shared GPU (32GB with vLLM)

---

## 11. Changelog

### 2026-08-12 — Documentation & Memory Bank Update
- Updated MEMORY.md to reflect current state (image-only, no video)
- Removed all video-related documentation (H3, LTX, Wan pipeline)
- Updated tool count from 39 to 36
- Updated ComfyUI lifecycle section with idle timeout details
- Removed hardcoded paths from Run Configuration section

### 2026-08-12 — Hardcoded Paths Removed
- `config.py`: ComfyUI paths now default to empty strings (must be set via env vars)
- `README.md`: MCP config example uses generic placeholders
- `README.md`: Environment Variables table documents all required/optional vars
- Project is now ready for public repository (no personal paths exposed)

### 2026-08-12 — FLUX 2 Klein Step Limit Documented
- Added detailed warning: quality diminishes after ~6 steps (artifacts, over-smoothing)
- Latency increases linearly with each additional step
- Model is designed for fast, low-step generation (4-6 steps optimal)

### 2026-08-12 — Idle Timeout Auto-Kill (180s)
- Replaced immediate auto-kill with configurable idle timeout
- `COMFYUI_IDLE_TIMEOUT` (default 180s) — kills ComfyUI after N seconds of inactivity
- Tool chaining works: timer resets on each new workflow call
- Errors trigger immediate kill (no idle delay)
- `_cancel_idle_kill()` called at start of each workflow
- `_schedule_idle_kill()` called on successful completion
- Tested: `generate_image` → `img2img` chain works without restart

### 2026-08-12 — Video Generation Removed
- **Removed from `comfy_client.py`:** `run_workflow_no_kill()`, `run_workflow_dual_output()`, SaveVideo node classification, `_cached_video_bytes` pre-fetch, `file_type="videos"` fallback
- **Removed from `.clinerules/`:** 5 MiniMax H3 prompt template files (`h3-i2va-template.md`, `h3-prompt-formatter.md`, `h3-ref2va-template.md`, `h3-t2va-template.md`, `minimax-h3-prompt-guide.md`)
- **Updated `README.md`:** Removed Video Generation section, video tool docs, video model prerequisites, video examples
- **Reason:** H3 (~13B) + LTX (~22B) models cause major OOM when shared with vLLM on 32GB GPU; video tools were never in active use

### 2026-08-12 — All print() Replaced with Logging
- Replaced all `print()` calls with Python `logging` module throughout codebase
- Uses `logger.info()`, `logger.warning()`, `logger.debug()`, `logger.error()` as appropriate
- Configurable via `LOG_LEVEL` environment variable

### 2026-08-07 — VRAM Pressure Management
- **Restored VRAM pressure heuristic** — After each generation, `free_or_kill_based_on_pressure()` checks free VRAM and either kills ComfyUI (low VRAM) or calls free_memory() (sufficient VRAM). This avoids OOM when vLLM is running alongside ComfyUI.
- **Config setting**: `VRAM_PRESSURE_THRESHOLD_MB` (default: 8192) controls when to kill vs free
- **Function**: `free_or_kill_based_on_pressure()` reads VRAM from ComfyUI system_stats or nvidia-smi
- **Removed external test scripts** — `test_single.py` and `test_generation.py` deleted. All testing should be done through MCP tool calls.

### 2026-08-07 — Video Timeout Fix (VIDEO_WEBSOCKET_TIMEOUT)
- Added `VIDEO_WEBSOCKET_TIMEOUT=1800` (30 min) for video workflows
- `run_workflow_and_wait()` accepts `timeout` parameter
- H3 model init takes 5-10 min, sampling takes 15-20 min total

### 2026-08-07 — Video Two-Stage Refactoring Complete (NOW REMOVED)
- ~~All 3 video tools ran H3 and LTX as separate `/prompt` calls~~
- ~~`_run_video_two_stage()` orchestrated H3 → extract frame → LTX pipeline~~
- Video tools removed 2026-08-12 (see above)

### 2026-08-07 — ANIMA Fix + Inpaint/Outpaint Tested
- Fixed ANIMA EmptyLatentImage resolution bug (pixel dims, not latent dims)
- Tested inpaint (424x600) and outpaint (524x600) — both working

### 2026-08-07 — Video Workflow Rewrite (Phase 7) (NOW REMOVED)
- ~~Rewrote all 3 video workflow builders to match real ComfyUI node schemas~~
- Video tools removed 2026-08-12 (see above)

### 2026-08-07 — MiniMax H3 Prompt Compiler (NOW REMOVED)
- ~~`_format_h3_prompt()`, `_validate_h3_prompt()`, `_detect_scene_context()`, etc.~~
- H3 prompt templates and compiler removed 2026-08-12 (see above)

### 2026-08-07 — Video VRAM Fix (Unconditional Hard-Kill) (NOW REMOVED)
- ~~Changed `_run_video_two_stage()` finally block to always call `kill_comfyui()`~~
- Video tools removed 2026-08-12 (see above)

### 2026-08-07 — Remove Bogus Frame Images (NOW REMOVED)
- ~~Added `include_save_image` parameter to `_build_h3_stage()`~~
- Video tools removed 2026-08-12 (see above)

### 2026-08-06 — NoobAI → ANIMA Migration
- Replaced NoobAI checkpoint with ANIMA UNET-based workflow
- Separate UNETLoader + CLIPLoader + VAELoader architecture