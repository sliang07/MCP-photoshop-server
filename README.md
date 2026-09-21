# MCP Photoshop Server

A local MCP (Model Context Protocol) server that combines **Photoshop-style image editing** with **AI image generation** powered by ComfyUI.

> **Designed for use with [Cline](https://github.com/cline/cline)** — an AI-powered coding assistant (stdio transport) — **and Open WebUI** (MCP streamable-HTTP transport). This server integrates as an MCP tool provider to enable image generation and editing directly from your Cline workflow or Open WebUI chat.

## Instruction editing upgrade

- `preview_canvas` returns an image the assistant can inspect.
- `get_editing_capabilities` starts ComfyUI if needed and reports live model/node availability without loading generation models.
- `edit_image` defaults to Qwen Image 2.1 with reference guidance, model-visible edit masks, protected outside pixels, RGBA previews, and undoable replacement layers. Original layers remain hidden and can be restored with undo.
- `upscale` now honors its size factor and preserves layer/mask alignment and transparency.

The legacy Flux.1-era tools (`img2img`, `character_transform`, `inpaint`, `controlnet_generate`, `style_transfer`) were removed 2026-09-20 — their models (Flux Kontext, ControlNet, Redux/CLIPVision) are no longer installed. Masked fills, restyling, and style/identity guidance all go through `edit_image` (`region`/`mask_path` for local edits, `reference_paths` for style/identity).

Reconnect the MCP server after saving any in-memory work to load the new tools.

ComfyUI does not need to be running beforehand. Call the requested generation/editing tool directly; it starts ComfyUI and waits for readiness. `get_comfyui_status` and `get_editing_capabilities` also start it by default. Pass `start_if_needed=False` only for a passive check. A stopped backend is normal with the five-second idle shutdown; it does not mean the tools are unavailable. Startup messages stay out of the MCP protocol stream.

## Features

### Canvas Management
- `new_canvas` — Create a blank canvas with custom dimensions and background color
- `open_image` — Load an existing image file
- `export` — Save to file (PNG/JPG/WEBP); auto-named when `path` is omitted
- `get_info` — View canvas dimensions, layers, undo/redo state

### AI Image Generation & Editing
- `generate_image` — Text-to-image via Flux2 (photorealistic) or ANIMA (anime)
- `outpaint` — AI extend canvas in a direction (left, right, top, bottom), Flux2 chain

### Instruction Editing
- `edit_image` — Qwen Image 2.1 (`qwen21`, default, 25 steps) or FLUX.2 (`flux2`, fast, 4 steps). Qwen 2511 and the original Qwen backend are retired. The canvas is `<image1>`; references follow in order. An optional white-to-edit mask is appended last and also preserves outside pixels exactly. Layer visibility masks are separate; supply `mask_path` or `region=[x,y,width,height]` explicitly. `max_side=1024` controls working resolution; use 2048 for more detail. Qwen accepts up to 16 total images in the installed node (10 recommended), including canvas and mask. The Qwen timeout defaults to 1800 seconds.
- `get_editing_capabilities` — Live ComfyUI model/node availability for every editing backend (notes also carry the GPU batching rule)
- `preview_canvas` — Render the current canvas so the assistant can inspect results

### Sessions & Batch (multi-document)
- Every canvas tool accepts an optional `session_id` (default `"default"`), so multiple documents can be edited independently in one server process
- `list_sessions` — List open sessions with size, layer count, and undo depth
- `close_session` — Close a session by `session_id`, freeing its canvas
- `batch_generate` — Queue a whole job list on ComfyUI in one pass (single WebSocket connection) and export each result to disk as it completes

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
- `semantic_select` — SAM 3 semantic object selection (text prompt via SAM 3.1, or point/box) — sets the active layer's mask
- `clear_mask` — Remove layer mask

### History
- `undo` / `redo` — Full operation history (up to 20 steps)

### System
- `get_comfyui_status` — Check ComfyUI connection (starts it by default; `start_if_needed=False` for a passive check)
- `clear_vram` — Free GPU memory

> **GPU batching rule:** the host GPU is shared with the `qwen38` LLM docker and Open WebUI. For multiple or long ComfyUI generations, queue the whole batch to run in the background, then ask the user for explicit approval to stop the `qwen38` container (stopping it ends the LLM session; the batch keeps running on the host). `searxng` is CPU-only and never needs stopping. `batch_generate` implements the queue side server-side: it submits the whole job list up front on one WebSocket connection and exports each result as it completes; the `docker stop qwen38` approval step remains a conversation-level decision. Full procedure: `MEMORY.md` → "GPU Contention & Batching Rule".

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

  **Qwen Image 2.1 (default instruction editor):**
  - `qwen_image_2.1_int8_convrot.safetensors` (diffusion_models)
  - `qwen3vl_8b_int8_convrot.safetensors` (text_encoders)
  - `qwen_image_2.1_vae_bf16.safetensors` (vae)
  - Requires native `TextEncodeQwenImage21`, `QwenImage21Cache`, and `JoinImageWithAlpha` nodes. The old Qwen VAE remains necessary for ANIMA; Qwen 2511 models and Lightning adapters are not used by the editor.

  **Upscaling:**
  - `RealESRGAN_x4plus_anime_6B.pth` (upscale_models) — Anime upscaling
  - `4xFaceUpDAT.pth` (upscale_models) — Face upscaling

  **Semantic Selection (`semantic_select`):**
  - `sam3.pt` (checkpoints) — SAM 3 point/box prompts via `ImageOnlyCheckpointLoader`
  - `sam3.1_multiplex_fp16.safetensors` (checkpoints) — SAM 3.1 text prompts via `CheckpointLoaderSimple` + `CLIPTextEncode`

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

### Using with Open WebUI

The server can also serve Open WebUI over MCP **streamable-HTTP** — it registers as an **external tool server**, so it appears in chat as the **"Mcp Photoshop" tool** (enabled per-chat via the Tools panel, which requires the "Direct Tool Servers" user permission). Each client gets its own server process, so Cline's stdio connection is unaffected.

1. Double-click `run_openwebui.bat` in this folder. It sets `MCP_TRANSPORT=streamable-http` and serves `127.0.0.1:8000` (endpoint `/mcp`); logs append to `mcp_http.log`.
2. Register the tool server in Open WebUI — admin only, the personal Integrations page accepts OpenAPI servers only. Two ways:
   - **UI (current `open-webui:main` builds):** Settings → **Admin** → **Integrations** → **External Tool Servers** → **Manage Direct Connections** → **Add Connection** → click the **OpenAPI** toggle to switch it to **MCP** (badge shows "MCP Streamable HTTP") → URL `http://host.docker.internal:8000/mcp`, API key left blank (optional), Name `Mcp Photoshop` → Save.
   - **Direct DB (required on v0.11.3 — its connection dialog was type-locked to OpenAPI and could not create an MCP connection):** stop the container, upsert this entry into the SQLite config (`/app/backend/data/webui.db`, table `config`, key `tool_server.connections`), then start it:
     ```json
     [{
       "url": "http://host.docker.internal:8000/mcp",
       "path": "",
       "type": "mcp",
       "auth_type": "none",
       "config": { "enable": true },
       "info": {
         "id": "mcp-photoshop",
         "name": "Mcp Photoshop",
         "description": "Photoshop MCP server (Streamable HTTP): canvas, layers, text, effects, AI generation via ComfyUI"
       }
     }]
     ```

Known limitations (details in `MEMORY.md`):

- Open WebUI discards the MCP `initialize` instructions, so the GPU batching rule never reaches the OWUI model. Compensate in the OWUI model's system prompt: use a distinct `session_id` per chat and include a summary of the GPU batching rule. Cline receives the instructions natively.
- OWUI caps tool calls at `AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER` (default 300 s). Long Qwen edits (up to 1800 s server-side) fail unless the container is run with `AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER=1800`. The current container already has it; if you ever recreate the container (e.g. via the Desktop Docker manager script), include that env var in `docker run`.
- OWUI chat uploads land in OWUI storage, not Windows paths — `open_image` cannot see them directly. Use `export` to a shared folder, then `open_image` with that path.

### Environment Variables
See [`.env_example`](.env_example) for a complete reference. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TRANSPORT` | `stdio` | MCP transport: `stdio` (default, e.g. Cline) or `streamable-http` (Open WebUI; served on `127.0.0.1:8000` at `/mcp`) |
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
├── server.py           # MCP server + 38 tool registrations (canvas tools accept session_id; FastMCP initialize instructions carry the GPU batching rule)
├── editing.py          # Instruction editing backends (qwen21 / flux2) + semantic selection + live capabilities
├── config.py           # Configuration & model names
├── comfy_client.py     # ComfyUI API client (REST + WebSocket)
├── canvas.py           # Layered document with blend modes + undo/redo
├── session.py          # Per-session document management
├── requirements.txt    # Python dependencies
├── run_openwebui.bat   # Streamable-HTTP launcher for Open WebUI (sets MCP_TRANSPORT=streamable-http)
├── MEMORY.md           # Project memory bank
├── tests/              # Regression suite (81 tests)
├── .env_example        # Environment variable template
├── .gitignore          # Git ignore rules
└── README.md
```

### Design Philosophy
- **AI operations** (generation, outpainting, upscaling, instruction editing) → routed through ComfyUI
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
2. edit_image(prompt="change background to a forest at sunset; preserve the person")
3. undo  # if not satisfied
4. edit_image(prompt="match the jacket color in <image1> to <image2>; preserve the face and pose", reference_paths=["ref.jpg"])
5. preview_canvas()
6. adjust(contrast=1.1)
7. export(path="output/edited.jpg", format="JPG")
# Inspect the returned preview. Undo a failed attempt before editing again.
```

### Masked Fill (inpaint-style)
```
1. open_image(path="photo.jpg")
2. edit_image(prompt="fill with a red rose bouquet", region=[100, 50, 200, 150])
3. preview_canvas()   # inspect before keeping
4. export(path="output/filled.png")
```

### Upscaling
```
1. open_image(path="small_photo.jpg")
2. upscale(factor=2, model="anime")
3. export(path="output/upscaled.png")
```

### Style / Identity Guidance (reference images)
```
1. open_image(path="photo.jpg")
2. edit_image(prompt="restyle <image1> in the style of <image2>: watercolor, muted palette", reference_paths=["C:/refs/painting.jpg"])
3. export(path="output/styled.png")
```

### Multi-Document & Batch
```
1. new_canvas(width=1024, height=1024, session_id="doc_a")
2. open_image(path="photo.jpg", session_id="doc_b")
3. edit_image(prompt="...", session_id="doc_a")   # doc_b stays untouched
4. batch_generate(jobs=[{"prompt": "a lighthouse", "steps": 4},
                         {"prompt": "a paper crane", "steps": 4}],
                  export_dir="output/batch")
5. list_sessions()
6. close_session(session_id="doc_b")
```

## Future Work
- [x] SAM-based semantic selection — delivered via the `semantic_select` tool (SAM 3 text/point/box prompts, live-verified 2026-09-17)
- [x] Multi-session support — all canvas tools accept `session_id`; `list_sessions`/`close_session` manage multiple open documents (2026-09-18)
- [x] Batch processing — `batch_generate` queues a whole job list on one WebSocket connection and exports each result as it completes (live-verified 2026-09-18)
- [x] Legacy ControlNet/Redux workflow repair — Flux2 pixel-dim latents, `ControlNetApply.conditioning`, `CLIPVisionLoader` + `crop`/`strength_type`, plus live capability pre-checks in `get_editing_capabilities` (2026-09-18)
- [ ] Additional ComfyUI custom nodes integration (beyond ControlNet/Redux/SAM3 — e.g. new node packs)
