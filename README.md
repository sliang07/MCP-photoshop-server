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
- `generate_image` — Text-to-image with `model="flux2"`, `"qwen21"`, or `"anima"`
- `outpaint` — Extend left/right/top/bottom with `backend="flux2"` or `"qwen21"`; restores the original pixels exactly after generating the new margin

| Model | Generate / batch | Edit / masked edit / references | Outpaint | Default steps / CFG | Sampler | Scheduler |
|---|---|---|---|---|---|---|
| `flux2` (Klein 9B distilled) | Yes; fast | Yes | Yes | 4 / 1 | Euler | Native `Flux2Scheduler` |
| `qwen21` (Qwen Image 2.1; custom preset) | Yes; detail, typography, alpha | Yes | Yes | 30 / 3 | Euler | Simple |
| `anima` (installed Aesthetic v1.1) | Yes; anime/illustration | No | No | 30 / 4 | `er_sde` | Simple |

These are the active MCP presets for the installed models. Qwen generation, editing and outpainting all use the requested **30 steps / CFG 3** preset. This is a user preference, not the official Qwen 2.1 recommendation: ComfyUI's current 2.1 templates use **25 steps / CFG 1 / Euler / Simple**. Older Qwen base and Lightning/Flash recipes do not define the 2.1 defaults; no Lightning LoRA or AuraFlow shift is applied.

All generation paths use a full denoising schedule (`denoise=1.0` on KSampler). Text-to-image defaults to 1024x1024; Anima also suits approximately 1MP portrait/landscape sizes such as 896x1152 and 1152x896. Anima Aesthetic's documented tuning range is 30–50 steps and CFG 4–5; the preset uses the lower end. Klein uses `flux2-vae.safetensors` and the Qwen3 8B encoder. Its installed checkpoint is distilled, so the base-model 20–30-step recipe does not apply. Anima Turbo and Qwen Lightning/Flash are not active backends.

Model selection is per call; batch jobs can each choose a different model. Omit `steps`/`cfg` to use the selected model's defaults; explicit values are preserved. `get_editing_capabilities` reports readiness separately for generation, editing and outpainting, including missing files/nodes. Missing or unsupported choices return an error without substituting another model. Defaults remain Flux for generation/outpaint and Qwen for editing. Upscaling and semantic selection retain their dedicated models.

Generation uses native ComfyUI nodes, including `Flux2Scheduler` for Klein and `TextEncodeQwenImage21` for Qwen. Qwen generation defaults to a 1800-second timeout; batch timeouts account for every job. PNG exports preserve generated alpha; existing visible canvas layers still composite normally. Outpainting uses reference-guided editing, with original pixels restored by the server rather than relying on the model to preserve them.

Qwen outpaint uses an opaque white margin and the guide's approximately 1-megapixel editing resolution before returning to the requested canvas size. Low-resolution transparent-margin tests produced blank/noisy borders; ordinary `edit_image` retains its existing resolution handling.

References: [Qwen 2.1 guide](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1), [Flux Klein guide](https://docs.comfy.org/tutorials/flux/flux-2-klein#flux-2-klein-9b-workflows), [Anima guide](https://docs.comfy.org/tutorials/image/anima/anima), [Anima model settings](https://huggingface.co/circlestone-labs/Anima#generation-settings).

### Master prompt integration

The image tools apply the relevant rules from the master files in `MASTER_PROMPT_DIR` (default: the repository-local `masters/` folder; set the variable in `.env` to point elsewhere):

| Tool/model | Source and application |
|---|---|
| Flux generation, edits and outpaint | `flux2prompt.txt`: connected prose, preserve explicit details/lighting, exact lettering, no unnecessary interview or tag suffix |
| Anima generation and batch jobs | `anima_prompt.txt`: hybrid tags/prose, explicit character-to-attribute binding, separate compatible negatives, no score tags for Aesthetic |
| Qwen generation and batch jobs | `qwen_image_2.1_system_prompt_t2i.txt`: detailed English observer prose (roughly 400–500 words), spatial layout, lighting, exact lettering in its original script |
| Qwen edits and outpaint | `qwen_image_2.1_system_prompt_edit.txt`: clear requested changes with untargeted content preserved, separate prose/lettering language rules, correct reference roles |
| `add_text` | Preserve exact supplied lettering; do not paraphrase or add image-prompt tags |

`get_prompt_guidance(model="anima", task="generation")` reads the applicable full master on demand, with its source path and SHA-256. It works through both MCP transports without accessing ComfyUI or the GPU. Essential rules are also included directly in `generate_image`, `edit_image`, `outpaint` and `batch_generate` descriptions, because some clients ignore initialize instructions. Reading the full guide is optional, not an extra prerequisite for every image.

For Qwen, use `get_prompt_guidance(model="qwen21", task="generation")` for the t2i master, or `task="editing"` / `"outpaint"` for the edit master. Pass the master's `rewritten_prompt` text as `prompt`, not its JSON wrapper. Map `wh_ratio` to generation `width`/`height`; `ratio_follow` identifies which image to open as the editing canvas. `edit_image` returns the canvas's dimensions, and `max_side` controls working detail, not aspect ratio. Use outpaint `direction`/`amount` or an explicit canvas operation for framing changes. The guides do not silently change the 1024 defaults, outpaint's internal resolution, or active sampling presets; `max_side=2048` remains available for detailed edits.

Qwen generation descriptions are English; edit descriptions are Chinese for Chinese instructions and English otherwise. Exact rendered text keeps the requested spelling/language. For edits without a specified text language, use the input's dominant text language, then the user's instruction language if the input has no text. Single-image edits/outpaint use natural image references; edits with references or an appended mask use numbered `<imageN>` tags.

The server removes a single surrounding prompt code fence. For Anima Aesthetic/unknown checkpoints, it removes standalone comma-separated `score_*` tags from both positive and negative prompts, while preserving quoted text verbatim; known Anima base checkpoints retain scores. Generation/batch results disclose any normalization. It does not truncate prompts to editorial word targets or automatically append negative tags that could conflict with the request.

Semantic requirements—intent, lighting, composition, character identity, suitable negatives and inspecting results—remain instructions for the calling LLM, not a guaranteed visual validator. The compact descriptions reflect the masters reviewed September 21, including the two newly supplied Qwen guides; full-guide reads always return current file contents. If a master changes, refresh the corresponding compact rules in `prompt_rules.py` and restart/reconnect the server. The H3 video, audio/music, historical review and backup files do not supply still-image syntax or override active MCP sampling presets. Original master files are unchanged.

### Instruction Editing
- `edit_image` — Qwen Image 2.1 (`qwen21`, default, custom 30 steps/CFG 3) or FLUX.2 (`flux2`, fast, 4 steps/CFG 1). Qwen 2511 and the original Qwen backend are retired. For multiple Qwen inputs, the canvas is `<image1>` and references follow in order; a lone canvas uses natural wording without a tag. An optional white-to-edit mask is appended last and also preserves outside pixels exactly. Layer visibility masks are separate; supply `mask_path` or `region=[x,y,width,height]` explicitly. `max_side=1024` controls working resolution; use 2048 for more detail. Qwen accepts up to 16 total images in the installed node (10 recommended), including canvas and mask. The Qwen timeout defaults to 1800 seconds.
- `get_editing_capabilities` — Live ComfyUI model/node availability per task (generation, editing, outpaint; notes also carry the GPU batching rule)
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
  - `flux-2-klein-9b.safetensors` or `flux-2-klein-9b-fp8.safetensors` (diffusion_models) — distilled model; defaults to 4 steps
  - `qwen_3_8b_fp8mixed.safetensors` (text_encoders) — Flux2 Klein text encoder
  - `flux2-vae.safetensors` (vae) — Flux2 Klein VAE

  **ANIMA (anime generation):**
  - `anima-aesthetic-v1.1.safetensors` (diffusion_models)
  - `qwen_3_06b_base.safetensors` (text_encoders)
  - `qwen_image_vae.safetensors` (vae)

  **Qwen Image 2.1 (generation and default instruction editor):**
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
- OWUI caps tool calls at `AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER` (default 300 s). Long Qwen edits (up to 1800 s server-side) fail unless the container is run with `AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER=1800`. The current container already has it; if you ever recreate the container (e.g. via a local Docker manager script), include that env var in `docker run`.
- OWUI chat uploads land in OWUI storage, not Windows paths — `open_image` cannot see them directly. Use `export` to a shared folder, then `open_image` with that path.

### Environment Variables
See [`.env_example`](.env_example) for a complete reference. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TRANSPORT` | `stdio` | MCP transport: `stdio` (default, e.g. Cline) or `streamable-http` (Open WebUI; served on `127.0.0.1:8000` at `/mcp`) |
| `MASTER_PROMPT_DIR` | `masters/` (repository-local, next to `server.py`) | Directory containing the current Flux, Anima and Qwen master files; read by `get_prompt_guidance` |
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
├── server.py           # MCP server; 39 tools including editing registrations (canvas tools accept session_id)
├── editing.py          # Instruction editing backends (qwen21 / flux2) + semantic selection + live capabilities
├── prompt_rules.py     # Tool-level master guidance, source reader and conservative prompt normalization
├── config.py           # Configuration & model names
├── comfy_client.py     # ComfyUI API client (REST + WebSocket)
├── canvas.py           # Layered document with blend modes + undo/redo
├── session.py          # Per-session document management
├── requirements.txt    # Python dependencies
├── masters/            # Master prompt files (Flux/Anima/Qwen) read by get_prompt_guidance
├── run_openwebui.bat   # Streamable-HTTP launcher for Open WebUI (sets MCP_TRANSPORT=streamable-http)
├── MEMORY.md           # Project memory bank
├── tests/              # Regression suite (106 tests)
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
4. batch_generate(jobs=[{"prompt": "a lighthouse", "model": "flux2"},
                         {"prompt": "a poster reading HELLO", "model": "qwen21"},
                         {"prompt": "an anime paper crane illustration", "model": "anima"}],
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
