# MCP Photoshop Server

A local MCP (Model Context Protocol) server that combines **Photoshop-style image editing** with **AI image generation** powered by ComfyUI.

This is a standalone editor; it does not control the Adobe Photoshop application.

> **Designed for use with [Cline](https://github.com/cline/cline)** — an AI-powered coding assistant (stdio transport) — **and Open WebUI** (MCP streamable-HTTP transport). This server integrates as an MCP tool provider to enable image generation and editing directly from your Cline workflow or Open WebUI chat.

## Instruction editing upgrade

- `preview_canvas` returns an image the assistant can inspect.
- `get_editing_capabilities` starts ComfyUI if needed and reports live model/node availability without loading generation models.
- `edit_image` defaults to Qwen Image 2.1 with reference guidance, model-visible edit masks, protected outside pixels, and RGBA previews. Composite edits hide originals beneath a new replacement layer; use `set_layer_visibility` or undo to restore them. Pass `layer_index` to edit one layer, or `output_mode="extract"` to add an extracted subject while retaining originals.
- `upscale` now honors its size factor and preserves layer/mask alignment and transparency.

The legacy Flux.1-era tools (`img2img`, `character_transform`, `inpaint`, `controlnet_generate`, `style_transfer`) were removed 2026-09-20 — their models (Flux Kontext, ControlNet, Redux/CLIPVision) are no longer installed. Masked fills, restyling, and style/identity guidance all go through `edit_image` (`region`/`mask_path` for local edits, `reference_paths` for style/identity).

Reconnect the MCP server after saving any in-memory work to load the new tools.

Idle shutdown starts after a backend availability check or the end of a workflow/upload, including failures and cancellation. It preserves an existing timer rather than postponing it on every status probe. An empty queue permits stopping this MCP's own backend or an adopted local process whose executable, main script and creation time match the configured ComfyUI; busy or unknown queue state defers shutdown. Unrelated listeners and remote backends are not adopted.

The timer lives in the MCP process: keep that process running through the idle interval. After an MCP restart, an active backend check can adopt and clean up a verified orphan. To check whether ComfyUI stayed off, use `get_comfyui_status(start_if_needed=False)`; the default status check starts it again.

ComfyUI does not need to be running beforehand. Call the requested generation/editing tool directly; it starts ComfyUI and waits for readiness. `get_comfyui_status` and `get_editing_capabilities` also start it by default. Pass `start_if_needed=False` only for a passive check. A stopped backend is normal with the five-second idle shutdown; it does not mean the tools are unavailable. Startup messages stay out of the MCP protocol stream.

## Features

### Canvas Management
- `new_canvas` — Create a blank canvas with custom dimensions and background color
- `open_image` — Load an existing image file
- `save_project`, `open_project` — Save/reopen layers, masks, names, opacity, blend modes, visibility, dimensions, and the active layer in a local `.mcpproj` file
- `export` — Save to file (PNG/JPG/WEBP); auto-named when `path` is omitted
- `get_info` — View canvas dimensions, layers, undo/redo state

Projects are versioned JSON plus lossless PNGs in a ZIP archive, not PSD files. Saving replaces the target atomically; a failed load leaves the current document intact. Opening starts a fresh undo history. Supply a path in an existing directory.

### AI Image Generation & Editing
- `generate_image` — Text-to-image with `model="flux2"`, `"qwen21"`, `"anima"`, or `"minimax_h3"`
- `outpaint` — Extend left/right/top/bottom with `backend="flux2"` or `"qwen21"`; restores the original pixels exactly after generating the new margin

| Model | Generate / batch | Edit / masked edit / references | Outpaint | Default steps / guidance | Sampler | Scheduler |
|---|---|---|---|---|---|---|
| `flux2` (FLUX.2 Dev NVFP4, 32B) | Yes | Yes | Yes | 50 / embedded guidance 4 | Euler | Native `Flux2Scheduler` |
| `qwen21` (Qwen Image 2.1; custom preset) | Yes; detail, typography, alpha | Yes | Yes | 30 / 3 | Euler | Simple |
| `anima` (installed Aesthetic v1.1) | Yes; anime/illustration | No | No | 30 / 4 | `er_sde` | Simple |
| `minimax_h3` (installed ref2va INT8) | Yes; experimental stills | Yes; RGB | No | 20 / BasicGuider | `res_multistep` | Simple |

H3 is available through `generate_image`, `batch_generate`, `submit_generation_job` and `edit_image(backend="minimax_h3")`. It generates the minimum five-frame block and saves frame 0. Editing uses `<Picture 1>` for the canvas, subsequent pictures for references, and the final picture for an optional edit mask; the MCP restores outside-mask pixels afterward. `cfg` and `negative_prompt` are unused and reported when supplied with non-default values. Dimensions round up to multiples of 32 for generation; edits return the original canvas size. H3 allows 3600 seconds per image and does not provide transparent extraction or outpaint.

This adapts `minimax_h3_ref2img_single.json` to installed weights: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`, `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, and `minimax_h3_video_vae_fp16.safetensors`. The supplied file's hybrid model, Turbo LoRA and custom speed patches are not required or substituted silently. The installed H3 node makes the audio VAE optional for image references. Five frames are below its documented trained video duration, so still quality is experimental. A live 640x384, 20-step check generated a red mug and edited it blue successfully.

[Nine ready-to-open workflows](workflows/README.md) include H3 text-to-image, one-reference edits, two-reference composition, Flux/Qwen generation and edits, Anima generation, and general 4x upscaling. Each has UI and API JSON versions. UI copies are installed under ComfyUI's **MCP Image Presets** folder. Save in-memory MCP work and reconnect the server to load the new H3 tool options.

These are the active MCP presets for the installed models. Qwen generation, editing and outpainting all use the requested **30 steps / CFG 3** preset. This is a user preference, not the official Qwen 2.1 recommendation: ComfyUI's current 2.1 templates use **25 steps / CFG 1 / Euler / Simple**. Older Qwen base and Lightning/Flash recipes do not define the 2.1 defaults; no Lightning LoRA or AuraFlow shift is applied.

All generation paths use a full denoising schedule (`denoise=1.0` on KSampler). Text-to-image defaults to 1024x1024; Anima also suits approximately 1MP portrait/landscape sizes such as 896x1152 and 1152x896. Anima Aesthetic's documented tuning range is 30–50 steps/CFG 4–5; the preset uses the lower end. Anima Turbo and Qwen Lightning/Flash are not active backends.

FLUX.2 Dev uses `flux2-dev-nvfp4.safetensors`, `mistral_3_small_flux2_fp8.safetensors` with CLIP type `flux2`, and `flux2-vae.safetensors`. Its `cfg` argument controls embedded `FluxGuidance`, followed by `BasicGuider`; it is not conventional positive/negative CFG. The preset is 50 steps/guidance 4; 28 steps is a supported faster trade-off. NVFP4 quantizes weights and does not make Dev a four-step model. Negative prompts are unused and disclosed when supplied; express desired constraints positively. Klein's Qwen3 encoder, LoRAs, AuraFlow shift and old sampling recipes are not used.

Model selection is per call; batch jobs can each choose a different model. Omit `steps`/`cfg` to use the selected model's defaults; explicit values are preserved. `get_editing_capabilities` reports readiness separately for generation, editing and outpainting, including missing files/nodes. Missing or unsupported choices return an error without substituting another model. Defaults remain Flux for generation/outpaint and Qwen for editing. Upscaling and semantic selection retain their dedicated models.

Generation uses native ComfyUI nodes, including `Flux2Scheduler` for Dev and `TextEncodeQwenImage21` for Qwen. Flux/Qwen generation, editing and outpaint default to 1800-second timeouts; batch defaults allocate that time for each Flux/Qwen job, including jobs with an omitted model. PNG exports preserve generated alpha; existing visible canvas layers still composite normally. Outpainting uses reference-guided editing, with original pixels restored by the server rather than relying on the model to preserve them.

Qwen outpaint uses an opaque white margin and the guide's approximately 1-megapixel editing resolution before returning to the requested canvas size. Low-resolution transparent-margin tests produced blank/noisy borders; ordinary `edit_image` retains its existing resolution handling.

References: [FLUX.2 Dev model and sampling](https://huggingface.co/black-forest-labs/FLUX.2-dev), [ComfyUI Dev guide](https://docs.comfy.org/tutorials/flux/flux-2-dev), [BFL FLUX.2 prompting guide (Pro/Max; shared prose guidance)](https://docs.bfl.ai/guides/prompting_guide_flux2), [Qwen 2.1 guide](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1), [Anima model settings](https://huggingface.co/circlestone-labs/Anima#generation-settings).

The four saved `flux2_klein*.json` ComfyUI workflows now run Dev; filenames are retained for bookmarks. They use the same model/encoder/VAE and 50-step native scheduler, with shared width/height controls. Negative branches, the AuraFlow patch and the incompatible Klein N LoRA were removed. The N workflow therefore runs without its former adapter. Reference/prop branches, bypass states, output names and stitching remain available. The i2i workflow now opens an existing character sample instead of the missing `rx78.png`, and its view prompts preserve the supplied character. Select your own source in its first LoadImage node.

Migration backup and live evidence: `verification/flux2_dev/`. All four saved workflows executed at half dimensions with 50 steps; saved production dimensions were preserved. Real MCP checks covered 1024x768 generation, recoloring, masked references, exact outside-mask preservation, outpaint to 1280x768 with exact original pixels, undo/redo, and a Flux/Qwen/Anima batch. These are functional and visual smoke checks; character/accessory fidelity remains model-dependent. Save canvases and reconnect persistent MCP processes to load the changes, and reopen the saved workflows in ComfyUI.

### Master prompt integration

The image tools apply the relevant rules from the master files in `MASTER_PROMPT_DIR` (default: the repository-local `masters/` folder; set the variable in `.env` to point elsewhere):

| Tool/model | Source and application |
|---|---|
| Flux generation, edits and outpaint | `flux2prompt.txt` adapted to Dev: connected prose, preserve explicit details/lighting, exact lettering, positive desired outcomes, no negative branch; the source's historical Klein heading does not define the installed backend |
| Anima generation and batch jobs | `anima_prompt.txt`: hybrid tags/prose, explicit character-to-attribute binding, separate compatible negatives, no score tags for Aesthetic |
| Qwen generation and batch jobs | `qwen_image_2.1_system_prompt_t2i.txt`: detailed English observer prose (roughly 400–500 words), spatial layout, lighting, exact lettering in its original script |
| Qwen edits and outpaint | `qwen_image_2.1_system_prompt_edit.txt`: clear requested changes with untargeted content preserved, separate prose/lettering language rules, correct reference roles |
| `add_text` | Preserve exact supplied lettering; do not paraphrase or add image-prompt tags |

`get_prompt_guidance(model="anima", task="generation")` reads the applicable full master on demand, with its source path and SHA-256. It works through both MCP transports without accessing ComfyUI or the GPU. Essential rules are also included directly in `generate_image`, `edit_image`, `outpaint` and `batch_generate` descriptions, because some clients ignore initialize instructions. Reading the full guide is optional, not an extra prerequisite for every image.

For Qwen, use `get_prompt_guidance(model="qwen21", task="generation")` for the t2i master, or `task="editing"` / `"outpaint"` for the edit master. Pass the master's `rewritten_prompt` text as `prompt`, not its JSON wrapper. Map `wh_ratio` to generation `width`/`height`; `ratio_follow` identifies which image to open as the editing canvas. `edit_image` returns the canvas's dimensions, and `max_side` controls working detail, not aspect ratio. Use outpaint `direction`/`amount` or an explicit canvas operation for framing changes. The guides do not silently change the 1024 defaults, outpaint's internal resolution, or active sampling presets; `max_side=2048` remains available for detailed edits.

Qwen generation descriptions are English; edit descriptions are Chinese for Chinese instructions and English otherwise. Exact rendered text keeps the requested spelling/language. For edits without a specified text language, use the input's dominant text language, then the user's instruction language if the input has no text. Single-image edits/outpaint use natural image references; edits with references or an appended mask use numbered `<imageN>` tags.

The server removes a single surrounding prompt code fence. For Anima Aesthetic/unknown checkpoints, it removes standalone comma-separated `score_*` tags from both positive and negative prompts, while preserving quoted text verbatim; known Anima base checkpoints retain scores. Generation/batch results disclose any normalization. It does not truncate prompts to editorial word targets or automatically append negative tags that could conflict with the request.

Semantic requirements—intent, lighting, composition, character identity, suitable negatives and inspecting results—remain instructions for the calling LLM, not a guaranteed visual validator. The compact descriptions reflect the masters reviewed September 21, including the two newly supplied Qwen guides; full-guide reads always return current file contents. If a master changes, refresh the corresponding compact rules in `prompt_rules.py` and restart/reconnect the server. H3 stills use built-in image guidance; H3 video, audio/music, historical review and backup files do not override active MCP sampling presets. Original master files are unchanged.

### Instruction Editing
- `edit_image` — Qwen Image 2.1 (`qwen21`, default, custom 30 steps/CFG 3), FLUX.2 Dev (`flux2`, 50 steps/embedded guidance 4), or experimental MiniMax H3 (`minimax_h3`, 20 steps/BasicGuider). Qwen 2511 and the original Qwen backend are retired. For multiple Qwen inputs, the canvas is `<image1>` and references follow in order; a lone canvas uses natural wording without a tag. An optional white-to-edit mask is appended last and also preserves outside pixels exactly. Layer visibility masks are separate; supply `mask_path` or `region=[x,y,width,height]` explicitly. `max_side=1024` controls working resolution; use 2048 for more detail. Qwen accepts up to 16 total images in the installed node (10 recommended), including canvas and mask. Flux/Qwen default to a 1800-second timeout; H3 uses 3600 seconds.
- `get_editing_capabilities` — Live ComfyUI model/node availability per task (generation, editing, outpaint; notes also carry the GPU batching rule)
- `preview_canvas` — Render the current canvas so the assistant can inspect results

### Sessions & Batch (multi-document)
- Every canvas tool accepts an optional `session_id` (default `"default"`), so multiple documents can be edited independently in one server process
- `list_sessions` — List open sessions with size, layer count, and undo depth
- `close_session` — Close a session by `session_id`, freeing its canvas
- `batch_generate` — Queue a whole job list on ComfyUI in one pass (single WebSocket connection), await full batch completion, then export results to disk. Existing files are preserved using numeric suffixes such as `_1`.
- `submit_generation_job` — Return a job ID immediately; process images one at a time and export each result before starting the next
- `get_job_status`, `list_jobs` — Read completed-image counts, current item, results, and exported paths
- `cancel_job` — Let the current image finish and export, then skip remaining images; other jobs are unaffected

Background jobs and their status belong to one MCP server process. They survive an HTTP client disconnect while that process remains running; closing a stdio server or restarting the server loses the worker and status. Already exported files remain. Background jobs run serially and do not interrupt unrelated ComfyUI work. Use the returned job ID with the same server instance.

### Deterministic Editing (Pillow — no GPU needed)
- `crop`, `resize` — Transform every layer and mask; aspect-preserving resize centers the fitted document in transparent padding
- `rotate`, `flip` — Transform the active layer and mask; expanded rotation pads the other layers without rotating them
- `adjust` — Brightness, contrast, saturation, hue in degrees, sharpness; hue preserves alpha
- `levels` — Black point, mid point (gamma), white point adjustment
- `curves` — Per-channel tone curves (R/G/B) with control points
- `apply_filter` — 17 filters: blur, gaussian_blur, sharpen, contour, detail, edge_enhance, edge_enhance_more, find_edges, smooth, smooth_more, emboss, pixelate, posterize, solarize, invert, grayscale, sepia
- `add_text` — Text overlay with font, color, stroke support. The default font honors `font_size` (Pillow 10.1+); supply a font file for scripts outside its limited glyph coverage.

`apply_filter` accepts optional `radius` (Gaussian blur, default 2), `size` (pixelate, default 8), and `levels` (posterize, default 4). Color-only filters preserve alpha.

### Upscaling
- `upscale` — AI upscaling via ComfyUI: `general` uses `RealESRGAN_x4plus.pth`, `anime` and `face` retain their existing models. An installed model filename is also accepted; the default remains `anime`.

### Layer System
- `add_layer`, `select_layer`, `delete_layer`, `merge_down`, `reorder_layer`
- `set_blend_mode` — 12 modes: normal, multiply, screen, overlay, darken, lighten, color_dodge, color_burn, hard_light, soft_light, difference, exclusion
- `set_layer_opacity` — Per-layer transparency
- `set_layer_visibility`, `rename_layer` — Show/hide or rename an addressed layer
- `duplicate_layer` — Copy pixels, mask, and settings into an independent layer immediately above the original, and select it
- `translate_layer` — Move a layer and its mask by relative pixel offsets, clipping at the canvas edges

These layer operations accept an optional `index` (defaults to the active layer), preserve document dimensions, and are undoable.

Merge-down applies both layers' opacity, masks, and visibility once. Normal blends may have small 8-bit rounding differences. Combinations whose blend depends on deeper layers return an error without changing the document; keep those layers separate or explicitly flatten the document. Undo restores the original layers.

`edit_image` accepts optional `cfg`: Qwen uses CFG, while FLUX.2 uses embedded guidance. Omitting it preserves the current Qwen 30-step/CFG-3 and FLUX.2 50-step/guidance-4 presets. With `layer_index`, the source is that layer's raw pixels and replacement preserves its mask and other settings. With `output_mode="extract"` (Qwen only), the result becomes a new layer while originals retain their pixels and visibility. Inspect the returned preview and `has_transparency`; actual extraction quality depends on the model output. Qwen Image 2.1 already provides the required RGBA capability, so no separate Layered download is needed.

### Selections & Masks
- `select_rect` — Rectangular mask
- `select_ellipse` — Elliptical mask
- `select_object` — Heuristic color/region selection (red, blue, sky, dark, etc.)
- `semantic_select` — SAM 3 semantic object selection (text prompt via SAM 3.1, or point/box) — sets the active layer's mask
- `clear_mask` — Remove layer mask
- `export_mask` — Copy a layer mask to an independent grayscale PNG, optionally invert, grow/shrink with signed `expand`, then `feather`; leaves the document and history unchanged

For reusable AI masks: create a selection, call `export_mask(path="selection.png")`, then pass that path to `edit_image(mask_path="selection.png")`. White permits editing and black protects original pixels. The layer visibility mask remains separate; call `clear_mask` after export if it should no longer hide pixels. Mask export needs no GPU and replaces the explicitly named output file.

### History
- `undo` / `redo` — Full operation history (up to 20 steps)

### System
- `get_comfyui_status` — Check ComfyUI connection (starts it by default; `start_if_needed=False` for a passive check)
- `clear_vram` — Free GPU memory

> **GPU batching rule:** the host GPU is shared with the LLM backend docker (Ollama-compatible API on :11434; container name kept out of the repo) and Open WebUI. Check contention before proposing changes. `submit_generation_job` retains the complete input list in the MCP server and exports between images, but its server process must remain running independently of the LLM connection. If that lifetime is uncertain, use a detached host runner. `batch_generate` waits for the whole batch before exporting and does not detach. Stopping the LLM container ends the LLM session and still requires explicit user approval after background ownership is established. `searxng` is CPU-only and never needs stopping. Full procedure: `MEMORY.md` → "GPU Contention & Batching Rule".

## Installation

### Prerequisites
- **Python 3.10+**
- **ComfyUI** installed (auto-start supported — the server automatically launches ComfyUI when needed via `COMFYUI_PYTHON` + `COMFYUI_MAIN` env vars. You can also start ComfyUI manually on port 8188 if preferred.)
- **⚠️ VRAM Warning:** Running ComfyUI without auto-kill causes major OOM on shared GPUs (e.g., 32GB GPU with vLLM). Set `COMFYUI_AUTO_KILL=1` (recommended) with `COMFYUI_IDLE_TIMEOUT=5` to kill ComfyUI 5 seconds after each task, preventing resource exhaustion and Cline freezes.
- Required models installed in ComfyUI:

  **FLUX.2 Dev (generation, editing and outpaint):**
  - `flux2-dev-nvfp4.safetensors` (diffusion_models) — 32B, quantized; 50 steps / embedded guidance 4
  - `mistral_3_small_flux2_fp8.safetensors` (text_encoders) — Mistral Small, CLIP type `flux2`
  - `flux2-vae.safetensors` (vae)

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
  - `RealESRGAN_x4plus.pth` (upscale_models) — General/photo upscaling
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

> **Local artifacts (gitignored):** `.env`, `mcp_http.log`, `unittest_run.txt`, `batch_output/` and `verification/` hold machine-specific paths and runtime outputs (e.g. `.env` points at this machine's ComfyUI install and carries the tailnet `MCP_HTTP_ALLOWED_HOSTS` values; verification status JSONs contain local file paths). They are excluded by `.gitignore` — never commit or share them.

### Using with Open WebUI

The server can also serve Open WebUI over MCP **streamable-HTTP** — it registers as an **external tool server**, so it appears in chat as the **"Mcp Photoshop" tool** (enabled per-chat via the Tools panel, which requires the "Direct Tool Servers" user permission). Each client gets its own server process, so Cline's stdio connection is unaffected.

1. Double-click `run_openwebui.bat` in this folder. It sets `MCP_TRANSPORT=streamable-http` and `MCP_HTTP_HOST=0.0.0.0` (endpoint `/mcp`); the transport allowlist admits local/Docker clients by default and tailnet peers configured in `.env` (`MCP_HTTP_ALLOWED_HOSTS`), keeping the LAN itself blocked; logs append to `mcp_http.log`.
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
- OWUI caps tool calls at `AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER` (default 300 s). Long Flux/Qwen edits (up to 1800 s server-side) fail unless the container is run with `AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER=1800`. The current container already has it; if you ever recreate the container (e.g. via a local Docker manager script), include that env var in `docker run`.
- OWUI chat uploads land in OWUI storage, not Windows paths — `open_image` cannot see them directly. Use `export` to a shared folder, then `open_image` with that path.

### Environment Variables
See [`.env_example`](.env_example) for a complete reference. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TRANSPORT` | `stdio` | MCP transport: `stdio` (default, e.g. Cline) or `streamable-http` (Open WebUI; served at `/mcp`) |
| `MCP_HTTP_HOST` | `127.0.0.1` | Bind address for the streamable-HTTP server; `0.0.0.0` (set by `run_openwebui.bat`) also admits tailnet peers via the allowlist |
| `MCP_HTTP_ALLOWED_HOSTS` | *(empty)* | Optional comma-separated `host:port` patterns (e.g. a tailnet IP and MagicDNS name) admitted as Host + `http://` Origin; machine-specific, belongs in `.env` |
| `MASTER_PROMPT_DIR` | `masters/` (repository-local, next to `server.py`) | Directory containing the current Flux, Anima and Qwen master files; read by `get_prompt_guidance` |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI API endpoint |
| `COMFYUI_START_CMD` | *(optional)* | Path to ComfyUI start .bat (alternative to COMFYUI_PYTHON + COMFYUI_MAIN) |
| `COMFYUI_PYTHON` | *(required for auto-start)* | Path to ComfyUI's embedded `python.exe` |
| `COMFYUI_MAIN` | *(required for auto-start)* | Path to ComfyUI's `main.py` |
| `COMFYUI_ARGS` | `--windows-standalone-build` | Extra startup arguments |
| `COMFYUI_AUTO_KILL` | code default `0`; `1` recommended | Kill after generation (`0`=VRAM pressure mode, `1`=idle timeout mode) |
| `COMFYUI_IDLE_TIMEOUT` | code default `60`; `5` recommended | Seconds of inactivity before auto-kill (when AUTO_KILL=1); 5 s prevents Cline freezes |
| `COMFYUI_START_TIMEOUT` | `180` | Seconds to wait for ComfyUI to start |
| `VRAM_PRESSURE_THRESHOLD_MB` | `8192` | Kill ComfyUI if free VRAM drops below this (MB) |
| `WEBSOCKET_TIMEOUT` | `600` | Seconds to wait for workflow completion |
| `MAX_UNDO_STEPS` | `20` | Maximum undo history entries |
| `LOG_LEVEL` | `WARNING` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

## Architecture

```
mcp-photoshop-server/
├── server.py           # MCP server; 50 tools including editing registrations (canvas tools accept session_id)
├── editing.py          # Instruction editing backends (qwen21 / flux2) + semantic selection + live capabilities
├── prompt_rules.py     # Tool-level master guidance, source reader and conservative prompt normalization
├── config.py           # Configuration & model names
├── comfy_client.py     # ComfyUI API client (REST + WebSocket)
├── canvas.py           # Layered document with blend modes + undo/redo
├── session.py          # Per-session document management
├── project.py          # Atomic layered-project save/load (versioned ZIP, JSON and PNG)
├── jobs.py             # Process-owned background generation, progress and graceful cancellation
├── requirements.txt    # Python dependencies
├── masters/            # Master prompt files (Flux/Anima/Qwen) read by get_prompt_guidance
├── run_openwebui.bat   # Streamable-HTTP launcher for Open WebUI (sets MCP_TRANSPORT=streamable-http)
├── MEMORY.md           # Project memory bank
├── tests/              # Regression suite (202 tests)
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
- [x] Layer controls, local layered projects, reusable edit masks, selected-layer edits, and Qwen extraction (2026-09-22; CPU coverage plus real Qwen extraction/masked-layer smoke checks)
- [x] Background generation with progress, graceful cancellation, and per-image exports (2026-09-22; survives HTTP disconnect while the MCP process runs)
- [x] SAM-based semantic selection — delivered via the `semantic_select` tool (SAM 3 text/point/box prompts, live-verified 2026-09-17)
- [x] Multi-session support — all canvas tools accept `session_id`; `list_sessions`/`close_session` manage multiple open documents (2026-09-18)
- [x] Batch processing — `batch_generate` queues a whole job list on one WebSocket connection, awaits completion, and exports all results (live-verified 2026-09-18)
- [x] Legacy ControlNet/Redux workflow repair — Flux2 pixel-dim latents, `ControlNetApply.conditioning`, `CLIPVisionLoader` + `crop`/`strength_type`, plus live capability pre-checks in `get_editing_capabilities` (2026-09-18)
- [ ] Additional ComfyUI custom nodes integration (beyond ControlNet/Redux/SAM3 — e.g. new node packs)
