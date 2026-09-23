# Memory Bank — MCP Photoshop Server

> Last updated: 2026-09-22
> Current editing behavior: `edit_image` defaults to Qwen Image 2.1 (`qwen21`, user-requested 30 steps, CFG 3, Euler/simple). `qwen` and `qwen2511` are retired. `flux2` now selects FLUX.2 Dev NVFP4, 32B, at 50 steps/embedded guidance 4 with Mistral Small FP8. Earlier Klein sampling values below are historical.
> Location: project root of this repository (mcp-photoshop-server)

## H3 redo review and sizing clarification

- Reviewed the supplied September 22 redo images/report and current code without running generation or tests. Anime local output cleanly replaces the lantern with a glowing maple-leaf lantern. The realistic retry removes the first attempt's white-out, but the crop comparison does not clearly establish a light-oak armchair replacement; do not repeat 4/4 semantic success as independently verified. Global preservation looks good; detail gains are not quantified.
- The active master's latest sizing edit conflated two paths. Corrected both copies: MCP uses `prepare_image`/`max_side`, nearest-32 working axes, and final resize to source size; standalone reference graphs use 0.75MP `ImageScaleToTotalPixels` and save their generated dimensions. `adapt_canvas` only handles reference-video frames. A 1024x1024 final file alone does not prove internal resolution.
- Current `make_edit_mask` feathers inward and final compositing restores fully black mask pixels. Global unmasked edits can drift; that differs from outside-selection pixels in a masked MCP edit. Added acceptance checks for both the requested change and integration, including insufficient edits that merely resemble the source. Preserved the original reports as supplied; recorded qualifications in workflow guidance and the master collection changelog.

## H3 initial supplied benchmark visual review

- User requested no further tests and dropped Qwen/Flux benchmarking. Reviewed existing PNGs only; no generation, LLM check or test rerun. The folder currently contains eight PNGs (seven individual renders plus grid), not the reported nine.
- The summary overstates local-edit success: `edit_anime_local_h3_attempt1_fail.png` has a white panel, `edit_anime_local_h3.png` still has a black panel, and `edit_realistic_local_h3.png` has a visible rectangular seam through the floor/left chair. Do not record the anime retry or realistic local edit as clean. Global max edits largely preserve structure; no controlled max/match comparison or measured speed data supports a universal ranking.
- Updated H3 master and compact edit guidance to inspect the full region and boundary, reject unintended panels/seams, restore the original before an authorized fresh-seed retry, and report persistent defects. Exact outside-mask compositing is not proof of visual integration. See `workflows/README.md` for the qualified review; prior 206-test results predate these wording-only changes.

## 2026-09-22 H3 pseudo-image master

- Added `masters/minimax_h3_pseudo_image_master.txt` and installed it in the configured `MASTER_PROMPT_DIR`. `get_prompt_guidance` reads it fresh, with path/SHA-256, for H3 generation and editing. Compact task-specific rules are embedded in generation, batch and edit descriptions; H3 outpaint remains unsupported.
- The original local master draws on the linked pseudo-image graph, its author's infographic examples and official H3 documentation. It returns `rewritten_prompt`, `wh_ratio`, `ratio_follow`; only the decoded prompt goes to MCP/ComfyUI. Ratios map to tool dimensions or an explicit canvas operation, not new tool arguments. It covers medium-aware still composition, exact text/data and `<Picture N>` roles, without importing the video/audio format or changing sampling presets.
- The three H3 UI presets include usage notes; the master is copied beside them and included in the nine-preset ZIP. Existing masters are unchanged. Local Qwen passed four authoring-format examples; 206 tests and fresh MCP stdio source/hash checks passed. This validates integration and sampled prompt behavior, not GPU image quality. Reconnect persistent MCP clients after saving canvases to load the new descriptions.
- The H3 pseudo-image master is the only tracked master file: it is original repo content, and `tests/test_h3_stills.py` reads it from the repo `masters/` folder (the README links it the same way). Upstream-supplied Flux, Anima and Qwen masters remain local and untracked; when `MASTER_PROMPT_DIR` points elsewhere, install the H3 master there too.

## 2026-09-22 Idle cleanup follow-up

- Reproduced two gaps after the adopted-PID change: attaching to an already-running backend did not arm an idle timer, and a dead owned Popen handle prevented fallback to an adopted process.
- Backend checks now arm missing timers without extending existing ones. Workflow and upload activity suspend cleanup; completion, failure and cancellation rearm the queue-guarded check. Finished workflows no longer wait for unrelated queue entries before returning their output.
- Adoption is local-only and verifies the configured Python executable, absolute main script and process creation time. The listener and identity are checked again before termination. Stale handles no longer block adoption, failed termination can retry, and a new operation invalidates an in-flight idle check or waits for an already-started shutdown.
- The timer still belongs to the MCP process. If that process is forcibly closed, its timer cannot run; a later active backend check adopts a verified orphan. Passive status checks (`start_if_needed=False`) do not start ComfyUI again.
- Regression suite: 202 tests pass, including 15 focused lifecycle regressions. Real ComfyUI checks passed for both an adopted process and a fresh launch followed by a model-free workflow: the configured port stopped listening after 6.28 s and 5.83 s respectively with a five-second idle timeout. Passive status stayed off. An isolated CPU backend passed the same checks, including its non-default port. Evidence: local `mcp_idle_shutdown_live_8188.json`, `mcp_idle_shutdown_live_8189.json` and `idle-shutdown-unit-tests.txt` artifacts. No image models were loaded or containers stopped.

## 2026-09-22 Repo sanitization for git push

- Removed all machine-specific values from tracked files before push: tailnet IP, MagicDNS hostname, LAN IP examples, node names, the LLM container name, PIDs.
- Tailnet transport admission now comes from `MCP_HTTP_ALLOWED_HOSTS` (comma-separated `host:port` patterns) in the gitignored `.env`; `server.py` appends them to the Host allowlist (421) and mirrored `http://` Origins (403). `run_openwebui.bat` keeps the `MCP_HTTP_HOST=0.0.0.0` bind but no longer embeds tailnet values in comments.
- The LLM docker's container name is generalized to "the LLM container" in `server.py` instructions, `editing.py`, README and this file (the name stays out of the repo; identify it via `docker ps` / `:11434`).
- `unittest_run.txt` added to `.gitignore`; README documents `MCP_HTTP_HOST` / `MCP_HTTP_ALLOWED_HOSTS`; `.env_example` carries commented placeholders.
- Verification: full unit suite green (187). HTTP server restarted via `run_openwebui.bat` (the temporary scheduled task that had kept it alive was deleted); live raw-socket probes: MCP `initialize` with tailnet-IP and MagicDNS Host headers -> 200, LAN Host -> 421, plain GET -> 406 "Not Acceptable: Client must accept text/event-stream" (the expected Mac-probe response). Staged-tree token sweep shows no machine-specific values (only a generic "MacBook" mention).

## 2026-09-22 Tailnet HTTP access for the Mac (MCP_HTTP_HOST)

- Made the streamable-HTTP server reachable from the user's MacBook over Tailscale. `run_openwebui.bat` sets `MCP_HTTP_HOST=0.0.0.0`; `server.py` reads it (`os.getenv("MCP_HTTP_HOST", "127.0.0.1")`) and passes it as `host=` to `FastMCP`. Note: `MCP_HOST`/`FASTMCP_HOST` env vars do NOT work here — mcp SDK 1.29.0's `FastMCP.__init__` always passes the constructor host explicitly to `Settings`, which overrides env-based settings.
- `tailscale serve` (the no-code alternative) is broken on this PC: the CLI hangs on every variant (any port, --bg/--yes, stdin closed) and never writes a serve config; GUI restart did not help. Needs an admin service reset or Tailscale reinstall to revisit.
- Tailnet peers are admitted via `MCP_HTTP_ALLOWED_HOSTS` in the gitignored `.env` (comma-separated `host:port` patterns for the PC's tailnet IP and MagicDNS name). `server.py` appends them to the `_TRANSPORT_SECURITY` Host allowlist (421) and mirrored `http://` Origins (403). The SDK validates Host/Origin headers only, not the peer IP.
- Bound 0.0.0.0 (not the tailnet IP alone) so Open WebUI's `host.docker.internal` -> loopback path keeps working; the allowlist keeps LAN out (verified: a LAN-style Host header -> 421).
- Mac client URL: the PC's tailnet IP or MagicDNS name, port 8000, path `/mcp` (exact values in `.env`), Streamable HTTP, no auth. If a Mac probe hangs instead of erroring, Windows Firewall is eating it: add inbound TCP 8000 restricted to the tailnet (`remoteip=100.64.0.0/10`).
- Verification: 187 unit tests pass (new `run_tests.bat`, plain `unittest`); log shows `Uvicorn running on http://0.0.0.0:8000` and a 406 response from the tailnet IP.

## 2026-09-22 Auto-kill fix: adopted PIDs and probe-safe idle timer

- Fixed the reported ComfyUI-not-killed failure mode. Two defects in `comfy_client.py`: `_kill_process()`/`kill_comfyui()` only terminated the live Popen handle started by the current MCP server, so a ComfyUI started in a previous session (orphaned, holding port 8188) was never killed; and `start_comfyui()` cancelled the pending idle-kill timer on every call, including read-only probes, so the timer never fired.
- New behavior: `start_comfyui()` adopts an already-running ComfyUI (PID found via `netstat -ano` on the `COMFYUI_URL` port) into `_adopted_pid`; the pending idle timer is cancelled only on the fresh-launch path; `_kill_process()` prefers the owned Popen and falls back to the adopted PID via `taskkill /F /T`; `kill_comfyui()` clears both handles and logs when nothing was killed.
- Verification: full unit suite green (187 tests, including 6 new adopted-PID / idle-timer / netstat-parsing tests). The stuck external ComfyUI (port 8188) was terminated and the port confirmed free.

## 2026-09-22 Live GPU validation and connection confirmation

- The running HTTP MCP server exposes all 50 tools; Open WebUI's native client connection and a tool call passed. The user separately confirmed Cline's live connection.
- Real Qwen 2.1 extraction and masked selected-layer editing passed at the unchanged 30 steps/CFG 3, Euler/simple, 1024×768. Extraction: 27.27 s, sampled peak whole-device memory 22,114 MiB (21.6 GiB). Masked recolour: 22.36 s, peak 25,731 MiB (25.1 GiB). These are individual smoke measurements, not a broad performance benchmark.
- Extraction preserves the original layer and yields a usable transparent cabin cutout; mild coloured edge fringe and background alpha noise (typically 0–1/255) remain. Masked editing preserved every outside-mask pixel, the other layer, metadata and visibility mask exactly. Both undo/redo checks passed. Regenerated pixels inside the edit mask have alpha 248–255; their original alpha is not guaranteed.
- Diagnostic artifacts: `gpu-validation-results.md`, `gpu-validation-restoration.json`, and the two GPU run folders in the local orchestration artifacts. The runner's helper argument collision was fixed after extraction; only the unstarted second test was resumed. No production code change was needed.
- With explicit user approval, the local LLM container was stopped temporarily, ComfyUI models were unloaded after testing, and the container was restarted. Its models endpoint returned HTTP 200 with `qwen3.8-27b`. Open WebUI and searxng stayed running. Test documents were isolated and closed after saving; user documents and model presets were unchanged.

## 2026-09-22 Layer, project, mask and background-job tools

- Completed the remaining Qwen-assisted tasks Q07–Q11. Added undoable visibility, rename, independent duplication, and translation with masks; atomic local `.mcpproj` save/open preserves layered state and opens with fresh undo history. This format is ZIP/JSON/PNG, not PSD.
- `export_mask` creates an independent grayscale PNG with optional invert, grow/shrink, and feather. `edit_image(layer_index=...)` replaces only that layer's pixels; `output_mode="extract"` uses installed Qwen 2.1/RGBA to add a subject layer while retaining originals. Check its preview and actual `has_transparency`; no Layered model download was needed.
- `submit_generation_job` returns an ID and exports one image before starting the next. `get_job_status`/`list_jobs` expose progress; `cancel_job` finishes the current image and skips the rest without interrupting unrelated work. Workers/status survive HTTP disconnect only while the same MCP process runs, not restart or closure of a stdio process. Exported files persist. Existing `batch_generate` still waits for its complete batch before exporting.
- Verification: 176 unit tests pass; fresh stdio and streamable-HTTP MCP checks pass with 50 tools. The HTTP check disconnected one client during a mocked generation, then retrieved status and the exported file through another. Evidence: `feature-unit-tests.txt` and `feature_mcp_smoke.json` in the local orchestration artifacts. Initial inference tests were mocked; real GPU validation is recorded in the entry above.
- No persistent MCP restart, container stop, model download/removal, sampling-preset change, or ComfyUI core edit. Preserve open work before reloading server processes and reconnecting clients. Changes are uncommitted; all Qwen coding requests have finished.

## 2026-09-21 MCP reliability fixes

- Reviewed Qwen3.8-27B proposals fix batch filename collisions, overly broad process shutdown, document transforms/mask alignment, filter MCP arguments, color alpha/hue, merge-down opacity/masks/visibility, and default text size. Merge-down refuses blend combinations that depend on deeper layers rather than silently changing the image.
- Added `upscale(model="general")` for the installed RealESRGAN model and optional `edit_image(cfg=...)`; existing sampling defaults are unchanged. Batch exports happen after the batch wait, and `batch_generate` does not detach a background worker. Automatic shutdown targets only this MCP process's managed ComfyUI handle and defers when queue state is busy or unknown.
- Validation at this stage: 150 unit tests and a fresh CPU-only stdio MCP smoke check passed. No new GPU-generation test, container stop, model download/removal, persistent MCP restart, or ComfyUI core change. The planned layer/project/mask/background-job additions were completed in the September 22 entry above.

## 2026-09-21 FLUX.2 Dev NVFP4 migration

- Active Flux weights: `flux2-dev-nvfp4.safetensors`, `mistral_3_small_flux2_fp8.safetensors` (CLIP type `flux2`), `flux2-vae.safetensors`. Native `FluxGuidance(4)` → `BasicGuider` → Euler/`Flux2Scheduler(50)` → `SamplerCustomAdvanced`. `cfg` means embedded guidance for Dev, not conventional CFG. No negative branch, Klein LoRA, Qwen3 encoder or AuraFlow shift. NVFP4 is quantization, not step distillation. BFL's Dev card uses 50/4 and offers 28 steps as a speed trade-off; see https://huggingface.co/black-forest-labs/FLUX.2-dev and https://docs.comfy.org/tutorials/flux/flux-2-dev .
- The partial migration already on disk used the correct model/encoder and guidance graph. Fixed remaining stale four-step instructions, negative-prompt guidance and timeout gaps. Flux generation/edit/outpaint now use 1800 s by default; batches budget 1800 s per Flux/Qwen job, including omitted-model Flux jobs. Nonempty Flux negatives are unused and explicitly reported. Capabilities state guidance type and lack of negative support. Prompt rules adapt the personal `flux2prompt.txt` (historical Klein heading) to Dev; full-source access still works.
- Migrated all four saved `ComfyUI/user/default/workflows/flux2_klein*.json`; kept filenames, output paths, canvas sizes and optional bypass states. Replaced Klein loader/encoder and KSamplers with Dev's native graph, linked scheduler dimensions to latent controls, removed old negative encoders, AuraFlow patch and incompatible Klein N LoRA. Removed stale embedded JSON recipes from notes. N now runs without its former adapter; old model/LoRA files were not deleted. Backup: `verification/flux2_dev/workflows_before_dev.zip`.
- Fixed pre-existing i2i missing `rx78.png` default using the existing `character_sheet_front_debug_00001_.png`; its prompts now preserve the actual reference instead of forcing an unrelated elf. Optional reference slots remain bypassed; select real files before enabling them. The prop bow prompt now explicitly includes both tips and a bowstring after visual review found an incomplete bow.
- Verification: 108 regression tests passed. Real MCP stdio: 39 tools; live capabilities and prompt guide; 1024x768 Dev generation, blue-cabin edit, undo/redo, reference + mask edit, outpaint to 1280x768, and mixed Flux/Qwen/Anima generation all passed. Outside-mask and original outpaint pixels were checked exactly. All four saved workflows produced images at half dimensions with the full 50 steps, including separate character views, props and stitched sheets; their saved production dimensions were preserved. Evidence: `verification/flux2_dev/live_mcp_status.json`, `live_workflows_status.json`, `workflow_validation.json` and PNGs. Functional smoke tests do not guarantee perfect costume/prop consistency.
- Existing 2048x2048 Dev job was allowed to finish. No containers were stopped, no new model downloads or ComfyUI core changes. Persistent MCP sessions were not restarted. Save canvases, reconnect MCP clients and reopen workflows to load changes.

## 2026-09-21 Klein 9B Base swap — historical verification & process pitfalls

- Live verification of `flux-2-klein-base-9b-fp8.safetensors` (28 steps / CFG 4 / Euler / `Flux2Scheduler`, seed 42, 1024x1024) passed standalone: ~30 s generation with the local LLM container stopped, output at `klein_base_verify/result.png` in a local scratch dir (script bug fixed: ComfyUI history entries use the `filename` key, not `name`).
- MCP-level smoke test then exercised the real on-disk `server.generate_image` path outside Cline (`mcp_smoke_test.py` in a local scratch dir): profiles resolve `flux2` → fp8 base at 28/4.0 for generation/editing/outpaint, and a 512x512 cabin image was produced with the local LLM container running concurrently. Evidence: `klein_base_verify/mcp_smoke.png`.
- Pitfall 1: a long-lived Cline MCP server process keeps pre-edit code in memory. After the model swap it kept returning `flux2 is unavailable: ['model']` even though ComfyUI listed the new checkpoint — fixed by killing the stale `python.exe` MCP process (Cline shows the server "Not connected" afterwards; the user must reconnect it).
- Pitfall 2: a ComfyUI child process spawned by that stale server survives its parent (orphaned, port 8188 held, idle-kill timer dies with the parent). `kill_comfyui`/auto-kill can only free processes the current client spawned itself. Check `netstat -ano | findstr :8188` after killing MCP processes and `taskkill` any orphan.
- Note: `start "" python ...` from the chat shell hangs for 30 s and loses the child (console inheritance); use a `subprocess.Popen` launcher with `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`.

## 2026-09-21 Master prompt integration

- Reviewed the active image masters and collection guide (the folder is now referenced by `MASTER_PROMPT_DIR`, defaulting to the repository-local `masters/`). `flux2prompt.txt` applies to Flux; `anima_prompt.txt` applies to Anima. The newly supplied `qwen_image_2.1_system_prompt_t2i.txt` applies to Qwen generation/batches; `qwen_image_2.1_system_prompt_edit.txt` applies to Qwen editing/outpaint. Production/numbered H3 and music/audio files are different tasks; do not inject their schemas into Photoshop image prompts. Original master files were not edited.
- `prompt_rules.py` adds compact master rules to the actual `generate_image`, `batch_generate`, `edit_image` and `outpaint` descriptions, covering clients that discard initialize instructions. `get_prompt_guidance(model, task)` exposes the current full applicable Flux/Anima/Qwen master, fixed source path and SHA-256, without GPU startup. Tool count is 39. `MASTER_PROMPT_DIR` overrides the source folder. Full guide reads are fresh; compact rules must be reviewed when the masters change. Missing Qwen files report an error rather than falling back to unrelated masters.
- Preserve explicit subjects, count, identity, outfits, setting, light, composition and exact visible text; bind attributes to each character. Flux uses connected prose (usually 30–80 words, never a hard limit). Anima uses hybrid tags/prose with checkpoint-aware score rules and compatible separate negatives. A sufficient brief/YOLO proceeds directly; ask one question only for material ambiguity or requested guidance. MCP arguments contain raw prompt text, not the standalone masters' code-block wrappers or H3 fields.
- Mechanical enforcement strips a single surrounding prompt fence and removes standalone comma-separated score tags from BOTH prompts for Anima Aesthetic/unknown checkpoints. Quoted lettering and known base scores are preserved. Generation/batch results disclose normalized prompts. No length truncation, semantic rewriting, or automatic negative-tag injection. Artistic/semantic compliance still depends on the calling LLM; these rules are not a guaranteed visual validator.
- `add_text` explicitly instructs exact lettering. Existing sampling presets stay Qwen 30/3, Flux 4/1, Anima 30/4. Tests verify quoted text, character prose, separate negatives, mixed batches, source refresh and tool descriptions. Save canvases before restarting/reconnecting the persistent MCP processes to load these changes.
- Qwen generation uses detailed English observer prose (roughly 400–500 words), spatial inventory, explicit light/materials and exact lettering in its original script. Qwen editing uses decisive requested changes, preservation without weakening the effect, and separate prose/text language decisions. Single-image edits/outpaint no longer add `<image1>`; multi-input edits retain numbered canvas/reference/mask roles, including mask-only cases. The wrapper adds no paragraph breaks. Language and artistic rules remain LLM responsibilities.
- Adapt Qwen `rewritten_prompt` to raw `prompt`; do not send master JSON to the image model. `wh_ratio` maps to generation dimensions; `ratio_follow` maps to the intended canvas. Edit output size remains the canvas size; `max_side` is working resolution. Outpaint uses `direction`/`amount`. No unsupported size fields, automatic JSON parsing, 2K default change or sampling changes. Both task-specific guides are available through `get_prompt_guidance(model="qwen21", task=...)`.
- Verification: all 103 regression tests passed. A fresh real MCP stdio connection listed 39 tools, confirmed master rules in all four image-tool descriptions, and successfully read Flux, Anima and Qwen guidance without starting ComfyUI. Evidence: `verification/master_prompt_mcp_status.json`. The persistent HTTP server was not restarted.
- After adding the Qwen files: all 106 tests passed, including task-based source selection, missing-file errors, single-image wording and mask-only numbering. Real MCP stdio calls returned exact full text and matching SHA-256 for Qwen generation/editing/outpaint and unchanged Flux/Anima sources; all four tool descriptions selected the correct Qwen master. Evidence: `verification/qwen_master_prompt_mcp_status.json`. No GPU inference or persistent-server restart was needed.

## Active sampling presets (updated 2026-09-21)

| Installed backend | Steps | CFG | Sampler | Scheduler |
|---|---|---|---|---|
| `qwen21` | 30 | 3 | Euler | Simple |
| `flux2` (Dev NVFP4, embedded guidance) | 50 | 4 | Euler | Native `Flux2Scheduler` |
| `anima` (Aesthetic v1.1) | 30 | 4 | `er_sde` | Simple |

- User requested these presets. Qwen's 30/3 is a custom preference; the [official Qwen 2.1 workflow](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1#sampler-settings) still recommends 25/1. Do not describe 30/3 as official or attach older Lightning/Flash/AuraFlow recipes to 2.1.
- Sampling defaults live in `editing.model_profiles` and are consumed by generation, batch generation, editing and outpainting. Qwen editing/outpaint CFG now reads that profile instead of a hard-coded 1. The capability report includes sampler, scheduler and denoise. Removed unused global 20-step/CFG-1.5 constants from config.py.
- Denoise is 1.0 (full schedule); generation defaults to 1024x1024. Anima's documented range is 30–50 steps/CFG 4–5 at roughly 1MP, e.g. 1024x1024 or 896x1152. Existing explicit steps/CFG arguments remain honored where exposed; samplers/schedulers use the profile.
- [FLUX.2 Dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) uses embedded guidance 4 with 50 steps, Mistral Small and flux2 VAE. Both Klein checkpoints are retired from these workflows and the MCP backend. [Anima Aesthetic](https://huggingface.co/circlestone-labs/Anima#generation-settings) is not Turbo. README lists the same active values.
- Verification: all 92 tests passed. Live Qwen 30/3 generation, cabin recolor, and outpaint completed; outpaint preserved original pixels exactly. Samples and settings are in `verification/sampling_presets/`. These are smoke checks, not a quality comparison against the official preset.

## 2026-09-20 Model selection by task

- `generate_image(model=...)` and per-job `batch_generate` support `flux2`, `qwen21`, and `anima`. `edit_image(backend=...)` and `outpaint(backend=...)` support `qwen21` and `flux2`; Anima remains text-to-image only. Upscaling/SAM retain their dedicated models. Defaults remain Flux generation/outpaint and Qwen editing.
- Shared live model discovery checks the required nodes separately for generation/editing/outpaint. Tool descriptions, model enums and `get_editing_capabilities` expose the choices to Cline and Open WebUI. Unknown/unavailable choices fail before submission without silent substitution.
- Omitted sampling settings use Flux 4 steps/CFG 1, Qwen custom 30/3, Anima Aesthetic 30/4 (`er_sde`/simple, per the model card). Explicit values are preserved. Removed the Flux step cap and arbitrary two-reference limit. Generation uses native encoders and Flux2Scheduler, replacing the custom sectioned encoder/BasicScheduler path. Qwen text-to-image follows the official TextEncodeQwenImage21 + EmptyLatentImage graph with the user's sampling preset.
- Outpaint now uses reference conditioning and restores original pixels exactly, including alpha. Qwen uses an opaque white margin and resolution=1024 in its encoder, then resizes back to the requested canvas size. Low-resolution transparent/masked margins failed visual checks; those approaches were removed. Ordinary Qwen instruction editing remains unchanged and was live-tested with a cabin recolor.
- A live mixed-model batch generated all three images; Flux and corrected Qwen outpaint passed visual and original-pixel checks. 92 regression tests passed. Evidence and samples: `verification/model_selection/`. Sources: the ComfyUI Qwen Image 2.1, Flux Klein 9B and Anima guides, installed official workflow templates, and CircleStone's Anima model card. No model downloads or ComfyUI core changes.
- Save canvases and restart/reconnect the Photoshop MCP processes to load new schemas; persistent HTTP processes keep old definitions until restarted.

## 2026-09-20 Automatic startup checks

- ComfyUI-dependent tools start ComfyUI automatically. `get_editing_capabilities` and `get_comfyui_status` now do so by default too; previously they returned offline errors that encouraged clients to require manual startup. Both accept `start_if_needed=False` for a passive probe and explain that dependent tools can start the backend.
- Automatic startup is stated in individual tool descriptions, because Open WebUI may ignore MCP initialize instructions. Startup cancels a pending idle timer before health checks. Child stdout/stderr no longer inherit the MCP protocol stream.
- Verified from fully stopped ComfyUI using the real MCP stdio transport: automatic startup in 45.8 seconds, Qwen 2.1 available, status and all 38 tool definitions readable, model-free workflow output retrieved. No LLM container was stopped and no generation model was loaded. Evidence: `verification/autostart_stdio_status.json`; regression suite at that point: 81 tests passed (later model-selection work brought it to 92).
- The already-running HTTP server on port 8000 still has the previous tool definitions. Save open canvases, restart the Photoshop server processes, then refresh/reconnect the client tools to activate the changes.

## 2026-09-20 Tool Cleanup (dead/legacy removal)

- Removed 5 tools: `img2img` + `character_transform` (legacy Flux2 denoise transforms; `edit_image` is the instruction-edit path), `inpaint` (Flux Kontext chain — its UNET/clip_l/t5xxl files are no longer installed; `edit_image` with `mask_path`/`region` is the masked-edit path), `controlnet_generate` (`models/controlnet` empty), `style_transfer` (`models/style_models` + `models/clip_vision` empty — `edit_image` with `reference_paths` is the style/identity path). Tool count 43 → 38.
- `outpaint` was dead on the same missing Flux.1 chain and was rewritten onto the installed Flux2 Klein chain: `LoadImage` → `ImageScale` → `UNETLoader`/`CLIPLoader`(flux2)/`VAELoader` + `Flux2KleinSectionedEncoder` → `VAEEncode` → `SetLatentNoiseMask` (LoadImage MASK output: transparent padding = generate, original content = protected). `steps` default 30 → 6 (Flux2 distilled cap), `guidance` param dropped.
- Fixed a latent `outpaint` bug the live test exposed: only the active layer was resized, leaving sibling layers (and masks) at the old size so `composite()` crashed with "images do not match". `outpaint_tool` now extends every layer (and every mask) to the new canvas size, transparent where new. 3 regression tests in `tests/test_outpaint.py`.
- Live-verified 2026-09-20 end-to-end on ComfyUI 0.33.0 (`verification/_live_outpaint_flux2_test.py` → `verification/outpaint_flux2_live.png`): 512x512 canvas outpainted right by 256px → 768x512, original content preserved exactly, new region filled. Fill content was semantically poor because the local LLM container was holding VRAM (GPU contention — see §11); the chain/graph itself is proven.
- Dead code removed: `build_img2img_kontext_workflow`, `build_inpaint_workflow`, `build_controlnet_workflow`, `build_style_transfer_workflow` (server.py); `controlnet_capabilities`, `style_transfer_capabilities` + their capability-report sections (editing.py); `MODEL_KONTEXT` (config.py); `tests/test_custom_nodes.py` (tested only the removed tools).
- Docs synced: README (feature list, model list, examples, tool counts) and this file's §5/§7/§9/§10.

## 2026-09-20 Open WebUI integration (streamable-HTTP)

- The server now serves two MCP clients: Cline (stdio, unchanged — `app.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))`) and Open WebUI (streamable-HTTP on `127.0.0.1:8000`, path `/mcp`), launched by `run_openwebui.bat` (sets `MCP_TRANSPORT=streamable-http`, logs to `mcp_http.log`).
- The `FastMCP` constructor now passes explicit `TransportSecuritySettings` allowing `host.docker.internal:*` — the SDK's auto DNS-rebinding protection only allows 127.0.0.1/localhost/::1 and would answer 421 to the Docker container's Host header.
- `ComfyUIClient._do_idle_kill` is now guarded by `_idle_kill_guarded()`: it checks ComfyUI `/queue` (queue_running/queue_pending) before killing and defers by `COMFYUI_IDLE_TIMEOUT` while any job is active — one client's idle timer can no longer kill ComfyUI under another client's running job.
- Open WebUI 0.11.3 connects under Settings → **Admin** → Integrations → External Tool Servers → Type "MCP (Streamable HTTP)", URL `http://host.docker.internal:8000/mcp`, Auth None. The personal (user) Integrations page only accepts OpenAPI servers — MCP is admin-only.
- OWUI 0.11.3's MCP client calls `initialize()` but DISCARDS the returned instructions — the GPU_BATCH_RULE text does not reach the OWUI model. Compensate in the OWUI model's system prompt: (1) use a distinct `session_id` per chat, (2) a summary of the GPU batching rule. Cline keeps receiving the instructions natively.
- OWUI caps tool calls at `AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER` (falls back to `AIOHTTP_CLIENT_TIMEOUT` = 300s). Long Qwen edits (server default up to 1800s) fail in OWUI unless the container is recreated with `AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER=1800` (see `verification/recreate_open_webui.bat`).
- Verified 2026-09-20: full handshake from inside the container (43 tools + canvas round-trip via `verification/_probe_mcp.py`, run with `docker cp` + `docker exec`), and flux2 generation through the HTTP endpoint (`verification/_probe_gpu.py`).
- Known gap: OWUI chat uploads land in OWUI storage, not Windows paths — `open_image` cannot see them directly. Use `export` to a shared folder, then `open_image` with that path, until attachment mapping is added.
- **2026-09-20 (session 3) — Cline stdio + ComfyUI live functional verification.** Confirmed the stdio path is intact after the transport work: canvas round-trip (`new_canvas` 512×512 → `add_text` "Cline stdio OK" → `export` → `verification/clined_stdio_check.png` → `preview_canvas`, text legible in preview). With ComfyUI stopped, `generate_image` (flux2, 512×512, 4 steps) auto-started ComfyUI from the `.env`-configured portable install and completed; `get_comfyui_status` then reported connected (ComfyUI 0.37.0, RTX 5090 32GB, torch 2.13.0+cu130); `img2img` (strength 0.35) exercised the upload→workflow path. No new failure modes; no `MEMORY.md` config changes needed.
- **2026-09-20 (session 2) — "stuck on OpenAPI" root cause + fix.** OWUI 0.11.3's Edit-Connection dialog renders Type as a read-only label, so a connection created as OpenAPI can never be switched to MCP in the UI. Worse: saving an OpenAPI connection first fetches the OpenAPI spec from the URL; our endpoint answers 406 to `GET /mcp`, so the save is aborted and **no connection is persisted** (the error toast is all the user sees; the dialog just keeps the typed values). In 0.11.3 the config no longer lives in `config.json` — it's in the SQLite DB (`/app/backend/data/webui.db`, table `config`, row `key='tool_server.connections'`, value = JSON list). MCP runtime path: `utils/middleware.py:connect_mcp_server` looks up the connection by `info.id`, connects with the top-level `url` field (no path join for MCP), `auth_type: none` → no headers. Fix applied: stop container → `verification/_fix_owui_mcp_conn.py` (run in a throwaway container with `-v open-webui:/data`) upserts the entry `{url: http://host.docker.internal:8000/mcp, path: "", type: "mcp", auth_type: "none", config: {enable: true}, info: {id: "mcp-photoshop", name: "Mcp Photoshop", ...}}` → start container. Pre-edit DB backup: `verification/webui.db.bak` (+ `-wal`/`-shm`). Verified E2E with OWUI's own `MCPClient` from inside the container (`verification/_owui_mcp_e2e.py`, needs `WEBUI_SECRET_KEY` env just for the `open_webui.env` import): 43 tools listed. UI now shows the connection as Type **MCP** (still edit-locked, but correct).

## 2026-09-20 editing migration

- Installed 2.1 stack: `qwen_image_2.1_int8_convrot.safetensors`, `qwen3vl_8b_int8_convrot.safetensors`, `qwen_image_2.1_vae_bf16.safetensors`. Use the native encoder's positive/negative/latent outputs and cache patch; no old Lightning adapters, AuraFlow shift, or CFGNorm.
- Masks are now the last reference image with an explicit white-to-edit instruction, followed by exact compositing outside the mask. Reference numbering is stable. RGBA survives preprocessing, ComfyUI loading, and application. Edits create a replacement layer and hide originals in one undo step. The canvas compositor now honors the bottom layer's visibility, opacity, and mask.
- Open an image first and keep the same session ID. Unknown sessions fail clearly. Inspect the returned preview and undo a bad attempt before retrying. Working dimensions match Qwen's 32-pixel grid. `max_side=2048` is accepted; default remains 1024. Qwen waits up to 1800 seconds by default.
- Migrated all three saved Qwen 2511 character-sheet workflows in ComfyUI; backups are in `verification/qwen2511_workflows_before_migration.zip`. The shared old VAE stays on the ANIMA branch only.
- Verification: 79 tests pass; all four UI graphs pass ComfyUI prompt validation. Actual MCP stdio → ComfyUI → GPU tests: a masked/reference circle recolor completed in 28.3 seconds; an elf's silver-to-blue hair edit completed in 9.9 seconds. Both changed zero pixels outside the mask. Evidence: `verification/migration_live_status.json`, `migration_character_status.json`, `workflow_validation.json`.
- Added `user/default/workflows/qwen_image_2_1_edit.json` as a simple standalone editor. Old model files are retained on disk but no active saved user workflow or MCP editing backend uses Qwen 2511.
- GPU instructions now require checking actual contention. A normal edit on an available GPU needs no container shutdown. Stopping the local LLM container still requires explicit approval because it may serve the client LLM.

---

## 1. Project Overview

**What:** A local MCP (Model Context Protocol) server that combines Photoshop-style image editing with AI image generation powered by ComfyUI.

**Why:** Provides an AI model (e.g., Claude) with a comprehensive set of image manipulation tools through a single MCP interface, enabling multi-step image editing workflows.

**Core Architecture — Two-Track Design:**
- **AI Operations** (generation, inpainting, transforms, ControlNet, style transfer) → Routed through ComfyUI API
- **Deterministic Edits** (crop, resize, adjustments, filters) → Executed locally with Pillow (fast, no GPU)

**ComfyUI Lifecycle Management:**
- **Auto-start**: Automatically launches ComfyUI via embedded Python when needed (no manual startup required)
- **Idle timeout auto-kill**: When `COMFYUI_AUTO_KILL=1`, ComfyUI is killed after `COMFYUI_IDLE_TIMEOUT` (default 60s) of inactivity instead of immediately. This allows chaining multiple ComfyUI tools without restarting each time, while still freeing VRAM when idle. The kill targets the process this server started or adopted (an already-running instance found via `netstat -ano`), so an externally started ComfyUI is terminated too.
- **Port management**: Handles Windows TIME_WAIT issues by waiting for port 8188 to be fully bindable
- **3-strategy kill**: Direct process handle → port-based (netstat) → command-line matching (wmic)
- **Pre-fetch output**: Downloads result bytes BEFORE killing ComfyUI when auto-kill is enabled
- **Retry logic**: If ComfyUI fails to start, stops only its owned process and retries once after the port clears; a foreign listener or failed cleanup returns an error

**VRAM Pressure Management:**
- **Adaptive cleanup**: After each generation, checks free VRAM via ComfyUI system_stats or nvidia-smi
- **Kill threshold**: If free VRAM < 8192 MB, kills ComfyUI entirely to release all VRAM for vLLM
- **Free memory**: If sufficient VRAM, calls free_memory() to unload models while keeping process warm
- **Configured via**: `VRAM_PRESSURE_THRESHOLD_MB` (default: 8192)

**OOM Issue:** Running ComfyUI directly (without auto-kill) causes major OOM when shared with vLLM on a 32GB GPU. The idle timeout (60s) balances convenience (chaining tools) with VRAM management (killing when idle).

---

## 2. File Architecture

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `config.py` | Configuration constants, model names, ComfyUI lifecycle | `COMFYUI_URL`, `COMFYUI_START_CMD`, `COMFYUI_PYTHON`, `COMFYUI_MAIN`, `COMFYUI_ARGS`, `COMFYUI_AUTO_KILL`, `COMFYUI_IDLE_TIMEOUT`, `COMFYUI_START_TIMEOUT`, `WEBSOCKET_TIMEOUT`, `VRAM_PRESSURE_THRESHOLD_MB` |
| `comfy_client.py` | ComfyUI API wrapper with auto-start/idle-kill lifecycle | `ComfyUIClient`: `start_comfyui()`, `kill_comfyui()`, `run_workflow_and_wait()`, `batch_run_workflows()`, `_listen_for_many()`, `_schedule_idle_kill()`, `_cancel_idle_kill()`, `submit_workflow()`, `upload_image()`, `get_output_file()`, `free_memory()` |
| `canvas.py` | Layered document model with blend modes | `Canvas`: layers with 12 blend modes (all non-normal modes alpha-aware via `_blend_with_alpha` since 2026-09-18), masks, undo/redo stack (20 steps), `resize_canvas()`, `composite()`, `composite_rgb()`, `BLEND_MODES` dict |
| `session.py` | Per-session document management | `SessionManager`: `get_or_create()`, `get()`, `create()`, `replace()`, `delete()`, `get_default_session()`, `list_sessions()` |
| `project.py` | Local layered-project persistence | `save_project()`, `load_project()`; atomic save, versioned manifest and lossless PNG members |
| `jobs.py` | Process-owned background generation | `GenerationJobs`: submit, status, list, graceful cancel; serial workers and per-image results |
| `server.py` | MCP server + all tool registrations + workflow builders | 45 `@app.tool()` registrations (canvas tools take `session_id`), `GPU_BATCH_RULE` passed as FastMCP `instructions`, `build_upscale_workflow` (outpaint reuses `editing.build_edit_workflow` with server-side PIL padding), `run_workflow()` helper, `free_or_kill_based_on_pressure()` |
| `editing.py` | Generation and instruction editing backends + live capabilities | 5 `@app.tool()` registrations: `edit_image` (qwen21 default / flux2 Dev; references, masks, layer editing and extraction), `export_mask`, `semantic_select` (SAM 3 text/point/box), `get_editing_capabilities`, `preview_canvas`; `model_profiles`, `build_generation_workflow`, `build_edit_workflow` / `build_semantic_select_workflow` |
| `prompt_rules.py` | Master prompt integration | Tool description rules, current master-file reader, outer-fence and Anima score-tag normalization |
| `requirements.txt` | Python dependencies | `mcp<2.0.0`, `Pillow>=10.1.0`, `httpx>=0.27.0`, `websockets>=12.0`, `numpy>=1.24.0`, `python-dotenv>=1.0.0` |
| `README.md` | User documentation | Installation, usage examples, architecture overview |
| `MEMORY.md` | This file — project memory bank |

---

## 3. Tool Inventory (50 Tools)

### Canvas Management (6)
| Tool | Signature | Description |
|------|-----------|-------------|
| `new_canvas` | `(width=1024, height=1024, bg_color="white")` | Create blank canvas |
| `open_image` | `(path: str)` | Load existing image file as active document |
| `save_project` | `(path: str)` | Atomically save the layered document as ZIP/JSON/PNG |
| `open_project` | `(path: str)` | Replace the session after a complete valid load; fresh undo history |
| `export` | `(path=None, format="PNG", quality=95)` | Save to file (PNG/JPG/WEBP); auto-named when `path` is omitted |
| `get_info` | `()` | Canvas dimensions, layers, undo/redo state |

### AI Image Generation (1)
| Tool | Signature | Description |
|------|-----------|-------------|
| `generate_image` | `(prompt, model="flux2", width=1024, height=1024, steps=None, cfg=None, seed=None, negative_prompt="", session_id="default", timeout=None)` | txt2img via Flux2, Qwen 2.1, Anima, or MiniMax H3 (experimental stills); model-specific sampling defaults |

### Prompt Guidance (1)
| Tool | Signature | Description |
|------|-----------|-------------|
| `get_prompt_guidance` | `(model="flux2", task="generation")` | Current applicable master and tool-specific rules; no ComfyUI or GPU required |

### AI Editing + Instruction Editing (4)
| Tool | Signature | Description |
|------|-----------|-------------|
| `outpaint` | `(prompt, direction="right", amount=256, steps=None, seed=None, session_id="default", backend="flux2", timeout=None)` | Extend canvas via Flux2 or Qwen 2.1 reference conditioning; preserve original pixels and layer/mask alignment |
| `edit_image` | `(prompt, backend="qwen21", reference_paths=None, mask_path=None, region=None, feather=0, steps=None, seed=None, max_side=1024, session_id="default", timeout=None, cfg=None, layer_index=None, output_mode="replace", h3_reference_detail="match")` | Qwen Image 2.1 (custom 30/3), FLUX.2 Dev (50/4) or MiniMax H3 ref2va (20/BasicGuider, experimental); 1800 s timeout (3600 s for H3); references, masks, composite/selected-layer replacement or Qwen extraction to a new layer; H3 detail `max` keeps full-resolution references independently of output `max_side` |
| `get_editing_capabilities` | `(start_if_needed=True)` | Live model/node availability per task: generation, editing, outpaint; notes carry the GPU batching rule |
| `preview_canvas` | `(max_size=1024, session_id="default")` | Render current canvas as an image for assistant inspection |

### Transforms (4)
| Tool | Signature | Description |
|------|-----------|-------------|
| `crop` | `(x, y, width, height)` | Crop every layer and mask |
| `resize` | `(width, height, maintain_aspect=False)` | Resize all layers/masks; aspect-preserving mode fits and centers in transparent padding |
| `rotate` | `(degrees, expand=True, bg_color="transparent")` | Rotate active layer and mask; expanded bounds pad other layers |
| `flip` | `(axis="horizontal")` | Flip active layer and mask horizontally or vertically |

### Color Adjustments (3)
| Tool | Signature | Description |
|------|-----------|-------------|
| `adjust` | `(brightness, contrast, saturation, hue, sharpness)` | Multi-parameter adjustment (multipliers, hue in degrees) |
| `levels` | `(black_point=0, mid_point=1.0, white_point=255)` | Levels/gamma adjustment via 256-point LUT |
| `curves` | `(red="", green="", blue="")` | Per-channel tone curves (format: "0,0 128,140 255,255") |

### Filters (1) — 17 filters available
| Tool | Signature | Description |
|------|-----------|-------------|
| `apply_filter` | `(name, session_id="default", *, radius=2, size=8, levels=4)` | 17 filters; explicit Gaussian radius, pixel size, posterize levels; color-only filters preserve alpha |

### Text (1)
| Tool | Signature | Description |
|------|-----------|-------------|
| `add_text` | `(text, x=50, y=50, font_size=48, color="white", font=None, stroke_width=0, stroke_color="black")` | Text overlay with font, color, stroke |

### Upscaling (1)
| Tool | Signature | Description |
|------|-----------|-------------|
| `upscale` | `(factor=2, model="anime")` | AI upscaling via ComfyUI ("general", "anime", "face", or an installed filename) |

### Layer System (11)
| Tool | Signature | Description |
|------|-----------|-------------|
| `add_layer` | `(name=None, source_path=None, opacity=1.0, blend_mode="normal")` | Add new layer, optionally from file |
| `select_layer` | `(index: int)` | Select layer by index (0 = bottom) |
| `set_blend_mode` | `(mode, index=None)` | Set blend mode (12 modes supported) |
| `set_layer_opacity` | `(opacity, index=None)` | Set layer transparency (0.0-1.0) |
| `merge_down` | `()` | Merge active layer into layer below |
| `delete_layer` | `(index=None)` | Delete layer by index or active |
| `reorder_layer` | `(index=None, direction="up")` | Move layer up/down in stack |
| `set_layer_visibility` | `(visible, index=None)` | Show/hide a layer |
| `rename_layer` | `(name, index=None)` | Rename a layer |
| `duplicate_layer` | `(index=None, name=None)` | Independent image/mask/settings copy above original; selects copy |
| `translate_layer` | `(dx, dy, index=None)` | Move image and mask together, clipping at canvas bounds |

### Selections & Masks (6)
| Tool | Signature | Description |
|------|-----------|-------------|
| `select_rect` | `(x, y, width, height)` | Rectangular mask |
| `select_ellipse` | `(x, y, rx, ry)` | Elliptical mask centered at (x,y) |
| `select_object` | `(description, threshold=128)` | Heuristic color/region selection |
| `semantic_select` | (prompt=None, point=None, box=None, threshold=0.5, refine=2, timeout=None, session_id="default") | SAM 3 semantic object mask; text prompts run `sam3.1_multiplex_fp16.safetensors` (via `CheckpointLoaderSimple` + `CLIPTextEncode`), point/box run `sam3.pt` (via `ImageOnlyCheckpointLoader`) — both in `models/checkpoints/`; applies the union mask to the active layer |
| `clear_mask` | `(index=None)` | Remove layer mask |
| `export_mask` | `(path, index=None, invert=False, expand=0, feather=0)` | Export an independent edit-mask PNG; invert, grow/shrink, then feather; document unchanged |

**select_object supported descriptions:** `red, blue, green, sky, dark, shadow, light, white, black, yellow, purple, orange, cyan, pink, brown`

### History (2)
| Tool | Signature | Description |
|------|-----------|-------------|
| `undo` | `()` | Undo last operation (up to 20 steps) |
| `redo` | `()` | Redo last undone operation |

### System (2)
| Tool | Signature | Description |
|------|-----------|-------------|
| `get_comfyui_status` | `(start_if_needed=True)` | Check ComfyUI connection and system info; starts ComfyUI by default |
| `clear_vram` | `()` | Free GPU VRAM by unloading cached models |

### Sessions & Batch (7)
| Tool | Signature | Description |
|------|-----------|-------------|
| `list_sessions` | `()` | All open document sessions with size/layers/active layer/undo depth |
| `close_session` | `(session_id="default")` | Close a session, freeing its canvas |
| `batch_generate` | `(jobs: list[dict], export_dir=None, export_format="PNG", timeout=None)` | Mixed-model txt2img queue (flux2/qwen21/anima/minimax_h3) on one WebSocket; returns exported paths, model, steps, CFG and seed |
| `submit_generation_job` | `(jobs, export_dir=None, export_format="PNG", timeout_per_image=None)` | Return an ID; generate and export one image at a time in the MCP process |
| `get_job_status` | `(job_id)` | Progress and per-image results for this server instance |
| `list_jobs` | `()` | Job summaries for this server instance |
| `cancel_job` | `(job_id)` | Finish/export current image, then skip remaining images |

> Every canvas tool takes an optional `session_id` (default `"default"`) so multiple documents can be edited independently in one server process.

---

## 4. ComfyUI Lifecycle Management

### Auto-Start Flow
1. `run_workflow_and_wait()` calls `_cancel_idle_kill()` then `start_comfyui()` before each workflow
2. `start_comfyui()` checks if ComfyUI is already running via `/history` endpoint
3. If already running: adopts a local listener only after verifying the configured Python/main script and creation time; arms a missing idle timer without postponing an existing one
4. If not running: cancels any stale pending idle timer, checks the configured port, stopping only its own stale process; foreign listeners return an error
5. Launches ComfyUI via embedded Python: `COMFYUI_PYTHON -s COMFYUI_MAIN COMFYUI_ARGS...`
6. Falls back to `.bat` if Python path doesn't exist
7. Polls `/history` every 2s until reachable (up to `COMFYUI_START_TIMEOUT`=180s)
8. On timeout: stops the owned process, verifies cleanup and port availability, then retries once

### Idle Timeout Auto-Kill (when `COMFYUI_AUTO_KILL=1`)
1. After a backend check, workflow or upload ends: arms idle cleanup when no operation is active
2. Idle timer set to `COMFYUI_IDLE_TIMEOUT` (default 60s / 1 minute)
3. If a new workflow starts within the timeout: `_cancel_idle_kill()` cancels pending timer
4. After timeout: checks the shared queue; busy, malformed or unreachable queue state defers shutdown. An empty queue permits stopping this server's owned backend, or an adopted local PID whose executable, script and creation time still match.
5. Failure and cancellation also rearm guarded idle cleanup; they do not immediately kill other clients' queued work
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
| `COMFYUI_START_CMD` | *(optional)* | Path to ComfyUI start .bat (alternative to COMFYUI_PYTHON + COMFYUI_MAIN) |
| `COMFYUI_ARGS` | `--windows-standalone-build` | Launch args |
| `COMFYUI_AUTO_KILL` | code default `0`; `1` recommended | Kill after generation (0=use free_memory, 1=idle timeout) |
| `COMFYUI_IDLE_TIMEOUT` | code default `60`; `5` recommended | Seconds of inactivity before auto-killing (when AUTO_KILL=1) — 5s recommended to prevent Cline freezes |
| `COMFYUI_START_TIMEOUT` | `180` | Seconds to wait for ComfyUI startup |
| `WEBSOCKET_TIMEOUT` | `600` | Seconds to wait for workflow completion (10 min) |
| `VRAM_PRESSURE_THRESHOLD_MB` | `8192` | Kill ComfyUI if free VRAM below this (MB) |

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI API endpoint |
| `COMFYUI_START_CMD` | *(optional)* | Path to ComfyUI start .bat (alternative to COMFYUI_PYTHON + COMFYUI_MAIN) |
| `COMFYUI_INPUT_DIR` | `None` (auto-detected) | ComfyUI input directory |
| `COMFYUI_OUTPUT_DIR` | `None` (auto-detected) | ComfyUI output directory |
| `MAX_UNDO_STEPS` | `20` | Maximum undo history entries |

### Windows-Specific Handling
- Redirects stdin from `/dev/null` to prevent console crashes
- Uses `DETACHED_PROCESS` flag for independent process lifecycle
- Sets `PYTHONIOENCODING=utf-8` to prevent UnicodeEncodeError from emoji logging
- Does NOT redirect stdout/stderr (embedded Python crashes on file redirect)

---

## 5. Workflow Builders (editing.py / server.py)

| Builder | Description | Key Nodes |
|---------|-------------|-----------|
| `build_generation_workflow()` | Text-to-image for flux2/qwen21/anima via per-model profiles | flux2: **UNETLoader**, **CLIPLoader** (flux2), **VAELoader**, **CLIPTextEncode** (positive only — Dev does not encode negatives), **FluxGuidance** (embedded guidance), **BasicGuider**, **RandomNoise**, **Flux2Scheduler**, **KSamplerSelect**, **EmptyFlux2LatentImage**, **SamplerCustomAdvanced**, **VAEDecode**, **SaveImage** (50 steps/embedded guidance 4, Euler); qwen21: **TextEncodeQwenImage21** (positive + negative) + **EmptyLatentImage** + **KSampler**; anima: **CLIPTextEncode** (positive + negative) + **EmptyLatentImage** + **KSampler** |
| Outpaint (in the `outpaint` tool) | Padded canvas built server-side (PIL), then run through `build_edit_workflow` reference conditioning; original pixels restored by the server | flux2: **LoadImage**, **VAEEncode**, **ReferenceLatent** chain, **CLIPTextEncode** (positive only) → **FluxGuidance** → **BasicGuider**, **RandomNoise**, **Flux2Scheduler**, **KSamplerSelect**, **EmptyFlux2LatentImage**, **SamplerCustomAdvanced**, **VAEDecode**, **SaveImage** (50 steps/embedded guidance 4, Euler) |
| `build_upscale_workflow()` | AI upscaling | **LoadImage**, **UpscaleModelLoader**, **ImageUpscaleWithModel**, **SaveImage** |

---

## 6. Design Decisions

- **Pillow for deterministic edits** — Faster (no GPU round-trip), more precise, simpler
- **Layer Model** — Full non-destructive layers with 12 blend modes, masks, opacity
- **Mask Handling** — PIL "L" mode images (0=protected, 255=selected)
- **Undo/Redo** — Full state snapshots (deepcopy via Layer.to_dict/from_dict), limited to 20 steps
- **Session Management** — Canvas instances keyed by session ID, default session: "default"
- **ComfyUI Auto-Start** — Eliminates manual ComfyUI startup
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
| `flux2-dev-nvfp4.safetensors` | diffusion_models | FLUX.2 Dev 32B, generation/editing/outpaint; 50 steps / embedded guidance 4 |
| `mistral_3_small_flux2_fp8.safetensors` | text_encoders | Dev Mistral Small encoder; CLIP type `flux2` |
| `flux2-vae.safetensors` | vae | FLUX.2 VAE |
| `anima-aesthetic-v1.1.safetensors` | diffusion_models | Text-to-image (anime — ANIMA) |
| `qwen_3_06b_base.safetensors` | text_encoders | ANIMA text encoder |
| `qwen_image_vae.safetensors` | vae | ANIMA VAE |

### Instruction Editing ("edit_image" backends)
| Model | Directory | Purpose |
|-------|-----------|---------|
| `qwen_image_2.1_int8_convrot.safetensors` | diffusion_models | `qwen21` backend UNET (Qwen Image 2.1) |
| `qwen3vl_8b_int8_convrot.safetensors` | text_encoders | Qwen 2.1 text encoder |
| `qwen_image_2.1_vae_bf16.safetensors` | vae | Qwen 2.1 VAE |

### Upscaling
| Model | Directory | Purpose |
|-------|-----------|---------|
| `RealESRGAN_x4plus_anime_6B.pth` | upscale_models | Anime upscaling |
| `4xFaceUpDAT.pth` | upscale_models | Face upscaling |

### Detection / Segmentation
| Model | Directory | Purpose |
|-------|-----------|---------|
| `sam3.pt` | detection | SAM 3 open-vocabulary segmentation — ComfyUI 0.33.0 native `SAM3_Detect` node; checkpoint deployed and smoke-tested 2026-09-17; integrated via the `semantic_select` MCP tool (live-verified same day) |
| `sam3.1_multiplex_fp16.safetensors` | checkpoints | SAM 3.1 text-prompted segmentation — Comfy-Org open repack, deployed 2026-09-17 (SHA-256 verified); feeds `semantic_select(prompt=...)`; listed live without restart |

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
        "COMFYUI_IDLE_TIMEOUT": "5",
        "VRAM_PRESSURE_THRESHOLD_MB": "8192"
      }
    }
  }
}
```

---

## 9. Testing Status

### Verified Working (50 tools)
- Prompt Guidance: `get_prompt_guidance` (full Flux/Anima masters, task-specific Qwen generation/edit masters and the H3 still master; real MCP stdio verified 2026-09-21, H3 master re-verified 2026-09-22) ✅
- Canvas Management: `new_canvas`, `export`, `get_info` ✅
- Local Projects: `save_project`, `open_project` (atomic `.mcpproj` ZIP/JSON/PNG, fresh undo history; unit + live MCP verified 2026-09-22) ✅
- Transforms: `crop`, `resize`, `rotate`, `flip` ✅
- Color Adjustments: `adjust`, `levels`, `curves` ✅
- Filters: `apply_filter` (all 17 filters) ✅
- Text: `add_text` ✅
- Layer System: all 11 tools (`duplicate_layer`, `translate_layer`, `set_layer_visibility`, `rename_layer` added 2026-09-22) ✅
- Selections: `select_rect`, `select_ellipse`, `clear_mask`, `export_mask` (independent edit-mask PNG with invert/expand/feather; 2026-09-22) ✅
- History: `undo`, `redo` ✅
- System: `get_comfyui_status`, `clear_vram` ✅
- AI Generation: `generate_image` (Flux2 + ANIMA) ✅
- AI Editing: `outpaint` (Flux2 chain, rewritten + live-verified 2026-09-20) ✅
- Instruction Editing: `edit_image` (qwen21 + flux2), `get_editing_capabilities`, `preview_canvas` ✅
- Removed 2026-09-20 (models deleted from ComfyUI / superseded): `img2img`, `character_transform`, `inpaint`, `controlnet_generate`, `style_transfer` — see "2026-09-20 Tool Cleanup"
- Upscale: `upscale` (anime + face + x4plus general-photo model, live MCP test 2026-09-17) ✅
- Semantic Selection: `semantic_select` (SAM 3 text/point/box, live CPU MCP test 2026-09-17) ✅
- Sessions & Batch: `list_sessions`, `close_session`, `batch_generate` + `session_id` on all canvas tools (unit-verified 2026-09-18, 23 new tests; live batch run in `verification/_live_sessions_batch_nodes_test.py`) ✅
- Background Jobs: `submit_generation_job`, `get_job_status`, `list_jobs`, `cancel_job` (process-owned serial workers, per-image export, graceful cancel; unit + live HTTP disconnect/recovery checks 2026-09-22) ✅

### Idle Timeout Chaining (tested 2026-08-12)
- `generate_image` → `img2img` chained successfully without ComfyUI restart (`img2img` removed 2026-09-20; chaining works between any two ComfyUI tools)
- Idle timer reset on second call, no restart needed

---

## 10. Known Limitations & Future Work

- `select_object` uses color/region heuristics (not SAM); `semantic_select` covers SAM 3 text, point, and box prompts (SAM 3.1 checkpoint deployed 2026-09-17; open Comfy-Org repack, not gated)
- Single ComfyUI instance (no concurrent workflows)
- Undo stores full snapshots (memory-intensive)
- No lasso/freehand selection
- No brush/paint tools
- ControlNet/Redux style tools were removed 2026-09-20 (`models/controlnet`, `models/style_models`, `models/clip_vision` hold only placeholder files). Style/identity/reference guidance is covered by `edit_image` `reference_paths`; a Flux2-compatible ControlNet could re-introduce guided generation if ever needed
- Running ComfyUI without auto-kill causes major OOM on shared GPU (32GB with vLLM)

---

## 11. GPU Contention & Batching Rule

> Added 2026-09-17. This is a standing rule for any LLM that calls this MCP server. It is delivered to every client in the MCP `initialize` response (server `instructions`, see `GPU_BATCH_RULE` in `server.py`) and is also repeated in the `get_editing_capabilities` notes.

**The problem.** One 32 GB GPU (RTX 5090). The LLM docker runs the LLM itself (Ollama-compatible API on `:11434`; the container name is machine-local and kept out of this repo - identify it via `docker ps`) and holds most of the VRAM while serving. ComfyUI is a host process (`:8188`, python from `COMFYUI_PYTHON` in `.env`), not a container. Generation sharing the GPU with a loaded LLM slows exponentially (qwen2511 alone peaked at ~32.1 GB).

**Container inventory:**

| Container | Port | Uses GPU? | Action before a generation batch |
|-----------|------|-----------|----------------------------------|
| LLM container | `:11434` | Yes — the LLM backend | **Ask the user for explicit approval, then `docker stop` that container** |
| `open-webui` | `:3000` | No (frontend for the LLM) | Optional stop; only useful if the LLM container stops too |
| `searxng` | `:8080` | No (CPU-only) | Never stop it for GPU speed |

**Procedure (any LLM client):**
1. Single generation with sufficient GPU memory: run it directly. Dev defaults to 50 steps; the four-step Klein recipe is retired.
2. Multiple generations or any long job (e.g. `qwen21` edits, full batches):
   a. Plan the complete job list with the user first.
   b. Hand the complete input list to `submit_generation_job` on an independently running host MCP server, or to a detached host runner. Verify that its process survives loss of the LLM connection; sequential awaited tool calls do not establish background ownership.
   c. Export each result as it completes (`submit_generation_job` does this automatically). Its job/status state is process-local; server restart or closing a stdio process loses it. Existing `batch_generate` waits for the whole batch before exporting.
   d. **Ask the user for explicit approval** to run `docker stop` on the LLM container (plus `open-webui` if desired). Stop only after approval, and only once the batch is fully backgrounded.
   e. The batch then runs on the full GPU; results accumulate in ComfyUI's output dir / exported files.
3. Afterwards, tell the user to `docker start` the LLM container (and `open-webui`) to restore the LLM.

**Self-kill warning.** The LLM container serves the LLM that is driving the MCP calls. Stopping it terminates the session — that is exactly what the rule anticipates, and it is only acceptable because step 2b guarantees the batch continues on the host. Never stop that container mid-conversation with work that still depends on the LLM.

**Lifetime note.** HTTP client disconnection does not stop a submitted worker while its MCP server remains running. A client-managed stdio server may close with its client; do not assume it can outlive that client. No job persistence or automatic resume across server restart is implemented.

---

## 12. Changelog

### 2026-09-22 — Master-prompt compliance audit and live Qwen21/Anima proof
- Re-verified all four local master files through live MCP `get_prompt_guidance` (path + SHA-256 + full text per call): qwen21 t2i/edit and anima hashes matched the previously verified records; `flux2prompt.txt` had drifted to the 2026-09-19 FLUX.2 [dev] revision (SHA `eda66664…870f`) and the served text now reflects it.
- Dropped the stale "historical Klein heading" sentence from `prompt_rules` `MODEL_RULES["flux2"]` (and resynced the changelog line above); repo-wide Klein scan confirmed all remaining references are correctly historical. 176 tests pass.
- Live master-compliant proofs through the MCP server (RTX 5090, 30.9 GB VRAM free, qwen3.8 LLM container untouched): one Qwen21 image from a t2i-master observer-style prompt (1152x768, quoted "MEADOW" sign rendered exactly) and one Anima image from an Aesthetic hybrid tag+prose prompt (768x1152, quoted "PLATFORM 3" sign rendered exactly, no `score_*` tags, conflicting "blurry" negative removed). Artifacts: `verification/master_compliance/` (gitignored, on disk).

### 2026-09-22 — Auto-kill now manages externally started ComfyUI (adopted PID)

- **`comfy_client.py` adopted-PID kill:** `_kill_process()` previously required the live Popen handle started by this server, so a ComfyUI started outside the current MCP session (an orphan holding port 8188) was never killed. New `_find_listening_pid()` resolves the PID via `netstat -ano`; `start_comfyui()` records it in `_adopted_pid` when ComfyUI is already running, and `_kill_process()`/`kill_comfyui()` fall back to `taskkill /F /T` on that PID (owned Popen takes precedence).
- **`comfy_client.py` probe-safe idle timer:** `start_comfyui()` no longer cancels the pending idle-kill timer on read-only probes (status/capability checks); only the fresh-launch path cancels it.
- **Verification:** 187 unit tests green (6 new: adopted-PID fallback kill, handle clearing, no-handle no-op, netstat parsing, port-absent None, probe-keeps-timer/launch-cancels). Stuck external ComfyUI terminated; port 8188 confirmed free.

### 2026-09-21 — FLUX.2 Dev NVFP4 (32B) replaces the Klein checkpoints
- `config.py`: `MODEL_FLUX2` → `flux2-dev-nvfp4.safetensors`, text encoder → `mistral_3_small_flux2_fp8.safetensors` (CLIP type `flux2`); VAE unchanged. `model_profiles` flux2 preset 28/4.0 → 50 steps/embedded guidance 4.
- Flux generation/edit/outpaint graphs: positive-only `CLIPTextEncode` → `FluxGuidance` → `BasicGuider`; `CFGGuider` and the negative branch are retired, and a supplied `negative_prompt` is reported as unused. 1800 s default timeouts; batches budget 1800 s per Flux/Qwen job.
- The local `flux2prompt.txt` is now the 2026-09-19 FLUX.2 [dev] master revision (SHA-256 `eda66664…870f`, replacing the last live-verified copy `75c08d3f…418`); `prompt_rules` adapts it to the installed NVFP4 backend. Four saved `flux2_klein*.json` ComfyUI workflows migrated to Dev (backup: `verification/flux2_dev/workflows_before_dev.zip`).
- Tests 106 → 108; README model table and MEMORY §2/§3/§5 resynced.

### 2026-09-21 — FLUX.2 Klein 9B Base (fp8) replaces the distilled Klein 9B
- Replaced `flux-2-klein-9b.safetensors` (distilled, 18.2 GB — deleted) with `flux-2-klein-base-9b-fp8.safetensors` (undistilled base, fp8, 9.57 GB); SHA256 `a9f5028c…42cf4` verified against the official `black-forest-labs/FLUX.2-klein-base-9b-fp8` blob. Text encoder (`qwen_3_8b_fp8mixed`) and VAE (`flux2-vae`) unchanged — both are structural pipeline requirements.
- `config.py` `MODEL_FLUX2` → fp8 base (full-precision `flux-2-klein-base-9b.safetensors` kept as fallback name). `model_profiles` flux2 preset 4 steps/CFG 1 → **28 steps/CFG 4** (base recipe — BFL card default is 50/4.0, tuned to the user's 28-step sweet spot); "Klein 9B distilled" description → "Klein 9B Base, undistilled".
- `build_edit_workflow` flux2 branch: `BasicGuider` → `CFGGuider` (empty negative) so CFG 4 applies to instruction edits and outpainting; generation already used `CFGGuider`.
- 4x `flux2_klein_*` ComfyUI workflows: UNETLoader → fp8 base, all KSamplers 6 steps/CFG 1 → 28/4.0 (euler/simple kept).
- README model table + sampling notes and MEMORY §2/§3 resynced; tests updated (fp8 name, 28/4.0, `CFGGuider`).

### 2026-09-21 — Master prompt integration; local machine paths removed from tracked files
- New `prompt_rules.py` + `tests/test_prompt_rules.py`: compact master rules injected into `generate_image`, `batch_generate`, `edit_image` and `outpaint` tool descriptions (covers clients that discard initialize instructions); new `get_prompt_guidance(model, task)` tool — tool count 39 — returns the applicable Flux/Anima/Qwen master with source path and SHA-256 without starting ComfyUI; conservative normalization strips one surrounding prompt fence and, for Anima Aesthetic/unknown checkpoints, standalone `score_*` tags from both prompts (quoted lettering preserved); generation/batch results disclose `prompt_adjustments`/`effective_prompt`.
- `config.py`: `MASTER_PROMPT_DIR` now defaults to the repository-local `masters/` folder (tracked via `.gitkeep`); the personal-machine default was removed, overridable via env / `.env`.
- Public-repo hygiene (2026-09-18 precedent): local machine paths removed from the README (master-prompt section + `MASTER_PROMPT_DIR` env row) and from this file (2026-09-21 entry, historical changelog); the live local value now lives only in the gitignored `.env`.
- Verification: 106 regression tests passed; fresh MCP stdio listed 39 tools with master rules in all four image-tool descriptions; full masters read with matching SHA-256 (evidence JSONs in gitignored `verification/`).

### 2026-09-20 — Docs Resync: MEMORY §2/§3 to 38 tools; README + Open WebUI docs corrected
- `MEMORY.md` §2/§3 resynced to the current 38-tool server. The earlier 2026-09-20 "Tool Cleanup" entry claimed docs were synced, but §3 still listed 43 tools (the 5 removed tools were still present) and §2 said "43 @app.tool() … 12 workflow builder functions". Current: 34 `@app.tool()` in `server.py` + 4 in `editing.py`; builders are txt2img/anima/outpaint/upscale (`server.py`) + `build_edit_workflow`/`build_semantic_select_workflow` (`editing.py`).
- README corrections: `export` is file-only (auto-named when `path` is omitted) — no base64 output; `semantic_select` added to the feature list; `get_comfyui_status` notes auto-start; SAM 3 (`sam3.pt`) + SAM 3.1 (`sam3.1_multiplex_fp16.safetensors`) checkpoints added to Prerequisites (both in `models/checkpoints`); architecture tree — garbled `◜───` glyphs fixed, dead `verification/` line removed (gitignored local artifacts, no longer on disk; `verification/…` references in older changelog entries are historical).
- README "Using with Open WebUI" rewritten against the live system: the server registers as a chat **tool** ("Mcp Photoshop"; per-chat Tools panel; "Direct Tool Servers" permission), the real UI path (Admin → Integrations → External Tool Servers → Manage Direct Connections → Add Connection → OpenAPI→MCP toggle), the direct-DB upsert actually used on v0.11.3 (exact `tool_server.connections` JSON), and the timeout note (live container verified to run with `AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER=1800`; a local `start-qwen-openwebui-.ps1` recreation script lacks it).
- `comfyui_nodes.json` (2 MB, tracked) is referenced by no code — left as-is.

### 2026-09-18 — Public Repo Hygiene: Local Machine Paths Removed from Docs & History
- `MEMORY.md` no longer carries absolute local paths (the "Location" header and the 2026-09-17 editing-upgrade changelog entries were rewritten path-free); the README's dead `EDITING_UPGRADE.md` references (intro line + architecture-tree line) were removed — that file was never committed and no longer exists.
- Git history rewritten with `git filter-repo --replace-text` so no commit, message, or ref contains local machine paths; `master` force-pushed.

### 2026-09-18 — Multi-Session Support, `batch_generate`, ControlNet/Redux Workflow Repair
- **Multi-session support:** every canvas tool (34 in `server.py`, 3 in `editing.py`) now takes `session_id: str = "default"`; bodies resolve via `sessions.get_or_create(session_id)` (the shared `DEFAULT_SESSION` constant removed, `get_default_session()` no longer referenced by any tool). New tools `list_sessions` (size/layers/active layer/undo depth per session) and `close_session(session_id)`. `SessionManager.list_sessions()` added. `character_transform` passes `session_id` through to `img2img`.
- **`batch_generate` (batch processing):** new tool queuing a whole job list (txt2img flux2/anima, per-job prompt/model/size/steps/cfg/seed/filename) on ONE WebSocket via `ComfyUIClient.batch_run_workflows()` — all `/prompt` submits up front so the batch runs unattended (GPU batching rule pattern), `_listen_for_many()` tracks every prompt_id with per-job error capture, queue-drain wait, per-job `_cached_file_bytes` pre-fetch before the idle kill, and a polling fallback if the WS is unavailable/closed. Results export to `export_dir` (default `<server>/batch_output`) as PNG/JPG; the tool returns a JSON summary (per-job status/file/seed). Flux2 jobs capped at 6 steps.
- **ControlNet/Redux repair:** both legacy builders fixed against the live node schemas — `EmptyLatentImage(width//8)` → `EmptyFlux2LatentImage(pixel dims)` (the Flux2 convention `build_txt2img_workflow` already used), `ControlNetApply` now passes `conditioning` (the old `positive` input no longer exists in current ComfyUI), `StyleModelApply` gained `strength_type="multiply"`, `CLIPVisionEncode` gained its required `CLIPVisionLoader` upstream + `crop="center"`. `controlnet_generate`/`style_transfer` now pre-validate live model availability via `controlnet_capabilities()`/`style_transfer_capabilities()` (new, in `editing.py`, also surfaced in `get_editing_capabilities`) and fail fast with actionable errors before any upload; `style_transfer` gained a `style_model` parameter and `clip_vision` is resolved from live options.
- **Docs:** README updated (Sessions & Batch section, future-work checkboxes, model prerequisites, batch usage example); 23 new tests — `tests/test_sessions.py` (8), `tests/test_batch_generate.py` (5), `tests/test_custom_nodes.py` (10, incl. builder-schema pins against the live node list and pre-check fail-fast paths). Full suite: **69 tests, 0 failures** (`python -m unittest discover -s tests`).
- **Live status (2026-09-18):** `verification/live_status_2026-09-18_sessions_batch_nodes.json` (script `verification/_live_sessions_batch_nodes_test.py`) — all green: multi-session tools (created/inspected/closed two live sessions), capabilities report (flux2 + qwen2511 + SAM3 available), ControlNet pre-check failed fast as designed, live batch 2/2 flux2 jobs OK with both PNGs exported to `verification/batch_live/`; style_transfer live run **skipped** — `models/style_models` and `models/clip_vision` hold only 0-byte placeholder files (no Redux/SigCLIP installed), so the pre-check correctly reported unavailable

### 2026-09-18 — Blend-Mode Alpha Fix (F2) + ComfyUI Health-Probe Validation (F4)
- **`canvas.py` — alpha-aware blend modes:** all 10 non-normal modes (`multiply`, `screen`, `overlay`, `darken`, `lighten`, `color_dodge`, `color_burn`, `hard_light`, `soft_light`, `exclusion`) previously ran per-channel ops on raw RGBA data, ignoring the top layer's per-pixel alpha — a masked edit layer set to `multiply` altered ~99.99% of the canvas and zeroed composite alpha outside its region (found in the 2026-09-17/18 MCP test campaign, F2, high severity). Now all modes route through `_blend_with_alpha()`: the formula applies to RGB only, top alpha (folded with layer opacity) scales the result toward the bottom (`out = bottom + a·(op(bottom,top) − bottom)/255`), output alpha is standard source-over. Transparent top pixels pass the bottom through **bit-exactly**; fully-opaque top is exactly the legacy formula (regression-pinned). `color_burn` gained a `t=0 → 255` guard (previously a latent division by zero). `merge_down` now applies the top layer's mask **before** blending (mirroring `composite()`).
- **`comfy_client.py` — health-probe validation:** `is_running()` previously treated any HTTP 200 as "running", so a foreign process occupying port 8188 (e.g. a stub server) bypassed the stale-port self-heal and later failed with an opaque JSON-decode error (F4/T9.x). It now requires `/history` to return a JSON object — a fresh ComfyUI returns `{}`, which is valid (the `queue` keys belong to `/queue`, not `/history`); non-JSON or non-object 200 responses are logged as "foreign process occupies the port" and treated as not-running so the existing kill/stale-port/relaunch flow runs. `start_comfyui`'s final TimeoutError now hints at `netstat -ano | findstr :8188`.
- **Tests:** new `tests/test_blend_modes.py` (8 tests: all modes × transparent/partial/zero-alpha, opaque-top oracles vs ImageChops + pinned formulas, masked-edit multiply F2 scenario, `merge_down` == `composite`); 4 `is_running()` tests in `tests/test_editing.py` (non-JSON 200 rejected, JSON non-object rejected, valid history accepted, fresh empty `{}` history accepted — the last added after the re-check caught regression F5). Full suite: **46 tests, 0 failures** (Python 3.10, `python -m unittest discover -s tests`).
- **Live status (2026-09-18, post-restart): LIVE-VERIFIED.** After the user restarted the `photoshop` MCP server: (1) **F2** — masked `multiply` layer confined its effect to the mask (0 changed px outside, alpha extrema (255,255)); unmasked control shows full-canvas multiply. (2) **F3** — `semantic_select(point)` on a synthetic red circle returned the exact bbox (70,797 px, 1 object); `undo` cleared `has_mask`, `redo` restored it — the 2026-09-18 anomaly confirmed as version skew, fully resolved. (3) The re-check caught a **regression in the F4 fix (F5)**: `is_running()` required a `queue` key in the `/history` JSON, but a fresh ComfyUI returns `{}` — healthy fresh instances were misread as foreign port occupants, causing a kill/relaunch loop ending in "ComfyUI did not become reachable within 180s". Fixed: accept any JSON object (new unit test; suite 45 → 46). Also raised the Cline MCP client `timeout` for `photoshop` 300 → 1800 s (SAM 3 CPU inference exceeds 300 s). Evidence: test-campaign `REPORT.md` §5.4–5.6 (`fix2_check_masked_multiply.png`, `fix2_check_nomask_multiply.png`, `comfy_poll.log`).

### 2026-09-17 — RealESRGAN x4plus + SAM 3 Supporting Models
- `RealESRGAN_x4plus.pth` (67,040,989 bytes; SHA-256 `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1`) installed in `models/upscale_models` from the official Real-ESRGAN release (v0.1.0 asset per the project README; the v0.2.0 download link 404s). Live MCP stdio test `verification/_realesrgan_test.py` upscaled a 512px canvas to 1024px (15.1s cold / 7.7s warm, alpha intact); evidence `verification/realesrgan_test_status.json`, `verification/realesrgan_upscaled.png`.
- `sam3.pt` (3,450,062,241 bytes; SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`) installed in `models/detection/`. Official `facebook/sam3` HF repo is access-gated (local token not approved), so the `cubert-gmbh/sam3` mirror was used — verified byte-identical by matching LFS pointer OID.
- ComfyUI 0.33.0 `SAM3_Detect` smoke-tested via `/prompt` on a temporary CPU instance (GPU saturated by the LLM container): point-prompt mask in 42.2s, white fraction 0.110 matching the synthetic circle; evidence `verification/sam3_smoke_status.json`, `verification/sam3_smoke_mask.png`. Found: `CheckpointLoaderSimple` fails on the original SAM 3 clip (wrapper expects SAM 3.1 shapes); `ImageOnlyCheckpointLoader` + point/box prompts work; text prompting needs the SAM 3.1 checkpoint. MCP semantic-mask integration remains pending (no server code changed, no MCP restart needed).
- ComfyUI model folders rescan per request (`folder_paths` mtime cache) — new model files appear in loader dropdowns without a restart.

### 2026-09-17 — semantic_select (SAM 3) MCP Tool
- Added `semantic_select(point, box, threshold, refine, timeout)` in `editing.py` (`register_editing_tools`): uploads the canvas composite, runs `ImageOnlyCheckpointLoader` → `SAM3_Detect` → `MaskToImage` → `SaveImage`, and sets the union mask as the active layer's selection (undoable, same mechanism as `select_rect`). Returns coverage/bbox/object-count plus a mask preview; an empty detection leaves the layer untouched. Tool count 39 → 40.
- Deployment fix: `ImageOnlyCheckpointLoader` lists `ckpt_name` only from `models/checkpoints` (the `detection` folder is used only by MediaPipe nodes), so `sam3.pt` is hardlinked into `models/checkpoints/` (same inode, zero extra disk); ComfyUI's per-request rescan listed it without a restart.
- Live-verified through a fresh MCP stdio session against a temporary CPU-only ComfyUI on :8189 (`verification/_semantic_live_test.py` → `verification/semantic_live_status.json`): point prompt (256,256) and box prompt [160,160,192,192] both segmented the synthetic red circle exactly (bbox [160,161,353,353], coverage 0.1107, 1 object, ~42s each on CPU); `get_info` shows `has_mask: true`; temp instance torn down, :8189 released, :8188 untouched. Evidence PNGs: `verification/semantic_mask_point.png`, `verification/semantic_mask_box.png`.
- `get_editing_capabilities` now reports a `sam3` availability block (node + checkpoint). 11 unit tests added in `tests/test_semantic_select.py`; suite now 28 tests, all passing.
- Still pending: text-prompted SAM 3 needs the SAM 3.1 checkpoint (`sam3.1_multiplex_fp16.safetensors`), which is not installed locally; `select_object` remains the fast heuristic fallback.

### 2026-09-17 — Editing-Upgrade Guide Synced (Out-of-Tree Copy)
- `EDITING_UPGRADE.md` was moved out of the project root to a local, out-of-tree copy (not shipped in this repo); the README's stale architecture-tree reference to it was removed on 2026-09-18
- RealESRGAN x4plus and SAM 3 sections updated from todo to deployed + live-verified (sizes, SHA-256, evidence files in `verification/`); SAM 3 heading now reflects checkpoint deployed, smoke-tested, MCP integration pending.
- Corrected the record: the SAM 3.1 checkpoint (`sam3.1_multiplex_fp16.safetensors`) is not present in `models/checkpoints/` — text-prompted SAM 3 needs that separate gated download; point/box prompting works with the deployed `sam3.pt`.

### 2026-09-17 — Documentation Synced to 39 Tools
- README.md: added `### Instruction Editing` feature subsection (`edit_image`, `get_editing_capabilities`, `preview_canvas`); AI Edit Workflow example now uses `edit_image` ("flux2" fast path, "qwen2511" reference path) with `preview_canvas` and a GPU batching rule comment; architecture tree lists 39 tools plus `editing.py`, `EDITING_UPGRADE.md`, `tests/`, `verification/`.
- MEMORY.md: tool inventory updated to 39, file architecture adds `editing.py` and notes `GPU_BATCH_RULE` in `instructions`, Qwen 2511 model files added under Instruction Editing, testing status now 39 tools.

### 2026-09-17 — GPU Contention & Batching Rule
- Added standing rule: batch all ComfyUI generations in the background, then (with explicit user approval) stop the LLM docker so generation gets the full GPU. Stopping it ends the LLM session, which is acceptable only after the batch is fully queued; remind the user to `docker start` the LLM (and `open-webui`) afterwards.
- Delivered via three channels: FastMCP `instructions` (MCP `initialize` response, `GPU_BATCH_RULE` in `server.py`), `get_editing_capabilities` notes (`editing.py`), and this MEMORY.md section.
- Context: qwen2511 FP8 `e4m3fn` (1038lab) deployed and live-verified (see EDITING_UPGRADE.md); running it alongside the LLM causes severe GPU contention.

### 2026-09-17 — SAM 3.1 Text Prompting for semantic_select
- Deployed `sam3.1_multiplex_fp16.safetensors` (1,745,546,848 bytes; SHA-256 `9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03`) to `models/checkpoints/` from the open `Comfy-Org/sam3.1` HF repo (not gated). Header parses (1,590 tensors incl. `language_backbone`); live :8188 ComfyUI listed it under `CheckpointLoaderSimple` + `ImageOnlyCheckpointLoader` via per-request rescan — no restart.
- `semantic_select` now takes `prompt` alongside `point`/`box` (at least one required, combinations allowed): text runs `CheckpointLoaderSimple` → `CLIPTextEncode` → `SAM3_Detect(conditioning=...)` on the 3.1 checkpoint; point/box-only calls keep using `sam3.pt` via `ImageOnlyCheckpointLoader`. Missing 3.1 + `prompt` raises before upload/inference. `sam3_capabilities` reports `sam3_1` and `text_prompt_available`; capabilities note updated.
- 6 unit tests added to `tests/test_semantic_select.py` (text graph wiring, prompt+point+box, missing-3.1 pre-upload error, blank prompt, 3.1 capability reporting); full suite now 34 tests, all passing.
- Live-verified via fresh MCP stdio session against a temporary CPU-only ComfyUI on :8189 (`verification/_semantic31_live_test.py` → `verification/semantic31_live_status.json`): capabilities `text_prompt_available: true`; text "red circle" 64.9s, bbox [160, 161, 353, 353], coverage 0.1107; point regression (256,256) 42.9s, still on `sam3.pt` with identical segmentation; temp instance torn down (:8189 released), :8188 untouched.
- Docs synced: out-of-tree `EDITING_UPGRADE.md` copy (check-off + SAM 3.1 verification section) and README future-work item ticked. `photoshop` MCP entry restarted and verified: the live `semantic_select` schema exposes `prompt`, and `get_editing_capabilities` against :8188 reports `sam3_1: sam3.1_multiplex_fp16.safetensors` and `text_prompt_available: true`.

### 2026-08-12 — Bug Fixes: Auto-Start, ControlNet, Auto-Kill
- **Auto-start fix** (`comfy_client.py`): Added `await self.start_comfyui()` at the top of `upload_image()` so all tools that upload images before running workflows (img2img, inpaint, outpaint, upscale, controlnet_generate, style_transfer) now auto-start ComfyUI. Previously only `generate_image` triggered auto-start.
- **ControlNet fix** (`server.py`): Added `ImageScale` node in `build_controlnet_workflow()` to resize the control image to match target `width x height` before passing to `ControlNetApply`. Fixes "images do not match" error when canvas size differs from requested output size.
- **Auto-kill enabled** (`.env`): Set `COMFYUI_AUTO_KILL=1` and `COMFYUI_IDLE_TIMEOUT=5` to kill ComfyUI 5 seconds after each task, preventing resource exhaustion and Cline freezes.
- **python-dotenv integration** (`config.py`, `requirements.txt`): Added `.env` file loading via `python-dotenv` so configuration is automatically read from `.env` on startup.
- **All AI tools tested**: generate_image (flux2+anima), img2img, inpaint, upscale, controlnet_generate, outpaint — all working with auto-start and auto-kill.

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

### 2026-08-12 — Idle Timeout Auto-Kill (60s)
- Replaced immediate auto-kill with configurable idle timeout
- `COMFYUI_IDLE_TIMEOUT` (default 60s) — kills ComfyUI after N seconds of inactivity
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
