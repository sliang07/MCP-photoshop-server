# MCP Image Presets

Open a file from `ui/` in ComfyUI, or find it in **Workflows → MCP Image Presets** on this installation. Files in `api/` are executable prompt graphs for ComfyUI's API; they are not the visual editor format. Existing workflows are unchanged.

| File stem | Purpose |
|---|---|
| `minimax_h3_text_to_image` | H3 still from text, using the installed fl2va checkpoint without keyframes |
| `minimax_h3_reference_edit` | H3 edit/restyle from one image (`<Picture 1>`) |
| `minimax_h3_two_references` | H3 composition from two images (`<Picture 1>` and `<Picture 2>`) |
| `flux2_text_to_image` | FLUX.2 Dev generation, 50 steps/embedded guidance 4 |
| `flux2_reference_edit` | FLUX.2 Dev reference-guided editing |
| `qwen21_text_to_image` | Qwen Image 2.1 generation, existing custom 30 steps/CFG 3 preset |
| `qwen21_reference_edit` | Qwen Image 2.1 instruction editing |
| `anima_text_to_image` | Anima Aesthetic illustration, 30 steps/CFG 4 |
| `upscale_general_4x` | General image upscaling with the installed RealESRGAN_x4plus model |

Replace the mug example in each `LoadImage` node with your own image. The two-reference example initially uses the same mug in both slots; select two intended references and describe their roles. `examples/mcp_h3_example.png` is included and already installed in ComfyUI's input folder. When moving the pack to another installation, upload it or select another input. All model files must already exist; this pack downloads nothing.

H3 text generation uses the installed FL2VA INT8 checkpoint with `MiniMaxH3ImageToVideo` and no first/last keyframe. Reference editing uses REF2VA INT8 with `MiniMaxH3ReferenceToVideo`. Both use the MiniMax Qwen3-VL AWQ encoder and FP16 video VAE. This task selection follows the [native ComfyUI H3 guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native).

The still presets sample five frames and save only frame zero as an RGB PNG, with 20 steps, `res_multistep`, `simple` and `BasicGuider`. To compare all five frames in ComfyUI, set `ImageFromBatch.length` to 5; to select one, change `batch_index` (0–4) and leave length at 1. Five-frame still generation is experimental, as discussed by the [H3 Image Studio author](https://github.com/astropuzzo/ComfyUI-MiniMax-H3-Image-Studio). No audio VAE, video saver, custom nodes or Turbo LoRA is needed. The supplied hybrid/Turbo workflow requires different weights; the base presets do not silently substitute them.

Generation examples start at 1024x768. Keep H3 dimensions at multiples of 32. H3 reference workflows derive output proportions from the first image, rounded to 32px; adjust `ImageScaleToTotalPixels.megapixels` to change output area (default 0.75MP). Original references feed the encoder separately: `ref_image_size=match` limits reference area to output area; `max` retains more detail, up to a 2048px short edge, at higher memory and sampling cost. Reference fidelity remains model-dependent.

The standalone reference graphs save the generated dimensions. MCP `edit_image` instead computes its working dimensions with `max_side`, then resizes the result back to the source canvas dimensions before compositing. It does not use the standalone 0.75MP sizing node. Both paths approximately follow source proportions after grid rounding. Native `adapt_canvas` applies to reference-video frames only; it does not force still-image canvases to a 768px short edge.

For Flux, shared width/height nodes keep the latent and scheduler in sync. Seeds are fixed for repeatability. Qwen edits use the first image's resolution; select an appropriate input or change its encoder's resolution setting. Images are saved in ComfyUI's output folder under the MCP filename prefixes.

The standalone JSONs contain model graphs. MCP canvas layers, undo, masked compositing and exact outside-mask preservation are implemented by the MCP tools, not by these standalone reference-edit graphs.

For H3 prompt authoring, give your LLM `minimax_h3_pseudo_image_master.txt` (in the ZIP, beside the installed UI presets, and in the repository's `masters/` folder) and your image brief. It returns `rewritten_prompt`, `wh_ratio` and `ratio_follow` in JSON, matching the existing authoring convention. Paste only the decoded `rewritten_prompt` into the H3 prompt widget; apply the requested sizing separately. The visual graph does not execute an LLM master automatically. The master covers generation, references, local edits and text-heavy layouts without adding video timelines or audio fields.

After reconnecting the MCP, use:

```json
{"tool":"generate_image","arguments":{"model":"minimax_h3","prompt":"A red ceramic mug on a cream tabletop","width":1024,"height":768}}
```

For edits, first call `open_image` with your source, then:

```json
{"tool":"edit_image","arguments":{"backend":"minimax_h3","prompt":"Change the mug in <Picture 1> to cobalt blue. Keep the shape, lighting and background."}}
```

Use the same `session_id` for related calls. `reference_paths` adds `<Picture 2>` onward. `get_editing_capabilities` checks installed files and nodes; `get_prompt_guidance(model="minimax_h3", task="editing")` provides the still-image rules. H3 also works in batch/background generation jobs. Its `cfg` and `negative_prompt` arguments are unused, and its default timeout is 3600 seconds.

For higher-detail MCP edits, add `"h3_reference_detail":"max"`. The default `match` retains the previous sizing behavior. `max_side` still controls working output resolution; the result returns to the original canvas size, and outside-mask pixels remain protected. The higher-detail mode uploads original-resolution canvas/reference images rather than reducing them to `max_side` first.

Prior integration validation: 206 tests passed, including task-specific model/node availability, reference sizing, EXIF orientation, mask alignment, outside-mask pixels and MCP tool schemas. All nine UI files round-tripped through the installed frontend with matching API graphs; native CPU checks verified landscape/portrait sizing and the example input. A fresh MCP stdio connection read the installed H3 master with matching content/hash for both tasks. Local Qwen passed four master-format examples. The later benchmark-guidance wording changes were not retested, as requested by the user.

## Supplied H3 quality benchmark

The initial user-supplied H3-only run used FL2VA generation, REF2VA edits, 20 steps, `res_multistep/simple`, BasicGuider and frame 0 of 5. The initial folder contained eight PNGs: two sources, two global edits, three local-edit attempts and one comparison grid. It contained no measured per-operation timings. Its summary said nine files and classified both final local edits as clean; visual inspection differed. The subsequent redo is reviewed separately below.

| Image(s) | Visual review |
|---|---|
| `h3_gen_realistic.png`, `h3_gen_anime.png` | Usable photographic and anime compositions in the supplied comparison grid. No matched comparison with another model was supplied. |
| `edit_realistic_global_h3.png`, `edit_anime_global_h3.png` | The global `max` examples largely retain scene layout and subject appearance while changing lighting. These support this use case, not a universal fidelity guarantee. |
| `edit_realistic_local_h3.png` | Replacement furniture appears, but a hard rectangular boundary is visible through the floor and left chair. Not a clean integration. |
| `edit_anime_local_h3_attempt1_fail.png` | Replacement leaves appear against an unintended white rectangular panel. Failed integration. |
| `edit_anime_local_h3.png` | The retry still has an unintended black rectangular panel around the replacement. It is not a clean retry. |

Inspect the complete mask interior and boundary before accepting an edit. Exact preservation outside the mask is a compositing property; it does not guarantee a plausible interior or seamless join. When a retry is authorized, restore the original source and try a fresh seed, then inspect again. Do not accept an artifact just because its color changed. Persistent defects must be reported rather than silently accepted or retried indefinitely.

The examples do not establish a measured speed difference, a controlled `max` versus `match` comparison, or a reliability ranking between generation and editing. No new generation, benchmark or automated test was run for this review.

### September 22 redo

The supplied redo used REF2VA for all four edit stages, `h3_reference_detail=max`, `max_side=1024`, 20 steps and frame 0 of 5; local masks used feather 16. Its 1024x1024 outputs are consistent with MCP returning the source canvas dimensions. Output dimensions alone do not establish the internal working size, the standalone 0.75MP path, or an effect from a documentation edit.

| Redo output | Review of supplied images |
|---|---|
| `anime_global_h3.png`, `realistic_global_h3.png` | Scene structure and subject appearance are largely retained. This visual review does not quantify the claimed detail improvement. |
| `anime_local_h3.png` | The glowing maple-leaf lantern is visibly changed and blends into the scene without the earlier rectangular panel. |
| `realistic_local_h3.png` | First attempt has a white-out covering the selected region; failed. |
| `realistic_local_h3_attempt2.png` | Retry removes the white-out and looks integrated, but the source/crop comparison does not clearly show the requested light-oak armchair replacement. Armrests and a distinctive material/design change are not evident enough to verify the reported full pass. |

The report's 4/4 final-stage pass should distinguish visible integration from instruction adherence. A near-copy can be seamless while failing the requested replacement. Check both criteria; identify a target by its position relative to other objects and state the defining features of its replacement.

The report's statement about unmasked-area drift needs a distinction: global edits can change the entire image, but the current MCP restores source pixels wherever a supplied mask is fully black. Its feathering is inward and blends within the selection. This follows the current `make_edit_mask` and final compositing code; no new pixel test was run.

The redo supports that changing a seed can remove a white-out in a particular attempt. It does not guarantee correct object replacement. Multiple settings and the edit goals differ from the initial round, so improvement cannot be assigned to `max`, feathering or the master wording alone. No new generation or automated test was run for this review.
