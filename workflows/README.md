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

Validation: 206 tests pass, including task-specific model/node availability, reference sizing, EXIF orientation, mask alignment, outside-mask pixels and MCP tool schemas. All nine UI files round-trip through the installed frontend with matching API graphs; native CPU checks verify landscape/portrait sizing and the example input. A fresh MCP stdio connection reads the installed H3 master with matching content/hash for both tasks. Local Qwen passed four master-format examples covering illustration, infographic and single/multiple-reference edits. These checks do not establish image quality. The original REF2VA generation/recolor pair completed on the GPU at 640x384 and 20 steps. That earlier result does not validate the new FL2VA generation route or compare `match` against `max`; those quality/speed comparisons have not been run while the local LLM occupies the GPU.
