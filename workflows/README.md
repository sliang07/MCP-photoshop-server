# MCP Image Presets

Open a file from `ui/` in ComfyUI, or find it in **Workflows → MCP Image Presets** on this installation. Files in `api/` are executable prompt graphs for ComfyUI's API; they are not the visual editor format. Existing workflows are unchanged.

| File stem | Purpose |
|---|---|
| `minimax_h3_text_to_image` | H3 still from text, using the installed ref2va checkpoint without references |
| `minimax_h3_reference_edit` | H3 edit/restyle from one image (`<Picture 1>`) |
| `minimax_h3_two_references` | H3 composition from two images (`<Picture 1>` and `<Picture 2>`) |
| `flux2_text_to_image` | FLUX.2 Dev generation, 50 steps/embedded guidance 4 |
| `flux2_reference_edit` | FLUX.2 Dev reference-guided editing |
| `qwen21_text_to_image` | Qwen Image 2.1 generation, existing custom 30 steps/CFG 3 preset |
| `qwen21_reference_edit` | Qwen Image 2.1 instruction editing |
| `anima_text_to_image` | Anima Aesthetic illustration, 30 steps/CFG 4 |
| `upscale_general_4x` | General image upscaling with the installed RealESRGAN_x4plus model |

Replace the mug example in each `LoadImage` node with your own image. The two-reference example initially uses the same mug in both slots; select two intended references and describe their roles. `examples/mcp_h3_example.png` is included and already installed in ComfyUI's input folder. When moving the pack to another installation, upload it or select another input. All model files must already exist; this pack downloads nothing.

H3 uses the installed ref2va INT8 checkpoint, MiniMax Qwen3-VL AWQ encoder and FP16 video VAE. It samples five frames and saves only frame zero as an RGB PNG, with 20 steps, `res_multistep`, `simple` and `BasicGuider`. It needs no audio VAE, video saver, custom nodes or Turbo LoRA. Five-frame still generation is experimental. These presets deliberately use the installed base model settings; the original supplied hybrid/Turbo workflow cannot run unchanged with this installation's weights.

Generation examples start at 1024x768. Keep H3 dimensions at multiples of 32. For Flux, shared width/height nodes keep the latent and scheduler in sync. Seeds are fixed for repeatability. Qwen edits use the first image's resolution; select an appropriate input or change its encoder's resolution setting. Images are saved in ComfyUI's output folder under the MCP filename prefixes.

The standalone JSONs contain model graphs. MCP canvas layers, undo, masked compositing and exact outside-mask preservation are implemented by the MCP tools, not by these standalone reference-edit graphs.

After reconnecting the MCP, use:

```json
{"tool":"generate_image","arguments":{"model":"minimax_h3","prompt":"A red ceramic mug on a cream tabletop","width":1024,"height":768}}
```

For edits, first call `open_image` with your source, then:

```json
{"tool":"edit_image","arguments":{"backend":"minimax_h3","prompt":"Change the mug in <Picture 1> to cobalt blue. Keep the shape, lighting and background."}}
```

Use the same `session_id` for related calls. `reference_paths` adds `<Picture 2>` onward. `get_editing_capabilities` checks installed files and nodes; `get_prompt_guidance(model="minimax_h3", task="editing")` provides the still-image rules. H3 also works in batch/background generation jobs. Its `cfg` and `negative_prompt` arguments are unused, and its default timeout is 3600 seconds.
