"""One-time migration of the four saved Klein workflows to native FLUX.2 Dev."""
import copy
import json
import os
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ComfyUI root: prefer an explicit COMFYUI_DIR, otherwise derive it from
# COMFYUI_MAIN (.../ComfyUI/main.py) as configured in .env. No machine paths
# are stored in this file.
_comfy_dir = os.environ.get('COMFYUI_DIR', '').strip()
_comfy_main = os.environ.get('COMFYUI_MAIN', '').strip()
if _comfy_dir:
    COMFY = Path(_comfy_dir)
elif _comfy_main:
    COMFY = Path(_comfy_main).resolve().parent
else:
    raise SystemExit('Set COMFYUI_MAIN (or COMFYUI_DIR) to locate the ComfyUI root; see .env')
SCHEMA = json.loads((HERE / 'object_info.json').read_text())
FILES = sorted((COMFY / 'user/default/workflows').glob('flux2_klein*.json'))
BACKUP = HERE / 'workflows_before_dev.zip'
PROMPT_NOTE = """FLUX.2 Dev prompting: describe subject, action, style and context in connected prose. Put the most important requirement first, preserve exact counts/colors/lettering, and specify reference roles for edits. Usually 30–80 words, longer when needed. Describe desired outcomes positively: this guidance-distilled model has no negative-prompt branch. Embedded guidance 4; 50 Euler steps (28 is a faster trade-off). NVFP4 is weight quantization, not four-step distillation. See https://huggingface.co/black-forest-labs/FLUX.2-dev and https://docs.comfy.org/tutorials/flux/flux-2-dev ."""
MIGRATION_NOTE = """Migrated to FLUX.2 Dev NVFP4 + Mistral Small FP8 + flux2 VAE. Uses FluxGuidance(4), BasicGuider, Euler, Flux2Scheduler(50), full denoising. Width/height controls feed both latent and schedule. Negative conditioning and AuraFlow shift are removed. The old embedded JSON is in workflows_before_dev.zip, not an alternate executable recipe. Filenames are retained for existing bookmarks. Klein LoRAs are incompatible with Dev; the N variant now runs without its old Klein adapter."""


def migrate(doc):
    doc = copy.deepcopy(doc)
    nodes = {n['id']: n for n in doc['nodes']}
    links = {v[0]: v[:] for v in doc['links']}
    node_id = max(nodes)
    link_id = max(links, default=0)

    def input_link(nid, name):
        slot = next(i for i, p in enumerate(nodes[nid].get('inputs', [])) if p['name'] == name)
        return next(v for v in links.values() if v[3:5] == [nid, slot])

    def connect(src, slot, dst, field):
        nonlocal link_id
        target_slot = next(i for i, p in enumerate(nodes[dst]['inputs']) if p['name'] == field)
        link_id += 1
        links[link_id] = [link_id, src, slot, dst, target_slot, nodes[src]['outputs'][slot]['type']]

    def make(kind, values=None, pos=(0, 0), mode=0, nid=None, title=None):
        nonlocal node_id
        if nid is None:
            node_id += 1
            nid = node_id
        schema = SCHEMA[kind]
        values = values or {}
        inputs, widgets, named = [], [], {}
        for group in ('required', 'optional'):
            for name, spec in schema['input'].get(group, {}).items():
                typ = 'COMBO' if isinstance(spec[0], list) else spec[0]
                options = spec[1] if len(spec) > 1 else {}
                entry = {'name': name, 'type': typ, 'link': None}
                if typ in ('INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO'):
                    entry['widget'] = {'name': name}
                    value = values[name] if name in values else options['default']
                    widgets.append(value)
                    named[name] = value
                    if options.get('control_after_generate'):
                        control = values.get('control_after_generate', 'fixed')
                        widgets.append(control)
                        named['control_after_generate'] = control
                inputs.append(entry)
        outputs = [{'name': name, 'type': typ, 'links': None, 'slot_index': i}
                   for i, (name, typ) in enumerate(zip(schema['output_name'], schema['output']))]
        node = {'id': nid, 'type': kind, 'pos': list(pos), 'size': [300, 180],
                'flags': {}, 'order': 0, 'mode': mode, 'inputs': inputs, 'outputs': outputs,
                'properties': {'cnr_id': 'comfy-core', 'Node name for S&R': kind},
                'widgets_values': widgets, 'widgets_values_named': named}
        if title:
            node['title'] = title
        nodes[nid] = node
        return nid

    # Remove architecture-specific adapters, preserving their input connections.
    for nid, n in list(nodes.items()):
        if n['type'] not in ('LoraLoader', 'ModelSamplingAuraFlow'):
            continue
        routes = {0: input_link(nid, 'model')[1:3]}
        if n['type'] == 'LoraLoader':
            routes[1] = input_link(nid, 'clip')[1:3]
        for link in links.values():
            if link[1] == nid:
                link[1:3] = routes[link[2]]
        links = {k: v for k, v in links.items() if v[3] != nid}
        del nodes[nid]

    for n in nodes.values():
        kind = n['type']
        if kind == 'UNETLoader':
            n['widgets_values'] = ['flux2-dev-nvfp4.safetensors', 'default']
            n['widgets_values_named'] = {'unet_name': n['widgets_values'][0], 'weight_dtype': 'default'}
            n['title'] = 'FLUX.2 Dev NVFP4'
        elif kind == 'CLIPLoader':
            n['widgets_values'] = ['mistral_3_small_flux2_fp8.safetensors', 'flux2', 'default']
            n['widgets_values_named'] = dict(zip(('clip_name', 'type', 'device'), n['widgets_values']))
        elif kind == 'FluxGuidance':
            n['widgets_values'] = [4.0]
            n['widgets_values_named'] = {'guidance': 4.0}
        elif kind == 'Note':
            old = n['widgets_values'][0]
            if old.lstrip().startswith('{'):
                text = MIGRATION_NOTE
            elif old.startswith('You are an expert'):
                text = PROMPT_NOTE
            elif old.startswith('WHAT CHANGED:'):
                text = 'Generate one front view, then reference it for side/back views and prop crops. Slot 1 is active; slots 2–3 remain bypassed templates. Enable and describe only props present in your character. This uses the same FLUX.2 Dev model for every view; no second editing model is needed.'
            else:
                text = old.replace('Flux Klein', 'FLUX.2 Dev')
            n['widgets_values'] = [text]
            n.pop('widgets_values_named', None)

    size_controls = {}
    for nid, n in list(nodes.items()):
        if n['type'] != 'EmptyFlux2LatentImage':
            continue
        width, height, batch = n['widgets_values']
        n['widgets_values_named'] = {'width': width, 'height': height, 'batch_size': batch}
        x, y = n['pos']
        w = make('PrimitiveInt', {'value': width}, (x - 350, y), title='Output width')
        h = make('PrimitiveInt', {'value': height}, (x - 350, y + 210), title='Output height')
        connect(w, 0, nid, 'width')
        connect(h, 0, nid, 'height')
        size_controls[nid] = w, h

    negatives = set()
    samplers = [n for n in list(nodes.values()) if n['type'] == 'KSampler']
    for n in samplers:
        nid, mode = n['id'], n.get('mode', 0)
        model = input_link(nid, 'model')[1:3]
        positive = input_link(nid, 'positive')[1:3]
        latent = input_link(nid, 'latent_image')[1:3]
        negatives.add(input_link(nid, 'negative')[1])
        seed, control = n['widgets_values'][:2]
        x, y = n['pos']
        if nodes[positive[0]]['type'] != 'FluxGuidance':
            guide = make('FluxGuidance', {'guidance': 4.0}, (x, y), mode)
            connect(*positive, guide, 'conditioning')
            positive = [guide, 0]
        guider = make('BasicGuider', pos=(x + 340, y), mode=mode)
        connect(*model, guider, 'model')
        connect(*positive, guider, 'conditioning')
        noise = make('RandomNoise', {'noise_seed': seed, 'control_after_generate': control}, (x, y + 200), mode)
        selector = make('KSamplerSelect', {'sampler_name': 'euler'}, (x, y + 420), mode)
        w, h = size_controls[latent[0]]
        scheduler = make('Flux2Scheduler', {'steps': 50, 'width': nodes[w]['widgets_values'][0],
                                          'height': nodes[h]['widgets_values'][0]}, (x + 340, y + 220), mode)
        connect(w, 0, scheduler, 'width')
        connect(h, 0, scheduler, 'height')
        links = {k: v for k, v in links.items() if v[3] != nid}
        make('SamplerCustomAdvanced', pos=(x + 680, y), mode=mode, nid=nid)
        for src, field in ((noise, 'noise'), (guider, 'guider'), (selector, 'sampler'), (scheduler, 'sigmas')):
            connect(src, 0, nid, field)
        connect(*latent, nid, 'latent_image')
        for edge in list(links.values()):
            if edge[1] == nid and nodes[edge[3]]['type'] == 'VAEDecode':
                nodes[edge[3]]['pos'] = [x + 1020, y]
                for downstream in links.values():
                    if downstream[1] == edge[3] and nodes[downstream[3]]['type'] == 'SaveImage':
                        nodes[downstream[3]]['pos'] = [x + 1340, y]

    for nid in negatives:
        assert not any(v[1] == nid for v in links.values())
        links = {k: v for k, v in links.items() if v[3] != nid}
        del nodes[nid]
    # Rebuild both ends; stale output-link lists otherwise break canvas editing.
    for n in nodes.values():
        for p in n.get('inputs', []): p['link'] = None
        for p in n.get('outputs', []): p['links'] = None
    for lid, src, slot, dst, target, typ in links.values():
        assert nodes[src]['outputs'][slot]['type'] == nodes[dst]['inputs'][target]['type'] == typ
        assert nodes[dst]['inputs'][target]['link'] is None
        nodes[dst]['inputs'][target]['link'] = lid
        port = nodes[src]['outputs'][slot]
        if port['links'] is None: port['links'] = []
        port['links'].append(lid)
    pending = set(nodes)
    done = set()
    order = 0
    while pending:
        ready = sorted(n for n in pending if all(e[1] in done for e in links.values() if e[3] == n))
        assert ready, 'Cycle in migrated workflow'
        for nid in ready:
            nodes[nid]['order'] = order
            order += 1
        pending.difference_update(ready)
        done.update(ready)
    doc['nodes'] = list(nodes.values())
    doc['links'] = list(links.values())
    doc['last_node_id'], doc['last_link_id'] = node_id, link_id
    doc['revision'] = doc.get('revision', 0) + 1
    return doc


if __name__ == '__main__':
    assert len(FILES) == 4
    if not BACKUP.exists():
        with zipfile.ZipFile(BACKUP, 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in FILES: archive.write(path, path.name)
    for path in FILES:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if not any(n['type'] == 'KSampler' for n in data['nodes']):
            continue
        result = migrate(data)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(path.name, len(result['nodes']), 'nodes', len(result['links']), 'links')
