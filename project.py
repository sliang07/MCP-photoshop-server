"""Local layered project files: a versioned manifest and lossless PNG images."""

import json
import math
import os
import tempfile
import zipfile

from PIL import Image

from canvas import BLEND_MODES, Canvas, Layer

MANIFEST_FORMAT = "mcp-photoshop-project"
MANIFEST_VERSION = 1


def _validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a dict")
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"invalid format: {manifest.get('format')}")
    if not isinstance(manifest.get("version"), int) or isinstance(manifest.get("version"), bool) or manifest["version"] != MANIFEST_VERSION:
        raise ValueError(f"invalid version: {manifest.get('version')}")

    width = manifest.get("width")
    height = manifest.get("height")
    for dim_name, dim_val in [("width", width), ("height", height)]:
        if not isinstance(dim_val, int) or isinstance(dim_val, bool) or dim_val <= 0:
            raise ValueError(f"invalid {dim_name}: {dim_val}")

    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) == 0:
        raise ValueError("layers must be a non-empty list")

    active = manifest.get("active_layer")
    if not isinstance(active, int) or isinstance(active, bool) or active < 0 or active >= len(layers):
        raise ValueError(f"invalid active_layer: {active}")

    for i, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"layer {i} must be a dict")
        name = layer.get("name")
        if not isinstance(name, str):
            raise ValueError(f"layer {i} name must be a string")
        visible = layer.get("visible")
        if not isinstance(visible, bool):
            raise ValueError(f"layer {i} visible must be a boolean")
        opacity = layer.get("opacity")
        if isinstance(opacity, bool) or not isinstance(opacity, (int, float)) or not math.isfinite(opacity):
            raise ValueError(f"layer {i} opacity must be a finite number")
        blend_mode = layer.get("blend_mode")
        if not isinstance(blend_mode, str):
            raise ValueError(f"layer {i} blend_mode must be a string")
        normalized_blend = blend_mode.lower().replace(" ", "_")
        if normalized_blend not in BLEND_MODES:
            raise ValueError(f"layer {i} invalid blend_mode: {blend_mode}")
        image_path = layer.get("image")
        expected_image = f"layers/{i}.png"
        if image_path != expected_image:
            raise ValueError(f"layer {i} invalid image path: {image_path}")
        if "mask" not in layer:
            raise ValueError(f"layer {i} missing mask key")
        mask_path = layer["mask"]
        if mask_path is not None:
            expected_mask = f"masks/{i}.png"
            if mask_path != expected_mask:
                raise ValueError(f"layer {i} invalid mask path: {mask_path}")


def _read_png(archive, name, size, mode):
    if name not in archive.namelist():
        raise ValueError(f"missing member: {name}")
    with archive.open(name) as opened, Image.open(opened) as img:
        if img.format != "PNG":
            raise ValueError(f"{name} must be PNG")
        if img.mode != mode:
            raise ValueError(f"{name} must be {mode} mode")
        if img.size != size:
            raise ValueError(f"{name} size mismatch")
        return img.copy()


def save_project(canvas, path):
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "width": canvas.width,
        "height": canvas.height,
        "active_layer": canvas.active_layer_index,
        "layers": []
    }

    for i, layer in enumerate(canvas.layers):
        entry = {
            "name": layer.name,
            "opacity": layer.opacity,
            "blend_mode": layer.blend_mode,
            "visible": layer.visible,
            "image": f"layers/{i}.png",
            "mask": f"masks/{i}.png" if layer.mask else None
        }
        manifest["layers"].append(entry)

    _validate_manifest(manifest)

    for i, layer in enumerate(canvas.layers):
        if layer.image.mode != "RGBA":
            raise ValueError(f"layer {i} image must be RGBA")
        if layer.image.size != (canvas.width, canvas.height):
            raise ValueError(f"layer {i} image size mismatch")
        if layer.mask is not None:
            if layer.mask.mode != "L":
                raise ValueError(f"layer {i} mask must be L mode")
            if layer.mask.size != (canvas.width, canvas.height):
                raise ValueError(f"layer {i} mask size mismatch")

    parent_dir = os.path.dirname(os.path.abspath(path))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=parent_dir, suffix=".tmp", delete=False) as temp:
            temp_path = temp.name
            with zipfile.ZipFile(temp, "w") as zf:
                manifest_bytes = json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode("utf-8")
                zf.writestr("manifest.json", manifest_bytes)
                for i, layer in enumerate(canvas.layers):
                    data = layer.to_dict()
                    zf.writestr(f"layers/{i}.png", data["image_data"])
                    if data["mask_data"]:
                        zf.writestr(f"masks/{i}.png", data["mask_data"])
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def load_project(path):
    if not os.path.isfile(path):
        raise ValueError(f"file not found: {path}")

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member names")

        try:
            manifest_data = zf.read("manifest.json")
        except KeyError:
            raise ValueError("missing manifest.json")

        try:
            manifest = json.loads(manifest_data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"invalid manifest JSON: {e}")

        _validate_manifest(manifest)

        width = manifest["width"]
        height = manifest["height"]
        active_layer = manifest["active_layer"]
        layers_manifest = manifest["layers"]

        layers = []
        for i, layer_manifest in enumerate(layers_manifest):
            img = _read_png(zf, f"layers/{i}.png", (width, height), "RGBA")
            mask = None
            if layer_manifest["mask"] is not None:
                mask = _read_png(zf, f"masks/{i}.png", (width, height), "L")

            layer = Layer(
                name=layer_manifest["name"],
                image=img,
                opacity=layer_manifest["opacity"],
                blend_mode=layer_manifest["blend_mode"],
                visible=layer_manifest["visible"],
                mask=mask
            )
            layers.append(layer)

        canvas = Canvas(1, 1)
        canvas.width = width
        canvas.height = height
        canvas.layers = layers
        canvas.active_layer_index = active_layer
        canvas.undo_stack.clear()
        canvas.redo_stack.clear()
        canvas._save_state()

        return canvas
