"""
Layered document canvas with blend modes, masks, and undo/redo history.
Mirrors the mental model of a Photoshop document.
"""

import copy
import io
from typing import Optional, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from config import DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT, DEFAULT_BG_COLOR, MAX_UNDO_STEPS


# ----------------------------------------------------------------------- #
#  Blend mode implementations
# ----------------------------------------------------------------------- #

def _clamp(value: int) -> int:
    return max(0, min(255, int(value)))


def blend_normal(bottom: Image.Image, top: Image.Image, opacity: float = 1.0) -> Image.Image:
    return Image.blend(bottom, top, opacity)


def blend_multiply(bottom: Image.Image, top: Image.Image, opacity: float = 1.0) -> Image.Image:
    result = ImageChops.multiply(bottom, top)
    return Image.blend(bottom, result, opacity)


def blend_screen(bottom: Image.Image, top: Image.Image, opacity: float = 1.0) -> Image.Image:
    result = ImageChops.screen(bottom, top)
    return Image.blend(bottom, result, opacity)


def blend_overlay(bottom: Image.Image, top: Image.Image, opacity: float = 1.0) -> Image.Image:
    """Overlay: multiply if bottom < 128, screen if bottom >= 128."""
    b_arr = bottom.split()
    t_arr = top.split()
    channels = []
    for b_ch, t_ch in zip(b_arr, t_arr):
        b_data = b_ch.getdata()
        t_data = t_ch.getdata()
        out = []
        for b, t in zip(b_data, t_data):
            if b < 128:
                out.append(_clamp((b * t) / 255))
            else:
                out.append(_clamp(255 - ((255 - b) * (255 - t)) / 255))
        ch = b_ch.copy()
        ch.putdata(out)
        channels.append(ch)
    result = Image.merge(bottom.mode, channels)
    return Image.blend(bottom, result, opacity)


def blend_darken(bottom: Image.Image, top: Image.Image, opacity: float = 1.0) -> Image.Image:
    result = ImageChops.darker(bottom, top)
    return Image.blend(bottom, result, opacity)


def blend_lighten(bottom: Image.Image, top: Image.Image, opacity: float = 1.0) -> Image.Image:
    result = ImageChops.lighter(bottom, top)
    return Image.blend(bottom, result, opacity)


def blend_color_dodge(bottom: Image.Image, top: Image.Image, opacity: float = 1.0) -> Image.Image:
    b_arr = bottom.split()
    t_arr = top.split()
    channels = []
    for b_ch, t_ch in zip(b_arr, t_arr):
        b_data = b_ch.getdata()
        t_data = t_ch.getdata()
        out = []
        for b, t in zip(b_data, t_data):
            if t >= 255:
                out.append(255)
            else:
                out.append(_clamp((b * 255) / (255 - t)))
        ch = b_ch.copy()
        ch.putdata(out)
        channels.append(ch)
    result = Image.merge(bottom.mode, channels)
    return Image.blend(bottom, result, opacity)


def blend_color_burn(bottom: Image.Image, top: Image.Image, opacity: float = 1.0) -> Image.Image:
    b_arr = bottom.split()
    t_arr = top.split()
    channels = []
    for b_ch, t_ch in zip(b_arr, t_arr):
        b_data = b_ch.getdata()
        t_data = t_ch.getdata()
        out = []
        for b, t in zip(b_data, t_data):
            if b <= 0:
                out.append(0)
            else:
                out.append(_clamp(255 - ((255 - b) * 255) / t))
        ch = b_ch.copy()
        ch.putdata(out)
        channels.append(ch)
    result = Image.merge(bottom.mode, channels)
    return Image.blend(bottom, result, opacity)


def blend_hard_light(bottom: Image.Image, top: Image.Image, opacity: float = 1.0) -> Image.Image:
    """Same as overlay but swap roles of bottom and top."""
    return blend_overlay(top, bottom, opacity)


def blend_soft_light(bottom: Image.Image, top: Image.Image, opacity: float = 1.0) -> Image.Image:
    b_arr = bottom.split()
    t_arr = top.split()
    channels = []
    for b_ch, t_ch in zip(b_arr, t_arr):
        b_data = b_ch.getdata()
        t_data = t_ch.getdata()
        out = []
        for b, t in zip(b_data, t_data):
            dark = (b * t) / 255
            light = 255 - ((255 - b) * (255 - t)) / 255
            result = t * dark + b * light - dark * light
            out.append(_clamp(result))
        ch = b_ch.copy()
        ch.putdata(out)
        channels.append(ch)
    result = Image.merge(bottom.mode, channels)
    return Image.blend(bottom, result, opacity)


def blend_difference(bottom: Image.Image, top: Image.Image, opacity: float = 1.0) -> Image.Image:
    result = ImageChops.difference(bottom, top)
    return Image.blend(bottom, result, opacity)


def blend_exclusion(bottom: Image.Image, top: Image.Image, opacity: float = 1.0) -> Image.Image:
    b_arr = bottom.split()
    t_arr = top.split()
    channels = []
    for b_ch, t_ch in zip(b_arr, t_arr):
        b_data = b_ch.getdata()
        t_data = t_ch.getdata()
        out = []
        for b, t in zip(b_data, t_data):
            result = b + t - (2 * b * t) / 255
            out.append(_clamp(result))
        ch = b_ch.copy()
        ch.putdata(out)
        channels.append(ch)
    result = Image.merge(bottom.mode, channels)
    return Image.blend(bottom, result, opacity)


BLEND_MODES = {
    "normal": blend_normal,
    "multiply": blend_multiply,
    "screen": blend_screen,
    "overlay": blend_overlay,
    "darken": blend_darken,
    "lighten": blend_lighten,
    "color_dodge": blend_color_dodge,
    "color_burn": blend_color_burn,
    "hard_light": blend_hard_light,
    "soft_light": blend_soft_light,
    "difference": blend_difference,
    "exclusion": blend_exclusion,
}


# ----------------------------------------------------------------------- #
#  Layer
# ----------------------------------------------------------------------- #

class Layer:
    """A single layer in the document."""

    def __init__(self, name: str = "", image: Optional[Image.Image] = None,
                 opacity: float = 1.0, blend_mode: str = "normal",
                 visible: bool = True, mask: Optional[Image.Image] = None):
        self.name = name
        self.image = image or Image.new("RGBA", (DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT), (0, 0, 0, 0))
        self.opacity = opacity
        self.blend_mode = blend_mode
        self.visible = visible
        self.mask = mask  # Grayscale image: 255 = visible, 0 = hidden

    def to_dict(self) -> dict:
        """Serialize layer for undo history (compact form)."""
        buffer = io.BytesIO()
        self._get_effective_image().save(buffer, format="PNG")
        return {
            "name": self.name,
            "image_data": buffer.getvalue(),
            "opacity": self.opacity,
            "blend_mode": self.blend_mode,
            "visible": self.visible,
            "mask_data": self._save_mask() if self.mask else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Layer":
        """Deserialize layer from undo history."""
        layer = cls(name=data["name"])
        layer.image = Image.open(io.BytesIO(data["image_data"]))
        layer.opacity = data["opacity"]
        layer.blend_mode = data["blend_mode"]
        layer.visible = data["visible"]
        if data.get("mask_data"):
            layer.mask = Image.open(io.BytesIO(data["mask_data"]))
        return layer

    def _get_effective_image(self) -> Image.Image:
        """Get image with opacity applied."""
        img = self.image.copy()
        if self.opacity < 1.0:
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            r, g, b, a = img.split()
            a = a.point(lambda x: int(x * self.opacity))
            img = Image.merge("RGBA", (r, g, b, a))
        return img

    def _save_mask(self) -> bytes:
        buffer = io.BytesIO()
        self.mask.save(buffer, format="PNG")
        return buffer.getvalue()


# ----------------------------------------------------------------------- #
#  Canvas (Document)
# ----------------------------------------------------------------------- #

class Canvas:
    """
    Layered document with undo/redo history.
    The core "document" that all editing operations work against.
    """

    def __init__(self, width: int = DEFAULT_CANVAS_WIDTH, height: int = DEFAULT_CANVAS_HEIGHT,
                 bg_color: Tuple[int, ...] = DEFAULT_BG_COLOR):
        self.width = width
        self.height = height
        self.layers: list[Layer] = []
        self.active_layer_index: int = -1
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self._max_undo = MAX_UNDO_STEPS

        # Start with a background layer
        bg = Image.new("RGBA", (width, height), bg_color + (255,) if len(bg_color) == 3 else bg_color)
        self.layers.append(Layer(name="Background", image=bg))
        self.active_layer_index = 0
        self._save_state()  # Initial state

    # ------------------------------------------------------------------ #
    #  Undo / Redo
    # ------------------------------------------------------------------ #

    def _save_state(self):
        """Push current state to undo stack."""
        state = {
            "width": self.width,
            "height": self.height,
            "layers": [l.to_dict() for l in self.layers],
            "active_layer_index": self.active_layer_index,
        }
        self.undo_stack.append(state)
        if len(self.undo_stack) > self._max_undo:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self) -> bool:
        """Undo last operation. Returns True if successful."""
        if len(self.undo_stack) <= 1:
            return False  # Can't undo initial state
        self.redo_stack.append(self.undo_stack.pop())
        self._restore_state(self.undo_stack[-1])
        return True

    def redo(self) -> bool:
        """Redo last undone operation. Returns True if successful."""
        if not self.redo_stack:
            return False
        state = self.redo_stack.pop()
        self.undo_stack.append(state)
        self._restore_state(state)
        return True

    def _restore_state(self, state: dict):
        """Restore canvas from a saved state."""
        self.width = state["width"]
        self.height = state["height"]
        self.layers = [Layer.from_dict(d) for d in state["layers"]]
        self.active_layer_index = state["active_layer_index"]

    # ------------------------------------------------------------------ #
    #  Layer management
    # ------------------------------------------------------------------ #

    def add_layer(self, name: str = "", image: Optional[Image.Image] = None,
                  opacity: float = 1.0, blend_mode: str = "normal") -> int:
        """Add a new layer on top. Returns layer index."""
        if image is None:
            image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        elif image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.LANCZOS)
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        layer = Layer(name=name, image=image, opacity=opacity, blend_mode=blend_mode)
        self.layers.append(layer)
        self.active_layer_index = len(self.layers) - 1
        self._save_state()
        return self.active_layer_index

    def select_layer(self, index: int) -> bool:
        """Select a layer by index. Returns True if valid."""
        if 0 <= index < len(self.layers):
            self.active_layer_index = index
            return True
        return False

    def delete_layer(self, index: Optional[int] = None) -> bool:
        """Delete a layer. Uses active layer if index not specified."""
        idx = index if index is not None else self.active_layer_index
        if len(self.layers) <= 1:
            return False  # Don't delete last layer
        if 0 <= idx < len(self.layers):
            self.layers.pop(idx)
            if self.active_layer_index >= len(self.layers):
                self.active_layer_index = len(self.layers) - 1
            self._save_state()
            return True
        return False

    def merge_down(self) -> bool:
        """Merge active layer into the layer below it."""
        idx = self.active_layer_index
        if idx <= 0:
            return False
        bottom = self.layers[idx - 1]
        top = self.layers[idx]
        effective = self._apply_blend(bottom.image, top.image, top.blend_mode, top.opacity)
        if top.mask is not None:
            effective = self._apply_mask(effective, top.mask)
        bottom.image = effective
        self.layers.pop(idx)
        self.active_layer_index = idx - 1
        self._save_state()
        return True

    def merge_all(self) -> Image.Image:
        """Merge all layers into a single flattened image (doesn't modify layers)."""
        return self.composite()

    def reorder_layer(self, index: int, direction: str = "up") -> bool:
        """Move layer up or down in the stack."""
        if direction == "up" and index < len(self.layers) - 1:
            self.layers[index], self.layers[index + 1] = self.layers[index + 1], self.layers[index]
            self.active_layer_index = index + 1
            self._save_state()
            return True
        elif direction == "down" and index > 0:
            self.layers[index], self.layers[index - 1] = self.layers[index - 1], self.layers[index]
            self.active_layer_index = index - 1
            self._save_state()
            return True
        return False

    def set_layer_opacity(self, index: Optional[int] = None, opacity: float = 1.0) -> bool:
        idx = index if index is not None else self.active_layer_index
        if 0 <= idx < len(self.layers):
            self.layers[idx].opacity = max(0.0, min(1.0, opacity))
            self._save_state()
            return True
        return False

    def set_blend_mode(self, mode: str, index: Optional[int] = None) -> bool:
        if mode not in BLEND_MODES:
            return False
        idx = index if index is not None else self.active_layer_index
        if 0 <= idx < len(self.layers):
            self.layers[idx].blend_mode = mode
            self._save_state()
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Composite (flatten)
    # ------------------------------------------------------------------ #

    def _apply_blend(self, bottom: Image.Image, top: Image.Image,
                     blend_mode: str, opacity: float) -> Image.Image:
        """Apply a blend mode between two images."""
        mode = blend_mode.lower().replace(" ", "_")
        fn = BLEND_MODES.get(mode, blend_normal)
        return fn(bottom, top, opacity)

    def _apply_mask(self, img: Image.Image, mask: Image.Image) -> Image.Image:
        """Apply a grayscale mask to an image."""
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        if mask.mode != "L":
            mask = mask.convert("L")
        mask = mask.resize(img.size, Image.LANCZOS)
        r, g, b, a = img.split()
        new_a = Image.blend(a, Image.new("L", a.size, 0), 1.0)
        # Use mask as alpha multiplier
        a_data = a.getdata()
        m_data = mask.getdata()
        new_a_data = [int(a_val * m_val / 255) for a_val, m_val in zip(a_data, m_data)]
        new_a = mask.copy()
        new_a.putdata(new_a_data)
        return Image.merge("RGBA", (r, g, b, new_a))

    def composite(self) -> Image.Image:
        """Flatten all visible layers into a single RGBA image."""
        if not self.layers:
            return Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))

        result = self.layers[0].image.copy()
        if result.mode != "RGBA":
            result = result.convert("RGBA")

        for layer in self.layers[1:]:
            if not layer.visible:
                continue
            img = layer.image.copy()
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            # Apply mask
            if layer.mask is not None:
                img = self._apply_mask(img, layer.mask)

            # Apply blend mode
            result = self._apply_blend(result, img, layer.blend_mode, layer.opacity)

        return result

    def composite_rgb(self) -> Image.Image:
        """Flatten to RGB (for saving as JPG)."""
        return self.composite().convert("RGB")

    # ------------------------------------------------------------------ #
    #  Canvas operations
    # ------------------------------------------------------------------ #

    def resize_canvas(self, width: int, height: int, position: str = "top-left") -> None:
        """Resize the canvas and all layers."""
        old_w, old_h = self.width, self.height
        self.width, self.height = width, height

        for layer in self.layers:
            old_img = layer.image
            new_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            if position == "top-left":
                new_img.paste(old_img, (0, 0))
            elif position == "center":
                x = (width - old_w) // 2
                y = (height - old_h) // 2
                new_img.paste(old_img, (x, y))
            elif position == "bottom-right":
                x = width - old_w
                y = height - old_h
                new_img.paste(old_img, (x, y))
            else:
                new_img.paste(old_img, (0, 0))
            layer.image = new_img

            if layer.mask:
                layer.mask = layer.mask.resize((width, height), Image.LANCZOS)

        self._save_state()

    # ------------------------------------------------------------------ #
    #  Info
    # ------------------------------------------------------------------ #

    def get_info(self) -> dict:
        """Get document info."""
        return {
            "width": self.width,
            "height": self.height,
            "layer_count": len(self.layers),
            "active_layer": self.active_layer_index,
            "layers": [
                {
                    "index": i,
                    "name": l.name,
                    "visible": l.visible,
                    "opacity": l.opacity,
                    "blend_mode": l.blend_mode,
                    "has_mask": l.mask is not None,
                }
                for i, l in enumerate(self.layers)
            ],
            "undo_available": len(self.undo_stack) > 1,
            "redo_available": len(self.redo_stack) > 0,
        }