import unittest
from PIL import Image, ImageChops

from canvas import BLEND_MODES, Canvas


def solid(color, size=(64, 64)):
    return Image.new("RGBA", size, color)


def square_layer(color, box=(16, 16, 48, 48), size=(64, 64)):
    """A transparent layer with an opaque square - the shape of a masked edit layer."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    x0, y0, x1, y1 = box
    layer.paste(solid(color, (x1 - x0, y1 - y0)), (x0, y0))
    return layer


def outside_box(x, y, box=(16, 16, 48, 48)):
    x0, y0, x1, y1 = box
    return not (x0 <= x < x1 and y0 <= y < y1)


class BlendModeAlphaTests(unittest.TestCase):
    """F2 regression: non-normal blend modes must respect the top layer's per-pixel alpha."""

    def test_transparent_top_is_noop_for_every_mode(self):
        bottom = solid((100, 150, 200, 255))
        top = solid((200, 50, 50, 0))
        for name, fn in BLEND_MODES.items():
            self.assertEqual(fn(bottom, top, 1.0).tobytes(), bottom.tobytes(), name)

    def test_partial_alpha_leaves_outside_untouched_for_every_mode(self):
        bottom = solid((100, 150, 200, 255))
        top = square_layer((200, 50, 50, 255))
        for name, fn in BLEND_MODES.items():
            out = fn(bottom, top, 1.0)
            for y in range(64):
                for x in range(64):
                    if outside_box(x, y):
                        self.assertEqual(
                            out.getpixel((x, y)), bottom.getpixel((x, y)),
                            f"{name} at ({x}, {y})")

    def test_opacity_zero_is_noop_for_every_mode(self):
        bottom = solid((100, 150, 200, 255))
        top = square_layer((200, 50, 50, 255))
        for name, fn in BLEND_MODES.items():
            self.assertEqual(fn(bottom, top, 0.0).tobytes(), bottom.tobytes(), name)

    def test_opaque_top_matches_legacy_channel_ops(self):
        bottom = solid((100, 150, 200, 255))
        top = solid((200, 50, 50, 255))
        oracles = {
            "multiply": ImageChops.multiply,
            "screen": ImageChops.screen,
            "darken": ImageChops.darker,
            "lighten": ImageChops.lighter,
            "difference": ImageChops.difference,
        }
        for name, op in oracles.items():
            expected = op(bottom.convert("RGB"), top.convert("RGB"))
            self.assertEqual(
                BLEND_MODES[name](bottom, top, 1.0).convert("RGB").tobytes(),
                expected.tobytes(), name)
            self.assertEqual(BLEND_MODES[name](bottom, top, 1.0).getchannel("A").getextrema(), (255, 255), name)

    def test_opaque_top_matches_loop_mode_formulas(self):
        bottom = solid((100, 150, 200, 255))
        top = solid((200, 50, 50, 255))
        expected = {
            "overlay": (78, 170, 210, 255),
            "color_dodge": (255, 186, 248, 255),
            "color_burn": (57, 0, 0, 255),
            "soft_light": (255, 255, 255, 255),
            "exclusion": (143, 141, 171, 255),
            "hard_light": (221, 29, 39, 255),
        }
        for name, pixel in expected.items():
            self.assertEqual(BLEND_MODES[name](bottom, top, 1.0).getpixel((0, 0)), pixel, name)

    def test_opacity_folds_top_alpha_and_keeps_outside_bit_exact(self):
        bottom = solid((100, 150, 200, 255))
        top = square_layer((200, 50, 50, 255))
        out = BLEND_MODES["multiply"](bottom, top, 0.5)
        # Outside the square: unchanged, alpha intact.
        for y in range(64):
            for x in range(64):
                if outside_box(x, y):
                    self.assertEqual(out.getpixel((x, y)), bottom.getpixel((x, y)))
        # Inside: strictly between the bottom color and the full multiply result.
        full = ImageChops.multiply(solid((100, 150, 200)).convert("RGB"),
                                   solid((200, 50, 50)).convert("RGB")).getpixel((0, 0))
        inside = out.getpixel((32, 32))
        self.assertEqual(inside[3], 255)
        for c in range(3):
            lo, hi = sorted((bottom.getpixel((32, 32))[c], full[c]))
            self.assertGreaterEqual(inside[c], lo)
            self.assertLessEqual(inside[c], hi)


class BlendModeCanvasTests(unittest.TestCase):
    def test_masked_edit_layer_multiply_leaves_outside_bit_identical(self):
        """The exact T5.7 failure: a masked edit layer set to multiply used to
        alter 99.99% of the canvas and zero its composite alpha."""
        canvas = Canvas(64, 64, (10, 20, 30))
        canvas.add_layer("edit", square_layer((200, 50, 50, 255)))
        normal = canvas.composite()
        canvas.set_blend_mode("multiply", 1)
        multiplied = canvas.composite()
        for y in range(64):
            for x in range(64):
                if outside_box(x, y):
                    self.assertEqual(
                        multiplied.getpixel((x, y)), normal.getpixel((x, y)),
                        f"outside at ({x}, {y})")
        expected_inside = ImageChops.multiply(
            solid((10, 20, 30)).convert("RGB"), solid((200, 50, 50)).convert("RGB")
        ).getpixel((0, 0))
        self.assertEqual(multiplied.getpixel((32, 32)), expected_inside + (255,))

    def test_merge_down_equals_composite_with_masked_blend_layer(self):
        canvas = Canvas(64, 64, (10, 20, 30))
        canvas.add_layer("edit", square_layer((200, 50, 50, 255)))
        canvas.set_blend_mode("multiply", 1)
        expected = canvas.composite()
        canvas.merge_down()
        self.assertEqual(len(canvas.layers), 1)
        self.assertEqual(canvas.composite().tobytes(), expected.tobytes())
        self.assertEqual(canvas.composite().getpixel((0, 0)), (10, 20, 30, 255))


if __name__ == "__main__":
    unittest.main()
