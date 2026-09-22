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

    def test_merge_down_opaque_top_over_semi_transparent_bottom(self):
        canvas = Canvas(4, 2, (0, 0, 0))
        bottom_img = Image.new("RGBA", (4, 2), (255, 0, 0, 255))
        top_img = Image.new("RGBA", (4, 2), (0, 0, 255, 255))
        canvas.add_layer("lower", bottom_img)
        canvas.layers[1].opacity = 0.5
        canvas.add_layer("top", top_img)
        before_composite = canvas.composite()
        before_snapshot = [l.to_dict() for l in canvas.layers]
        before_undo_len = len(canvas.undo_stack)
        canvas.merge_down()
        after_composite = canvas.composite()
        self.assertEqual(after_composite.tobytes(), before_composite.tobytes())
        self.assertEqual(len(canvas.layers), 2)
        self.assertEqual(canvas.active_layer_index, 1)
        self.assertEqual(len(canvas.undo_stack), before_undo_len + 1)
        self.assertEqual(canvas.layers[0].to_dict(), before_snapshot[0])
        merged = canvas.layers[1]
        self.assertEqual(merged.name, "lower")
        self.assertEqual(merged.opacity, 1.0)
        self.assertIsNone(merged.mask)
        self.assertEqual(merged.blend_mode, "normal")
        self.assertTrue(merged.visible)
        for y in range(2):
            for x in range(4):
                self.assertEqual(after_composite.getpixel((x, y))[3], 255)

    def test_merge_down_nonuniform_masks_and_opacity_history(self):
        canvas = Canvas(4, 2, (0, 0, 0))
        bottom_img = Image.new("RGBA", (4, 2), (255, 0, 0, 255))
        bottom_mask = Image.new("L", (4, 2), 128)
        bottom_mask.putdata([255, 0, 128, 64, 32, 96, 16, 240])
        top_img = Image.new("RGBA", (4, 2), (0, 255, 0, 255))
        top_mask = Image.new("L", (4, 2), 255)
        top_mask.putdata([0, 255, 128, 64, 32, 96, 16, 240])
        canvas.layers[0].image = bottom_img
        canvas.layers[0].mask = bottom_mask
        canvas.layers[0].opacity = 0.7
        canvas.add_layer("top", top_img)
        canvas.layers[1].mask = top_mask
        canvas.layers[1].opacity = 0.8
        canvas._save_state()
        before_composite = canvas.composite()
        before_layers = [l.to_dict() for l in canvas.layers]
        before_undo_len = len(canvas.undo_stack)
        canvas.merge_down()
        after_composite = canvas.composite()
        self.assertEqual(after_composite.tobytes(), before_composite.tobytes())
        self.assertEqual(len(canvas.layers), 1)
        self.assertEqual(canvas.active_layer_index, 0)
        self.assertEqual(len(canvas.undo_stack), before_undo_len + 1)
        self.assertTrue(canvas.undo())
        restored_composite = canvas.composite()
        self.assertEqual(restored_composite.tobytes(), before_composite.tobytes())
        self.assertEqual(len(canvas.layers), 2)
        self.assertEqual(canvas.active_layer_index, 1)
        self.assertEqual([l.to_dict() for l in canvas.layers], before_layers)
        self.assertTrue(canvas.redo())
        redo_composite = canvas.composite()
        self.assertEqual(redo_composite.tobytes(), before_composite.tobytes())
        self.assertEqual(len(canvas.layers), 1)
        self.assertEqual(canvas.active_layer_index, 0)

    def test_merge_down_hidden_layers(self):
        backdrop_color = (100, 100, 100)
        cases = [
            ("top-hidden", True, False, "normal"),
            ("bottom-hidden", False, True, "multiply"),
            ("both-hidden", False, False, "multiply"),
        ]
        for name, bottom_visible, top_visible, top_mode in cases:
            with self.subTest(name=name):
                canvas = Canvas(4, 2, backdrop_color)
                bottom_img = Image.new("RGBA", (4, 2), (255, 0, 0, 255))
                top_img = Image.new("RGBA", (4, 2), (0, 0, 255, 255))
                canvas.add_layer("lower", bottom_img)
                canvas.layers[1].visible = bottom_visible
                canvas.add_layer("top", top_img)
                canvas.layers[2].visible = top_visible
                canvas.set_blend_mode(top_mode, 2)
                canvas.layers[2].opacity = 0.6
                canvas.layers[2].mask = Image.new("L", (4, 2), 200)
                canvas.layers[2].mask.putpixel((0, 0), 0)
                before_composite = canvas.composite()
                before_lower = canvas.layers[1].to_dict()
                before_top = canvas.layers[2].to_dict()
                before_deeper = canvas.layers[0].to_dict()
                canvas.select_layer(2)
                canvas.merge_down()
                after_composite = canvas.composite()
                self.assertEqual(after_composite.tobytes(), before_composite.tobytes())
                self.assertEqual(len(canvas.layers), 2)
                self.assertEqual(canvas.layers[0].to_dict(), before_deeper)
                self.assertEqual(canvas.layers[1].name, "lower")
                if not top_visible:
                    self.assertEqual(canvas.layers[1].to_dict(), before_lower)
                else:
                    before_top["name"] = "lower"
                    self.assertEqual(canvas.layers[1].to_dict(), before_top)

    def test_merge_down_partial_alpha_tolerance(self):
        canvas = Canvas(4, 2, (50, 50, 50))
        bottom_img = Image.new("RGBA", (4, 2), (200, 100, 50, 255))
        bottom_mask = Image.new("L", (4, 2), 255)
        bottom_mask.putdata([255, 128, 64, 32, 16, 8, 4, 2])
        top_img = Image.new("RGBA", (4, 2), (10, 20, 30, 255))
        top_mask = Image.new("L", (4, 2), 255)
        top_mask.putdata([255, 200, 150, 100, 50, 25, 10, 5])
        canvas.add_layer("lower", bottom_img)
        canvas.layers[1].mask = bottom_mask
        canvas.layers[1].opacity = 0.9
        canvas.add_layer("top", top_img)
        canvas.layers[2].mask = top_mask
        canvas.layers[2].opacity = 0.8
        before_composite = canvas.composite()
        canvas.merge_down()
        after_composite = canvas.composite()
        for y in range(2):
            for x in range(4):
                b_px = before_composite.getpixel((x, y))
                a_px = after_composite.getpixel((x, y))
                for c in range(4):
                    self.assertLessEqual(abs(b_px[c] - a_px[c]), 1, f"channel {c} at ({x}, {y})")

    def test_merge_down_multiply_with_fullwhite_mask(self):
        canvas = Canvas(4, 2, (100, 100, 100))
        bottom_img = Image.new("RGBA", (4, 2), (255, 255, 255, 255))
        bottom_mask = Image.new("L", (4, 2), 255)
        top_img = Image.new("RGBA", (4, 2), (128, 128, 128, 255))
        canvas.add_layer("lower", bottom_img)
        canvas.layers[1].mask = bottom_mask
        canvas.add_layer("top", top_img)
        canvas.set_blend_mode("multiply", 2)
        before_composite = canvas.composite()
        canvas.merge_down()
        after_composite = canvas.composite()
        self.assertEqual(after_composite.tobytes(), before_composite.tobytes())

    def test_merge_down_multiply_with_transparent_bottom_raises(self):
        canvas = Canvas(4, 2, (100, 100, 100))
        bottom_img = Image.new("RGBA", (4, 2), (255, 255, 255, 0))
        top_img = Image.new("RGBA", (4, 2), (128, 128, 128, 255))
        canvas.add_layer("lower", bottom_img)
        canvas.add_layer("top", top_img)
        canvas.set_blend_mode("multiply", 2)
        before_layers = [l.to_dict() for l in canvas.layers]
        before_active = canvas.active_layer_index
        before_undo_len = len(canvas.undo_stack)
        with self.assertRaises(ValueError):
            canvas.merge_down()
        self.assertEqual([l.to_dict() for l in canvas.layers], before_layers)
        self.assertEqual(canvas.active_layer_index, before_active)
        self.assertEqual(len(canvas.undo_stack), before_undo_len)

    def test_merge_down_bottom_non_normal_above_backdrop_raises(self):
        canvas = Canvas(4, 2, (100, 100, 100))
        bottom_img = Image.new("RGBA", (4, 2), (255, 255, 255, 255))
        top_img = Image.new("RGBA", (4, 2), (128, 128, 128, 255))
        canvas.add_layer("lower", bottom_img)
        canvas.set_blend_mode("screen", 1)
        canvas.add_layer("top", top_img)
        before_layers = [l.to_dict() for l in canvas.layers]
        before_active = canvas.active_layer_index
        before_undo_len = len(canvas.undo_stack)
        with self.assertRaises(ValueError):
            canvas.merge_down()
        self.assertEqual([l.to_dict() for l in canvas.layers], before_layers)
        self.assertEqual(canvas.active_layer_index, before_active)
        self.assertEqual(len(canvas.undo_stack), before_undo_len)


if __name__ == "__main__":
    unittest.main()
