import json
import unittest
from unittest.mock import AsyncMock

from PIL import Image
from session import SessionManager


class SessionManagerTests(unittest.TestCase):
    def setUp(self):
        self.sessions = SessionManager()

    def test_list_sessions_empty_until_used(self):
        self.assertEqual(self.sessions.list_sessions(), {})
        self.sessions.get_default_session()
        self.assertIn("default", self.sessions.list_sessions())

    def test_sessions_are_isolated_documents(self):
        a = self.sessions.create("a", 64, 64, (1, 2, 3))
        b = self.sessions.create("b", 32, 32, (9, 9, 9))
        a.add_layer(image=None)
        self.assertEqual((a.width, a.height), (64, 64))
        self.assertEqual((b.width, b.height), (32, 32))
        self.assertEqual(len(a.layers), 2)
        self.assertEqual(len(b.layers), 1)

    def test_get_or_create_reuses_existing(self):
        first = self.sessions.get_or_create("x")
        second = self.sessions.get_or_create("x")
        self.assertIs(first, second)
        self.assertEqual(set(self.sessions.list_sessions()), {"x"})

    def test_close_session(self):
        self.sessions.create("temp", 16, 16)
        self.assertTrue(self.sessions.delete("temp"))
        self.assertFalse(self.sessions.delete("temp"))


class SessionToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import server
        self.server = server
        for sid in list(server.sessions.list_sessions()):
            server.sessions.delete(sid)

    async def test_list_and_close_session_tools(self):
        await self.server.new_canvas(width=64, height=64, session_id="photo")
        report = json.loads((await self.server.list_sessions_tool())[0].text)
        self.assertEqual(report["photo"]["size"], [64, 64])
        self.assertEqual(report["photo"]["layers"], 1)
        out = await self.server.close_session_tool(session_id="photo")
        self.assertIn("closed", out[0].text)
        report = json.loads((await self.server.list_sessions_tool())[0].text)
        self.assertNotIn("photo", report)
        out = await self.server.close_session_tool(session_id="photo")
        self.assertIn("No open session", out[0].text)

    async def test_canvas_tools_address_session_by_id(self):
        await self.server.new_canvas(width=64, height=64, session_id="s1")
        await self.server.new_canvas(width=32, height=32, session_id="s2")
        await self.server.adjust_tool(brightness=0.5, session_id="s1")
        info_s1 = json.loads((await self.server.get_info(session_id="s1"))[0].text)
        info_s2 = json.loads((await self.server.get_info(session_id="s2"))[0].text)
        self.assertEqual(info_s1["width"], 64)
        self.assertEqual(info_s2["width"], 32)
        self.assertTrue(info_s1["undo_available"])    # adjust pushed a state
        self.assertFalse(info_s2["undo_available"])   # untouched document

    async def test_preview_and_editing_tools_address_session_by_id(self):
        from editing import register_editing_tools

        class Registry:
            def __init__(self):
                self.tools = {}

            def tool(self, name):
                def register(function):
                    self.tools[name] = function
                    return function
                return register

        registry = Registry()
        run = AsyncMock(return_value=None)
        register_editing_tools(registry, AsyncMock(), self.server.sessions, run)
        await self.server.new_canvas(width=48, height=48, session_id="edit")
        out = await registry.tools["preview_canvas"](session_id="edit")
        self.assertEqual(json.loads(out[0].text)["width"], 48)
        out = await registry.tools["preview_canvas"]()  # default session, independent
        self.assertNotEqual(json.loads(out[0].text)["width"], 48)

    async def test_crop_right_half_of_left_masked_layer(self):
        await self.server.new_canvas(width=8, height=4, session_id="crop_test")
        canvas = self.server.sessions.get_or_create("crop_test")
        img = Image.new("RGBA", (8, 4), (255, 0, 0, 255))
        mask = Image.new("L", (8, 4), 0)
        mask.paste(255, (0, 0, 4, 4))
        canvas.layers[0].image = img
        canvas.layers[0].mask = mask
        canvas._save_state()
        out = await self.server.crop_tool(x=4, y=0, width=4, height=4, session_id="crop_test")
        self.assertIn("Cropped to 4x4 at (4,0)", out[0].text)
        self.assertEqual(canvas.width, 4)
        self.assertEqual(canvas.height, 4)
        self.assertEqual(canvas.layers[0].image.size, (4, 4))
        self.assertEqual(canvas.layers[0].mask.size, (4, 4))
        composite = canvas.composite()
        self.assertIsNone(composite.getchannel("A").getbbox())

    async def test_resize_two_layers_preserves_properties_and_history(self):
        await self.server.new_canvas(width=8, height=4, session_id="resize_test")
        canvas = self.server.sessions.get_or_create("resize_test")
        canvas.add_layer(image=None)
        canvas.layers[0].name = "Layer1"
        canvas.layers[0].opacity = 0.8
        canvas.layers[0].visible = True
        canvas.layers[0].mask = Image.new("L", (8, 4), 128)
        canvas.layers[0].mask.paste(255, (0, 0, 4, 2))
        canvas.layers[1].name = "Layer2"
        canvas.layers[1].opacity = 0.6
        canvas.layers[1].visible = False
        canvas.layers[1].mask = Image.new("L", (8, 4), 64)
        canvas.layers[1].mask.paste(255, (4, 2, 8, 4))
        canvas.active_layer_index = 1
        canvas._save_state()
        baseline_undo_count = len(canvas.undo_stack)
        source_pixels = [(layer.image.tobytes(), layer.mask.tobytes()) for layer in canvas.layers]
        out = await self.server.resize_tool(width=16, height=8, session_id="resize_test")
        self.assertIn("Resized to 16x8", out[0].text)
        self.assertEqual(len(canvas.undo_stack), baseline_undo_count + 1)
        self.assertEqual(canvas.width, 16)
        self.assertEqual(canvas.height, 8)
        self.assertEqual(canvas.layers[0].image.size, (16, 8))
        self.assertEqual(canvas.layers[0].mask.size, (16, 8))
        self.assertEqual(canvas.layers[1].image.size, (16, 8))
        self.assertEqual(canvas.layers[1].mask.size, (16, 8))
        resized_pixels = [(layer.image.tobytes(), layer.mask.tobytes()) for layer in canvas.layers]
        self.assertEqual(canvas.layers[0].name, "Layer1")
        self.assertEqual(canvas.layers[0].opacity, 0.8)
        self.assertTrue(canvas.layers[0].visible)
        self.assertEqual(canvas.layers[1].name, "Layer2")
        self.assertEqual(canvas.layers[1].opacity, 0.6)
        self.assertFalse(canvas.layers[1].visible)
        self.assertEqual(canvas.active_layer_index, 1)
        self.assertTrue(canvas.undo())
        self.assertEqual(canvas.width, 8)
        self.assertEqual(canvas.height, 4)
        self.assertEqual([(layer.image.tobytes(), layer.mask.tobytes()) for layer in canvas.layers], source_pixels)
        self.assertEqual(canvas.active_layer_index, 1)
        self.assertTrue(canvas.redo())
        self.assertEqual([(layer.image.tobytes(), layer.mask.tobytes()) for layer in canvas.layers], resized_pixels)
        self.assertEqual(canvas.width, 16)
        self.assertEqual(canvas.height, 8)
        self.assertEqual(canvas.layers[0].image.size, (16, 8))
        self.assertEqual(canvas.layers[0].mask.size, (16, 8))

    async def test_maintain_aspect_upscale_centers_content(self):
        await self.server.new_canvas(width=8, height=4, session_id="aspect_test")
        canvas = self.server.sessions.get_or_create("aspect_test")
        img = Image.new("RGBA", (8, 4), (0, 0, 0, 0))
        img.paste((255, 0, 0, 255), (0, 0, 8, 4))
        mask = Image.new("L", (8, 4), 0)
        mask.paste(255, (0, 0, 8, 4))
        canvas.layers[0].image = img
        canvas.layers[0].mask = mask
        canvas._save_state()
        out = await self.server.resize_tool(width=16, height=16, maintain_aspect=True, session_id="aspect_test")
        self.assertIn("Resized to 16x16", out[0].text)
        self.assertEqual(canvas.width, 16)
        self.assertEqual(canvas.height, 16)
        self.assertEqual(canvas.layers[0].image.size, (16, 16))
        self.assertEqual(canvas.layers[0].mask.size, (16, 16))
        top_padding = canvas.layers[0].image.crop((0, 0, 16, 4))
        bottom_padding = canvas.layers[0].image.crop((0, 12, 16, 16))
        self.assertIsNone(top_padding.getchannel("A").getbbox())
        self.assertIsNone(bottom_padding.getchannel("A").getbbox())
        top_mask_padding = canvas.layers[0].mask.crop((0, 0, 16, 4))
        bottom_mask_padding = canvas.layers[0].mask.crop((0, 12, 16, 16))
        self.assertIsNone(top_mask_padding.getbbox())
        self.assertIsNone(bottom_mask_padding.getbbox())
        content_area = canvas.layers[0].image.crop((0, 4, 16, 12))
        self.assertIsNotNone(content_area.getchannel("A").getbbox())

    async def test_rotate_90deg_expand_centers_active_layer(self):
        await self.server.new_canvas(width=8, height=4, session_id="rotate_test")
        canvas = self.server.sessions.get_or_create("rotate_test")
        canvas.add_layer(image=None)
        img1 = Image.new("RGBA", (8, 4), (255, 0, 0, 255))
        mask1 = Image.new("L", (8, 4), 255)
        img2 = Image.new("RGBA", (8, 4), (0, 0, 0, 0))
        img2.paste((0, 255, 0, 255), (0, 0, 4, 2))
        mask2 = Image.new("L", (8, 4), 0)
        mask2.paste(255, (0, 0, 4, 2))
        canvas.layers[0].image = img1
        canvas.layers[0].mask = mask1
        canvas.layers[1].image = img2
        canvas.layers[1].mask = mask2
        canvas.active_layer_index = 0
        canvas._save_state()
        out = await self.server.rotate_tool(degrees=90, expand=True, session_id="rotate_test")
        self.assertIn("Rotated 90 degrees", out[0].text)
        self.assertEqual(canvas.width, 8)
        self.assertEqual(canvas.height, 8)
        self.assertEqual(canvas.layers[0].image.size, (8, 8))
        self.assertEqual(canvas.layers[0].mask.size, (8, 8))
        self.assertEqual(canvas.layers[1].image.size, (8, 8))
        self.assertEqual(canvas.layers[1].mask.size, (8, 8))
        other_center_crop = canvas.layers[1].image.crop((0, 2, 8, 6))
        expected_other = Image.new("RGBA", (8, 4), (0, 0, 0, 0))
        expected_other.paste((0, 255, 0, 255), (0, 0, 4, 2))
        self.assertEqual(other_center_crop.tobytes(), expected_other.tobytes())
        other_mask_crop = canvas.layers[1].mask.crop((0, 2, 8, 6))
        expected_mask = Image.new("L", (8, 4), 0)
        expected_mask.paste(255, (0, 0, 4, 2))
        self.assertEqual(other_mask_crop.tobytes(), expected_mask.tobytes())
        self.assertIsNotNone(canvas.composite())

    async def test_rotate_30deg_no_expand_keeps_dimensions(self):
        await self.server.new_canvas(width=8, height=4, session_id="rotate_noexpand_test")
        canvas = self.server.sessions.get_or_create("rotate_noexpand_test")
        img = Image.new("RGBA", (8, 4), (0, 0, 0, 0))
        img.paste((255, 255, 255, 255), (2, 1, 6, 3))
        mask = img.getchannel("A").copy()
        canvas.layers[0].image = img
        canvas.layers[0].mask = mask
        canvas._save_state()
        out = await self.server.rotate_tool(degrees=30, expand=False, session_id="rotate_noexpand_test")
        self.assertIn("Rotated 30 degrees", out[0].text)
        self.assertEqual(canvas.width, 8)
        self.assertEqual(canvas.height, 4)
        self.assertEqual(canvas.layers[0].image.size, (8, 4))
        self.assertEqual(canvas.layers[0].mask.size, (8, 4))
        rotated_alpha = canvas.layers[0].image.getchannel("A")
        rotated_mask = canvas.layers[0].mask
        self.assertEqual(rotated_alpha.tobytes(), rotated_mask.tobytes())

    async def test_flip_moves_image_and_mask_together(self):
        await self.server.new_canvas(width=8, height=4, session_id="flip_test")
        canvas = self.server.sessions.get_or_create("flip_test")
        img = Image.new("RGBA", (8, 4), (0, 0, 0, 0))
        img.putpixel((0, 0), (255, 0, 0, 255))
        img.putpixel((7, 3), (0, 0, 255, 255))
        mask = Image.new("L", (8, 4), 0)
        mask.putpixel((0, 0), 255)
        mask.putpixel((7, 3), 255)
        canvas.layers[0].image = img
        canvas.layers[0].mask = mask
        canvas._save_state()
        out = await self.server.flip_tool(axis="horizontal", session_id="flip_test")
        self.assertIn("Flipped", out[0].text)
        layer = canvas.layers[0]
        self.assertEqual(layer.image.getpixel((7, 0))[0], 255)
        self.assertEqual(layer.mask.getpixel((7, 0)), 255)
        self.assertEqual(layer.image.getpixel((0, 3))[2], 255)
        self.assertEqual(layer.mask.getpixel((0, 3)), 255)
        canvas.undo()
        out = await self.server.flip_tool(axis="vertical", session_id="flip_test")
        self.assertIn("Flipped", out[0].text)
        layer = canvas.layers[0]
        self.assertEqual(layer.image.getpixel((0, 3))[0], 255)
        self.assertEqual(layer.mask.getpixel((0, 3)), 255)
        self.assertEqual(layer.image.getpixel((7, 0))[2], 255)
        self.assertEqual(layer.mask.getpixel((7, 0)), 255)

    async def test_invalid_zero_crop_returns_error(self):
        await self.server.new_canvas(width=8, height=4, session_id="invalid_test")
        canvas = self.server.sessions.get_or_create("invalid_test")
        initial_width = canvas.width
        initial_height = canvas.height
        initial_undo_count = len(canvas.undo_stack)
        out = await self.server.crop_tool(x=0, y=0, width=0, height=4, session_id="invalid_test")
        self.assertIn("Crop error", out[0].text)
        self.assertEqual(canvas.width, initial_width)
        self.assertEqual(canvas.height, initial_height)
        self.assertEqual(len(canvas.undo_stack), initial_undo_count)
        out = await self.server.resize_tool(width=8, height=0, session_id="invalid_test")
        self.assertIn("Resize error", out[0].text)
        self.assertEqual(canvas.width, initial_width)
        self.assertEqual(canvas.height, initial_height)
        self.assertEqual(len(canvas.undo_stack), initial_undo_count)

    async def test_apply_filter_schema(self):
        tools = await self.server.app.list_tools()
        tool = next(t for t in tools if t.name == "apply_filter")
        self.assertTrue(tool.inputSchema["required"])
        self.assertIn("name", tool.inputSchema["required"])
        self.assertNotIn("params", tool.inputSchema.get("properties", {}))
        for param in ("radius", "size", "levels"):
            self.assertIn(param, tool.inputSchema.get("properties", {}))
            self.assertNotIn(param, tool.inputSchema["required"])

    async def test_apply_filter_invert_preserves_alpha(self):
        await self.server.new_canvas(width=4, height=1, session_id="alpha_test")
        canvas = self.server.sessions.get_or_create("alpha_test")
        img = Image.new("RGBA", (4, 1), (255, 0, 0, 255))
        img.putpixel((0, 0), (255, 0, 0, 0))
        img.putpixel((1, 0), (0, 255, 0, 40))
        img.putpixel((2, 0), (0, 0, 255, 128))
        img.putpixel((3, 0), (128, 128, 128, 255))
        canvas.layers[0].image = img
        canvas._save_state()
        out = await self.server.app.call_tool("apply_filter", {"name": "invert", "session_id": "alpha_test"})
        self.assertNotIn("error", out[0].text.lower())
        result = canvas.layers[0].image
        self.assertEqual(result.getchannel("A").tobytes(), bytes([0, 40, 128, 255]))
        self.assertEqual(result.getpixel((0, 0))[0], 0)
        self.assertEqual(result.getpixel((1, 0))[1], 0)
        self.assertEqual(result.getpixel((2, 0))[2], 0)

    async def test_color_filters_preserve_nonuniform_alpha(self):
        await self.server.new_canvas(width=4, height=1, session_id="color_alpha_test")
        canvas = self.server.sessions.get_or_create("color_alpha_test")
        source_img = Image.new("RGBA", (4, 1), (255, 0, 0, 255))
        source_img.putpixel((0, 0), (255, 0, 0, 0))
        source_img.putpixel((1, 0), (0, 255, 0, 40))
        source_img.putpixel((2, 0), (0, 0, 255, 128))
        source_img.putpixel((3, 0), (128, 128, 128, 255))
        source_bytes = source_img.tobytes()
        expected_alpha = bytes([0, 40, 128, 255])
        for name in ("invert", "grayscale", "posterize", "solarize"):
            canvas.layers[0].image = Image.frombytes("RGBA", (4, 1), source_bytes)
            canvas._save_state()
            out = await self.server.app.call_tool("apply_filter", {"name": name, "session_id": "color_alpha_test"})
            self.assertNotIn("error", out[0].text.lower())
            self.assertEqual(canvas.layers[0].image.getchannel("A").tobytes(), expected_alpha)
        canvas.layers[0].image = Image.frombytes("RGBA", (4, 1), source_bytes)
        canvas._save_state()
        out = await self.server.app.call_tool("adjust", {"hue": 120, "session_id": "color_alpha_test"})
        self.assertNotIn("error", out[0].text.lower())
        self.assertEqual(canvas.layers[0].image.getchannel("A").tobytes(), expected_alpha)

    async def test_hue_rotation_preserves_pixels_and_colors(self):
        await self.server.new_canvas(width=1, height=1, session_id="hue_test")
        canvas = self.server.sessions.get_or_create("hue_test")
        source_img = Image.new("RGBA", (1, 1), (255, 0, 0, 255))
        source_bytes = source_img.tobytes()
        for angle in (360, 720):
            canvas.layers[0].image = Image.frombytes("RGBA", (1, 1), source_bytes)
            canvas._save_state()
            out = await self.server.app.call_tool("adjust", {"hue": angle, "session_id": "hue_test"})
            self.assertNotIn("error", out[0].text.lower())
            self.assertEqual(canvas.layers[0].image.tobytes(), source_bytes)
        canvas.layers[0].image = Image.frombytes("RGBA", (1, 1), source_bytes)
        canvas._save_state()
        out = await self.server.app.call_tool("adjust", {"hue": 120, "session_id": "hue_test"})
        self.assertNotIn("error", out[0].text.lower())
        r, g, b, a = canvas.layers[0].image.getpixel((0, 0))
        self.assertLess(abs(r - 0), 2)
        self.assertEqual(g, 255)
        self.assertLess(abs(b - 0), 2)
        self.assertEqual(a, 255)
        canvas.layers[0].image = Image.frombytes("RGBA", (1, 1), source_bytes)
        canvas._save_state()
        out = await self.server.app.call_tool("adjust", {"hue": 480, "session_id": "hue_test"})
        self.assertNotIn("error", out[0].text.lower())
        r2, g2, b2, a2 = canvas.layers[0].image.getpixel((0, 0))
        self.assertLess(abs(r - r2), 2)
        self.assertLess(abs(g - g2), 2)
        self.assertLess(abs(b - b2), 2)
        self.assertEqual(a2, 255)

    async def test_gaussian_blur_radius_changes_output(self):
        await self.server.new_canvas(width=4, height=4, session_id="blur_test")
        canvas = self.server.sessions.get_or_create("blur_test")
        source_img = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
        source_img.putpixel((0, 0), (255, 255, 255, 255))
        source_bytes = source_img.tobytes()
        canvas.layers[0].image = Image.frombytes("RGBA", (4, 4), source_bytes)
        canvas._save_state()
        default_out = await self.server.app.call_tool("apply_filter", {"name": "gaussian_blur", "session_id": "blur_test"})
        self.assertNotIn("error", default_out[0].text.lower())
        default_bytes = canvas.layers[0].image.tobytes()
        self.assertNotEqual(default_bytes, source_bytes)
        canvas.layers[0].image = Image.frombytes("RGBA", (4, 4), source_bytes)
        canvas._save_state()
        custom_out = await self.server.app.call_tool("apply_filter", {"name": "gaussian_blur", "radius": 5.0, "session_id": "blur_test"})
        self.assertNotIn("error", custom_out[0].text.lower())
        custom_bytes = canvas.layers[0].image.tobytes()
        self.assertNotEqual(custom_bytes, default_bytes)
        canvas.layers[0].image = Image.frombytes("RGBA", (4, 4), source_bytes)
        canvas._save_state()
        zero_out = await self.server.app.call_tool("apply_filter", {"name": "gaussian_blur", "radius": 0, "session_id": "blur_test"})
        self.assertNotIn("error", zero_out[0].text.lower())
        self.assertEqual(canvas.layers[0].image.tobytes(), source_bytes)

    async def test_posterize_levels_one_preserves_alpha(self):
        await self.server.new_canvas(width=2, height=1, session_id="posterize_test")
        canvas = self.server.sessions.get_or_create("posterize_test")
        source_img = Image.new("RGBA", (2, 1), (255, 0, 0, 255))
        source_img.putpixel((1, 0), (0, 255, 0, 128))
        source_bytes = source_img.tobytes()
        canvas.layers[0].image = Image.frombytes("RGBA", (2, 1), source_bytes)
        canvas._save_state()
        out = await self.server.app.call_tool("apply_filter", {"name": "posterize", "levels": 1, "session_id": "posterize_test"})
        self.assertNotIn("error", out[0].text.lower())
        result = canvas.layers[0].image
        self.assertEqual(result.getchannel("A").tobytes(), bytes([255, 128]))
        for x in range(2):
            r, g, b, _ = result.getpixel((x, 0))
            for c in (r, g, b):
                self.assertIn(c, (0, 128))


    async def test_add_text_font_sizing_and_history(self):
        canvas = self.server.sessions.create("text_test", 512, 256, (0, 0, 0, 0))
        initial_bytes = canvas.layers[0].image.tobytes()
        initial_undo_count = len(canvas.undo_stack)

        out_small = await self.server.app.call_tool("add_text", {"text": "Test", "x": 10, "y": 10, "font_size": 12, "session_id": "text_test"})
        self.assertNotIn("error", out_small[0].text.lower())
        small_bbox = canvas.layers[0].image.getchannel("A").getbbox()
        self.assertIsNotNone(small_bbox)
        small_width = small_bbox[2] - small_bbox[0]
        small_height = small_bbox[3] - small_bbox[1]
        self.assertEqual(len(canvas.undo_stack), initial_undo_count + 1)

        canvas.undo()
        self.assertEqual(len(canvas.undo_stack), initial_undo_count)
        self.assertEqual(canvas.layers[0].image.tobytes(), initial_bytes)

        out_large = await self.server.app.call_tool("add_text", {"text": "Test", "x": 10, "y": 10, "font_size": 96, "session_id": "text_test"})
        self.assertNotIn("error", out_large[0].text.lower())
        large_bbox = canvas.layers[0].image.getchannel("A").getbbox()
        self.assertIsNotNone(large_bbox)
        large_width = large_bbox[2] - large_bbox[0]
        large_height = large_bbox[3] - large_bbox[1]
        self.assertGreaterEqual(large_width, small_width * 3)
        self.assertGreaterEqual(large_height, small_height * 3)
        self.assertEqual(len(canvas.undo_stack), initial_undo_count + 1)

        large_bytes = canvas.layers[0].image.tobytes()
        outside_region = canvas.layers[0].image.crop((large_bbox[2] + 10, large_bbox[3] + 10, min(512, large_bbox[2] + 50), min(256, large_bbox[3] + 50)))
        self.assertIsNone(outside_region.getchannel("A").getbbox())

        canvas.undo()
        self.assertEqual(len(canvas.undo_stack), initial_undo_count)
        self.assertEqual(canvas.layers[0].image.tobytes(), initial_bytes)
        canvas.redo()
        self.assertEqual(len(canvas.undo_stack), initial_undo_count + 1)
        self.assertEqual(canvas.layers[0].image.tobytes(), large_bytes)

    async def test_add_text_missing_font_fallback(self):
        canvas = self.server.sessions.create("text_fallback_test", 512, 256, (0, 0, 0, 0))
        initial_undo_count = len(canvas.undo_stack)

        out = await self.server.app.call_tool("add_text", {"text": "Test", "x": 10, "y": 10, "font_size": 48, "font": "/nonexistent/font.ttf", "session_id": "text_fallback_test"})
        self.assertNotIn("error", out[0].text.lower())
        bbox = canvas.layers[0].image.getchannel("A").getbbox()
        self.assertIsNotNone(bbox)
        self.assertGreater(bbox[2] - bbox[0], 60)
        self.assertGreater(bbox[3] - bbox[1], 25)
        self.assertEqual(len(canvas.undo_stack), initial_undo_count + 1)


if __name__ == "__main__":
    unittest.main()
