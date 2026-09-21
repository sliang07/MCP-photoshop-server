"""Regression tests for outpaint (Flux2 chain, 2026-09-20 rewrite)."""
import io
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

import server
from test_editing import node_info


def png_bytes(size=(16, 16), color="red"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class OutpaintTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from session import SessionManager
        self.sessions = SessionManager()

    def canvas(self):
        canvas = self.sessions.create("op", 16, 16, (10, 20, 30))
        canvas.add_layer("detail", Image.new("RGBA", (16, 16), (80, 90, 100, 128)))
        return canvas

    def run_tool(self, canvas, result_bytes):
        client = AsyncMock()
        client.get_object_info.return_value = node_info()
        client.upload_image = AsyncMock(return_value="op.png")
        return (patch.object(server, "sessions", self.sessions),
                patch.object(server, "comfy", client),
                patch.object(server, "run_workflow", AsyncMock(return_value=result_bytes)))

    async def test_right_outpaint_extends_all_layers(self):
        canvas = self.canvas()
        a, b, c = self.run_tool(canvas, png_bytes((32, 16), "blue"))
        with a, b, c:
            out = await server.outpaint_tool("extend the scene", direction="right", amount=16, steps=6, session_id="op")
        self.assertIn("32x16", out[0].text)
        self.assertEqual((canvas.width, canvas.height), (32, 16))
        for layer in canvas.layers:
            self.assertEqual(layer.image.size, (32, 16))
        # Regression: composite must work after outpaint ("images do not match")
        self.assertEqual(canvas.composite().size, (32, 16))

    async def test_top_outpaint_extends_masks(self):
        canvas = self.canvas()
        canvas.layers[1].mask = Image.new("L", (16, 16), 200)
        a, b, c = self.run_tool(canvas, png_bytes((16, 32), "green"))
        with a, b, c:
            await server.outpaint_tool("extend the scene", direction="top", amount=16, steps=6, session_id="op")
        for layer in canvas.layers:
            self.assertEqual(layer.image.size, (16, 32))
        self.assertEqual(canvas.layers[1].mask.size, (16, 32))
        self.assertEqual(canvas.composite().size, (16, 32))

    async def test_explicit_steps_and_uploaded_filename_are_preserved(self):
        canvas = self.canvas()
        client = AsyncMock()
        client.get_object_info.return_value = node_info()
        client.upload_image = AsyncMock(return_value="renamed/op.png")
        run = AsyncMock(return_value=png_bytes((32, 16), "blue"))
        with patch.object(server, "sessions", self.sessions), \
             patch.object(server, "comfy", client), \
             patch.object(server, "run_workflow", run):
            await server.outpaint_tool("extend the scene", direction="right", amount=16, steps=30, session_id="op")
        workflow = run.await_args.args[0]
        scheduler = next(n for n in workflow.values() if n["class_type"] == "Flux2Scheduler")
        self.assertEqual(scheduler["inputs"]["steps"], 30)
        self.assertIn("ReferenceLatent", [n["class_type"] for n in workflow.values()])
        load = next(n for n in workflow.values() if n["class_type"] == "LoadImage")
        self.assertEqual(load["inputs"]["image"], "renamed/op.png")

    async def test_both_backends_preserve_original_pixels_and_undo(self):
        for backend in ("flux2", "qwen21"):
            for direction, offset in (("left", (16, 0)), ("right", (0, 0)), ("top", (0, 16)), ("bottom", (0, 0))):
                with self.subTest(backend=backend, direction=direction):
                    canvas = self.canvas()
                    layer = canvas.layers[canvas.active_layer_index]
                    original = layer.image.tobytes()
                    composite = canvas.composite().tobytes()
                    size = (32, 16) if direction in ("left", "right") else (16, 32)
                    a, b, c = self.run_tool(canvas, png_bytes(size, "blue"))
                    with a, b, c:
                        out = await server.outpaint_tool("continue", direction=direction, amount=16, backend=backend, session_id="op")
                    self.assertIn(f"Model: {backend}", out[0].text)
                    x, y = offset
                    self.assertEqual(layer.image.crop((x, y, x + 16, y + 16)).tobytes(), original)
                    self.assertEqual(canvas.composite().size, size)
                    canvas.undo()
                    self.assertEqual(canvas.composite().tobytes(), composite)

    async def test_anima_outpaint_is_rejected_before_upload(self):
        client = AsyncMock()
        with patch.object(server, "comfy", client):
            out = await server.outpaint_tool("continue", backend="anima")
        self.assertIn("generation only", out[0].text)
        client.upload_image.assert_not_awaited()

    async def test_qwen_uses_opaque_margin_and_native_edit_resolution(self):
        self.canvas()
        client = AsyncMock()
        client.get_object_info.return_value = node_info()
        client.upload_image.return_value = "qwen.png"
        run = AsyncMock(return_value=png_bytes((64, 64), "blue"))
        with patch.object(server, "sessions", self.sessions), patch.object(server, "comfy", client), \
             patch.object(server, "run_workflow", run):
            await server.outpaint_tool("continue", backend="qwen21", amount=16, session_id="op")
        with Image.open(io.BytesIO(client.upload_image.await_args.args[0])) as uploaded:
            self.assertEqual(uploaded.getpixel((uploaded.width - 1, 0)), (255, 255, 255, 255))
        nodes = {n["class_type"]: n["inputs"] for n in run.await_args.args[0].values()}
        self.assertEqual(nodes["TextEncodeQwenImage21"]["resolution"], 1024)
        self.assertNotIn("images.image_2", nodes["TextEncodeQwenImage21"])
        self.assertTrue(nodes["TextEncodeQwenImage21"]["prompt"].startswith("Outpaint the image:"))
        self.assertNotIn("<image1>", nodes["TextEncodeQwenImage21"]["prompt"])
        self.assertEqual(nodes["KSampler"]["steps"], 30)
        self.assertEqual(nodes["KSampler"]["cfg"], 3.0)


if __name__ == "__main__":
    unittest.main()
