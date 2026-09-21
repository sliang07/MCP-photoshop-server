"""Regression tests for outpaint (Flux2 chain, 2026-09-20 rewrite)."""
import io
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

import server


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

    async def test_steps_capped_at_six(self):
        canvas = self.canvas()
        client = AsyncMock()
        client.upload_image = AsyncMock(return_value="op.png")
        run = AsyncMock(return_value=png_bytes((32, 16), "blue"))
        with patch.object(server, "sessions", self.sessions), \
             patch.object(server, "comfy", client), \
             patch.object(server, "run_workflow", run):
            await server.outpaint_tool("extend the scene", direction="right", amount=16, steps=30, session_id="op")
        workflow = run.await_args.args[0]
        scheduler = next(n for n in workflow.values() if n["class_type"] == "BasicScheduler")
        self.assertEqual(scheduler["inputs"]["steps"], 6)
        self.assertIn("Flux2KleinSectionedEncoder", [n["class_type"] for n in workflow.values()])


if __name__ == "__main__":
    unittest.main()
