import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

import server
from editing import png_bytes
from session import SessionManager


class UpscaleTests(unittest.IsolatedAsyncioTestCase):
    async def test_factor_alignment_alpha_and_undo(self):
        sessions = SessionManager()
        canvas = sessions.create("default", 16, 16)
        index = canvas.add_layer("detail", Image.new("RGBA", (16, 16), (80, 90, 100, 128)))
        canvas.layers[index].mask = Image.new("L", (16, 16), 200)
        canvas._save_state()
        before = canvas.composite().tobytes()
        native_4x = png_bytes(Image.new("RGB", (64, 64), "red"))
        with patch.object(server, "sessions", sessions), patch.object(server, "comfy", AsyncMock()), patch.object(server, "run_workflow", AsyncMock(return_value=native_4x)):
            result = await server.upscale_tool(factor=2)
        self.assertIn("32x32 (2x)", result[0].text)
        self.assertEqual((canvas.width, canvas.height), (32, 32))
        self.assertTrue(all(layer.image.size == (32, 32) for layer in canvas.layers))
        self.assertEqual(canvas.layers[index].mask.size, (32, 32))
        self.assertEqual(canvas.layers[index].image.getpixel((0, 0))[3], 128)
        self.assertTrue(canvas.undo())
        self.assertEqual(canvas.composite().tobytes(), before)

    async def test_invalid_factor_does_not_call_gpu(self):
        with patch.object(server, "comfy", AsyncMock()) as client:
            result = await server.upscale_tool(factor=0)
        self.assertIn("factor must", result[0].text)
        client.upload_image.assert_not_called()


if __name__ == "__main__":
    unittest.main()
