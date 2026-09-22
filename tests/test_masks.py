import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock

from PIL import Image

from editing import png_bytes, register_editing_tools
from session import SessionManager
from test_editing import Registry, node_info


class MaskExportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = Registry()
        self.sessions = SessionManager()
        self.canvas = self.sessions.create("mask", 32, 32, (10, 20, 30, 255))
        self.canvas.layers[0].mask = Image.new("L", (32, 32), 0)
        self.canvas.layers[0].mask.paste(255, (12, 12, 20, 20))
        self.canvas._save_state()
        self.client = AsyncMock()
        self.client.get_object_info.return_value = node_info()
        self.client.upload_image.return_value = "mask.png"
        self.run = AsyncMock(return_value=png_bytes(Image.new("RGBA", (32, 32), (255, 0, 0, 100))))
        register_editing_tools(self.registry, self.client, self.sessions, self.run)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "edit-mask.png"

    async def export(self, **arguments):
        return await self.registry.tools["export_mask"](str(self.path), session_id="mask", **arguments)

    async def test_independent_mask_can_drive_edit_with_exact_outside_protection(self):
        before = self.canvas.layers[0].to_dict()
        history_size = len(self.canvas.undo_stack)
        result = await self.export()
        self.assertEqual(json.loads(result[0].text)["path"], str(self.path))
        self.assertEqual(result[1].type, "image")
        self.assertEqual(self.canvas.layers[0].to_dict(), before)
        self.assertEqual(len(self.canvas.undo_stack), history_size)
        self.client.start_comfyui.assert_not_awaited()
        self.run.assert_not_awaited()
        saved = self.path.read_bytes()
        self.canvas.layers[0].mask = None
        self.canvas._save_state()
        await self.registry.tools["edit_image"]("red patch", mask_path=str(self.path), session_id="mask")
        self.assertEqual(self.path.read_bytes(), saved)
        result_image = self.canvas.composite()
        self.assertEqual(result_image.getpixel((0, 0)), (10, 20, 30, 255))
        self.assertEqual(result_image.getpixel((16, 16)), (255, 0, 0, 100))

    async def test_invert_grow_shrink_and_feather_without_document_mutation(self):
        before = self.canvas.layers[0].to_dict()
        for options, bbox in [({"expand": 2}, (10, 10, 22, 22)),
                              ({"expand": -2}, (14, 14, 18, 18))]:
            await self.export(**options)
            with Image.open(self.path) as mask:
                self.assertEqual(mask.mode, "L")
                self.assertEqual(mask.getbbox(), bbox)
        await self.export(invert=True)
        with Image.open(self.path) as mask:
            self.assertEqual(mask.getpixel((0, 0)), 255)
            self.assertEqual(mask.getpixel((16, 16)), 0)
        await self.export(feather=2)
        with Image.open(self.path) as mask:
            self.assertGreater(mask.getpixel((11, 16)), 0)
            self.assertLess(mask.getpixel((12, 16)), 255)
        self.assertEqual(self.canvas.layers[0].to_dict(), before)
        self.assertEqual(len(self.canvas.undo_stack), 2)

    async def test_invalid_inputs_preserve_existing_output_and_document(self):
        self.path.write_bytes(b"previous mask")
        before = self.canvas.layers[0].to_dict()
        for options in ({"index": -1}, {"index": 10}, {"feather": -1}):
            with self.subTest(options=options), self.assertRaises((ValueError, IndexError)):
                await self.export(**options)
            self.assertEqual(self.path.read_bytes(), b"previous mask")
            self.assertEqual(self.canvas.layers[0].to_dict(), before)
        self.canvas.layers[0].mask = None
        with self.assertRaisesRegex(ValueError, "no mask"):
            await self.export()
        self.assertEqual(self.path.read_bytes(), b"previous mask")
