import unittest
from unittest.mock import patch

from PIL import Image

import server
from session import SessionManager


def document(canvas):
    return (canvas.width, canvas.height, canvas.active_layer_index,
            [layer.to_dict() for layer in canvas.layers])


class LayerControlTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sessions = SessionManager()
        patcher = patch.object(server, "sessions", self.sessions)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.canvas = self.sessions.create("layers", 4, 3, (0, 0, 0, 0))

    async def call(self, tool_name, **arguments):
        return await server.app.call_tool(tool_name, {**arguments, "session_id": "layers"})

    async def test_visibility_rename_history_and_session_isolation(self):
        other = self.sessions.create("other", 2, 2)
        other_before = document(other)
        original = document(self.canvas)
        await self.call("set_layer_visibility", visible=False)
        self.assertFalse(self.canvas.layers[0].visible)
        self.assertEqual(len(self.canvas.undo_stack), 2)
        await self.call("rename_layer", name="Original")
        renamed = document(self.canvas)
        self.assertEqual(self.canvas.layers[0].name, "Original")
        self.assertEqual(len(self.canvas.undo_stack), 3)
        self.assertTrue(self.canvas.undo())
        self.assertFalse(self.canvas.layers[0].visible)
        self.assertEqual(self.canvas.layers[0].name, "Background")
        self.assertTrue(self.canvas.undo())
        self.assertEqual(document(self.canvas), original)
        self.assertTrue(self.canvas.redo())
        self.assertTrue(self.canvas.redo())
        self.assertEqual(document(self.canvas), renamed)
        await self.call("set_layer_visibility", visible=True, index=0)
        self.assertTrue(self.canvas.layers[0].visible)
        self.assertEqual(document(other), other_before)

    async def test_duplicate_copies_pixels_mask_settings_and_history(self):
        canvas = self.canvas
        canvas.add_layer("Subject", Image.new("RGBA", (4, 3), (20, 30, 40, 128)), 0.4, "multiply")
        source = canvas.layers[1]
        source.mask = Image.new("L", (4, 3), 170)
        source.visible = False
        canvas.add_layer("Top")
        before = document(canvas)
        history_size = len(canvas.undo_stack)
        out = await self.call("duplicate_layer", index=1)
        self.assertIn("index 2", out[0].text)
        self.assertEqual(len(canvas.layers), 4)
        self.assertEqual(canvas.active_layer_index, 2)
        expected = source.to_dict()
        expected["name"] = "Subject copy"
        self.assertEqual(canvas.layers[2].to_dict(), expected)
        self.assertEqual(canvas.layers[3].name, "Top")
        self.assertEqual(len(canvas.undo_stack), history_size + 1)
        duplicated = document(canvas)
        canvas.layers[2].image.putpixel((0, 0), (1, 2, 3, 4))
        canvas.layers[2].mask.putpixel((0, 0), 0)
        self.assertEqual(source.image.getpixel((0, 0)), (20, 30, 40, 128))
        self.assertEqual(source.mask.getpixel((0, 0)), 170)
        self.assertTrue(canvas.undo())
        self.assertEqual(document(canvas), before)
        self.assertTrue(canvas.redo())
        self.assertEqual(document(canvas), duplicated)
        await self.call("duplicate_layer", index=1, name="")
        self.assertEqual(canvas.layers[2].name, "")

    async def test_translation_aligns_mask_preserves_alpha_and_clips(self):
        canvas = self.canvas
        image = Image.new("RGBA", (4, 3))
        image.putpixel((1, 1), (20, 30, 40, 128))
        image.putpixel((3, 2), (50, 60, 70, 255))
        canvas.add_layer("Subject", image)
        canvas.layers[1].mask = Image.new("L", (4, 3))
        canvas.layers[1].mask.putpixel((1, 1), 200)
        canvas._save_state()
        before = document(canvas)
        history_size = len(canvas.undo_stack)
        await self.call("translate_layer", dx=1, dy=-1)
        layer = canvas.layers[1]
        self.assertEqual(layer.image.getpixel((2, 0)), (20, 30, 40, 128))
        self.assertEqual(layer.image.getpixel((1, 1)), (0, 0, 0, 0))
        self.assertEqual(layer.image.getchannel("A").getbbox(), (2, 0, 3, 1))
        self.assertEqual(layer.mask.getpixel((2, 0)), 200)
        self.assertEqual(layer.mask.getpixel((1, 1)), 0)
        self.assertEqual(canvas.layers[0].to_dict(), before[3][0])
        self.assertEqual((canvas.width, canvas.height), (4, 3))
        self.assertEqual(len(canvas.undo_stack), history_size + 1)
        translated = document(canvas)
        self.assertTrue(canvas.undo())
        self.assertEqual(document(canvas), before)
        self.assertTrue(canvas.redo())
        self.assertEqual(document(canvas), translated)
        canvas.layers[1].mask = None
        await self.call("translate_layer", dx=-1, dy=1, index=1)
        self.assertIsNone(canvas.layers[1].mask)
        self.assertEqual(canvas.layers[1].image.getpixel((1, 1)), (20, 30, 40, 128))

    async def test_invalid_indices_leave_document_and_history_unchanged(self):
        canvas = self.canvas
        canvas.add_layer("Top")
        canvas.undo()
        before = document(canvas)
        history = (len(canvas.undo_stack), len(canvas.redo_stack))
        for tool, arguments in [("set_layer_visibility", {"visible": False}),
                                ("rename_layer", {"name": "Bad"}), ("duplicate_layer", {}),
                                ("translate_layer", {"dx": 2, "dy": 1})]:
            for index in (-1, 10):
                with self.subTest(tool=tool, index=index):
                    out = await self.call(tool, **arguments, index=index)
                    self.assertIn("Invalid", out[0].text)
                    self.assertEqual(document(canvas), before)
                    self.assertEqual((len(canvas.undo_stack), len(canvas.redo_stack)), history)
