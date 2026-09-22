import base64
import io
import json
import unittest
from unittest.mock import AsyncMock

from PIL import Image

from canvas import Canvas
from editing import png_bytes, register_editing_tools
from session import SessionManager
from test_editing import Registry, node_info


class LayerEditingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = Registry()
        self.sessions = SessionManager()
        self.canvas = self.sessions.create("default", 32, 32, (10, 20, 30, 255))
        self.canvas.add_layer("Subject", Image.new("RGBA", (32, 32), (50, 60, 70, 120)), 0.4, "multiply")
        self.canvas.layers[1].mask = Image.new("L", (32, 32), 180)
        self.canvas.layers[1].visible = False
        self.canvas.active_layer_index = 0
        self.canvas._save_state()
        self.client = AsyncMock()
        self.client.get_object_info.return_value = node_info()
        self.client.upload_image.return_value = "source.png"
        self.generated = Image.new("RGBA", (32, 32), (255, 0, 0, 90))
        self.run = AsyncMock(return_value=png_bytes(self.generated))
        register_editing_tools(self.registry, self.client, self.sessions, self.run)

    async def test_explicit_layer_reads_raw_pixels_and_preserves_other_state(self):
        before = [layer.to_dict() for layer in self.canvas.layers]
        count = len(self.canvas.undo_stack)
        result = await self.registry.tools["edit_image"]("recolor", layer_index=1, region=[8, 8, 16, 16])
        report = json.loads(result[0].text)
        self.assertEqual(report["source_layer"], 1)
        self.assertEqual(report["layer"], 1)
        self.assertTrue(report["has_transparency"])
        with Image.open(io.BytesIO(self.client.upload_image.await_args_list[0].args[0])) as uploaded:
            self.assertEqual(uploaded.getpixel((0, 0)), (50, 60, 70, 120))
        self.assertEqual(len(self.canvas.layers), 2)
        self.assertEqual(self.canvas.active_layer_index, 0)
        self.assertEqual(self.canvas.layers[0].to_dict(), before[0])
        layer = self.canvas.layers[1]
        self.assertEqual(layer.image.getpixel((0, 0)), (50, 60, 70, 120))
        self.assertEqual(layer.image.getpixel((16, 16)), (255, 0, 0, 90))
        for key in ("name", "opacity", "blend_mode", "visible", "mask_data"):
            self.assertEqual(layer.to_dict()[key], before[1][key])
        with Image.open(io.BytesIO(base64.b64decode(result[1].data))) as preview:
            self.assertEqual(preview.tobytes(), layer.image.tobytes())
        edited = [layer.to_dict() for layer in self.canvas.layers]
        self.assertEqual(len(self.canvas.undo_stack), count + 1)
        self.assertTrue(self.canvas.undo())
        self.assertEqual([layer.to_dict() for layer in self.canvas.layers], before)
        self.assertTrue(self.canvas.redo())
        self.assertEqual([layer.to_dict() for layer in self.canvas.layers], edited)

    async def test_extraction_adds_cutout_keeps_originals_and_uses_qwen_rgba_graph(self):
        before = [layer.to_dict() for layer in self.canvas.layers]
        self.run.return_value = png_bytes(Image.new("RGBA", (32, 32), (255, 0, 0, 255)))
        result = await self.registry.tools["edit_image"]("the subject", output_mode="extract", region=[8, 8, 16, 16])
        report = json.loads(result[0].text)
        self.assertEqual(report["output_mode"], "extract")
        self.assertTrue(report["has_transparency"])
        self.assertEqual([layer.to_dict() for layer in self.canvas.layers[:2]], before)
        cutout = self.canvas.layers[2].image
        self.assertEqual(cutout.getpixel((0, 0)), (0, 0, 0, 0))
        self.assertEqual(cutout.getpixel((16, 16)), (255, 0, 0, 255))
        with Image.open(io.BytesIO(base64.b64decode(result[1].data))) as preview:
            self.assertEqual(preview.tobytes(), cutout.tobytes())
        graph = self.run.call_args.args[0]
        by_type = {node["class_type"]: node["inputs"] for node in graph.values()}
        self.assertEqual(by_type["VAELoader"]["vae_name"], "qwen_image_2.1_vae_bf16.safetensors")
        self.assertIn("JoinImageWithAlpha", by_type)
        self.assertIn("transparent background", report["effective_prompt"])
        self.assertIn("black marks the area to make transparent", report["effective_prompt"])
        self.assertTrue(self.canvas.undo())
        self.assertEqual([layer.to_dict() for layer in self.canvas.layers], before)
        self.assertTrue(self.canvas.redo())
        self.assertEqual(len(self.canvas.layers), 3)

    async def test_opaque_model_output_is_reported_as_opaque(self):
        self.run.return_value = png_bytes(Image.new("RGB", (32, 32), "red"))
        result = await self.registry.tools["edit_image"]("subject", output_mode="extract", layer_index=1)
        self.assertFalse(json.loads(result[0].text)["has_transparency"])
        self.assertFalse(self.canvas.layers[1].visible)

    async def test_invalid_modes_and_indices_fail_before_backend_start(self):
        before = [layer.to_dict() for layer in self.canvas.layers]
        for arguments in ({"layer_index": -1}, {"layer_index": 2}, {"output_mode": "unknown"},
                          {"output_mode": "extract", "backend": "flux2"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                await self.registry.tools["edit_image"]("subject", **arguments)
        self.client.start_comfyui.assert_not_awaited()
        self.run.assert_not_awaited()
        self.assertEqual([layer.to_dict() for layer in self.canvas.layers], before)

    async def test_changed_replaced_and_closed_documents_reject_stale_result(self):
        for change in ("visibility", "replace", "close"):
            with self.subTest(change=change):
                canvas = self.sessions.replace("default", Canvas(32, 32))

                async def generate(workflow, **kwargs):
                    if change == "visibility":
                        canvas.set_layer_visibility(False)
                    elif change == "replace":
                        self.sessions.replace("default", Canvas(16, 16))
                    else:
                        self.sessions.delete("default")
                    return png_bytes(self.generated)

                self.run.side_effect = generate
                with self.assertRaisesRegex(RuntimeError, "Canvas changed"):
                    await self.registry.tools["edit_image"]("subject", layer_index=0, output_mode="extract")
                if change == "close":
                    self.assertIsNone(self.sessions.get("default"))
                elif change == "replace":
                    self.assertEqual(self.sessions.get("default").width, 16)
                else:
                    self.assertFalse(canvas.layers[0].visible)
                    self.assertEqual(len(canvas.layers), 1)
