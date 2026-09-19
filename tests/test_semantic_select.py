import json
import unittest
from unittest.mock import AsyncMock

from PIL import Image, ImageDraw

from editing import (build_semantic_select_workflow, count_selection_regions,
                     png_bytes, register_editing_tools, sam3_capabilities)
from session import SessionManager


class Registry:
    def __init__(self):
        self.tools = {}

    def tool(self, name):
        def register(function):
            self.tools[name] = function
            return function
        return register


def sam3_info(ckpts=("sam3.pt",), node=True, sam31=None):
    data = {"ImageOnlyCheckpointLoader": {"input": {"required": {"ckpt_name": [list(ckpts)]}}}}
    if sam31:
        data["CheckpointLoaderSimple"] = {"input": {"required": {"ckpt_name": [list(sam31)]}}}
    if node:
        data["SAM3_Detect"] = {}
    return data


def circle_mask():
    mask = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(mask).ellipse((16, 16, 47, 47), fill=255)
    return mask


class SemanticSelectTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = Registry()
        self.sessions = SessionManager()
        self.canvas = self.sessions.create("default", 64, 64, (10, 20, 30))
        self.client = AsyncMock()
        self.client.get_object_info.return_value = sam3_info()
        self.client.get_system_stats.return_value = {"system": {"comfyui_version": "0.33.0"}, "devices": []}
        self.client.upload_image.return_value = "renamed/source.png"
        self.mask_ref = circle_mask()
        self.run = AsyncMock(return_value=png_bytes(self.mask_ref))
        register_editing_tools(self.registry, self.client, self.sessions, self.run)

    def active_layer(self):
        return self.canvas.layers[self.canvas.active_layer_index]

    async def test_point_prompt_builds_graph_and_applies_mask(self):
        result = await self.registry.tools["semantic_select"](point=[32, 32])
        workflow = self.run.await_args.args[0]
        self.assertEqual(sorted(n["class_type"] for n in workflow.values()),
                         ["ImageOnlyCheckpointLoader", "LoadImage", "MaskToImage", "SAM3_Detect", "SaveImage"])
        keys = {n["class_type"]: key for key, n in workflow.items()}
        loader, detect = workflow[keys["ImageOnlyCheckpointLoader"]], workflow[keys["SAM3_Detect"]]
        self.assertEqual(loader["inputs"]["ckpt_name"], "sam3.pt")
        self.assertEqual(workflow[keys["LoadImage"]]["inputs"]["image"], "renamed/source.png")
        self.assertEqual(detect["inputs"]["model"], [keys["ImageOnlyCheckpointLoader"], 0])
        self.assertEqual(detect["inputs"]["image"], [keys["LoadImage"], 0])
        self.assertEqual(detect["inputs"]["positive_coords"], '[{"x": 32, "y": 32}]')
        self.assertNotIn("bboxes", detect["inputs"])
        self.assertEqual(detect["inputs"]["individual_masks"], False)
        self.assertEqual(workflow[keys["MaskToImage"]]["inputs"]["mask"], [keys["SAM3_Detect"], 0])
        self.assertEqual(workflow[keys["SaveImage"]]["inputs"]["images"], [keys["MaskToImage"], 0])

        layer = self.active_layer()
        self.assertEqual(layer.mask.tobytes(), self.mask_ref.tobytes())
        report = json.loads(result[0].text)
        self.assertTrue(report["selected"])
        self.assertEqual(report["prompt"]["point"], [32, 32])
        self.assertEqual(report["mask_size"], [64, 64])
        self.assertEqual(report["selected_pixels"], sum(self.mask_ref.histogram()[128:]))
        self.assertEqual(report["objects"], 1)
        self.assertGreater(report["coverage"], 0.05)
        self.assertEqual(result[1].type, "image")

        self.assertTrue(self.canvas.undo())
        self.assertIsNone(self.active_layer().mask)
        self.assertTrue(self.canvas.redo())
        self.assertEqual(self.active_layer().mask.tobytes(), self.mask_ref.tobytes())

    async def test_box_prompt_builds_bbox_input(self):
        await self.registry.tools["semantic_select"](box=[8, 8, 32, 32])
        workflow = self.run.await_args.args[0]
        detect = next(n for n in workflow.values() if n["class_type"] == "SAM3_Detect")
        self.assertEqual(detect["inputs"]["bboxes"], {"x": 8, "y": 8, "width": 32, "height": 32})
        self.assertNotIn("positive_coords", detect["inputs"])

    async def test_missing_checkpoint_raises_before_upload(self):
        self.client.get_object_info.return_value = {"SAM3_Detect": {}}
        with self.assertRaises(ValueError) as ctx:
            await self.registry.tools["semantic_select"](point=[32, 32])
        self.assertIn("sam3.pt", str(ctx.exception))
        self.client.upload_image.assert_not_awaited()
        self.run.assert_not_awaited()

    async def test_requires_point_or_box(self):
        with self.assertRaises(ValueError):
            await self.registry.tools["semantic_select"]()

    async def test_point_outside_canvas_rejected(self):
        with self.assertRaises(ValueError):
            await self.registry.tools["semantic_select"](point=[32, 200])

    async def test_box_outside_canvas_rejected(self):
        with self.assertRaises(ValueError):
            await self.registry.tools["semantic_select"](box=[40, 40, 40, 40])

    async def test_empty_mask_leaves_layer_untouched(self):
        self.run.return_value = png_bytes(Image.new("L", (64, 64), 0))
        before = self.canvas.composite().tobytes()
        result = await self.registry.tools["semantic_select"](point=[32, 32])
        report = json.loads(result[0].text)
        self.assertFalse(report["selected"])
        self.assertIsNone(self.active_layer().mask)
        self.assertEqual(self.canvas.composite().tobytes(), before)

    async def test_capabilities_report_sam3(self):
        result = await self.registry.tools["get_editing_capabilities"]()
        report = json.loads(result[0].text)
        self.assertTrue(report["sam3"]["available"])
        self.assertEqual(report["sam3"]["checkpoint"], "sam3.pt")
        self.assertFalse(report["sam3"]["text_prompt_available"])
        self.assertTrue(any("semantic_select" in note for note in report["notes"]))

    async def test_text_prompt_builds_31_graph(self):
        self.client.get_object_info.return_value = sam3_info(sam31=["sam3.1_multiplex_fp16.safetensors"])
        result = await self.registry.tools["semantic_select"](prompt="red circle")
        workflow = self.run.await_args.args[0]
        self.assertEqual(sorted(n["class_type"] for n in workflow.values()),
                         ["CLIPTextEncode", "CheckpointLoaderSimple", "LoadImage", "MaskToImage", "SAM3_Detect", "SaveImage"])
        keys = {n["class_type"]: key for key, n in workflow.items()}
        loader, text, detect = (workflow[keys["CheckpointLoaderSimple"]],
                                workflow[keys["CLIPTextEncode"]], workflow[keys["SAM3_Detect"]])
        self.assertEqual(loader["inputs"]["ckpt_name"], "sam3.1_multiplex_fp16.safetensors")
        self.assertEqual(text["inputs"]["text"], "red circle")
        self.assertEqual(text["inputs"]["clip"], [keys["CheckpointLoaderSimple"], 1])
        self.assertEqual(detect["inputs"]["model"], [keys["CheckpointLoaderSimple"], 0])
        self.assertEqual(detect["inputs"]["conditioning"], [keys["CLIPTextEncode"], 0])
        self.assertNotIn("positive_coords", detect["inputs"])
        self.assertNotIn("bboxes", detect["inputs"])
        self.assertNotIn("ImageOnlyCheckpointLoader", keys)
        report = json.loads(result[0].text)
        self.assertTrue(report["selected"])
        self.assertEqual(report["checkpoint"], "sam3.1_multiplex_fp16.safetensors")
        self.assertEqual(report["prompt"]["text"], "red circle")

    async def test_text_prompt_combines_point_and_box(self):
        self.client.get_object_info.return_value = sam3_info(sam31=["sam3.1_multiplex_fp16.safetensors"])
        await self.registry.tools["semantic_select"](prompt="red circle", point=[32, 32], box=[8, 8, 32, 32])
        detect = next(n for n in self.run.await_args.args[0].values() if n["class_type"] == "SAM3_Detect")["inputs"]
        self.assertIn("conditioning", detect)
        self.assertEqual(detect["positive_coords"], '[{"x": 32, "y": 32}]')
        self.assertEqual(detect["bboxes"], {"x": 8, "y": 8, "width": 32, "height": 32})

    async def test_text_prompt_without_31_raises_before_upload(self):
        with self.assertRaises(ValueError) as ctx:
            await self.registry.tools["semantic_select"](prompt="red circle")
        self.assertIn("sam3.1_multiplex_fp16.safetensors", str(ctx.exception))
        self.client.upload_image.assert_not_awaited()
        self.run.assert_not_awaited()

    async def test_blank_prompt_rejected(self):
        with self.assertRaises(ValueError):
            await self.registry.tools["semantic_select"](prompt="   ")
        self.client.upload_image.assert_not_awaited()


class Sam3HelperTests(unittest.TestCase):
    def test_capabilities_reports_missing_parts(self):
        self.assertFalse(sam3_capabilities({"SAM3_Detect": {}})["available"])
        no_node = sam3_info()
        del no_node["SAM3_Detect"]
        self.assertFalse(sam3_capabilities(no_node)["node_present"])
        self.assertTrue(sam3_capabilities(sam3_info())["available"])

    def test_capabilities_report_31_and_text_availability(self):
        caps = sam3_capabilities(sam3_info(sam31=["sam3.1_multiplex_fp16.safetensors"]))
        self.assertEqual(caps["sam3_1"], "sam3.1_multiplex_fp16.safetensors")
        self.assertTrue(caps["text_prompt_available"])
        plain = sam3_capabilities(sam3_info())
        self.assertIsNone(plain["sam3_1"])
        self.assertFalse(plain["text_prompt_available"])
        no_node = sam3_info(sam31=["sam3.1_multiplex_fp16.safetensors"], node=False)
        self.assertFalse(sam3_capabilities(no_node)["text_prompt_available"])

    def test_text_workflow_only_adds_clip_nodes_for_prompts(self):
        text_graph = build_semantic_select_workflow("sam3.1_multiplex_fp16.safetensors", "src.png", prompt="chair")
        text_types = {n["class_type"] for n in text_graph.values()}
        self.assertIn("CLIPTextEncode", text_types)
        self.assertIn("CheckpointLoaderSimple", text_types)
        self.assertNotIn("ImageOnlyCheckpointLoader", text_types)
        point_graph = build_semantic_select_workflow("sam3.pt", "src.png", point=[1, 1])
        point_types = {n["class_type"] for n in point_graph.values()}
        self.assertIn("ImageOnlyCheckpointLoader", point_types)
        self.assertNotIn("CLIPTextEncode", point_types)
        self.assertNotIn("CheckpointLoaderSimple", point_types)

    def test_workflow_carries_point_and_box_together(self):
        graph = build_semantic_select_workflow("sam3.pt", "src.png", point=[10, 20], box=[0, 0, 8, 8])
        detect = next(n for n in graph.values() if n["class_type"] == "SAM3_Detect")
        self.assertEqual(detect["inputs"]["positive_coords"], '[{"x": 10, "y": 20}]')
        self.assertEqual(detect["inputs"]["bboxes"], {"x": 0, "y": 0, "width": 8, "height": 8})

    def test_count_regions_counts_separate_objects(self):
        two = Image.new("L", (64, 64), 0)
        draw = ImageDraw.Draw(two)
        draw.ellipse((8, 8, 23, 23), fill=255)
        draw.ellipse((40, 40, 55, 55), fill=255)
        self.assertEqual(count_selection_regions(two), 2)
        self.assertEqual(count_selection_regions(Image.new("L", (64, 64), 0)), 0)


if __name__ == "__main__":
    unittest.main()