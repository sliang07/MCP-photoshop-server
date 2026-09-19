import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from PIL import Image

from editing import controlnet_capabilities, resolve_model, style_transfer_capabilities


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def object_info(controlnet=(), style=(), clip_vision=()):
    info = {}
    if controlnet:
        info["ControlNetLoader"] = {"input": {"required": {"control_net_name": [list(controlnet)]}}}
    if style:
        info["StyleModelLoader"] = {"input": {"required": {"style_model_name": [list(style)]}}}
    if clip_vision:
        info["CLIPVisionLoader"] = {"input": {"required": {"clip_name": [list(clip_vision)]}}}
    return info


class CapabilityTests(unittest.TestCase):
    def test_controlnet_empty(self):
        caps = controlnet_capabilities(object_info())
        self.assertFalse(caps["available"])
        self.assertEqual(caps["models"], [])

    def test_controlnet_with_models(self):
        caps = controlnet_capabilities(object_info(controlnet=["flux2_controlnet.safetensors"]))
        self.assertTrue(caps["available"])
        self.assertEqual(caps["models"], ["flux2_controlnet.safetensors"])

    def test_style_transfer_needs_both(self):
        self.assertFalse(style_transfer_capabilities(
            object_info(style=["flux1-redux-dev.safetensors"]))["available"])
        self.assertFalse(style_transfer_capabilities(
            object_info(clip_vision=["clip_vision.safetensors"]))["available"])
        self.assertTrue(style_transfer_capabilities(object_info(
            style=["flux1-redux-dev.safetensors"],
            clip_vision=["sigclip_vision_patch14_384.safetensors"]))["available"])

    def test_resolve_model_subfolder(self):
        self.assertEqual(resolve_model(["subdir/flux1-redux-dev.safetensors"], "flux1-redux-dev.safetensors"),
                         "subdir/flux1-redux-dev.safetensors")
        self.assertIsNone(resolve_model(["a.safetensors"], "b.safetensors"))


class BuilderSchemaTests(unittest.TestCase):
    def test_controlnet_workflow_matches_live_schema(self):
        from server import build_controlnet_workflow
        wf = build_controlnet_workflow(prompt="a city", image_filename="c.png", width=512, height=512)
        by_type = {n["class_type"]: n for n in wf.values()}
        apply_node = by_type["ControlNetApply"]["inputs"]
        self.assertIn("conditioning", apply_node)
        self.assertNotIn("positive", apply_node)
        latent = by_type["EmptyFlux2LatentImage"]["inputs"]
        self.assertEqual(latent["width"], 512)
        self.assertEqual(latent["height"], 512)
        self.assertNotIn("EmptyLatentImage", by_type)

    def test_style_transfer_workflow_is_a_valid_graph(self):
        from server import build_style_transfer_workflow
        wf = build_style_transfer_workflow(prompt="p", content_filename="a.png", style_filename="b.png")
        by_type = {n["class_type"]: n for n in wf.values()}
        self.assertIn("CLIPVisionLoader", by_type)
        encode = by_type["CLIPVisionEncode"]["inputs"]
        self.assertIsInstance(encode["clip_vision"], list)
        self.assertEqual(encode["crop"], "center")
        apply_node = by_type["StyleModelApply"]["inputs"]
        self.assertEqual(apply_node["strength_type"], "multiply")
        latent = by_type["EmptyFlux2LatentImage"]["inputs"]
        self.assertEqual((latent["width"], latent["height"]), (1024, 1024))

    def test_style_transfer_builder_params(self):
        from server import build_style_transfer_workflow
        wf = build_style_transfer_workflow(prompt="p", content_filename="a.png", style_filename="b.png",
                                          style_model_name="custom.safetensors",
                                          clip_vision_name="clip.safetensors")
        by_type = {n["class_type"]: n for n in wf.values()}
        self.assertEqual(by_type["StyleModelLoader"]["inputs"]["style_model_name"], "custom.safetensors")
        self.assertEqual(by_type["CLIPVisionLoader"]["inputs"]["clip_name"], "clip.safetensors")



class PrecheckToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import server
        self.server = server
        self.client = AsyncMock()
        self.client.start_comfyui = AsyncMock()
        self.client.upload_image = AsyncMock(return_value="up.png")
        self.orig_client = server.comfy
        server.comfy = self.client
        self.addCleanup(setattr, self.server, "comfy", self.orig_client)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.style_path = Path(self.tmp.name) / "style.png"
        self.style_path.write_bytes(png_bytes())
        for sid in list(server.sessions.list_sessions()):
            server.sessions.delete(sid)

    async def test_controlnet_fails_fast_without_models(self):
        self.client.get_object_info.return_value = object_info()
        out = await self.server.controlnet_generate("a city", controlnet="depth")
        self.assertIn("no models in models/controlnet", out[0].text)
        self.client.upload_image.assert_not_awaited()

    async def test_controlnet_rejects_uninstalled_model(self):
        self.client.get_object_info.return_value = object_info(controlnet=["other_controlnet.safetensors"])
        out = await self.server.controlnet_generate("a city", controlnet="depth")
        self.assertIn("is not installed", out[0].text)
        self.client.upload_image.assert_not_awaited()

    async def test_style_transfer_fails_fast_without_clip_vision(self):
        self.client.get_object_info.return_value = object_info(style=["flux1-redux-dev.safetensors"])
        out = await self.server.style_transfer_tool("same scene", str(self.style_path))
        self.assertIn("models/clip_vision", out[0].text)
        self.client.upload_image.assert_not_awaited()

    async def test_style_transfer_happy_path_builds_valid_workflow(self):
        info = object_info(style=["flux1-redux-dev.safetensors"],
                           clip_vision=["sigclip_vision_patch14_384.safetensors"])
        self.client.get_object_info.return_value = info
        self.orig_run = self.server.run_workflow
        self.server.run_workflow = AsyncMock(return_value=png_bytes())
        self.addCleanup(setattr, self.server, "run_workflow", self.orig_run)
        out = await self.server.style_transfer_tool("same scene", str(self.style_path))
        self.assertIn("Style transfer complete", out[0].text)
        workflow = self.server.run_workflow.await_args.args[0]
        by_type = {n["class_type"]: n for n in workflow.values()}
        self.assertEqual(by_type["CLIPVisionLoader"]["inputs"]["clip_name"],
                         "sigclip_vision_patch14_384.safetensors")
        self.assertEqual(by_type["StyleModelLoader"]["inputs"]["style_model_name"],
                         "flux1-redux-dev.safetensors")


if __name__ == "__main__":
    unittest.main()
