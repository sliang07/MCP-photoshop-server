import asyncio
import base64
import io
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image

from comfy_client import ComfyUIClient
from editing import (build_edit_workflow, edit_profiles, make_edit_mask,
                     model_options, png_bytes, preview_content, register_editing_tools)
from session import SessionManager


class Registry:
    def __init__(self):
        self.tools = {}

    def tool(self, name):
        def register(function):
            self.tools[name] = function
            return function
        return register


def node_info():
    data = {}
    models = {
        "UNETLoader": ("unet_name", ["flux-2-klein-9b.safetensors", "qwen/qwen_image_edit_fp8_e4m3fn.safetensors"]),
        "CLIPLoader": ("clip_name", ["qwen_3_8b_fp8mixed.safetensors", "qwen/qwen_2.5_vl_7b_fp8_scaled.safetensors"]),
        "VAELoader": ("vae_name", ["flux2-vae.safetensors", "qwen_image_vae.safetensors"]),
    }
    for loader, (field, names) in models.items():
        data[loader] = {"input": {"required": {field: [names]}}}
    for name in ["LoadImage", "CLIPTextEncode", "VAEEncode", "ReferenceLatent", "BasicGuider",
                 "RandomNoise", "Flux2Scheduler", "KSamplerSelect", "EmptyFlux2LatentImage",
                 "SamplerCustomAdvanced", "VAEDecode", "SaveImage", "TextEncodeQwenImageEdit",
                 "ModelSamplingAuraFlow", "CFGNorm", "KSampler"]:
        data[name] = {}
    return data


class EditingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = Registry()
        self.sessions = SessionManager()
        self.canvas = self.sessions.create("default", 64, 64, (10, 20, 30))
        self.client = AsyncMock()
        self.client.get_object_info.return_value = node_info()
        self.client.upload_image.return_value = "renamed/source.png"
        self.run = AsyncMock(return_value=png_bytes(Image.new("RGB", (64, 64), "red")))
        register_editing_tools(self.registry, self.client, self.sessions, self.run)

    async def test_masked_edit_preserves_outside_and_undo_redo(self):
        original = self.canvas.composite().tobytes()
        result = await self.registry.tools["edit_image"]("red patch", region=[16, 16, 32, 32], feather=3, seed=0)
        edited = self.canvas.composite()
        for y in range(64):
            for x in range(64):
                if not (16 <= x < 48 and 16 <= y < 48):
                    self.assertEqual(edited.getpixel((x, y)), (10, 20, 30, 255))
        self.assertEqual(edited.getpixel((32, 32)), (255, 0, 0, 255))
        self.assertEqual(json.loads(result[0].text)["seed"], 0)
        self.assertEqual(result[1].type, "image")
        self.assertTrue(self.canvas.undo())
        self.assertEqual(self.canvas.composite().tobytes(), original)
        self.assertTrue(self.canvas.redo())
        self.assertEqual(self.canvas.composite().tobytes(), edited.tobytes())

    async def test_failed_inference_leaves_canvas_untouched(self):
        self.run.return_value = None
        before = self.canvas.composite().tobytes()
        with self.assertRaises(RuntimeError):
            await self.registry.tools["edit_image"]("change")
        self.assertEqual(len(self.canvas.layers), 1)
        self.assertEqual(self.canvas.composite().tobytes(), before)

    async def test_stale_result_does_not_modify_changed_document(self):
        async def change_document(workflow):
            self.canvas.add_layer("concurrent edit")
            return png_bytes(Image.new("RGB", (64, 64), "red"))
        self.run.side_effect = change_document
        with self.assertRaisesRegex(RuntimeError, "Canvas changed"):
            await self.registry.tools["edit_image"]("change")
        self.assertEqual(len(self.canvas.layers), 2)

    async def test_missing_model_fails_before_upload(self):
        with self.assertRaisesRegex(ValueError, "unavailable"):
            await self.registry.tools["edit_image"]("change", backend="qwen2511")
        self.client.upload_image.assert_not_called()

    async def test_uploaded_filename_is_used_in_graph(self):
        await self.registry.tools["edit_image"]("change")
        graph = self.run.call_args.args[0]
        loads = [n["inputs"]["image"] for n in graph.values() if n["class_type"] == "LoadImage"]
        self.assertEqual(loads, ["renamed/source.png"])

    async def test_legacy_qwen_rejects_additional_references(self):
        with self.assertRaisesRegex(ValueError, "at most 0"):
            await self.registry.tools["edit_image"]("change", backend="qwen", reference_paths=["unused.png"])
        self.client.upload_image.assert_not_called()

    def test_reference_images_are_connected_to_guider(self):
        profile = edit_profiles(node_info())["flux2"]
        graph = build_edit_workflow("flux2", profile, "edit", ["source.png", "ref.png"], 512, 512, 4, 0)
        guider = next(n for n in graph.values() if n["class_type"] == "BasicGuider")
        conditioning = guider["inputs"]["conditioning"]
        for expected in ("ref.png", "source.png"):
            ref = graph[conditioning[0]]
            self.assertEqual(ref["class_type"], "ReferenceLatent")
            encoded = graph[ref["inputs"]["latent"][0]]
            loaded = graph[encoded["inputs"]["pixels"][0]]
            self.assertEqual(loaded["inputs"]["image"], expected)
            conditioning = ref["inputs"]["conditioning"]

    def test_model_detection_handles_new_combo_and_nested_paths(self):
        info = {"Loader": {"input": {"required": {"model": ["COMBO", {"options": ["a.pth"]}]}}}}
        self.assertEqual(model_options(info, "Loader", "model"), ["a.pth"])
        self.assertTrue(edit_profiles(node_info())["qwen"]["available"])
        broken = node_info()
        del broken["ReferenceLatent"]
        self.assertFalse(edit_profiles(broken)["flux2"]["available"])

    def test_preview_is_valid_png_and_bounded(self):
        preview = preview_content(Image.new("RGB", (2048, 1024)), 512)
        decoded = Image.open(io.BytesIO(base64.b64decode(preview.data)))
        self.assertEqual(decoded.size, (512, 256))

    def test_mask_rejects_outside_and_empty_regions(self):
        for region in ([0, 0, 0, 10], [-1, 0, 10, 10], [50, 50, 20, 20]):
            with self.assertRaises(ValueError):
                make_edit_mask((64, 64), region=region)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = ComfyUIClient()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_binary_previews_and_node_events_do_not_finish_workflow(self):
        ws = AsyncMock()
        ws.recv.side_effect = [b"preview", json.dumps({"type": "executed", "data": {"prompt_id": "job"}}),
                               json.dumps({"type": "executing", "data": {"prompt_id": "job", "node": None}})]
        self.client.get_history = AsyncMock(return_value={"outputs": {"save": {"images": []}}})
        result = await self.client._listen_for_completion(ws, "job", timeout=5)
        self.assertIn("outputs", result)
        self.assertEqual(ws.recv.await_count, 3)

    async def test_timeout_never_resubmits_prompt(self):
        self.client.start_comfyui = AsyncMock()
        self.client.submit_workflow = AsyncMock(return_value="job")
        self.client._listen_for_completion = AsyncMock(side_effect=TimeoutError("timeout"))
        ws = AsyncMock()
        with patch("comfy_client.websockets.connect", AsyncMock(return_value=ws)), patch("comfy_client.COMFYUI_AUTO_KILL", False):
            with self.assertRaises(TimeoutError):
                await self.client.run_workflow_and_wait({})
        self.client.submit_workflow.assert_awaited_once()

    async def test_websocket_fallback_submits_once(self):
        self.client.start_comfyui = AsyncMock()
        self.client.submit_workflow = AsyncMock(return_value="job")
        self.client._wait_via_polling = AsyncMock(return_value={"outputs": {}})
        self.client._wait_for_queue_drain = AsyncMock()
        with patch("comfy_client.websockets.connect", AsyncMock(side_effect=OSError("offline"))), patch("comfy_client.COMFYUI_AUTO_KILL", False):
            await self.client.run_workflow_and_wait({})
        self.client.submit_workflow.assert_awaited_once()
        self.client._wait_via_polling.assert_awaited_once()

    async def test_history_reports_execution_errors(self):
        await self.client.session.aclose()
        def respond(request):
            return httpx.Response(200, json={"job": {"status": {"status_str": "error", "messages": ["bad model"]}}})
        self.client.session = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with self.assertRaisesRegex(RuntimeError, "bad model"):
            await self.client.get_history("job")

    async def test_is_running_rejects_foreign_listener_on_port(self):
        """T9.x regression: a non-ComfyUI HTTP listener answering 200 with a
        non-JSON body must not be mistaken for a running ComfyUI."""
        await self.client.session.aclose()
        self.client.session = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"stub")))
        self.assertFalse(await self.client.is_running())

    async def test_is_running_rejects_json_that_is_not_an_object(self):
        await self.client.session.aclose()
        self.client.session = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=["not", "an", "object"])))
        self.assertFalse(await self.client.is_running())

    async def test_is_running_accepts_comfyui_history(self):
        await self.client.session.aclose()
        self.client.session = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"prompt": {}, "queue": []})))
        self.assertTrue(await self.client.is_running())

    async def test_is_running_accepts_fresh_empty_history(self):
        """A brand-new ComfyUI returns {} from /history; that must count as running,
        otherwise start_comfyui would kill the healthy instance in a boot loop."""
        await self.client.session.aclose()
        self.client.session = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={})))
        self.assertTrue(await self.client.is_running())

    async def test_upload_uses_actual_server_name(self):
        await self.client.session.aclose()
        self.client.start_comfyui = AsyncMock()
        self.client.session = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"name": "renamed.png", "subfolder": "edits"})))
        self.assertEqual(await self.client.upload_image(b"test", "source.png"), "edits/renamed.png")


if __name__ == "__main__":
    unittest.main()
