import asyncio
import base64
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
from PIL import Image

from comfy_client import ComfyUIClient
from editing import (build_edit_workflow, edit_profiles, make_edit_mask,
                     model_options, png_bytes, prepare_image, preview_content, register_editing_tools)
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
        "UNETLoader": ("unet_name", ["flux2-dev-nvfp4.safetensors", "qwen/qwen_image_2.1_int8_convrot.safetensors", "anima-aesthetic-v1.1.safetensors"]),
        "CLIPLoader": ("clip_name", ["mistral_3_small_flux2_fp8.safetensors", "qwen/qwen3vl_8b_int8_convrot.safetensors", "qwen_3_06b_base.safetensors"]),
        "VAELoader": ("vae_name", ["flux2-vae.safetensors", "qwen_image_2.1_vae_bf16.safetensors", "qwen_image_vae.safetensors"]),
    }
    for loader, (field, names) in models.items():
        data[loader] = {"input": {"required": {field: [names]}}}
    for name in ["LoadImage", "CLIPTextEncode", "VAEEncode", "ReferenceLatent", "BasicGuider",
                 "RandomNoise", "Flux2Scheduler", "KSamplerSelect", "EmptyFlux2LatentImage",
                 "SamplerCustomAdvanced", "VAEDecode", "SaveImage", "TextEncodeQwenImage21",
                 "QwenImage21Cache", "JoinImageWithAlpha", "KSampler", "EmptyLatentImage", "CFGGuider", "FluxGuidance"]:
        data[name] = {}
    return data


class EditingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = Registry()
        self.sessions = SessionManager()
        self.canvas = self.sessions.create("default", 64, 64, (10, 20, 30))
        self.client = AsyncMock()
        self.client.get_object_info.return_value = node_info()
        self.client.get_system_stats.return_value = {"system": {}, "devices": []}
        self.client.upload_image.return_value = "renamed/source.png"
        self.run = AsyncMock(return_value=png_bytes(Image.new("RGB", (64, 64), "red")))
        register_editing_tools(self.registry, self.client, self.sessions, self.run)

    async def test_capabilities_starts_comfyui_before_reading_models(self):
        async def read_models():
            self.client.start_comfyui.assert_awaited_once()
            return node_info()
        self.client.get_object_info.side_effect = read_models
        result = await self.registry.tools["get_editing_capabilities"]()
        report = json.loads(result[0].text)
        self.assertTrue(report["connected"])
        self.assertTrue(report["editing"]["qwen21"]["available"])
        self.run.assert_not_awaited()

    async def test_passive_capabilities_explains_automatic_startup(self):
        self.client.get_object_info.side_effect = httpx.ConnectError("offline")
        result = await self.registry.tools["get_editing_capabilities"](start_if_needed=False)
        report = json.loads(result[0].text)
        self.assertFalse(report["connected"])
        self.assertFalse(report["auto_start_attempted"])
        self.assertIn("automatically starts ComfyUI", report["message"])
        self.client.start_comfyui.assert_not_awaited()

    async def test_capabilities_reports_actual_startup_failure(self):
        self.client.start_comfyui.side_effect = RuntimeError("configured Python executable missing")
        result = await self.registry.tools["get_editing_capabilities"]()
        report = json.loads(result[0].text)
        self.assertFalse(report["connected"])
        self.assertTrue(report["auto_start_attempted"])
        self.assertEqual(report["error"], "configured Python executable missing")
        self.client.get_object_info.assert_not_awaited()

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
        async def change_document(workflow, **kwargs):
            self.canvas.add_layer("concurrent edit")
            return png_bytes(Image.new("RGB", (64, 64), "red"))
        self.run.side_effect = change_document
        with self.assertRaisesRegex(RuntimeError, "Canvas changed"):
            await self.registry.tools["edit_image"]("change")
        self.assertEqual(len(self.canvas.layers), 2)

    async def test_missing_model_fails_before_upload(self):
        info = node_info()
        info["UNETLoader"]["input"]["required"]["unet_name"] = [[]]
        self.client.get_object_info.return_value = info
        with self.assertRaisesRegex(ValueError, "unavailable"):
            await self.registry.tools["edit_image"]("change")
        self.client.upload_image.assert_not_called()

    async def test_uploaded_filename_is_used_in_graph(self):
        await self.registry.tools["edit_image"]("change")
        graph = self.run.call_args.args[0]
        loads = [n["inputs"]["image"] for n in graph.values() if n["class_type"] == "LoadImage"]
        self.assertEqual(loads, ["renamed/source.png"])

    async def test_retired_backends_are_not_silently_substituted(self):
        for backend in ("qwen", "qwen2511"):
            with self.assertRaisesRegex(ValueError, "retired"):
                await self.registry.tools["edit_image"]("change", backend=backend)
        self.client.upload_image.assert_not_called()

    def test_reference_images_are_connected_to_guider(self):
        profile = edit_profiles(node_info())["flux2"]
        graph = build_edit_workflow("flux2", profile, "edit", ["source.png", "ref.png"], 512, 512, 4, 0)
        classes = [n["class_type"] for n in graph.values()]
        self.assertNotIn("CFGGuider", classes)
        guider = next(n for n in graph.values() if n["class_type"] == "BasicGuider")

        conditioning = guider["inputs"]["conditioning"]
        for expected in ("ref.png", "source.png"):
            ref = graph[conditioning[0]]
            self.assertEqual(ref["class_type"], "ReferenceLatent")
            encoded = graph[ref["inputs"]["latent"][0]]
            loaded = graph[encoded["inputs"]["pixels"][0]]
            self.assertEqual(loaded["inputs"]["image"], expected)
            conditioning = ref["inputs"]["conditioning"]
        guided = graph[conditioning[0]]
        self.assertEqual(guided["class_type"], "FluxGuidance")
        self.assertEqual(guided["inputs"]["guidance"], profile["default_cfg"])

    def test_model_detection_handles_new_combo_and_nested_paths(self):
        info = {"Loader": {"input": {"required": {"model": ["COMBO", {"options": ["a.pth"]}]}}}}
        self.assertEqual(model_options(info, "Loader", "model"), ["a.pth"])
        self.assertTrue(edit_profiles(node_info())["qwen21"]["available"])
        self.assertNotIn("qwen2511", edit_profiles(node_info()))
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

    def test_qwen21_graph_uses_cache_and_unified_encoder(self):
        info = node_info()
        info["UNETLoader"]["input"]["required"]["unet_name"] = [["qwen_image_2.1_int8_convrot.safetensors"]]
        info["CLIPLoader"]["input"]["required"]["clip_name"] = [["qwen3vl_8b_int8_convrot.safetensors"]]
        info["VAELoader"]["input"]["required"]["vae_name"] = [["qwen_image_2.1_vae_bf16.safetensors"]]
        info["QwenImage21Cache"] = {}
        info["TextEncodeQwenImage21"] = {}
        profile = edit_profiles(info)["qwen21"]
        self.assertTrue(profile["available"], profile["missing"])
        self.assertEqual(profile["max_additional_references"], 15)
        self.assertEqual(profile["default_steps"], 30)
        graph = build_edit_workflow("qwen21", profile, "edit", ["source.png", "ref.png"], 512, 512, 30, 0)
        classes = [n["class_type"] for n in graph.values()]
        self.assertEqual(classes.count("QwenImage21Cache"), 1)
        # Official 2.1 path: no VAEEncode (the encoder emits the latent),
        # no legacy ModelSamplingAuraFlow/CFGNorm nodes.
        self.assertNotIn("VAEEncode", classes)
        self.assertNotIn("ModelSamplingAuraFlow", classes)
        self.assertNotIn("CFGNorm", classes)
        cache = next(n for n in graph.values() if n["class_type"] == "QwenImage21Cache")
        self.assertEqual(cache["inputs"]["device"], "auto")
        self.assertEqual(cache["inputs"]["dtype"], "default")
        key = next(k for k, n in graph.items() if n["class_type"] == "TextEncodeQwenImage21")
        encoder = graph[key]
        self.assertEqual(encoder["inputs"]["resolution"], 0)
        self.assertEqual(encoder["inputs"]["negative_prompt"], "")
        self.assertEqual(
            [graph[graph[encoder["inputs"][f"images.image_{i}"][0]]["inputs"]["image"][0]]["inputs"]["image"] for i in (1, 2)],
            ["source.png", "ref.png"])
        sampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
        self.assertEqual(sampler["inputs"]["positive"], [key, 0])
        self.assertEqual(sampler["inputs"]["negative"], [key, 1])
        self.assertEqual(sampler["inputs"]["latent_image"], [key, 2])
        self.assertEqual(sampler["inputs"]["cfg"], 3.0)
        self.assertEqual(sampler["inputs"]["sampler_name"], "euler")
        self.assertEqual(sampler["inputs"]["scheduler"], "simple")
        self.assertEqual(sampler["inputs"]["denoise"], 1.0)
        self.assertEqual(sampler["inputs"]["model"], [next(
            k for k, n in graph.items() if n["class_type"] == "QwenImage21Cache"), 0])

    async def test_qwen21_fails_before_upload_when_nodes_missing(self):
        info = node_info()
        del info["TextEncodeQwenImage21"]
        self.client.get_object_info.return_value = info
        with self.assertRaisesRegex(ValueError, "unavailable"):
            await self.registry.tools["edit_image"]("change", backend="qwen21")
        self.client.upload_image.assert_not_called()

    async def test_qwen21_rejects_more_references_than_node_sockets(self):
        paths = [f"r{i}.png" for i in range(16)]
        with self.assertRaisesRegex(ValueError, "at most 15"):
            await self.registry.tools["edit_image"]("change", backend="qwen21", reference_paths=paths)
        self.client.upload_image.assert_not_called()

    async def test_default_uses_qwen21_and_longer_timeout(self):
        result = await self.registry.tools["edit_image"]("change", max_side=2048)
        report = json.loads(result[0].text)
        self.assertEqual(report["backend"], "qwen21")
        self.assertEqual(report["steps"], 30)
        self.assertEqual(report["cfg"], 3.0)
        self.assertEqual(self.run.call_args.kwargs["timeout"], 1800)

    async def test_mask_is_last_reference_and_user_references_keep_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "reference.png"
            Image.new("RGBA", (64, 64), (0, 0, 255, 128)).save(ref)
            result = await self.registry.tools["edit_image"](
                "use the blue from <image2>", reference_paths=[str(ref)], region=[16, 16, 32, 32])
        uploads = self.client.upload_image.call_args_list
        self.assertEqual(len(uploads), 3)
        reference = Image.open(io.BytesIO(uploads[1].args[0]))
        self.assertEqual(reference.getpixel((0, 0)), (0, 0, 255, 128))
        mask = Image.open(io.BytesIO(uploads[2].args[0]))
        self.assertEqual(mask.getpixel((0, 0)), (0, 0, 0, 255))
        self.assertEqual(mask.getpixel((32, 32)), (255, 255, 255, 255))
        self.assertIn("<image3> is an edit mask", json.loads(result[0].text)["effective_prompt"])

    async def test_qwen_mask_without_references_still_uses_multi_image_tags(self):
        result = await self.registry.tools["edit_image"]("recolor the selected object", region=[16, 16, 32, 32])
        instruction = json.loads(result[0].text)["effective_prompt"]
        self.assertTrue(instruction.startswith("Edit <image1>."))
        self.assertIn("<image2> is an edit mask for <image1>", instruction)
        self.assertNotIn("\n", instruction)

    async def test_transparent_result_replaces_source_and_undo_restores_it(self):
        before = self.canvas.composite().tobytes()
        output = Image.new("RGBA", (64, 64), (0, 0, 255, 0))
        output.putpixel((32, 32), (0, 0, 255, 128))
        self.run.return_value = png_bytes(output)
        await self.registry.tools["edit_image"]("extract the subject with transparency")
        self.assertEqual(self.canvas.composite().getpixel((0, 0))[3], 0)
        self.assertEqual(self.canvas.composite().getpixel((32, 32)), (0, 0, 255, 128))
        self.assertFalse(self.canvas.layers[0].visible)
        self.canvas.undo()
        self.assertEqual(self.canvas.composite().tobytes(), before)
        self.canvas.redo()
        self.assertEqual(self.canvas.composite().getpixel((0, 0))[3], 0)

    async def test_masked_transparent_edit_preserves_outside(self):
        self.run.return_value = png_bytes(Image.new("RGBA", (64, 64), (0, 0, 0, 0)))
        await self.registry.tools["edit_image"]("remove", region=[16, 16, 32, 32])
        self.assertEqual(self.canvas.composite().getpixel((0, 0)), (10, 20, 30, 255))
        self.assertEqual(self.canvas.composite().getpixel((32, 32))[3], 0)

    async def test_unknown_session_does_not_edit_a_blank_canvas(self):
        with self.assertRaisesRegex(ValueError, "Open an image"):
            await self.registry.tools["edit_image"]("change", session_id="missing")
        self.client.upload_image.assert_not_called()
        self.assertIsNone(self.sessions.get("missing"))

    async def test_undo_edit_preserves_source_layer_opacity(self):
        self.canvas.layers[0].opacity = 0.5
        self.canvas._save_state()
        before = self.canvas.composite().tobytes()
        await self.registry.tools["edit_image"]("change")
        self.canvas.undo()
        self.assertEqual(self.canvas.composite().tobytes(), before)
        self.assertEqual(self.canvas.layers[0].opacity, 0.5)

    def test_qwen_image_size_matches_encoder_rounding_and_preserves_alpha(self):
        image = prepare_image(Image.new("RGBA", (1001, 701), (10, 20, 30, 128)), 1024)
        self.assertEqual(image.size, (992, 704))
        self.assertEqual(image.getpixel((0, 0))[3], 128)

    async def test_cfg_resolution_and_workflow_nodes(self):
        cases = [("flux2", None, 4.0, 50), ("flux2", 0.0, 0.0, 50), ("flux2", 1.25, 1.25, 50),
                 ("qwen21", None, 3.0, 30), ("qwen21", 0.0, 0.0, 30), ("qwen21", 1.25, 1.25, 30)]
        for backend, cfg, expected_cfg, expected_steps in cases:
            with self.subTest(backend=backend, cfg=cfg):
                result = await self.registry.tools["edit_image"]("test", backend=backend, cfg=cfg)
                report = json.loads(result[0].text)
                self.assertEqual(report["cfg"], expected_cfg)
                self.assertEqual(report["steps"], expected_steps)
                graph = self.run.call_args.args[0]
                if backend == "flux2":
                    guidance_node = next(n for n in graph.values() if n["class_type"] == "FluxGuidance")
                    self.assertEqual(guidance_node["inputs"]["guidance"], expected_cfg)
                else:
                    sampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
                    self.assertEqual(sampler["inputs"]["cfg"], expected_cfg)

    def test_build_edit_workflow_positional_resolution_compatibility(self):
        info = node_info()
        profile = edit_profiles(info)["qwen21"]
        original_profile = dict(profile)
        graph = build_edit_workflow("qwen21", profile, "edit", ["source.png"], 512, 512, 30, 0, 1024, cfg=0.75)
        encoder = next(n for n in graph.values() if n["class_type"] == "TextEncodeQwenImage21")
        self.assertEqual(encoder["inputs"]["resolution"], 1024)
        sampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
        self.assertEqual(sampler["inputs"]["cfg"], 0.75)
        graph_default = build_edit_workflow("qwen21", profile, "edit", ["source.png"], 512, 512, 30, 0)
        encoder_default = next(n for n in graph_default.values() if n["class_type"] == "TextEncodeQwenImage21")
        self.assertEqual(encoder_default["inputs"]["resolution"], 0)
        sampler_default = next(n for n in graph_default.values() if n["class_type"] == "KSampler")
        self.assertEqual(sampler_default["inputs"]["cfg"], 3.0)
        self.assertEqual(profile, original_profile)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = ComfyUIClient()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_start_cancels_idle_shutdown_before_health_check(self):
        self.client._schedule_idle_kill()
        timer = self.client._idle_kill_timer
        async def healthy():
            self.assertTrue(timer.cancelled())
            return True
        self.client.is_running = healthy
        with patch.object(self.client, "_start_process") as launch:
            await self.client.start_comfyui()
        launch.assert_not_called()

    async def test_stopped_comfyui_is_launched_and_waited_for(self):
        self.client.is_running = AsyncMock(side_effect=[False, True])
        self.client._wait_for_port_free = AsyncMock(return_value=True)
        with patch.object(self.client, "_start_process") as launch:
            await self.client.start_comfyui()
        launch.assert_called_once()
        self.assertEqual(self.client.is_running.await_count, 2)

    def test_comfyui_startup_output_cannot_corrupt_mcp_stdio(self):
        for python_available in (True, False):
            with patch("comfy_client.os.path.exists", return_value=python_available), \
                 patch("comfy_client.COMFYUI_START_CMD", "start_comfyui.bat"), \
                 patch("comfy_client.subprocess.Popen") as launch:
                self.client._start_process()
            for stream in ("stdin", "stdout", "stderr"):
                self.assertEqual(launch.call_args.kwargs[stream], subprocess.DEVNULL)

    async def test_cancelled_shutdown_does_not_log_a_spurious_error(self):
        task = asyncio.create_task(asyncio.sleep(60))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.client._on_idle_kill_done(task)

    async def test_status_starts_comfyui_before_reading_stats(self):
        import server
        client = AsyncMock()
        async def read_stats():
            client.start_comfyui.assert_awaited_once()
            return {"system": {"comfyui_version": "test"}}
        client.get_system_stats.side_effect = read_stats
        with patch.object(server, "comfy", client):
            result = await server.get_comfyui_status_tool()
        self.assertEqual(json.loads(result[0].text)["system"]["comfyui_version"], "test")

    async def test_passive_status_does_not_require_manual_start(self):
        import server
        client = AsyncMock()
        client.get_system_stats.side_effect = httpx.ConnectError("offline")
        with patch.object(server, "comfy", client):
            result = await server.get_comfyui_status_tool(start_if_needed=False)
        report = json.loads(result[0].text)
        self.assertFalse(report["auto_start_attempted"])
        self.assertIn("automatically starts ComfyUI", report["message"])
        client.start_comfyui.assert_not_awaited()

    async def test_status_reports_startup_error(self):
        import server
        client = AsyncMock()
        client.start_comfyui.side_effect = TimeoutError("startup timed out")
        with patch.object(server, "comfy", client):
            result = await server.get_comfyui_status_tool()
        report = json.loads(result[0].text)
        self.assertTrue(report["auto_start_attempted"])
        self.assertEqual(report["error"], "startup timed out")
        client.get_system_stats.assert_not_awaited()

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

    async def test_no_owned_handle_means_subprocess_run_never_called(self):
        self.client._comfyui_process = None
        with patch("comfy_client.subprocess.run") as mock_run:
            result = self.client._kill_process()
        self.assertFalse(result)
        mock_run.assert_not_called()

    async def test_live_owned_handle_causes_exactly_one_taskkill(self):
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_proc.wait = Mock()
        self.client._comfyui_process = mock_proc

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = b""

        with patch("comfy_client.subprocess.run", return_value=mock_result) as mock_run:
            result = self.client._kill_process()

        self.assertTrue(result)
        mock_run.assert_called_once_with(["taskkill", "/F", "/T", "/PID", "12345"], capture_output=True, timeout=15)
        mock_proc.wait.assert_called_once_with(timeout=5)

    async def test_exited_handle_is_not_targeted(self):
        mock_proc = Mock()
        mock_proc.poll.return_value = 0  # Exited
        self.client._comfyui_process = mock_proc

        with patch("comfy_client.subprocess.run") as mock_run:
            result = self.client._kill_process()

        self.assertFalse(result)
        mock_run.assert_not_called()

    async def test_failed_taskkill_retains_ownership(self):
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        self.client._comfyui_process = mock_proc

        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = b"error"

        with patch("comfy_client.subprocess.run", return_value=mock_result):
            result = self.client._kill_process()

        self.assertFalse(result)
        self.assertIsNotNone(self.client._comfyui_process)

    async def test_timed_out_wait_retains_ownership(self):
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="taskkill", timeout=5)
        self.client._comfyui_process = mock_proc

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = b""

        with patch("comfy_client.subprocess.run", return_value=mock_result):
            result = self.client._kill_process()

        self.assertFalse(result)
        self.assertIsNotNone(self.client._comfyui_process)

    async def test_foreign_occupied_port_blocks_launch_and_doesnt_kill(self):
        self.client.is_running = AsyncMock(return_value=False)
        self.client._wait_for_port_free = AsyncMock(return_value=False)
        self.client._comfyui_process = None

        with patch.object(self.client, "_start_process") as mock_start, \
             patch.object(self.client, "_kill_process") as mock_kill:
            with self.assertRaises(RuntimeError) as ctx:
                await self.client.start_comfyui()

        self.assertIn("foreign process", str(ctx.exception))
        mock_start.assert_not_called()
        mock_kill.assert_not_called()

    async def test_failed_owned_cleanup_blocks_start_even_if_port_becomes_free(self):
        self.client.is_running = AsyncMock(return_value=False)
        self.client._wait_for_port_free = AsyncMock(side_effect=[False, True])
        self.client._comfyui_process = Mock()
        self.client._comfyui_process.poll.return_value = None
        with patch.object(self.client, "_kill_process", return_value=False), \
             patch.object(self.client, "_start_process") as launch:
            with self.assertRaisesRegex(RuntimeError, "Could not stop"):
                await self.client.start_comfyui()
        launch.assert_not_called()
        self.assertIsNotNone(self.client._comfyui_process)

    async def test_queue_error_defers_idle_kill(self):
        self.client.get_queue_status = AsyncMock(side_effect=Exception("network error"))
        self.client.kill_comfyui = AsyncMock()

        await self.client._idle_kill_guarded()

        self.client.kill_comfyui.assert_not_awaited()
        self.assertIsNotNone(self.client._idle_kill_timer)

    async def test_malformed_queue_defers_idle_kill(self):
        self.client.get_queue_status = AsyncMock(return_value={"queue_running": "invalid"})
        self.client.kill_comfyui = AsyncMock()

        await self.client._idle_kill_guarded()

        self.client.kill_comfyui.assert_not_awaited()
        self.assertIsNotNone(self.client._idle_kill_timer)

    async def test_busy_queue_defers_idle_kill(self):
        self.client.get_queue_status = AsyncMock(return_value={"queue_running": ["job1"], "queue_pending": []})
        self.client.kill_comfyui = AsyncMock()

        await self.client._idle_kill_guarded()

        self.client.kill_comfyui.assert_not_awaited()
        self.assertIsNotNone(self.client._idle_kill_timer)

    async def test_known_empty_queue_permits_kill(self):
        self.client.get_queue_status = AsyncMock(return_value={"queue_running": [], "queue_pending": []})
        self.client.kill_comfyui = AsyncMock()

        await self.client._idle_kill_guarded()

        self.client.kill_comfyui.assert_awaited_once()
        self.assertIsNone(self.client._idle_kill_timer)


if __name__ == "__main__":
    unittest.main()
