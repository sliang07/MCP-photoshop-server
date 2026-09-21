import json
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

import server
from editing import model_profiles, png_bytes
from session import SessionManager
from test_editing import node_info


class ModelSelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = AsyncMock()
        self.client.get_object_info.return_value = node_info()
        self.sessions = SessionManager()
        self.image = png_bytes(Image.new("RGBA", (64, 32), (30, 50, 70, 128)))
        self.run = AsyncMock(return_value=self.image)
        for name, value in (("comfy", self.client), ("sessions", self.sessions), ("run_workflow", self.run)):
            p = patch.object(server, name, value)
            p.start()
            self.addCleanup(p.stop)

    async def test_generation_switches_weights_encoders_and_defaults(self):
        expected = {
            "flux2": ("flux2-dev-nvfp4.safetensors", "flux2", 50, 4.0),
            "qwen21": ("qwen/qwen_image_2.1_int8_convrot.safetensors", "qwen_image", 30, 3.0),
            "anima": ("anima-aesthetic-v1.1.safetensors", "stable_diffusion", 30, 4.0),
        }
        for model, (checkpoint, encoder, steps, cfg) in expected.items():
            with self.subTest(model=model):
                result = await server.generate_image("a bird", model=model, width=64, height=32, seed=0)
                self.assertIn(f"Model: {model}", result[0].text)
                graph = self.run.await_args.args[0]
                nodes = {n["class_type"]: n["inputs"] for n in graph.values()}
                self.assertEqual(nodes["UNETLoader"]["unet_name"], checkpoint)
                self.assertEqual(nodes["CLIPLoader"]["type"], encoder)
                sampler = nodes["Flux2Scheduler"] if model == "flux2" else nodes["KSampler"]
                self.assertEqual(sampler["steps"], steps)
                self.assertEqual(nodes["FluxGuidance"]["guidance"] if model == "flux2" else sampler["cfg"], cfg)
                self.assertEqual(nodes["RandomNoise"]["noise_seed"] if model == "flux2" else sampler["seed"], 0)
                self.assertNotIn("Flux2KleinSectionedEncoder", nodes)
                if model == "qwen21":
                    self.assertNotIn("LoadImage", nodes)
                    self.assertEqual(nodes["EmptyLatentImage"]["width"], 64)
                    self.assertEqual(nodes["EmptyLatentImage"]["height"], 32)
                    self.assertEqual(self.run.await_args.kwargs["timeout"], 1800)
                if model == "flux2":
                    self.assertEqual(nodes["CLIPLoader"]["clip_name"], "mistral_3_small_flux2_fp8.safetensors")
                    self.assertIn("BasicGuider", nodes)
                    self.assertNotIn("CFGGuider", nodes)
                    self.assertEqual(self.run.await_args.kwargs["timeout"], 1800)
                self.assertEqual(self.sessions.get("default").layers[-1].image.getpixel((0, 0))[3], 128)

    async def test_new_canvas_keeps_generated_transparency(self):
        await server.generate_image("transparent bird", model="qwen21", session_id="alpha")
        self.assertEqual(self.sessions.get("alpha").composite().getpixel((0, 0))[3], 128)

    async def test_explicit_sampling_settings_and_negative_prompt_survive(self):
        for model in ("flux2", "qwen21", "anima"):
            with self.subTest(model=model):
                await server.generate_image("bird", model=model, steps=9, cfg=2.3,
                                            negative_prompt="watermark", timeout=42)
                nodes = {n["class_type"]: n["inputs"] for n in self.run.await_args.args[0].values()}
                self.assertEqual(self.run.await_args.kwargs["timeout"], 42)
                sampler = nodes["Flux2Scheduler"] if model == "flux2" else nodes["KSampler"]
                self.assertEqual(sampler["steps"], 9)
                self.assertEqual(nodes["FluxGuidance"]["guidance"] if model == "flux2" else sampler["cfg"], 2.3)
                if model == "qwen21":
                    self.assertEqual(nodes["TextEncodeQwenImage21"]["negative_prompt"], "watermark")
                else:
                    if model == "flux2":
                        self.assertEqual(nodes["CLIPTextEncode"]["text"], "bird")
                    else:
                        self.assertEqual(nodes["CLIPTextEncode"]["text"], "watermark")

    async def test_unknown_and_unavailable_models_never_fall_back(self):
        out = await server.generate_image("bird", model="typo")
        self.assertIn("model must be", out[0].text)
        self.client.start_comfyui.assert_not_awaited()
        info = node_info()
        del info["TextEncodeQwenImage21"]
        self.client.get_object_info.return_value = info
        out = await server.generate_image("bird", model="qwen21")
        self.assertIn("qwen21 is unavailable", out[0].text)
        self.run.assert_not_awaited()
        self.assertIsNone(self.sessions.get("default"))

    def test_availability_is_specific_to_the_task(self):
        info = node_info()
        del info["ReferenceLatent"]
        del info["JoinImageWithAlpha"]
        self.assertTrue(model_profiles(info, "generation")["flux2"]["available"])
        self.assertTrue(model_profiles(info, "generation")["qwen21"]["available"])
        self.assertFalse(model_profiles(info, "editing")["flux2"]["available"])
        self.assertFalse(model_profiles(info, "editing")["qwen21"]["available"])
        self.assertNotIn("anima", model_profiles(info, "editing"))
        self.assertNotIn("anima", model_profiles(info, "outpaint"))

    async def test_mixed_batch_reports_each_model_and_preserves_png_alpha(self):
        self.client.batch_run_workflows.return_value = [
            {"history": {"_cached_file_bytes": self.image}, "error": None} for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            out = await server.batch_generate_tool(
                [{"prompt": "bird", "model": m, "seed": 0} for m in ("flux2", "qwen21", "anima")],
                export_dir=tmp)
            report = json.loads(out[0].text)
            self.assertEqual(report["summary"]["ok"], 3)
            self.assertEqual([r["steps"] for r in report["results"]], [50, 30, 30])
            self.assertEqual([r["cfg"] for r in report["results"]], [4.0, 3.0, 4.0])
            for record in report["results"]:
                with Image.open(record["file"]) as img:
                    self.assertEqual(img.getpixel((0, 0))[3], 128)
        self.client.get_object_info.assert_awaited_once()
        self.assertEqual(len(self.client.batch_run_workflows.await_args.args[0]), 3)

    async def test_batch_preflights_every_model_before_submitting_any(self):
        info = node_info()
        del info["TextEncodeQwenImage21"]
        self.client.get_object_info.return_value = info
        out = await server.batch_generate_tool([
            {"prompt": "bird", "model": "flux2"}, {"prompt": "bird", "model": "qwen21"}])
        self.assertIn("Job 1: qwen21 is unavailable", out[0].text)
        self.client.batch_run_workflows.assert_not_awaited()

    async def test_default_flux_batch_uses_dev_timeout(self):
        self.client.batch_run_workflows.return_value = [{"history": None, "error": "test"}] * 2
        with tempfile.TemporaryDirectory() as tmp:
            await server.batch_generate_tool([{"prompt": "one"}, {"prompt": "two", "model": "flux2"}], export_dir=tmp)
        self.assertEqual(self.client.batch_run_workflows.await_args.kwargs["timeout"], 3600)

    async def test_all_failed_batch_returns_errors(self):
        self.client.batch_run_workflows.return_value = [{"history": None, "error": "out of VRAM"}]
        with tempfile.TemporaryDirectory() as tmp:
            result = await server.batch_generate_tool([{"prompt": "bird"}], export_dir=tmp)
        report = json.loads(result[0].text)
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(report["results"][0]["error"], "out of VRAM")


if __name__ == "__main__":
    unittest.main()
