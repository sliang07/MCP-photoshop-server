import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from PIL import Image
from test_editing import node_info


def png_bytes(size=(16, 16), color=(200, 30, 60)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class BatchGenerateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import server
        self.server = server
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.client = AsyncMock()
        self.client.get_object_info.return_value = node_info()
        self.client.batch_run_workflows = AsyncMock(return_value=[
            {"history": {"_cached_file_bytes": png_bytes()}, "error": None},
            {"history": None, "error": "execution_error: out of VRAM"},
        ])
        self.orig_client = server.comfy
        server.comfy = self.client
        self.addCleanup(setattr, self.server, "comfy", self.orig_client)

    async def test_batch_exports_each_result(self):
        out = await self.server.batch_generate_tool(
            jobs=[{"prompt": "a red dot", "width": 16, "height": 16, "steps": 4, "filename": "red_dot"},
                  {"prompt": "a blue dot", "width": 16, "height": 16, "steps": 4}],
            export_dir=self.tmp.name)
        report = json.loads(out[0].text)
        self.assertEqual(report["summary"]["jobs"], 2)
        self.assertEqual(report["summary"]["ok"], 1)
        self.assertEqual(report["summary"]["failed"], 1)
        first, second = report["results"]
        self.assertEqual(first["status"], "ok")
        self.assertTrue(Path(first["file"]).is_file())
        self.assertEqual(Path(first["file"]).name, "red_dot.png")
        self.assertEqual(second["status"], "failed")
        self.assertIn("out of VRAM", second["error"])
        workflows = self.client.batch_run_workflows.await_args.args[0]
        self.assertEqual(len(workflows), 2)
        self.assertEqual(workflows[0]["class_type"] if False else sorted(n["class_type"] for n in workflows[0].values()),
                         sorted(n["class_type"] for n in workflows[1].values()))

    async def test_flux2_jobs_use_flux2_conventions(self):
        self.client.batch_run_workflows.return_value = [
            {"history": {"_cached_file_bytes": png_bytes()}, "error": None}]
        await self.server.batch_generate_tool(
            jobs=[{"prompt": "x", "width": 16, "height": 16, "steps": 30}],
            export_dir=self.tmp.name)
        workflows = self.client.batch_run_workflows.await_args.args[0]
        by_type = {n["class_type"]: n for n in workflows[0].values()}
        self.assertIn("EmptyFlux2LatentImage", by_type)
        self.assertEqual(by_type["EmptyFlux2LatentImage"]["inputs"]["width"], 16)
        self.assertEqual(by_type["EmptyFlux2LatentImage"]["inputs"]["height"], 16)
        self.assertEqual(by_type["Flux2Scheduler"]["inputs"]["steps"], 30)

    async def test_rejects_invalid_jobs_before_submission(self):
        out = await self.server.batch_generate_tool(jobs=[{"prompt": "  "}], export_dir=self.tmp.name)
        self.assertIn("Job 0 is invalid", out[0].text)
        self.client.batch_run_workflows.assert_not_awaited()

    async def test_rejects_unknown_model(self):
        out = await self.server.batch_generate_tool(jobs=[{"prompt": "x", "model": "sd15"}], export_dir=self.tmp.name)
        self.assertIn("must be 'flux2', 'qwen21', 'anima' or 'minimax_h3'", out[0].text)
        self.client.batch_run_workflows.assert_not_awaited()

    async def test_empty_job_list_rejected(self):
        out = await self.server.batch_generate_tool(jobs=[], export_dir=self.tmp.name)
        self.assertIn("at least one job", out[0].text)
        self.client.batch_run_workflows.assert_not_awaited()

    async def test_repeated_calls_preserve_original_bytes(self):
        red_png = png_bytes(color=(200, 30, 60))
        blue_png = png_bytes(color=(30, 60, 200))
        self.client.batch_run_workflows.return_value = [
            {"history": {"_cached_file_bytes": red_png}, "error": None}]
        await self.server.batch_generate_tool(
            jobs=[{"prompt": "a", "width": 16, "height": 16, "steps": 4, "filename": "same"}],
            export_dir=self.tmp.name)
        first_path = Path(self.tmp.name) / "same.png"
        self.assertTrue(first_path.is_file())
        first_bytes = first_path.read_bytes()
        self.assertEqual(first_bytes, red_png)
        self.client.batch_run_workflows.return_value = [
            {"history": {"_cached_file_bytes": blue_png}, "error": None}]
        await self.server.batch_generate_tool(
            jobs=[{"prompt": "b", "width": 16, "height": 16, "steps": 4, "filename": "same"}],
            export_dir=self.tmp.name)
        second_path = Path(self.tmp.name) / "same_1.png"
        self.assertTrue(second_path.is_file())
        self.assertEqual(first_path.read_bytes(), red_png)
        self.assertEqual(second_path.read_bytes(), blue_png)
        self.assertNotEqual(first_path.read_bytes(), second_path.read_bytes())

    async def test_duplicate_basenames_in_one_batch(self):
        self.client.batch_run_workflows.return_value = [
            {"history": {"_cached_file_bytes": png_bytes()}, "error": None},
            {"history": {"_cached_file_bytes": png_bytes()}, "error": None}]
        await self.server.batch_generate_tool(
            jobs=[{"prompt": "a", "width": 16, "height": 16, "steps": 4, "filename": "dup"},
                  {"prompt": "b", "width": 16, "height": 16, "steps": 4, "filename": "dup"}],
            export_dir=self.tmp.name)
        self.assertTrue((Path(self.tmp.name) / "dup.png").is_file())
        self.assertTrue((Path(self.tmp.name) / "dup_1.png").is_file())

    async def test_unsupported_format_rejected_before_startup(self):
        out = await self.server.batch_generate_tool(
            jobs=[{"prompt": "x", "width": 16, "height": 16, "steps": 4}],
            export_dir=self.tmp.name,
            export_format="GIF")
        self.assertIn("export_format must be PNG, JPG, or JPEG", out[0].text)
        self.client.start_comfyui.assert_not_awaited()
        self.client.batch_run_workflows.assert_not_awaited()

    async def test_filename_sanitization_keeps_output_inside_export_dir(self):
        self.client.batch_run_workflows.return_value = [
            {"history": {"_cached_file_bytes": png_bytes()}, "error": None}]
        out = await self.server.batch_generate_tool(
            jobs=[{"prompt": "x", "width": 16, "height": 16, "steps": 4, "filename": "../../etc/passwd"}],
            export_dir=self.tmp.name)
        expected = Path(json.loads(out[0].text)["results"][0]["file"])
        self.assertTrue(expected.is_file())
        self.assertEqual(expected.parent, Path(self.tmp.name))
        self.assertNotIn("/", expected.name)
        self.assertNotIn("\\", expected.name)

    async def test_backslash_filename_sanitization(self):
        self.client.batch_run_workflows.return_value = [
            {"history": {"_cached_file_bytes": png_bytes()}, "error": None}]
        out = await self.server.batch_generate_tool(
            jobs=[{"prompt": "x", "width": 16, "height": 16, "steps": 4, "filename": "C:\\Windows\\System32"}],
            export_dir=self.tmp.name)
        expected = Path(json.loads(out[0].text)["results"][0]["file"])
        self.assertTrue(expected.is_file())
        self.assertEqual(expected.parent, Path(self.tmp.name))
        self.assertNotIn("/", expected.name)
        self.assertNotIn("\\", expected.name)

    async def test_jpg_format_accepted_and_saves_as_jpeg(self):
        self.client.batch_run_workflows.return_value = [
            {"history": {"_cached_file_bytes": png_bytes()}, "error": None}]
        await self.server.batch_generate_tool(
            jobs=[{"prompt": "x", "width": 16, "height": 16, "steps": 4, "filename": "test"}],
            export_dir=self.tmp.name,
            export_format="jpg")
        expected = Path(self.tmp.name) / "test.jpg"
        self.assertTrue(expected.is_file())
        with Image.open(expected) as img:
            self.assertEqual(img.format, "JPEG")

    async def test_jpeg_format_accepted_and_saves_as_jpeg(self):
        self.client.batch_run_workflows.return_value = [
            {"history": {"_cached_file_bytes": png_bytes()}, "error": None}]
        await self.server.batch_generate_tool(
            jobs=[{"prompt": "x", "width": 16, "height": 16, "steps": 4, "filename": "test"}],
            export_dir=self.tmp.name,
            export_format="jpeg")
        expected = Path(self.tmp.name) / "test.jpg"
        self.assertTrue(expected.is_file())
        with Image.open(expected) as img:
            self.assertEqual(img.format, "JPEG")


if __name__ == "__main__":
    unittest.main()
