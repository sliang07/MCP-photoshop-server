import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from PIL import Image


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
        # Distilled flux2 is capped at 6 steps even when asked for more.
        self.assertEqual(by_type["BasicScheduler"]["inputs"]["steps"], 6)

    async def test_rejects_invalid_jobs_before_submission(self):
        out = await self.server.batch_generate_tool(jobs=[{"prompt": "  "}], export_dir=self.tmp.name)
        self.assertIn("Job 0 is invalid", out[0].text)
        self.client.batch_run_workflows.assert_not_awaited()

    async def test_rejects_unknown_model(self):
        out = await self.server.batch_generate_tool(jobs=[{"prompt": "x", "model": "sd15"}], export_dir=self.tmp.name)
        self.assertIn("must be 'flux2' or 'anima'", out[0].text)
        self.client.batch_run_workflows.assert_not_awaited()

    async def test_empty_job_list_rejected(self):
        out = await self.server.batch_generate_tool(jobs=[], export_dir=self.tmp.name)
        self.assertIn("at least one job", out[0].text)
        self.client.batch_run_workflows.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
