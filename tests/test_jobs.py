import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

from comfy_client import ComfyUIClient
from editing import png_bytes
from jobs import GenerationJobs
import server
from test_editing import node_info


class GenerationJobTests(unittest.IsolatedAsyncioTestCase):
    async def test_progress_input_snapshot_and_result_snapshot(self):
        entered_second = asyncio.Event()
        release = asyncio.Event()
        seen = []

        async def run_item(item, index, *options):
            seen.append((item["prompt"], index))
            if index == 1:
                entered_second.set()
                await release.wait()
            return {"status": "ok", "file": f"{index}.png"}

        manager = GenerationJobs(run_item)
        inputs = [{"prompt": "first"}, {"prompt": "second"}]
        report = manager.submit(inputs)
        tasks = list(manager._tasks.values())
        self.assertEqual(report["status"], "queued")
        inputs[0]["prompt"] = "changed after submit"
        await asyncio.wait_for(entered_second.wait(), 2)
        status = manager.status(report["job_id"])
        self.assertEqual((status["finished"], status["current_index"]), (1, 1))
        self.assertEqual(status["results"][0]["file"], "0.png")
        status["results"][0]["file"] = "changed by caller"
        self.assertEqual(manager.status(report["job_id"])["results"][0]["file"], "0.png")
        self.assertNotIn("results", manager.list_jobs()[0])
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        self.assertEqual(seen, [("first", 0), ("second", 1)])
        self.assertEqual(manager.status(report["job_id"])["status"], "completed")
        self.assertEqual(manager._tasks, {})

    async def test_cancel_current_job_finishes_image_and_does_not_cancel_other_job(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        seen = []

        async def run_item(item, index, *options):
            seen.append(item["prompt"])
            if item["prompt"] == "current":
                entered.set()
                await release.wait()
            return {"status": "ok", "file": item["prompt"] + ".png"}

        manager = GenerationJobs(run_item)
        first = manager.submit([{"prompt": "current"}, {"prompt": "skip"}])["job_id"]
        await asyncio.wait_for(entered.wait(), 2)
        second = manager.submit([{"prompt": "other"}])["job_id"]
        tasks = list(manager._tasks.values())
        self.assertEqual(manager.cancel(first)["status"], "cancelling")
        self.assertEqual(manager.status(second)["status"], "queued")
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        self.assertEqual(seen, ["current", "other"])
        self.assertEqual(manager.status(first)["status"], "cancelled")
        self.assertEqual(manager.status(first)["finished"], 1)
        self.assertEqual(manager.cancel(first)["status"], "cancelled")
        self.assertEqual(manager.status(second)["status"], "completed")

    async def test_cancel_queued_job_never_calls_runner(self):
        runner = AsyncMock()
        manager = GenerationJobs(runner)
        job_id = manager.submit([{"prompt": "unused"}])["job_id"]
        tasks = list(manager._tasks.values())
        self.assertEqual(manager.cancel(job_id)["status"], "cancelled")
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        runner.assert_not_awaited()
        self.assertEqual(manager.status(job_id)["finished"], 0)

    async def test_cancelled_request_does_not_cancel_owned_worker(self):
        request_ready = asyncio.Event()
        release = asyncio.Event()
        submitted = {}

        async def run_item(*arguments):
            await release.wait()
            return {"status": "ok", "file": "completed.png"}

        manager = GenerationJobs(run_item)

        async def request():
            submitted.update(manager.submit([{"prompt": "continue"}]))
            request_ready.set()
            await asyncio.Future()

        request_task = asyncio.create_task(request())
        await asyncio.wait_for(request_ready.wait(), 2)
        workers = list(manager._tasks.values())
        request_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request_task
        release.set()
        await asyncio.wait_for(asyncio.gather(*workers), 2)
        self.assertEqual(manager.status(submitted["job_id"])["status"], "completed")

    async def test_cancelled_lock_waiter_does_not_release_running_jobs_lock(self):
        entered = asyncio.Event()

        async def run_item(*arguments):
            entered.set()
            await asyncio.Future()

        manager = GenerationJobs(run_item)
        first = manager.submit([{}])["job_id"]
        await asyncio.wait_for(entered.wait(), 2)
        running = manager._tasks[first]
        second = manager.submit([{}])["job_id"]
        waiting = manager._tasks[second]
        await asyncio.sleep(0)
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        self.assertTrue(manager._lock.locked())
        running.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await running
        self.assertFalse(manager._lock.locked())
        self.assertEqual(manager.status(first)["status"], "interrupted")
        self.assertEqual(manager.cancel(second)["status"], "interrupted")

    async def test_failure_is_recorded_and_next_image_runs(self):
        runner = AsyncMock(side_effect=[OSError("failed image"), {"status": "ok", "file": "second.png"}])
        manager = GenerationJobs(runner)
        job_id = manager.submit([{}, {}])["job_id"]
        await asyncio.wait_for(asyncio.gather(*list(manager._tasks.values())), 2)
        report = manager.status(job_id)
        self.assertEqual(report["status"], "completed_with_errors")
        self.assertEqual(report["finished"], 2)
        self.assertEqual(report["results"][0]["error"], "failed image")
        self.assertEqual(report["results"][1]["index"], 1)
        self.assertEqual(manager.cancel(job_id), report)
        with self.assertRaises(ValueError):
            manager.status("unknown")


class GenerationJobIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_job_exports_first_image_while_second_is_running(self):
        client = AsyncMock()
        client.get_object_info.return_value = node_info()
        image_bytes = png_bytes(Image.new("RGBA", (16, 16), (10, 20, 30, 128)))
        second_started = asyncio.Event()
        release_second = asyncio.Event()
        count = 0

        async def generate(workflows, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                second_started.set()
                await release_second.wait()
            return [{"history": {"_cached_file_bytes": image_bytes}, "error": None}]

        client.batch_run_workflows.side_effect = generate
        manager = GenerationJobs(server._generate_job_item)
        with tempfile.TemporaryDirectory() as directory, patch.object(server, "comfy", client), patch.object(server, "generation_jobs", manager):
            response = await server.app.call_tool("submit_generation_job", {
                "jobs": [{"prompt": "one"}, {"prompt": "two"}], "export_dir": directory})
            job_id = json.loads(response[0].text)["job_id"]
            tasks = list(manager._tasks.values())
            await asyncio.wait_for(second_started.wait(), 2)
            response = await server.app.call_tool("get_job_status", {"job_id": job_id})
            partial = json.loads(response[0].text)
            self.assertEqual(partial["finished"], 1)
            self.assertEqual(Path(partial["results"][0]["file"]).read_bytes(), image_bytes)
            self.assertEqual(len(list(Path(directory).glob("*.png"))), 1)
            release_second.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 2)
            self.assertEqual(manager.status(job_id)["status"], "completed")
            self.assertEqual(len(list(Path(directory).glob("*.png"))), 2)
            client.kill_comfyui.assert_not_awaited()

    async def test_completed_batch_does_not_wait_for_unrelated_queue_items(self):
        client = ComfyUIClient()
        self.addAsyncCleanup(client.close)
        client.start_comfyui = AsyncMock()
        client.submit_workflow = AsyncMock(return_value="own-prompt")
        client.get_queue_status = AsyncMock(return_value={"queue_running": ["other-client"], "queue_pending": []})
        client._schedule_idle_kill = Mock()
        client._listen_for_many = AsyncMock(return_value={0: {"outputs": {"1": {"images": [{"filename": "done.png"}]}}}})
        client.get_output_file = AsyncMock(return_value=b"completed image bytes")
        with patch("comfy_client.websockets.connect", AsyncMock(return_value=AsyncMock())), patch("comfy_client.COMFYUI_AUTO_KILL", True):
            result = await client.batch_run_workflows([{"1": {"class_type": "SaveImage"}}])
        self.assertEqual(result[0]["history"]["_cached_file_bytes"], b"completed image bytes")
        client.get_queue_status.assert_not_awaited()
