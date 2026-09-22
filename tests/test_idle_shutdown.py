import asyncio
import json
import subprocess
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from comfy_client import ComfyUIClient


class IdleShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = ComfyUIClient()
        self.os_calls = patch("comfy_client.subprocess.run", side_effect=AssertionError("Unmocked process operation"))
        self.os_calls.start()
        self.addCleanup(self.os_calls.stop)
        self.auto_kill = patch("comfy_client.COMFYUI_AUTO_KILL", True)
        self.auto_kill.start()
        self.addCleanup(self.auto_kill.stop)

    async def asyncTearDown(self):
        await self.client.close()

    def mock_running_backend(self):
        self.client.is_running = AsyncMock(return_value=True)
        self.client._find_listening_pid = Mock(return_value=4321)
        self.client._get_process_identity = Mock(return_value="created-1")

    async def test_adoption_arms_timer_and_probes_do_not_postpone_it(self):
        self.mock_running_backend()
        await self.client.start_comfyui()
        timer = self.client._idle_kill_timer
        self.assertIsNotNone(timer)
        self.assertEqual(self.client._adopted_creation_time, "created-1")
        await self.client.start_comfyui()
        self.assertIs(self.client._idle_kill_timer, timer)
        self.assertFalse(timer.cancelled())

    async def test_probe_started_backend_arms_timer(self):
        self.client.is_running = AsyncMock(side_effect=[False, True])
        self.client._wait_for_port_free = AsyncMock(return_value=True)
        with patch.object(self.client, "_start_process"):
            await self.client.start_comfyui()
        self.assertIsNotNone(self.client._idle_kill_timer)

    async def test_adopted_idle_backend_reaches_shutdown_without_a_generation(self):
        self.mock_running_backend()
        self.client.get_queue_status = AsyncMock(return_value={"queue_running": [], "queue_pending": []})
        killed = asyncio.Event()

        async def kill():
            self.client._adopted_pid = None
            killed.set()

        self.client.kill_comfyui = AsyncMock(side_effect=kill)
        with patch("comfy_client.COMFYUI_IDLE_TIMEOUT", 0.01):
            await self.client.start_comfyui()
            await asyncio.wait_for(killed.wait(), 1)
        self.client.kill_comfyui.assert_awaited_once()

    async def test_disabled_auto_shutdown_does_not_arm_a_timer(self):
        self.mock_running_backend()
        with patch("comfy_client.COMFYUI_AUTO_KILL", False):
            await self.client.start_comfyui()
        self.assertIsNone(self.client._idle_kill_timer)

    async def test_stale_handle_does_not_block_verified_adopted_pid(self):
        self.client._comfyui_process = Mock()
        self.client._comfyui_process.poll.return_value = 0
        self.client._adopted_pid = 4321
        self.client._adopted_creation_time = "created-1"
        self.client._find_listening_pid = Mock(return_value=4321)
        self.client._get_process_identity = Mock(return_value="created-1")
        with patch("comfy_client.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stderr=b"")) as run:
            await self.client.kill_comfyui()
        self.assertEqual(run.call_args.args[0], ["taskkill", "/F", "/T", "/PID", "4321"])
        self.assertIsNone(self.client._comfyui_process)
        self.assertIsNone(self.client._adopted_pid)

    async def test_pid_reuse_or_replaced_listener_is_never_killed(self):
        for listener, created in ((4321, "created-2"), (9876, "created-1"), (4321, None)):
            with self.subTest(listener=listener, created=created):
                self.client._adopted_pid = 4321
                self.client._adopted_creation_time = "created-1"
                self.client._find_listening_pid = Mock(return_value=listener)
                self.client._get_process_identity = Mock(return_value=created)
                self.assertFalse(self.client._kill_process())
                self.assertIsNone(self.client._adopted_pid)

    def test_remote_backend_cannot_adopt_a_local_port_owner(self):
        self.client.base_url = "http://192.0.2.10:8188"
        self.assertIsNone(self.client._find_listening_pid())

    async def test_startup_port_check_uses_configured_port(self):
        self.client.base_url = "http://127.0.0.1:8189"
        with patch("socket.socket") as socket_factory:
            self.assertTrue(await self.client._wait_for_port_free(timeout=1))
        socket_factory.return_value.bind.assert_called_once_with(("127.0.0.1", 8189))

    def test_identity_requires_configured_executable_and_main_script(self):
        executable = "C:/Comfy/python/python.exe"
        main = "C:/Comfy/ComfyUI/main.py"
        cases = [
            (executable, f'"{executable}" -s "{main}" --windows-standalone-build', "created-1"),
            (executable, f'"{executable}" -c "{main}"', None),
            (executable, f'"{executable}" C:/other/main.py "{main}"', None),
            ("C:/other/python.exe", f'python -s "{main}"', None),
            (executable, "python -s main.py", None),
        ]
        with patch("comfy_client.COMFYUI_PYTHON", executable), patch("comfy_client.COMFYUI_MAIN", main):
            for exe, command, expected in cases:
                with self.subTest(command=command):
                    metadata = {"ExecutablePath": exe, "CommandLine": command, "CreationDate": "created-1"}
                    result = subprocess.CompletedProcess([], 0, stdout=json.dumps(metadata))
                    with patch("comfy_client.subprocess.run", return_value=result):
                        self.assertEqual(self.client._get_process_identity(4321), expected)

    async def test_new_activity_cancels_an_inflight_idle_queue_check(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def queue():
            entered.set()
            await release.wait()
            return {"queue_running": [], "queue_pending": []}

        self.client.get_queue_status = queue
        self.client.kill_comfyui = AsyncMock()
        pending = asyncio.create_task(self.client._idle_kill_guarded())
        await entered.wait()
        self.client._cancel_idle_kill()
        release.set()
        await pending
        self.client.kill_comfyui.assert_not_awaited()

    async def test_start_waits_for_shutdown_in_progress_before_probing(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def kill():
            entered.set()
            await release.wait()

        self.client.get_queue_status = AsyncMock(return_value={"queue_running": [], "queue_pending": []})
        self.client.kill_comfyui = kill
        self.client._do_idle_kill()
        await entered.wait()
        self.mock_running_backend()
        starting = asyncio.create_task(self.client.start_comfyui())
        await asyncio.sleep(0)
        self.client.is_running.assert_not_awaited()
        release.set()
        await starting
        self.client.is_running.assert_awaited_once()

    async def test_upload_failure_rearms_cleanup_without_killing_during_upload(self):
        self.mock_running_backend()

        async def response(request):
            self.assertEqual(self.client._active_operations, 1)
            self.assertIsNone(self.client._idle_kill_timer)
            return httpx.Response(500)

        await self.client.session.aclose()
        self.client.session = httpx.AsyncClient(transport=httpx.MockTransport(response))
        with self.assertRaises(httpx.HTTPStatusError):
            await self.client.upload_image(b"image", "test.png", max_retries=1)
        self.assertEqual(self.client._active_operations, 0)
        self.assertIsNotNone(self.client._idle_kill_timer)

    async def test_workflow_cancellation_rearms_guarded_cleanup(self):
        self.mock_running_backend()
        self.client.submit_workflow = AsyncMock(side_effect=asyncio.CancelledError)
        self.client.kill_comfyui = AsyncMock()
        with patch("comfy_client.websockets.connect", AsyncMock(side_effect=OSError("offline"))):
            with self.assertRaises(asyncio.CancelledError):
                await self.client.run_workflow_and_wait({})
        self.assertEqual(self.client._active_operations, 0)
        self.assertIsNotNone(self.client._idle_kill_timer)
        self.client.kill_comfyui.assert_not_awaited()

    async def test_failed_batch_submission_rearms_cleanup(self):
        self.mock_running_backend()
        self.client.submit_workflow = AsyncMock(side_effect=RuntimeError("invalid workflow"))
        with patch("comfy_client.websockets.connect", AsyncMock(side_effect=OSError("offline"))):
            result = await self.client.batch_run_workflows([{}])
        self.assertIn("invalid workflow", result[0]["error"])
        self.assertEqual(self.client._active_operations, 0)
        self.assertIsNotNone(self.client._idle_kill_timer)

    async def test_finished_workflow_does_not_wait_for_other_clients_jobs(self):
        self.mock_running_backend()
        self.client.submit_workflow = AsyncMock(return_value="own-prompt")
        self.client._wait_via_polling = AsyncMock(return_value={"outputs": {}})
        self.client.get_queue_status = AsyncMock(return_value={"queue_running": ["other-client"], "queue_pending": []})
        with patch("comfy_client.websockets.connect", AsyncMock(side_effect=OSError("offline"))):
            result = await self.client.run_workflow_and_wait({})
        self.assertEqual(result, {"outputs": {}})
        self.client.get_queue_status.assert_not_awaited()
        self.assertIsNotNone(self.client._idle_kill_timer)


if __name__ == "__main__":
    unittest.main()
