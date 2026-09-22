"""
Thin wrapper for ComfyUI's REST + WebSocket API.
Handles workflow submission, progress tracking, and result retrieval.
"""

import asyncio
import base64
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx
import websockets

from config import (
    COMFYUI_URL, WEBSOCKET_TIMEOUT, POLLING_INTERVAL,
    COMFYUI_START_CMD, COMFYUI_PYTHON, COMFYUI_MAIN, COMFYUI_ARGS,
    COMFYUI_AUTO_KILL, COMFYUI_START_TIMEOUT, COMFYUI_IDLE_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Debug logging: set MCP_DEBUG_LOG to a file path to enable, or "0" to disable
_DEBUG_LOG_PATH = os.getenv("MCP_DEBUG_LOG", "0")

def _debug_log(msg: str):
    """Write to debug log file if enabled. No-op when MCP_DEBUG_LOG is not set or "0"."""
    if not _DEBUG_LOG_PATH or _DEBUG_LOG_PATH == "0":
        return
    try:
        import json as _json
        import time as _time
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(f"[{_time.strftime('%H:%M:%S')}] {msg}\n")
            f.flush()
    except Exception:
        pass


class ComfyUIClient:
    """Client for ComfyUI's API."""

    def __init__(self, base_url: str = COMFYUI_URL):
        self.base_url = base_url.rstrip("/")
        self.session = httpx.AsyncClient(timeout=120.0)
        self._comfyui_process: Optional[subprocess.Popen] = None
        self._adopted_pid: Optional[int] = None  # PID of a running ComfyUI this server did not start
        self._adopted_creation_time = None
        self._idle_kill_timer: Optional[asyncio.Handle] = None  # Background timer for idle kill
        self._idle_kill_task = None
        self._idle_generation = 0
        self._active_operations = 0

    async def close(self):
        self._cancel_idle_kill()
        await self.session.aclose()

    # ------------------------------------------------------------------ #
    #  Idle kill timer
    # ------------------------------------------------------------------ #

    def _cancel_idle_kill(self):
        """Cancel any pending idle kill timer."""
        self._idle_generation += 1
        if self._idle_kill_timer is not None:
            self._idle_kill_timer.cancel()
            self._idle_kill_timer = None

    def _schedule_idle_kill(self):
        """Schedule ComfyUI to be killed after COMFYUI_IDLE_TIMEOUT seconds of inactivity."""
        self._cancel_idle_kill()
        self._idle_kill_timer = asyncio.get_event_loop().call_later(
            COMFYUI_IDLE_TIMEOUT,
            self._do_idle_kill,
        )
        logger.info("Idle kill scheduled in %ds.", COMFYUI_IDLE_TIMEOUT)

    def _arm_idle_shutdown(self):
        if COMFYUI_AUTO_KILL and not self._active_operations and self._idle_kill_timer is None:
            self._schedule_idle_kill()

    def _do_idle_kill(self):
        """Called by the idle timer to kill ComfyUI after timeout.

        The kill is guarded: if another workflow is active in ComfyUI's queue -
        possibly submitted by ANOTHER server instance (Cline stdio vs Open
        WebUI HTTP) sharing the same ComfyUI process - the kill is deferred
        instead of interrupting that job.
        """
        self._idle_kill_timer = None
        logger.info("Idle timeout reached (%ds), checking queue before killing ComfyUI...", COMFYUI_IDLE_TIMEOUT)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            task = asyncio.create_task(self._idle_kill_guarded())
            self._idle_kill_task = task
            task.add_done_callback(self._on_idle_kill_done)
        else:
            loop.run_until_complete(self._idle_kill_guarded())

    async def _idle_kill_guarded(self):
        """Kill ComfyUI only when no job is running or queued.

        Multiple server instances (e.g. a stdio instance for Cline and a
        streamable-HTTP instance for Open WebUI) share one ComfyUI process on
        port 8188. An idle timer from one instance must not kill ComfyUI while
        the other's workflow is in progress - doing so fails that job.
        While the queue is busy the kill re-checks every COMFYUI_IDLE_TIMEOUT.
        """
        generation = self._idle_generation
        if self._active_operations:
            return
        try:
            queue = await self.get_queue_status()
            running = queue.get("queue_running")
            pending = queue.get("queue_pending")
            if not isinstance(running, list) or not isinstance(pending, list):
                raise ValueError("Malformed queue status response")
        except Exception as e:
            if generation != self._idle_generation:
                return
            # ComfyUI unreachable or malformed: defer kill to avoid interrupting unknown jobs.
            logger.warning("Queue check before idle kill failed: %s. Deferring kill.", e)
            self._idle_kill_timer = asyncio.get_event_loop().call_later(
                COMFYUI_IDLE_TIMEOUT, self._do_idle_kill)
            return
        if generation != self._idle_generation or self._active_operations:
            return
        if running or pending:
            logger.info("ComfyUI queue not empty (%d running, %d pending); deferring idle kill by %ds.",
                        len(running), len(pending), COMFYUI_IDLE_TIMEOUT)
            self._idle_kill_timer = asyncio.get_event_loop().call_later(
                COMFYUI_IDLE_TIMEOUT, self._do_idle_kill)
            return
        await self.kill_comfyui()
        if self._comfyui_process is not None or self._adopted_pid is not None:
            self._arm_idle_shutdown()

    def _on_idle_kill_done(self, task):
        """Callback to log errors from idle kill task."""
        if self._idle_kill_task is task:
            self._idle_kill_task = None
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as e:
            logger.error("Idle kill failed: %s", e, exc_info=True)

    # ------------------------------------------------------------------ #
    #  ComfyUI lifecycle management (auto-start / auto-kill)
    # ------------------------------------------------------------------ #

    async def is_running(self) -> bool:
        """Check if ComfyUI is reachable by hitting /history. Retries before giving up,
        since a busy ComfyUI can be briefly slow to respond.

        Verifies the response is ComfyUI's JSON history object. A foreign
        HTTP listener on port 8188 can answer 200 with a non-JSON body; without
        this check it would be mistaken for a running ComfyUI."""
        for attempt in range(3):
            try:
                resp = await self.session.get(f"{self.base_url}/history", timeout=8.0)
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(1.0)
                continue
            if resp.status_code != 200:
                return False
            try:
                data = resp.json()
            except Exception:
                logger.warning("Port 8188 answered /history with a non-JSON body; a foreign "
                               "process occupies the port.")
                return False
            if not isinstance(data, dict):
                logger.warning("Port 8188 answered /history with JSON that is not an object; "
                               "a foreign process occupies the port.")
                return False
            # Note: a fresh ComfyUI returns {} (empty object) - that is valid.
            # The /queue endpoint (not /history) is the one with queue_running/queue_pending.
            return True
        return False

    def _start_process(self):
        """Synchronous helper to launch ComfyUI via embedded Python directly."""
        # Build environment with UTF-8 encoding to prevent UnicodeEncodeError
        # when ComfyUI custom nodes log emoji/special characters
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        # Prefer direct Python launch over .bat for reliability
        if os.path.exists(COMFYUI_PYTHON) and os.path.exists(COMFYUI_MAIN):
            args = [COMFYUI_PYTHON, "-s", COMFYUI_MAIN] + COMFYUI_ARGS.split()
            cwd = str(Path(COMFYUI_MAIN).parent.parent)  # ComfyUI_windows_portable root
            logger.info("Starting ComfyUI via Python: %s ...", ' '.join(args[:3]))
            logger.info("Working directory: %s", cwd)
            # Child process output must not enter the MCP stdio protocol.
            self._comfyui_process = subprocess.Popen(
                args,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008,  # DETACHED_PROCESS
            )
        else:
            # Fallback to .bat
            logger.info("Starting ComfyUI via bat: %s", COMFYUI_START_CMD)
            cwd = str(Path(COMFYUI_START_CMD).parent)
            self._comfyui_process = subprocess.Popen(
                COMFYUI_START_CMD,
                shell=True,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
            )
        pid = self._comfyui_process.pid
        logger.info("ComfyUI launched with PID %d", pid)

    def _find_listening_pid(self, port=None):
        """Find the PID LISTENING on a TCP port via netstat -ano."""
        endpoint = urlsplit(self.base_url)
        if endpoint.hostname not in ("127.0.0.1", "localhost", "::1"):
            return None
        if port is None:
            port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
        try:
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=10,
            )
            suffix = ":" + str(port)
            for line in result.stdout.splitlines():
                parts = line.split()
                if (len(parts) >= 5 and parts[0] == "TCP"
                        and parts[1].endswith(suffix) and parts[3] == "LISTENING"):
                    try:
                        return int(parts[4])
                    except ValueError:
                        continue
        except Exception as e:
            logger.debug("netstat lookup for port %d failed: %s", port, e)
        return None

    def _get_process_identity(self, pid):
        """Return the creation time only for the configured Python/main.py process."""
        if not COMFYUI_PYTHON or not COMFYUI_MAIN:
            return None
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                 f"Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}' | "
                 "Select-Object ExecutablePath,CommandLine,CreationDate | ConvertTo-Json -Compress"],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode or not result.stdout.strip():
                return None
            process = json.loads(result.stdout)
            executable = process.get("ExecutablePath")
            command = process.get("CommandLine")
            if not executable or not command:
                return None
            if os.path.normcase(os.path.abspath(executable)) != os.path.normcase(os.path.abspath(COMFYUI_PYTHON)):
                return None
            arguments = [arg.strip('"') for arg in shlex.split(command, posix=False)][1:]
            while arguments and arguments[0] in ("-s", "-u", "-B", "-E", "-I", "-O", "-OO"):
                arguments.pop(0)
            if not arguments or not os.path.isabs(arguments[0]):
                return None
            if os.path.normcase(os.path.abspath(arguments[0])) != os.path.normcase(os.path.abspath(COMFYUI_MAIN)):
                return None
            return process.get("CreationDate")
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError, AttributeError) as error:
            logger.debug("Could not verify ComfyUI PID %s: %s", pid, error)
            return None

    def _kill_process(self) -> bool:
        """Synchronous helper to terminate the managed ComfyUI process tree.
        A live Popen handle started by this server takes precedence; otherwise
        an adopted PID (a ComfyUI already running, e.g. from a previous MCP
        server session) is terminated. Returns True only if a process was
        actually killed.
        """
        proc = self._comfyui_process
        if proc and proc.poll() is not None:
            self._comfyui_process = None
            proc = None

        if proc is not None:
            pid = proc.pid
            logger.info("Killing owned ComfyUI process tree (PID %d)...", pid)
            try:
                result = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=15,
                )
                if result.returncode == 0:
                    proc.wait(timeout=5)
                    logger.info("Owned ComfyUI process tree killed.")
                    return True
                logger.warning("taskkill failed for PID %d (rc=%d): %s", pid, result.returncode, result.stderr.decode(errors='ignore'))
            except subprocess.TimeoutExpired:
                logger.warning("taskkill timed out for PID %d", pid)
            except OSError as e:
                logger.warning("taskkill raised OSError for PID %d: %s", pid, e)
            # Keep ownership on failure so future attempts can retry
            return False

        if self._adopted_pid:
            pid = self._adopted_pid
            if (not self._adopted_creation_time
                    or self._find_listening_pid() != pid
                    or self._get_process_identity(pid) != self._adopted_creation_time):
                logger.warning("Adopted ComfyUI PID %d no longer matches the configured backend; leaving it alone.", pid)
                self._adopted_pid = None
                self._adopted_creation_time = None
                return False
            logger.info("Killing adopted ComfyUI process tree (PID %d)...", pid)
            try:
                result = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=15,
                )
                if result.returncode == 0:
                    logger.info("Adopted ComfyUI process tree killed.")
                    return True
                logger.warning("taskkill failed for adopted PID %d (rc=%d): %s", pid, result.returncode, result.stderr.decode(errors='ignore'))
            except subprocess.TimeoutExpired:
                logger.warning("taskkill timed out for adopted PID %d", pid)
            except OSError as e:
                logger.warning("taskkill raised OSError for adopted PID %d: %s", pid, e)

        logger.warning("No managed ComfyUI process to kill (none started by this server, no adopted PID).")
        return False

    async def _wait_for_port_free(self, port=None, timeout=60):
        """Wait until port is truly free (no LISTENING or TIME_WAIT sockets).
        Actually tries to bind to verify the port is available, since Windows
        cannot bind to ports with TIME_WAIT connections even after process is killed."""
        import socket
        if port is None:
            endpoint = urlsplit(self.base_url)
            port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
        deadline = time.monotonic() + timeout
        last_netstat_msg = ""
        while time.monotonic() < deadline:
            # First check: try to actually bind to the port (most reliable)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                s.close()
                logger.info("Port %d is bindable (free).", port)
                return True
            except OSError:
                pass  # Port still not bindable, continue waiting

            # Show what's holding the port
            try:
                result = subprocess.run(
                    ["netstat", "-ano"], capture_output=True, text=True, timeout=5,
                )
                states = set()
                for line in result.stdout.splitlines():
                    if f":{port}" in line:
                        for state in ["LISTENING", "TIME_WAIT", "ESTABLISHED", "CLOSE_WAIT"]:
                            if state in line:
                                states.add(state)
                msg = f"Port {port} states: {', '.join(states) if states else 'unknown'} - waiting..."
                if msg != last_netstat_msg:
                    logger.debug(msg)
                    last_netstat_msg = msg
            except Exception:
                pass

            await asyncio.sleep(1.0)

        logger.warning("Port %d not free after %ds", port, timeout)
        return False

    async def kill_comfyui(self):
        """Kill the managed ComfyUI process (owned or adopted) to free VRAM."""
        killed = await asyncio.get_event_loop().run_in_executor(None, self._kill_process)
        if killed:
            self._comfyui_process = None
            self._adopted_pid = None
            self._adopted_creation_time = None

    async def start_comfyui(self):
        """Start ComfyUI if not already running, then wait until it is reachable.
        Ensures port 8188 is free before starting. Cleans up owned stale processes
        and adopts a running ComfyUI started elsewhere (e.g. a previous MCP server
        session) so the idle kill can manage it. Raises RuntimeError if a foreign
        non-ComfyUI listener remains on the port."""
        if self._idle_kill_task is not None:
            await asyncio.shield(self._idle_kill_task)
        if await self.is_running():
            logger.info("ComfyUI is already running.")
            # Track an instance this server did not start itself, so the idle
            # kill can terminate it after inactivity (previously a silent no-op).
            if self._comfyui_process is None or self._comfyui_process.poll() is not None:
                self._comfyui_process = None
                pid = await asyncio.get_event_loop().run_in_executor(None, self._find_listening_pid)
                identity = await asyncio.get_event_loop().run_in_executor(
                    None, self._get_process_identity, pid) if pid is not None else None
                self._adopted_pid = pid if identity else None
                self._adopted_creation_time = identity
                if identity:
                    logger.info("ComfyUI (PID %d) was started outside this server; adopted for idle management.", self._adopted_pid)
            self._arm_idle_shutdown()
            return

        # A fresh launch must not be killed by a stale timer from an earlier instance.
        self._cancel_idle_kill()

        # Ensure port is free before starting
        if not await self._wait_for_port_free(timeout=5):
            # Attempt to clean up only our OWN stale process if we own one
            if self._comfyui_process and self._comfyui_process.poll() is None:
                logger.warning("Port in use by owned process, cleaning up...")
                killed = await asyncio.get_event_loop().run_in_executor(None, self._kill_process)
                if not killed:
                    raise RuntimeError("Could not stop the managed ComfyUI process; startup was not retried.")
                self._comfyui_process = None
                if not await self._wait_for_port_free(timeout=10):
                    raise RuntimeError("Failed to free port 8188 after killing owned process.")
            else:
                raise RuntimeError("Port 8188 is occupied by a foreign process. Check the configured backend/port.")

        # Launch in a thread to avoid blocking the event loop
        await asyncio.get_event_loop().run_in_executor(None, self._start_process)

        # Poll until ComfyUI is ready or timeout
        # ComfyUI may take time to start + load models (up to COMFYUI_START_TIMEOUT)
        deadline = time.monotonic() + COMFYUI_START_TIMEOUT
        attempts = 0
        while time.monotonic() < deadline:
            attempts += 1
            if await self.is_running():
                logger.info("ComfyUI is ready (attempt %d).", attempts)
                self._arm_idle_shutdown()
                return
            await asyncio.sleep(2.0)

        # ComfyUI may have failed to start (e.g., port still in TIME_WAIT).
        # Kill whatever we started, verify port is free, and retry once.
        logger.warning("ComfyUI not ready after %d polls, retrying after port clear...", attempts)
        await self.kill_comfyui()
        if self._comfyui_process is not None and self._comfyui_process.poll() is None:
            raise RuntimeError("Could not stop the managed ComfyUI process; startup was not retried.")
        if not await self._wait_for_port_free(timeout=60):
            raise RuntimeError("Failed to free port 8188 before retry launch.")
        await asyncio.get_event_loop().run_in_executor(None, self._start_process)

        deadline2 = time.monotonic() + COMFYUI_START_TIMEOUT
        while time.monotonic() < deadline2:
            if await self.is_running():
                logger.info("ComfyUI is ready (retry).")
                self._arm_idle_shutdown()
                return
            await asyncio.sleep(2.0)

        raise TimeoutError(
            f"ComfyUI did not become reachable within {COMFYUI_START_TIMEOUT}s (even after retry). "
            "Check the configured backend, port, and startup command."
        )

    # ------------------------------------------------------------------ #
    #  Workflow submission
    # ------------------------------------------------------------------ #

    async def submit_workflow(self, workflow: dict, client_id: Optional[str] = None) -> str:
        """
        POST a workflow graph to /prompt and return the prompt_id.
        workflow: dict in ComfyUI API format (nodes dict).
        client_id: optional client_id to correlate with WebSocket events.
        """
        if client_id is None:
            client_id = str(uuid.uuid4())
        response = await self.session.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        if response.status_code == 400:
            err = response.json()
            raise RuntimeError(f"ComfyUI rejected workflow (400): {err}")
        response.raise_for_status()
        data = response.json()
        return data["prompt_id"]

    # ------------------------------------------------------------------ #
    #  Progress / completion
    # ------------------------------------------------------------------ #

    async def run_workflow_and_wait(self, workflow: dict, progress_callback=None, timeout: Optional[int] = None) -> dict:
        """
        Submit workflow and wait for completion atomically.
        Auto-starts ComfyUI if not running. Idle cleanup is scheduled when the
        operation ends; a busy or unknown queue always defers shutdown.
        Connects WebSocket FIRST, then submits with the same client_id,
        preventing race conditions where workflow finishes before WebSocket connects.
        Returns the history entry with output file bytes pre-fetched (if auto-kill is enabled).
        timeout: optional override for WEBSOCKET_TIMEOUT.
        """
        # Cancel any pending idle kill — a new workflow is starting
        self._cancel_idle_kill()

        client_id = str(uuid.uuid4())
        ws_url = f"{self.base_url.replace('http', 'ws')}/ws?clientId={client_id}"
        result = None
        success = False

        self._active_operations += 1
        try:
            await self.start_comfyui()
            # Only connection establishment may fall back to polling. Once a prompt
            # is submitted, never resubmit it after a lost WebSocket or timeout.
            ws = None
            try:
                ws = await websockets.connect(ws_url, open_timeout=15)
            except Exception as ws_error:
                logger.warning("WebSocket unavailable; using polling: %s", ws_error)
            try:
                prompt_id = await self.submit_workflow(workflow, client_id=client_id)
                if ws is None:
                    result = await self._wait_via_polling(prompt_id, progress_callback, timeout=timeout)
                else:
                    try:
                        result = await self._listen_for_completion(ws, prompt_id, progress_callback, timeout=timeout)
                    except websockets.exceptions.ConnectionClosed:
                        result = await self._wait_via_polling(prompt_id, progress_callback, timeout=timeout)
                success = True
            finally:
                if ws is not None:
                    await ws.close()

            # Pre-fetch output file bytes BEFORE killing ComfyUI
            # This is needed because after kill, /view endpoint is unavailable
            if success and COMFYUI_AUTO_KILL and result:
                try:
                    outputs = result.get("outputs", {})
                    save_image_ids = {n for n, d in workflow.items() if d["class_type"] == "SaveImage"}

                    img_files = []
                    for node_id, node_output in outputs.items():
                        if node_id in save_image_ids:
                            img_files.extend(node_output.get("images", []))

                    logger.info("Pre-fetch: %d image files (from %d SaveImage nodes)", len(img_files), len(save_image_ids))

                    # Pre-fetch image output
                    if img_files:
                        first_img = img_files[0]
                        img_bytes = await self.get_output_file(
                            first_img.get("filename", ""),
                            subfolder=first_img.get("subfolder", ""),
                            output_dir=first_img.get("type", "output"),
                        )
                        result["_cached_file_bytes"] = img_bytes
                        result["_cached_file_info"] = first_img
                        logger.info("Pre-fetched image: %s (%d bytes)", first_img.get('filename'), len(img_bytes))

                    if not img_files:
                        logger.warning("no SaveImage outputs found in this workflow.")
                except Exception as e:
                    logger.warning("failed to pre-fetch output file: %s", e)

            return result
        finally:
            self._active_operations -= 1
            self._arm_idle_shutdown()

    async def _listen_for_completion(self, ws, prompt_id: str, progress_callback=None, timeout: Optional[int] = None) -> dict:
        """Listen on an already-open WebSocket for execution events."""
        effective_timeout = timeout if timeout is not None else WEBSOCKET_TIMEOUT
        deadline = asyncio.get_event_loop().time() + effective_timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                if isinstance(msg_raw, bytes):
                    continue  # Binary preview frame, not a JSON execution event.
                msg = json.loads(msg_raw)
                msg_type = msg.get("type")
                data = msg.get("data", {})

                # Ignore events for other prompts sharing this connection
                if data.get("prompt_id") not in (None, prompt_id):
                    continue

                if msg_type == "progress":
                    if progress_callback:
                        progress_callback(data.get("value", 0), data.get("max", 100))

                elif msg_type == "executing":
                    node = data.get("node")
                    if progress_callback:
                        progress_callback(None, None, node)
                    # node goes null when the queue finishes this prompt - success OR error
                    if node is None:
                        # Retry with backoff - history may not be ready immediately
                        for attempt in range(10):
                            await asyncio.sleep(0.5 * (attempt + 1))
                            history = await self.get_history(prompt_id)
                            if history is not None:
                                return history
                        # Fallback: return None if history never appears
                        logger.warning("history for %s not available after 55s retries", prompt_id)
                        return None

                elif msg_type == "executed":
                    # A node completed; downstream nodes may still be running.
                    continue

                elif msg_type == "execution_error":
                    raise RuntimeError(f"ComfyUI execution error: {data}")

                elif msg_type == "execution_interrupted":
                    raise RuntimeError(f"ComfyUI execution interrupted: {data}")

            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                raise

        raise TimeoutError(f"Workflow {prompt_id} did not complete within {effective_timeout}s")

    async def _listen_for_many(self, ws, prompt_map: dict, timeout: Optional[int] = None) -> dict:
        """Listen until every prompt in prompt_map (job_index -> prompt_id) completes.

        Returns {job_index: history_entry} for completed jobs and
        {job_index: {"_batch_error": str}} for jobs that errored or timed out.
        Mirrors _listen_for_completion() but tracks many prompts on one connection.
        """
        effective_timeout = timeout if timeout is not None else max(WEBSOCKET_TIMEOUT, 60 * len(prompt_map))
        deadline = asyncio.get_event_loop().time() + effective_timeout
        pid_to_idx = {pid: idx for idx, pid in prompt_map.items()}
        pending = set(pid_to_idx)
        out = {}
        while pending and asyncio.get_event_loop().time() < deadline:
            try:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                if isinstance(msg_raw, bytes):
                    continue  # Binary preview frame, not a JSON execution event.
                msg = json.loads(msg_raw)
                msg_type = msg.get("type")
                data = msg.get("data", {})
                pid = data.get("prompt_id")
                if pid not in pending:
                    continue
                if msg_type == "executing":
                    # node goes null when the queue finishes this prompt - success OR error
                    if data.get("node") is None:
                        history = None
                        error = None
                        for attempt in range(10):
                            try:
                                history = await self.get_history(pid)
                            except RuntimeError as he:
                                error = str(he)
                                break
                            if history is not None:
                                break
                            await asyncio.sleep(0.5 * (attempt + 1))
                        if history is not None:
                            out[pid_to_idx[pid]] = history
                        elif error is not None:
                            logger.warning("batch job %d failed: %s", pid_to_idx[pid], error)
                            out[pid_to_idx[pid]] = {"_batch_error": error}
                        else:
                            logger.warning("batch history for %s not available after retries", pid)
                            out[pid_to_idx[pid]] = {"_batch_error": "history unavailable after completion"}
                        pending.discard(pid)
                elif msg_type in ("execution_error", "execution_interrupted"):
                    logger.error("batch execution %s: %s", msg_type, data)
                    out[pid_to_idx[pid]] = {"_batch_error": f"{msg_type}: {data}"}
                    pending.discard(pid)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                raise

        # Final chance for anything still outstanding: check /history once more.
        for pid in list(pending):
            try:
                history = await self.get_history(pid)
            except RuntimeError:
                history = None
            if history is not None:
                out[pid_to_idx[pid]] = history
            else:
                out[pid_to_idx[pid]] = {"_batch_error": f"did not complete within {effective_timeout}s"}
        return out

    async def batch_run_workflows(self, workflows: list, timeout: Optional[int] = None) -> list:
        """Submit a whole batch and wait for every workflow on ONE WebSocket connection.

        Queues all jobs up front (back-to-back /prompt calls) so the batch keeps
        running unattended — the GPU batching rule pattern. Mirrors
        run_workflow_and_wait() per job:
          - returns one entry per job, aligned with `workflows`:
            {"history": <history entry or None>, "error": <str or None>}
          - when COMFYUI_AUTO_KILL is on, output bytes are pre-fetched into
            history["_cached_file_bytes"] before the idle kill,
          - schedules the idle kill after the batch (keeps ComfyUI warm for a follow-up).
        timeout: seconds for the whole batch wait (default: max(WEBSOCKET_TIMEOUT, 60 * jobs)).
        """
        self._cancel_idle_kill()

        n = len(workflows)
        effective_timeout = timeout if timeout is not None else max(WEBSOCKET_TIMEOUT, 60 * n)
        client_id = str(uuid.uuid4())
        ws_url = f"{self.base_url.replace('http', 'ws')}/ws?clientId={client_id}"

        results = [{"history": None, "error": None} for _ in range(n)]
        prompt_ids = [None] * n
        ws = None

        self._active_operations += 1
        try:
            await self.start_comfyui()
            try:
                ws = await websockets.connect(ws_url, open_timeout=15)
            except Exception as ws_error:
                logger.warning("WebSocket unavailable for batch; using polling: %s", ws_error)

            # Queue the entire batch up front.
            for i, wf in enumerate(workflows):
                try:
                    prompt_ids[i] = await self.submit_workflow(wf, client_id=client_id)
                except Exception as e:
                    logger.error("batch job %d failed to submit: %s", i, e)
                    results[i]["error"] = f"submission failed: {e}"

            remaining = {i: pid for i, pid in enumerate(prompt_ids) if pid is not None}
            if remaining:
                if ws is None:
                    for i, pid in remaining.items():
                        try:
                            history = await self._wait_via_polling(pid, None, timeout=effective_timeout)
                            results[i]["history"] = history
                            if history is None:
                                results[i]["error"] = "workflow did not complete"
                        except (RuntimeError, TimeoutError) as e:
                            results[i]["error"] = str(e)
                else:
                    many = {}
                    try:
                        many = await self._listen_for_many(ws, remaining, timeout=effective_timeout)
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("batch WebSocket closed; falling back to polling for %d jobs", len(remaining))
                        many = {}
                    for i, pid in remaining.items():
                        value = many.get(i)
                        if value is None:
                            # Not reported (e.g. WS dropped): poll this one job.
                            try:
                                history = await self._wait_via_polling(pid, None, timeout=effective_timeout)
                                results[i]["history"] = history
                                if history is None:
                                    results[i]["error"] = "workflow did not complete"
                            except (RuntimeError, TimeoutError) as e:
                                results[i]["error"] = str(e)
                        elif "_batch_error" in value:
                            results[i]["error"] = value["_batch_error"]
                        else:
                            results[i]["history"] = value
            if any(r["history"] for r in results):
                # Pre-fetch output bytes BEFORE the idle kill (same reason as run_workflow_and_wait).
                if COMFYUI_AUTO_KILL:
                    for i, res in enumerate(results):
                        history = res["history"]
                        if not history or history.get("_cached_file_bytes") is not None:
                            continue
                        try:
                            outputs = history.get("outputs", {})
                            save_image_ids = {nid for nid, d in workflows[i].items() if d["class_type"] == "SaveImage"}
                            img_files = []
                            for node_id, node_output in outputs.items():
                                if node_id in save_image_ids:
                                    img_files.extend(node_output.get("images", []))
                            if img_files:
                                first_img = img_files[0]
                                history["_cached_file_bytes"] = await self.get_output_file(
                                    first_img.get("filename", ""),
                                    subfolder=first_img.get("subfolder", ""),
                                    output_dir=first_img.get("type", "output"),
                                )
                                logger.info("batch pre-fetch job %d: %s", i, first_img.get("filename"))
                        except Exception as e:
                            logger.warning("failed to pre-fetch batch output for job %d: %s", i, e)
            return results
        finally:
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
            self._active_operations -= 1
            self._arm_idle_shutdown()

    async def wait_for_completion(self, prompt_id: str, progress_callback=None) -> dict:
        """
        Wait for a workflow to complete using WebSocket for real-time progress.
        Falls back to polling /history if WebSocket fails (but not for ComfyUI execution errors).
        Returns the history entry for the prompt_id.
        """
        try:
            return await self._wait_via_websocket(prompt_id, progress_callback)
        except RuntimeError:
            # ComfyUI-reported execution error - don't mask it by silently retrying via polling
            raise
        except Exception:
            return await self._wait_via_polling(prompt_id, progress_callback)

    async def _wait_via_websocket(self, prompt_id: str, progress_callback=None) -> dict:
        """Listen on WebSocket for execution events (legacy method)."""
        ws_url = f"{self.base_url.replace('http', 'ws')}/ws?clientId={uuid.uuid4()}"
        async with websockets.connect(ws_url) as ws:
            return await self._listen_for_completion(ws, prompt_id, progress_callback)

    async def _wait_via_polling(self, prompt_id: str, progress_callback=None, timeout: Optional[int] = None) -> dict:
        """Poll /history until the prompt_id appears."""
        effective_timeout = timeout if timeout is not None else WEBSOCKET_TIMEOUT
        deadline = asyncio.get_event_loop().time() + effective_timeout
        while asyncio.get_event_loop().time() < deadline:
            history = await self.get_history(prompt_id)
            if history:
                return history
            await asyncio.sleep(POLLING_INTERVAL)
        raise TimeoutError(f"Workflow {prompt_id} did not complete within {WEBSOCKET_TIMEOUT}s")

    # ------------------------------------------------------------------ #
    #  History
    # ------------------------------------------------------------------ #

    async def get_history(self, prompt_id: str) -> Optional[dict]:
        """Get history entry for a specific prompt_id."""
        import json as _json
        try:
            response = await self.session.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            data = response.json()
            _debug_log(f"[get_history] prompt_id={prompt_id}, keys={list(data.keys())}")
            entry = data.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI execution failed: {status.get('messages', [])}")
                if status.get("completed") is False:
                    return None
            return entry
        except Exception as e:
            _debug_log(f"[get_history] EXCEPTION: {type(e).__name__}: {e}")
            raise

    async def get_queue_status(self) -> dict:
        """Get the current queue status from ComfyUI."""
        response = await self.session.get(f"{self.base_url}/queue")
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------ #
    #  File retrieval
    # ------------------------------------------------------------------ #

    async def get_output_file(self, filename: str, subfolder: str = "", output_dir: str = "output") -> bytes:
        """
        Fetch an output file by name.
        Returns the file bytes.
        """
        params = {"filename": filename}
        if subfolder:
            params["subfolder"] = subfolder
        params["type"] = output_dir

        response = await self.session.get(f"{self.base_url}/view", params=params)
        response.raise_for_status()
        return response.content

    async def retrieve_result(self, history: dict, prefer_type: str = "images") -> Optional[bytes]:
        """
        Retrieve the first result file from history.
        Returns file bytes or None.
        """
        files = ComfyUIClient.get_output_files_from_history(history, file_type=prefer_type)
        if not files:
            return None

        first = files[0]
        return await self.get_output_file(
            first["filename"],
            subfolder=first.get("subfolder", ""),
            output_dir=first.get("type", "output"),
        )

    # ------------------------------------------------------------------ #
    #  Upload
    # ------------------------------------------------------------------ #

    async def upload_image(self, image_bytes: bytes, filename: str, max_retries: int = 3) -> str:
        """
        Upload an image to ComfyUI's input directory.
        Returns the filename as stored in ComfyUI.
        Retries on connection errors (e.g., after process restart or during free_memory).
        Auto-starts ComfyUI if not already running.
        """
        # An upload starts a new operation; cancel the previous idle timer.
        self._cancel_idle_kill()
        self._active_operations += 1
        try:
            # Ensure ComfyUI is running before attempting upload
            await self.start_comfyui()

            for attempt in range(max_retries):
                try:
                    files = {"image": (filename, image_bytes, "image/png")}
                    data = {"subfolder": "", "type": "input"}
                    response = await self.session.post(
                        f"{self.base_url}/upload/image",
                        data=data,
                        files=files,
                    )
                    response.raise_for_status()
                    uploaded = response.json()
                    name = uploaded["name"]
                    subfolder = uploaded.get("subfolder", "").strip("/\\")
                    return f"{subfolder}/{name}" if subfolder else name
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2.0 ** attempt
                        logger.warning("upload_image retry %d/%d for '%s': %s, waiting %.1fs...", attempt+1, max_retries, filename, e, wait)
                        await asyncio.sleep(wait)
                    else:
                        raise
        finally:
            self._active_operations -= 1
            self._arm_idle_shutdown()

    # ------------------------------------------------------------------ #
    #  System info
    # ------------------------------------------------------------------ #

    async def get_object_info(self) -> dict:
        """Read live nodes/model choices without auto-starting ComfyUI."""
        response = await self.session.get(f"{self.base_url}/object_info", timeout=15.0)
        response.raise_for_status()
        return response.json()

    async def get_system_stats(self) -> dict:
        """Get ComfyUI system statistics."""
        response = await self.session.get(f"{self.base_url}/system_stats")
        response.raise_for_status()
        return response.json()

    async def free_memory(self):
        """Call /free to release VRAM. Must explicitly request both flags —
        ComfyUI defaults unload_models and free_memory to False if omitted."""
        try:
            response = await self.session.post(
                f"{self.base_url}/free",
                json={"unload_models": True, "free_memory": True},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError:
            # Best-effort VRAM cleanup - some ComfyUI versions return 500
            pass
        except Exception:
            # ComfyUI may already be killed (auto-kill mode) - ignore
            pass

    # ------------------------------------------------------------------ #
    #  Helper: parse history outputs
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_output_files_from_history(history: dict, file_type: str = "images") -> list:
        """Static helper to extract file info from history.
        Only returns files matching the requested file_type."""
        files = []
        outputs = history.get("outputs", {})
        for node_output in outputs.values():
            if file_type in node_output:
                for item in node_output[file_type]:
                    files.append(item)
        return files


