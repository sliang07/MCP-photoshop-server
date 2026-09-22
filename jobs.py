"""Generation jobs owned by one running MCP server process."""
import asyncio
import copy
import uuid


class GenerationJobs:
    def __init__(self, run_item):
        self.run_item = run_item
        self._tasks = {}
        self._jobs = {}
        self._lock = asyncio.Lock()

    def submit(self, jobs, export_dir=None, export_format="PNG", timeout_per_image=None):
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("jobs must be a non-empty list")
        for item in jobs:
            if not isinstance(item, dict):
                raise ValueError("each job item must be a dict")
        job_id = uuid.uuid4().hex
        report = {
            "job_id": job_id,
            "status": "queued",
            "total": len(jobs),
            "finished": 0,
            "current_index": None,
            "cancel_requested": False,
            "results": [],
            "lifetime": "MCP server process; not persisted across restart"
        }
        self._jobs[job_id] = report
        inputs = copy.deepcopy(jobs)
        task = asyncio.create_task(self._worker(job_id, inputs, export_dir, export_format, timeout_per_image))
        self._tasks[job_id] = task
        return self.status(job_id)

    def status(self, job_id):
        if job_id not in self._jobs:
            raise ValueError(f"Unknown job ID: {job_id}")
        return copy.deepcopy(self._jobs[job_id])

    def list_jobs(self):
        summaries = []
        for report in self._jobs.values():
            summary = {key: copy.deepcopy(value) for key, value in report.items() if key != "results"}
            summaries.append(summary)
        return summaries

    def cancel(self, job_id):
        if job_id not in self._jobs:
            raise ValueError(f"Unknown job ID: {job_id}")
        report = self._jobs[job_id]
        if report["status"] in ("completed", "completed_with_errors", "cancelled", "interrupted"):
            return copy.deepcopy(report)
        if report["status"] == "queued":
            report["status"] = "cancelled"
            report["cancel_requested"] = True
            return copy.deepcopy(report)
        if report["status"] != "cancelling":
            report["cancel_requested"] = True
            report["status"] = "cancelling"
        return copy.deepcopy(report)

    async def _worker(self, job_id, inputs, export_dir, export_format, timeout_per_image):
        report = self._jobs[job_id]
        try:
            async with self._lock:
                if report["cancel_requested"]:
                    report["status"] = "cancelled"
                    return
                report["status"] = "running"
                for index, item in enumerate(inputs):
                    if report["cancel_requested"]:
                        break
                    report["current_index"] = index
                    try:
                        result = await self.run_item(item, index, export_dir, export_format, timeout_per_image)
                    except Exception as e:
                        result = {"status": "failed", "error": str(e), "file": None}
                    result["index"] = index
                    report["results"].append(result)
                    report["finished"] += 1
                if report["cancel_requested"]:
                    report["status"] = "cancelled"
                elif any(r.get("status") == "failed" for r in report["results"]):
                    report["status"] = "completed_with_errors"
                else:
                    report["status"] = "completed"
        except asyncio.CancelledError:
            report["status"] = "interrupted"
            raise
        finally:
            report["current_index"] = None
            self._tasks.pop(job_id, None)
