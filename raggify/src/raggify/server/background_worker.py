from __future__ import annotations

import asyncio
import contextlib
import datetime
import threading
import uuid
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Dict, Optional

from ..logger import logger

__all__ = ["BackgroundWorker", "Job", "JobPayload", "JobStatus", "get_worker"]

_worker: BackgroundWorker | None = None
_lock = threading.Lock()


class JobStatus(StrEnum):
    """Execution status of a job."""

    PENDING = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    FAILED = auto()


@dataclass(kw_only=True)
class JobPayload:
    """Payload passed to the worker."""

    kind: str  # e.g. ingest_path, ingest_url, etc.
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Job:
    """Job object."""

    job_id: str
    payload: JobPayload
    config_snapshot: dict[str, Any]
    created_at: str
    last_update: str
    status: JobStatus = JobStatus.PENDING
    message: str = ""


class BackgroundWorker:
    """Lightweight worker to execute ingest tasks asynchronously."""

    def __init__(self) -> None:
        """Constructor."""
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._jobs: Dict[str, Job] = {}
        self._worker_task: Optional[asyncio.Task[None]] = None

        self._jobs_lock = threading.Lock()

    async def start(self) -> None:
        """Start the worker."""
        if self._worker_task is not None:
            return

        self._worker_task = asyncio.create_task(
            self._worker_loop(), name="ingest-worker"
        )

    async def shutdown(self) -> None:
        """Shut down the worker."""
        if self._worker_task is None:
            return

        self._worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker_task

        self._worker_task = None

    def _update(self, job: Job, status: JobStatus) -> None:
        """Update the status of a job.

        Args:
            job (Job): Job to update.
        """
        job.status = status
        job.last_update = str(datetime.datetime.now())

    def submit(self, payload: JobPayload) -> Job:
        """Add a job to the queue.

        Args:
            payload (JobPayload): Job payload passed to the worker.

        Returns:
            Job: Created job.
        """
        from ..runtime import get_runtime as _rt

        job_id = str(uuid.uuid4())[:8]
        cfg_snapshot = _rt().cfg.get_dict()
        t = str(datetime.datetime.now())
        job = Job(
            job_id=job_id,
            payload=payload,
            config_snapshot=cfg_snapshot,
            created_at=t,
            last_update=t,
        )

        with self._jobs_lock:
            self._jobs[job_id] = job
            self._queue.put_nowait(job)

        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID.

        Args:
            job_id (str): Job ID.

        Returns:
            Optional[Job]: Job if found.
        """
        return self._jobs.get(job_id)

    def get_jobs(self) -> Dict[str, Job]:
        """Get all jobs.

        Returns:
            Dict[str, Job]: All jobs keyed by job_id.
        """
        return self._jobs.copy()

    def remove_job(self, job_id: str) -> None:
        """Remove a job from the queue.

        Args:
            job_id (str): Job ID.
        """
        with self._jobs_lock:
            self._jobs.pop(job_id, None)

    def remove_completed_jobs(self) -> None:
        """Remove jobs that have finished execution from the queue."""
        completed_ids = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED)
        ]
        for job_id in completed_ids:
            self.remove_job(job_id)

    async def _worker_loop(self) -> None:
        """Worker loop."""
        while True:
            job = await self._queue.get()
            await self._dispatch(job)
            self._queue.task_done()

    async def _dispatch(self, job: Job) -> None:
        """Dispatcher that executes a job.

        Args:
            job (Job): Job to execute.

        Raises:
            ValueError: Unknown job kind.
        """
        from ..ingest import ingest

        self._update(job=job, status=JobStatus.RUNNING)

        def is_canceled() -> bool:
            return self._jobs.get(job.job_id) is None

        try:
            match job.payload.kind:
                case "ingest_path":
                    await ingest.aingest_path(
                        **job.payload.kwargs, is_canceled=is_canceled
                    )
                case "ingest_path_list":
                    await ingest.aingest_path_list(
                        **job.payload.kwargs, is_canceled=is_canceled
                    )
                case "ingest_url":
                    await ingest.aingest_url(
                        **job.payload.kwargs, is_canceled=is_canceled
                    )
                case "ingest_url_list":
                    await ingest.aingest_url_list(
                        **job.payload.kwargs, is_canceled=is_canceled
                    )
                case _:
                    raise ValueError(f"unknown job kind: {job.payload.kind}")

            self._update(job=job, status=JobStatus.SUCCEEDED)
            logger.info(f"{job.payload.kind} (job_id: {job.job_id}) succeeded")
        except Exception as e:
            logger.exception(e)
            self._update(job=job, status=JobStatus.FAILED)
            job.message = str(e)


def get_worker() -> BackgroundWorker:
    """Getter for the background worker singleton.

    Returns:
        BackgroundWorker: Worker instance.
    """
    global _worker

    if _worker is None:
        with _lock:
            if _worker is None:
                _worker = BackgroundWorker()

    return _worker
