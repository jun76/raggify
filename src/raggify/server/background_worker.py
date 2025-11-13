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

__all__ = ["get_worker"]

_worker: BackgroundWorker | None = None
_lock = threading.Lock()


class JobStatus(StrEnum):
    """ジョブの実行状態。"""

    PENDING = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    FAILED = auto()


@dataclass(kw_only=True)
class JobPayload:
    """ワーカーに渡すジョブ内容。"""

    kind: str  # ingest_path 等
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Job:
    """ジョブ本体。"""

    job_id: str
    payload: JobPayload
    config_snapshot: dict[str, Any]
    created_at: str
    last_update: str
    status: JobStatus = JobStatus.PENDING
    message: str = ""


class BackgroundWorker:
    """ingest 処理を非同期に実行するための簡易ワーカー。"""

    def __init__(self) -> None:
        """コンストラクタ。"""
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._jobs: Dict[str, Job] = {}
        self._worker_task: Optional[asyncio.Task[None]] = None

        self._jobs_lock = threading.Lock()

    async def start(self) -> None:
        """ワーカーを起動する。"""
        if self._worker_task is not None:
            return

        self._worker_task = asyncio.create_task(
            self._worker_loop(), name="ingest-worker"
        )

    async def shutdown(self) -> None:
        """ワーカーを終了する。"""
        if self._worker_task is None:
            return

        self._worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker_task

        self._worker_task = None

    def _update(self, job: Job, status: JobStatus) -> None:
        """ジョブの状態を更新する。

        Args:
            job (Job): ジョブ
        """
        job.status = status
        job.last_update = str(datetime.datetime.now())

    def submit(self, payload: JobPayload) -> Job:
        """ジョブをキューに追加する。

        Args:
            payload (JobPayload): ワーカーに渡すジョブ内容

        Returns:
            Job: ジョブ
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
        """特定 ID のジョブを取得する。

        Args:
            job_id (str): ジョブ ID

        Returns:
            Optional[Job]: ジョブ
        """
        return self._jobs.get(job_id)

    def get_jobs(self) -> Dict[str, Job]:
        """ジョブを全件取得する。

        Returns:
            Dict[str, Job]: ジョブ
        """
        return self._jobs.copy()

    def remove_job(self, job_id: str) -> None:
        """ジョブをキューから削除する。

        Args:
            job_id (str): ジョブ ID
        """
        with self._jobs_lock:
            self._jobs.pop(job_id, None)

    def remove_completed_jobs(self) -> None:
        """実行完了しているジョブをキューから削除する。"""
        completed_ids = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED)
        ]
        for job_id in completed_ids:
            self.remove_job(job_id)

    async def _worker_loop(self) -> None:
        """ワーカーループ。"""
        while True:
            job = await self._queue.get()
            await self._dispatch(job)
            self._queue.task_done()

    async def _dispatch(self, job: Job) -> None:
        """ディスパッチャー。

        Args:
            job (Job): 次に実行するジョブ

        Raises:
            ValueError: 未知のジョブ種別
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
        except Exception as e:
            logger.exception(e)
            self._update(job=job, status=JobStatus.FAILED)
            job.message = str(e)


def get_worker() -> BackgroundWorker:
    """バックグラウンドワーカーシングルトンの getter。

    Returns:
        BackgroundWorker: ワーカー
    """
    global _worker

    if _worker is None:
        with _lock:
            if _worker is None:
                _worker = BackgroundWorker()

    return _worker
