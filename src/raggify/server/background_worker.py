from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Dict, Optional

from ..logger import logger


class JobStatus(StrEnum):
    """ジョブの実行状態。"""

    PENDING = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    FAILED = auto()


@dataclass
class JobPayload:
    """ワーカーに渡すジョブ内容。

    kind: ingest_path / ingest_url などの識別子
    kwargs: ingest に渡すパラメータ
    """

    kind: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    """ジョブ本体。"""

    job_id: str
    payload: JobPayload
    config_snapshot: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    message: str = ""


class BackgroundWorker:
    """ingest 処理を非同期に実行するための簡易ワーカー。"""

    def __init__(self) -> None:
        """コンストラクタ。"""
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._jobs: Dict[str, Job] = {}
        self._worker_task: Optional[asyncio.Task[None]] = None

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

    def submit(self, payload: JobPayload) -> Job:
        """ジョブをキューに追加する。

        Args:
            payload (JobPayload): ワーカーに渡すジョブ内容

        Returns:
            Job: ジョブ
        """
        from ..runtime import get_runtime as _rt

        job_id = str(uuid.uuid4())
        cfg_snapshot = _rt().cfg.get_dict()
        job = Job(job_id=job_id, payload=payload, config_snapshot=cfg_snapshot)
        self._jobs[job_id] = job
        self._queue.put_nowait(job)

        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """ジョブを参照する。

        Args:
            job_id (str): ジョブ ID

        Returns:
            Optional[Job]: ジョブ
        """
        return self._jobs.get(job_id)

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

        job.status = JobStatus.RUNNING
        try:
            match job.payload.kind:
                case "ingest_path":
                    await ingest.aingest_path(**job.payload.kwargs)
                case "ingest_path_list":
                    await ingest.aingest_path_list(**job.payload.kwargs)
                case "ingest_url":
                    await ingest.aingest_url(**job.payload.kwargs)
                case "ingest_url_list":
                    await ingest.aingest_url_list(**job.payload.kwargs)
                case _:
                    raise ValueError(f"unknown job kind: {job.payload.kind}")

            job.status = JobStatus.SUCCEEDED
        except Exception as e:
            logger.exception(e)
            job.status = JobStatus.FAILED
            job.message = str(e)
