from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Coroutine, Optional, TypeVar

T = TypeVar("T")

__all__ = ["async_loop_runner"]


class AsyncLoopRunner:
    """共有イベントループ上でコルーチンを同期実行するヘルパ。"""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop and self._loop.is_running():
            return self._loop

        def _loop_worker(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=_loop_worker, args=(loop,), daemon=True)
        thread.start()

        self._loop = loop
        self._thread = thread

        return loop

    def run(self, coro_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """同期コンテキストから非同期処理を実行する。

        Args:
            coro_factory: 実行したいコルーチンを返す関数

        Returns:
            コルーチンの戻り値
        """
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro_factory(), loop)

        return future.result()


async_loop_runner = AsyncLoopRunner()
