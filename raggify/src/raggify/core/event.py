from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Coroutine, Optional, TypeVar

T = TypeVar("T")

__all__ = ["async_loop_runner"]


class AsyncLoopRunner:
    """A helper class to run async coroutines in a separate event loop
    running in a background thread."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Ensure the background event loop is running.

        Returns:
            asyncio.AbstractEventLoop: The background event loop.
        """
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
        """Run a coroutine in the background event loop and wait for the result.

        Args:
            coro_factory (Callable[[], Coroutine[Any, Any, T]]):
                A factory function that returns the coroutine to run.

        Returns:
            T: The result of the coroutine.
        """
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro_factory(), loop)

        return future.result()


async_loop_runner = AsyncLoopRunner()
