from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

from llama_index.core.schema import TransformComponent

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode

__all__ = ["BaseTransform"]


class BaseTransform(TransformComponent):
    """Base class for all transform components."""

    def __init__(self, is_canceled: Callable[[], bool]) -> None:
        """Constructor.

        Args:
            is_canceled (Callable[[], bool]): Cancellation flag for the job.
        """
        self._record_nodes: (
            Callable[[TransformComponent, Sequence[BaseNode]], None] | None
        ) = None
        self._is_canceled = is_canceled

    def set_pipe_callback(
        self, record_nodes: Callable[[TransformComponent, Sequence[BaseNode]], None]
    ) -> None:
        """Set pipe callback.

        Args:
            record_nodes (Callable[[TransformComponent, Sequence[BaseNode]], None]):
                Callback to register transformed nodes in the pipeline.
        """
        self._record_nodes = record_nodes
