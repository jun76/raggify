from __future__ import annotations

from typing import Callable, Optional, Sequence

from ..core.event import async_loop_runner
from ..ingest.upsert import aupsert_nodes
from ..logger import logger
from ..runtime import get_runtime as _rt

__all__ = [
    "ingest_path",
    "aingest_path",
    "ingest_path_list",
    "aingest_path_list",
    "ingest_url",
    "aingest_url",
    "ingest_url_list",
    "aingest_url_list",
]


def _read_list(path: str) -> list[str]:
    """Read a list of paths or URLs from a file.

    Args:
        path (str): Path to the list file.

    Returns:
        list[str]: Loaded list.
    """
    lst = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            temp = []
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                temp.append(stripped)
            lst = temp
    except OSError as e:
        logger.warning(f"failed to read config file: {e}")

    return lst


def ingest_path(
    path: str,
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Ingest, embed, and store content from a local path (directory or file).

    Directories are traversed recursively to ingest multiple files.

    Args:
        path (str): Target path.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag for the job. Defaults to lambda:False.
    """
    async_loop_runner.run(
        lambda: aingest_path(
            path, pipe_batch_size=pipe_batch_size, is_canceled=is_canceled
        )
    )


async def aingest_path(
    path: str,
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Asynchronously ingest, embed, and store content from a local path.

    Directories are traversed recursively to ingest multiple files.

    Args:
        path (str): Target path.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag for the job. Defaults to lambda:False.
    """
    rt = _rt()
    texts, images, audios, videos = await rt.file_loader.aload_from_path(path)
    pipe_batch_size = pipe_batch_size or rt.cfg.pipeline.batch_size

    await aupsert_nodes(
        text_nodes=texts,
        image_nodes=images,
        audio_nodes=audios,
        video_nodes=videos,
        persist_dir=rt.cfg.pipeline.persist_dir,
        pipe_batch_size=pipe_batch_size,
        is_canceled=is_canceled,
    )


def ingest_path_list(
    lst: str | Sequence[str],
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Ingest, embed, and store content from multiple paths in a list.

    Args:
        lst (str | Sequence[str]): Text file path or in-memory sequence.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag for the job. Defaults to lambda:False.
    """
    async_loop_runner.run(
        lambda: aingest_path_list(
            lst, pipe_batch_size=pipe_batch_size, is_canceled=is_canceled
        )
    )


async def aingest_path_list(
    lst: str | Sequence[str],
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Asynchronously ingest, embed, and store content from multiple paths.

    Args:
        lst (str | Sequence[str]): Text file path or in-memory sequence.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag for the job. Defaults to lambda:False.
    """
    if isinstance(lst, str):
        lst = _read_list(lst)

    rt = _rt()
    texts, images, audios, videos = await rt.file_loader.aload_from_paths(
        paths=list(lst), is_canceled=is_canceled
    )
    pipe_batch_size = pipe_batch_size or rt.cfg.pipeline.batch_size
    await aupsert_nodes(
        text_nodes=texts,
        image_nodes=images,
        audio_nodes=audios,
        video_nodes=videos,
        persist_dir=rt.cfg.pipeline.persist_dir,
        pipe_batch_size=pipe_batch_size,
        is_canceled=is_canceled,
    )


def ingest_url(
    url: str,
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Ingest, embed, and store content from a URL.

    For sitemaps (.xml), traverse the tree to ingest multiple sites.

    Args:
        url (str): Target URL.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag for the job. Defaults to lambda:False.
    """
    async_loop_runner.run(
        lambda: aingest_url(
            url=url, pipe_batch_size=pipe_batch_size, is_canceled=is_canceled
        )
    )


async def aingest_url(
    url: str,
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Asynchronously ingest, embed, and store content from a URL.

    For sitemaps (.xml), traverse the tree to ingest multiple sites.

    Args:
        url (str): Target URL.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag for the job. Defaults to lambda:False.
    """
    rt = _rt()
    texts, images, audios, videos = await rt.web_page_loader.aload_from_url(
        url=url, is_canceled=is_canceled
    )
    pipe_batch_size = pipe_batch_size or rt.cfg.pipeline.batch_size

    await aupsert_nodes(
        text_nodes=texts,
        image_nodes=images,
        audio_nodes=audios,
        video_nodes=videos,
        persist_dir=rt.cfg.pipeline.persist_dir,
        pipe_batch_size=pipe_batch_size,
        is_canceled=is_canceled,
    )


def ingest_url_list(
    lst: str | Sequence[str],
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Ingest, embed, and store content from multiple URLs in a list.

    Args:
        lst (str | Sequence[str]): Text file path or in-memory URL list.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag for the job. Defaults to lambda:False.
    """
    async_loop_runner.run(
        lambda: aingest_url_list(
            lst, pipe_batch_size=pipe_batch_size, is_canceled=is_canceled
        )
    )


async def aingest_url_list(
    lst: str | Sequence[str],
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Asynchronously ingest, embed, and store content from multiple URLs.

    Args:
        lst (str | Sequence[str]): Text file path or in-memory URL list.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag for the job. Defaults to lambda:False.
    """
    if isinstance(lst, str):
        lst = _read_list(lst)

    rt = _rt()
    texts, images, audios, videos = await rt.web_page_loader.aload_from_urls(
        urls=list(lst), is_canceled=is_canceled
    )
    pipe_batch_size = pipe_batch_size or rt.cfg.pipeline.batch_size

    await aupsert_nodes(
        text_nodes=texts,
        image_nodes=images,
        audio_nodes=audios,
        video_nodes=videos,
        persist_dir=rt.cfg.pipeline.persist_dir,
        pipe_batch_size=pipe_batch_size,
        is_canceled=is_canceled,
    )
