from __future__ import annotations

import asyncio
from typing import Optional

import requests

from ...logger import logger

__all__ = ["arequest_get", "afetch_text"]


async def arequest_get(
    url: str,
    user_agent: str,
    timeout_sec: int,
    req_per_sec: int,
) -> requests.Response:
    """Async wrapper for HTTP GET.

    Args:
        url (str): Target URL.
        user_agent (str): User-Agent string.
        timeout_sec (int): Timeout in seconds.
        req_per_sec (int): Requests per second.

    Raises:
        requests.HTTPError: On HTTP errors.
        RuntimeError: When fetching fails.

    Returns:
        requests.Response: Response object.
    """
    headers = {"User-Agent": user_agent}
    res: Optional[requests.Response] = None

    try:
        res = await asyncio.to_thread(
            requests.get,
            url,
            timeout=timeout_sec,
            headers=headers,
        )
        res.raise_for_status()
    except requests.HTTPError as e:
        status = res.status_code if res is not None else "unknown"
        raise requests.HTTPError(f"HTTP {status}: {str(e)}") from e
    except requests.RequestException as e:
        raise RuntimeError("failed to fetch url") from e
    finally:
        await asyncio.sleep(1 / req_per_sec)

    return res


async def afetch_text(
    url: str, user_agent: str, timeout_sec: int, req_per_sec: int
) -> str:
    """Fetch HTML and return the response text.

    Args:
        url (str): Target URL.
        user_agent (str): User-Agent string.
        timeout_sec (int): Timeout in seconds.
        req_per_sec (int): Requests per second.

    Returns:
        str: Response body.
    """
    try:
        res = await arequest_get(
            url=url,
            user_agent=user_agent,
            timeout_sec=timeout_sec,
            req_per_sec=req_per_sec,
        )
    except Exception as e:
        logger.exception(e)
        return ""

    return res.text
