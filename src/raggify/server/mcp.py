from __future__ import annotations

from fastapi_mcp.server import FastApiMCP

from .fastapi import app as fastapi

__all__ = ["app"]


def _get_cfg():
    from ..runtime import get_runtime

    return get_runtime().cfg


# FastAPI アプリを MCP サーバとして公開
app = FastApiMCP(
    fastapi,
    name=_get_cfg().project_name,
    include_operations=[
        "query_text_text",
        "query_text_image",
        "query_image_image",
        "query_text_audio",
        "query_audio_audio",
    ],
)
