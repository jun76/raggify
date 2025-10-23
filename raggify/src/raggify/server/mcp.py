from __future__ import annotations

from fastapi_mcp.server import FastApiMCP

from ..config import cfg
from .fastapi import app as fastapi

__all__ = ["app"]

# FastAPI アプリを MCP サーバとして公開
app = FastApiMCP(
    fastapi,
    name=cfg.project_name,
    include_operations=[
        "query_text_text",
        "query_text_image",
        "query_image_image",
        "query_text_audio",
        "query_audio_audio",
    ],
)
