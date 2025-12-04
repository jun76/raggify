from __future__ import annotations

from typing import List

from raggify.ingest.parser import BaseParser


class DummyParser(BaseParser):
    """Simple parser stub for tests."""

    def __init__(self) -> None:
        self._exts = {".png", ".mp3"}

    @property
    def ingest_target_exts(self) -> set[str]:
        return self._exts

    async def aparse(self, root: str):
        return []

    def parse(self, root: str):
        return []
