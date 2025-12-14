from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from ..config.config_manager import ConfigManager
from ..core.event import async_loop_runner
from ..core.exts import Exts
from ..logger import logger
from .loader.file_reader import (
    AudioReader,
    DummyMediaReader,
    MultiPDFReader,
    VideoReader,
)

if TYPE_CHECKING:
    from llama_index.core.readers.base import BaseReader
    from llama_index.core.schema import Document


__all__ = ["BaseParser", "DefaultParser", "LlamaParser", "create_parser"]


class BaseParser:
    """Base parser that reads local files and generates documents."""

    def __init__(
        self,
        cfg: ConfigManager,
        is_known_source: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Constructor.

        Args:
            cfg (ConfigManager): Configuration manager.
            is_known_source (Optional[Callable[[str], bool]]):
                Function to check if a source is known to skip. Defaults to None.
        """
        self._ingest_target_exts = cfg.ingest_target_exts
        self._is_known_source = is_known_source
        self._readers: dict[str, BaseReader] = {}

        if cfg.general.audio_embed_provider is not None:
            # Convert audio files to mp3 for ingestion
            audio_reader = AudioReader()
            for ext in Exts.AUDIO:
                self._readers[ext] = audio_reader

            if cfg.general.use_modality_fallback:
                # add readers for audio transcription if supported in the future
                pass

        # For cases like video -> image + audio decomposition, use a reader
        if cfg.general.video_embed_provider is None:
            if cfg.general.use_modality_fallback:
                video_reader = VideoReader()
                for ext in Exts.VIDEO:
                    self._readers[ext] = video_reader

    @property
    def ingest_target_exts(self) -> set[str]:
        """Get allowed extensions for ingestion.

        Returns:
            set[str]: Allowed extensions.
        """
        return self._ingest_target_exts

    async def aparse(
        self,
        root: str,
    ) -> list[Document]:
        """Parse data asynchronously from the input path.
        Args:
            root (str): Target path.

        Returns:
            list[Document]: List of documents parsed from the input path(s).
        """
        from llama_index.core.readers.file.base import SimpleDirectoryReader

        try:
            path = Path(root).absolute()
            if path.is_file():
                ext = Exts.get_ext(root)
                if ext not in self._ingest_target_exts:
                    logger.warning(f"skip unsupported extension: {ext}")
                    return []

            reader = SimpleDirectoryReader(
                input_dir=root if path.is_dir() else None,
                input_files=[root] if path.is_file() else None,
                recursive=True,
                required_exts=list(self._ingest_target_exts),
                file_extractor=self._readers,
                raise_on_error=True,
            )

            if self._is_known_source is not None:
                filtered_files = []
                for file_path in reader.input_files:
                    if not self._is_known_source(str(file_path)):
                        filtered_files.append(file_path)
                    else:
                        logger.debug(f"skip known source: {file_path}")

                if not filtered_files:
                    logger.debug("no new files found, skipped parsing")
                    return []

                reader.input_files = filtered_files

            docs = await reader.aload_data()
        except Exception as e:
            logger.exception(e)
            raise ValueError("failed to parse from path") from e

        return docs

    def parse(
        self,
        root: str,
    ) -> list[Document]:
        """Parse data from the input path.
        Args:
            root (str): Target path.

        Returns:
            list[Document]: List of documents parsed from the input path(s).
        """
        return async_loop_runner.run(lambda: self.aparse(root))


class DefaultParser(BaseParser):
    """Default parser that reads local files and generates documents."""

    def __init__(
        self,
        cfg: ConfigManager,
        is_known_source: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Constructor.

        Args:
            cfg (ConfigManager): Configuration manager.
            is_known_source (Optional[Callable[[str], bool]]):
                Function to check if a source is known to skip. Defaults to None.
        """
        from .loader.file_reader.html_reader import HTMLReader

        super().__init__(cfg=cfg, is_known_source=is_known_source)

        if cfg.general.image_embed_provider is not None:
            # Dictionary of custom readers to pass to SimpleDirectoryReader
            self._readers[Exts.PDF] = MultiPDFReader()

            if cfg.general.use_modality_fallback:
                # add readers for image transcription if supported in the future
                pass

        # HTML content is loaded via a temporary .html file
        self._readers[Exts.HTML] = HTMLReader()

        # For other media types, use dummy reader to pass through
        dummy_reader = DummyMediaReader()
        for ext in Exts.PASS_THROUGH_MEDIA:
            self._readers.setdefault(ext, dummy_reader)


class LlamaParser(BaseParser):
    """Llama Cloud parser that uses Llama Cloud API to parse files."""

    def __init__(
        self,
        cfg: ConfigManager,
        is_known_source: Optional[Callable[[str], bool]] = None,
        *args,
        **kwargs,
    ) -> None:
        """Constructor.

        Args:
            cfg (ConfigManager): Configuration manager.
            is_known_source (Optional[Callable[[str], bool]]):
                Function to check if a source is known to skip. Defaults to None.
        """
        from llama_cloud_services import LlamaParse

        super().__init__(cfg=cfg, is_known_source=is_known_source)

        # https://developers.llamaindex.ai/python/cloud/llamaparse/features/supported_document_types/
        llama_supported_exts: set[str] = {
            # Base types
            ".pdf",
            # Documents and presentations
            ".602",
            ".abw",
            ".cgm",
            ".cwk",
            ".doc",
            ".docx",
            ".docm",
            ".dot",
            ".dotm",
            ".hwp",
            ".key",
            ".lwp",
            ".mw",
            ".mcw",
            ".pages",
            ".pbd",
            ".ppt",
            ".pptm",
            ".pptx",
            ".pot",
            ".potm",
            ".potx",
            ".rtf",
            ".sda",
            ".sdd",
            ".sdp",
            ".sdw",
            ".sgl",
            ".sti",
            ".sxi",
            ".sxw",
            ".stw",
            ".sxg",
            ".txt",
            ".uof",
            ".uop",
            ".uot",
            ".vor",
            ".wpd",
            ".wps",
            ".xml",
            ".zabw",
            ".epub",
            # Images
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".svg",
            ".tiff",
            ".webp",
            ".web",
            ".htm",
            ".html",
            # Spreadsheets
            ".xlsx",
            ".xls",
            ".xlsm",
            ".xlsb",
            ".xlw",
            ".csv",
            ".dif",
            ".sylk",
            ".slk",
            ".prn",
            ".numbers",
            ".et",
            ".ods",
            ".fods",
            ".uos1",
            ".uos2",
            ".dbf",
            ".wk1",
            ".wk2",
            ".wk3",
            ".wk4",
            ".wks",
            ".123",
            ".wq1",
            ".wq2",
            ".wb1",
            ".wb2",
            ".wb3",
            ".qpw",
            ".xlr",
            ".eth",
            ".tsv",
            # Audio
            ".mp3",
            ".mp4",
            ".mpeg",
            ".mpga",
            ".m4a",
            ".wav",
            ".webm",
        }

        exts = self._ingest_target_exts & llama_supported_exts
        try:
            parser = LlamaParse(*args, **kwargs)
        except Exception as e:
            raise ValueError("failed to initialize LlamaParse") from e

        for ext in exts:
            self._readers.setdefault(ext, parser)


def create_parser(
    cfg: ConfigManager,
    is_known_source: Optional[Callable[[str], bool]] = None,
    *args,
    **kwargs,
) -> BaseParser:
    """Factory method to create a parser instance based on configuration.

    Args:
        cfg (ConfigManager): Configuration manager.
        is_known_source (Optional[Callable[[str], bool]]):
            Function to check if a source is known to skip. Defaults to None.

    Returns:
        Parser: An instance of a parser.
    """
    from ..config.ingest_config import ParserProvider

    match cfg.general.parser_provider:
        case ParserProvider.LOCAL:
            return DefaultParser(cfg=cfg, is_known_source=is_known_source)
        case ParserProvider.LLAMA_CLOUD:
            return LlamaParser(
                cfg=cfg, is_known_source=is_known_source, *args, **kwargs
            )
        case _:
            raise ValueError("unsupported parser provider")
