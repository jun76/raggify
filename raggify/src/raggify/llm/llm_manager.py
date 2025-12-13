from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto

from llama_index.core.llms import LLM

from ..logger import logger


class LLMUsage(StrEnum):
    TEXT_SUMMARIZER = auto()
    IMAGE_SUMMARIZER = auto()
    AUDIO_SUMMARIZER = auto()
    VIDEO_SUMMARIZER = auto()


__all__ = ["LLMContainer", "LLMManager"]


@dataclass(kw_only=True)
class LLMContainer:
    """Container for LLM-related parameters per modality."""

    provider_name: str
    llm: LLM


class LLMManager:
    """Manager class for LLM."""

    def __init__(
        self,
        conts: dict[LLMUsage, LLMContainer],
    ) -> None:
        """Constructor.

        Args:
            conts (dict[LLMUsage, LLMContainer]):
                Mapping of LLMUsage to LLM container.
        """
        self._conts = conts

        for llm_usage, cont in self._conts.items():
            logger.debug(f"{cont.provider_name} {llm_usage} initialized")

    @property
    def name(self) -> str:
        """Provider names.

        Returns:
            str: Provider names.
        """
        return ", ".join([cont.provider_name for cont in self._conts.values()])

    @property
    def llm_usage(self) -> set[LLMUsage]:
        """LLM usages supported by this LLM manager.

        Returns:
            set[LLMUsage]: LLM usages.
        """
        return set(self._conts.keys())

    @property
    def text_summarizer(self) -> LLM:
        """Get the text summarize transform LLM.

        Raises:
            RuntimeError: If uninitialized.

        Returns:
            LLM: Text summarize transform LLM.
        """
        return self.get_container(LLMUsage.TEXT_SUMMARIZER).llm

    @property
    def image_summarizer(self) -> LLM:
        """Get the image summarize transform LLM.

        Raises:
            RuntimeError: If uninitialized.

        Returns:
            LLM: Image summarize transform LLM.
        """
        return self.get_container(LLMUsage.IMAGE_SUMMARIZER).llm

    @property
    def audio_summarizer(self) -> LLM:
        """Get the audio summarize transform LLM.

        Raises:
            RuntimeError: If uninitialized.

        Returns:
            LLM: Audio summarize transform LLM.
        """
        return self.get_container(LLMUsage.AUDIO_SUMMARIZER).llm

    @property
    def video_summarizer(self) -> LLM:
        """Get the video summarize transform LLM.

        Raises:
            RuntimeError: If uninitialized.

        Returns:
            LLM: Video summarize transform LLM.
        """
        return self.get_container(LLMUsage.VIDEO_SUMMARIZER).llm

    def get_container(self, llm_usage: LLMUsage) -> LLMContainer:
        """Get the LLM container for a llm usage.

        Args:
            llm_usage (LLMUsage): LLM usage.

        Raises:
            RuntimeError: If uninitialized.

        Returns:
            LLMContainer: LLM container.
        """
        cont = self._conts.get(llm_usage)
        if cont is None:
            raise RuntimeError(f"LLM {llm_usage} is not initialized")

        return cont
