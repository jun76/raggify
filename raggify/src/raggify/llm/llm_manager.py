from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto

from llama_index.core.llms import LLM

from ..logger import logger


class LLMUsage(StrEnum):
    IMAGE_CAPTIONER = auto()
    AUDIO_CAPTIONER = auto()
    VIDEO_CAPTIONER = auto()


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
    def image_captioner(self) -> LLM:
        """Get the image caption transform LLM.

        Raises:
            RuntimeError: If uninitialized.

        Returns:
            LLM: Image caption transform LLM.
        """
        return self.get_container(LLMUsage.IMAGE_CAPTIONER).llm

    @property
    def audio_captioner(self) -> LLM:
        """Get the audio caption transform LLM.

        Raises:
            RuntimeError: If uninitialized.

        Returns:
            LLM: Audio caption transform LLM.
        """
        return self.get_container(LLMUsage.AUDIO_CAPTIONER).llm

    @property
    def video_captioner(self) -> LLM:
        """Get the video caption transform LLM.

        Raises:
            RuntimeError: If uninitialized.

        Returns:
            LLM: Video caption transform LLM.
        """
        return self.get_container(LLMUsage.VIDEO_CAPTIONER).llm

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
