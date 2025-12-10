from __future__ import annotations

import asyncio
from io import BytesIO
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import numpy as np

from ...core.const import EXTRA_PKG_NOT_FOUND_MSG
from .multi_modal_base import AudioEmbedding, AudioType

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import Embedding

__all__ = ["ClapEmbedding"]


class ClapEmbedding(AudioEmbedding):
    """Embedding class dedicated to laion CLAP.

    Implemented with reference to MultiModalEmbedding;
    uses BaseEmbedding -> AudioEmbedding
    because MultiModalEmbedding lacks audio support.
    """

    @classmethod
    def class_name(cls) -> str:
        """Class name.

        Returns:
            str: Class name.
        """
        return "ClapEmbedding"

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: str = "cuda",
        embed_batch_size: int = 8,
    ) -> None:
        """Constructor."""

        torch = self._import_or_raise(
            module="torch",
            pkg="torch",
        )
        transformers = self._import_or_raise(
            module="transformers",
            pkg="transformers",
        )
        soundfile = self._import_or_raise(
            module="soundfile",
            pkg="soundfile",
        )

        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
        )

        self._device = device
        self._torch = torch
        self._soundfile = soundfile
        self._processor = transformers.AutoProcessor.from_pretrained(model_name)
        self._model = transformers.ClapModel.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

        feature_extractor = getattr(self._processor, "feature_extractor", None)
        sampling_rate = getattr(feature_extractor, "sampling_rate", 48000)
        self._target_sampling_rate = sampling_rate

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Embed a query string asynchronously.

        Args:
            query (str): Query string.

        Returns:
            Embedding: Embedding vector.
        """
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> Embedding:
        """Embed a single text synchronously.

        Args:
            text (str): Text content.

        Returns:
            Embedding: Embedding vector.
        """
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """Embed multiple texts synchronously.

        Args:
            texts (list[str]): Texts.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        model_inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return self._run_model_forward(self._model.get_text_features, model_inputs)

    def _get_query_embedding(self, query: str) -> Embedding:
        """Embed a query string synchronously.

        Args:
            query (str): Query string.

        Returns:
            Embedding: Embedding vector.
        """
        return self._get_text_embedding(query)

    def _get_audio_embeddings(
        self, audio_file_paths: list[AudioType]
    ) -> list[Embedding]:
        """Synchronous wrapper for the CLAP audio embedding API.

        Args:
            audio_file_paths (list[AudioType]): Audio file paths.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        audio_arrays = self._prepare_audio_inputs(audio_file_paths)
        model_inputs = self._processor(
            audio=audio_arrays,
            sampling_rate=self._target_sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        return self._run_model_forward(self._model.get_audio_features, model_inputs)

    async def aget_audio_embedding_batch(
        self, audio_file_paths: list[AudioType], show_progress: bool = False
    ) -> list[Embedding]:
        """Async batch interface for audio embeddings
        (modeled after `aget_image_embedding_batch`).

        Args:
            audio_file_paths (list[AudioType]): Audio file paths.
            show_progress (bool, optional): Show progress. Defaults to False.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        from llama_index.core.callbacks.schema import CBEventType, EventPayload

        cur_batch: list[AudioType] = []
        callback_payloads: list[tuple[str, list[AudioType]]] = []
        result_embeddings: list[Embedding] = []
        embeddings_coroutines: list[Coroutine] = []
        for idx, audio_file_path in enumerate(audio_file_paths):
            cur_batch.append(audio_file_path)
            if (
                idx == len(audio_file_paths) - 1
                or len(cur_batch) == self.embed_batch_size
            ):
                # flush
                event_id = self.callback_manager.on_event_start(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                )
                callback_payloads.append((event_id, cur_batch))
                embeddings_coroutines.append(self._aget_audio_embeddings(cur_batch))
                cur_batch = []

        # flatten the results of asyncio.gather, which is a list of embeddings lists
        nested_embeddings = []
        if show_progress:
            try:
                from tqdm.asyncio import tqdm_asyncio

                nested_embeddings = await tqdm_asyncio.gather(
                    *embeddings_coroutines,
                    total=len(embeddings_coroutines),
                    desc="Generating embeddings",
                )
            except ImportError:
                nested_embeddings = await asyncio.gather(*embeddings_coroutines)
        else:
            nested_embeddings = await asyncio.gather(*embeddings_coroutines)

        result_embeddings = [
            embedding for embeddings in nested_embeddings for embedding in embeddings
        ]

        for (event_id, audio_batch), embeddings in zip(
            callback_payloads, nested_embeddings
        ):
            self.callback_manager.on_event_end(
                CBEventType.EMBEDDING,
                payload={
                    EventPayload.CHUNKS: audio_batch,
                    EventPayload.EMBEDDINGS: embeddings,
                },
                event_id=event_id,
            )

        return result_embeddings

    async def _aget_audio_embeddings(
        self, audio_file_paths: list[AudioType]
    ) -> list[Embedding]:
        """Async wrapper for the CLAP audio embedding API.

        At implementation time, only synchronous CLAP interfaces exist.

        Args:
            audio_file_paths (list[AudioType]): Audio file paths.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        return await asyncio.to_thread(self._get_audio_embeddings, audio_file_paths)

    def _run_model_forward(
        self,
        forward: Callable[..., Any],
        processor_outputs: Any,
    ) -> list[Embedding]:
        """Run the model forward pass and return normalized embeddings.

        Args:
            forward (Callable[..., Any]): Model forward function.
            processor_outputs: Processor outputs.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        inputs = self._move_to_device(processor_outputs)
        with self._torch.no_grad():
            embeddings = forward(**inputs)

        normalized = self._torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return normalized.cpu().tolist()

    def _move_to_device(self, processor_outputs: Any) -> Any:
        """Move tensors to the specified device.

        Args:
            processor_outputs: Processor outputs.

        Returns:
            Any: Processor outputs moved to the specified device.
        """
        return {key: value.to(self._device) for key, value in processor_outputs.items()}

    def _prepare_audio_inputs(
        self, audio_file_paths: list[AudioType]
    ) -> list[np.ndarray]:
        """Prepare audio inputs by loading and resampling audio files.

        Args:
            audio_file_paths (list[AudioType]): List of audio file paths or buffers.

        Returns:
            list[np.ndarray]: List of audio waveforms as numpy arrays.
        """
        arrays: list[np.ndarray] = []
        for audio in audio_file_paths:
            waveform, sampling_rate = self._load_audio(audio)
            arrays.append(self._resample_waveform(waveform, sampling_rate))

        return arrays

    def _load_audio(self, audio: AudioType) -> tuple[np.ndarray, int]:
        """Load an audio file (path or buffer).

        Args:
            audio (AudioType): Audio file path or buffer.

        Returns:
            tuple[np.ndarray, int]: Loaded waveform and its sampling rate.
        """
        if isinstance(audio, BytesIO):
            audio.seek(0)

        data, sampling_rate = self._soundfile.read(audio)
        waveform = np.asarray(data, dtype=np.float32)

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        return waveform, sampling_rate

    def _resample_waveform(
        self, waveform: np.ndarray, sampling_rate: int
    ) -> np.ndarray:
        """Resample the waveform to the target sampling rate using simple linear interpolation.

        Args:
            waveform (np.ndarray): Input waveform.
            sampling_rate (int): Sampling rate of the input waveform.

        Returns:
            np.ndarray: Resampled waveform.
        """
        if sampling_rate == self._target_sampling_rate:
            return waveform

        if waveform.size == 0:
            return waveform

        duration = waveform.shape[0] / sampling_rate
        target_length = max(1, int(round(duration * self._target_sampling_rate)))
        origin = np.linspace(0.0, duration, num=waveform.shape[0], endpoint=False)
        target = np.linspace(0.0, duration, num=target_length, endpoint=False)

        return np.interp(target, origin, waveform).astype(np.float32)

    @staticmethod
    def _import_or_raise(*, module: str, pkg: str) -> ModuleType:
        """Import a dependency package, and raise an informative error if it fails.

        Args:
            module (str): Module name to import.
            pkg (str): Package name for error message.

        Raises:
            ImportError: If the module cannot be imported.

        Returns:
            ModuleType: Imported module.
        """
        try:
            return __import__(module)
        except ImportError as e:
            raise ImportError(
                EXTRA_PKG_NOT_FOUND_MSG.format(
                    pkg=pkg,
                    extra="localmodel",
                    feature="ClapEmbedding",
                )
            ) from e
