from __future__ import annotations

import asyncio
import io
import json
import sys
from types import SimpleNamespace
from typing import cast

import pytest
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.schema import QueryBundle

from raggify.core.exts import Exts
from raggify.llama_like.core.indices.multi_modal.retriever import (
    AudioEncoders,
    AudioRetriever,
    VideoRetriever,
)
from raggify.llama_like.embeddings.bedrock import MultiModalBedrockEmbedding
from raggify.llama_like.embeddings.clap import (
    ClapEmbedding,
    ClapModels,
    _AudioEncoderModel,
)
from raggify.llama_like.embeddings.multi_modal_base import AudioType, VideoType
from tests.utils.mock_llama_like import (
    DummyEmbedModel,
    DummyVectorStoreIndex,
    setup_bedrock_mock,
    setup_clap_mock,
)

from .config import configure_test_env

configure_test_env()


def test_audio_encoders_from_embed_model():
    embed = DummyEmbedModel(
        {
            "text": [1.0, 0.0],
            "audio": [0.5, 0.5],
            "image": [0.0, 1.0],
            "video": [0.2, 0.8],
        }
    )
    enc = AudioEncoders.from_embed_model(cast(BaseEmbedding, embed))
    assert enc.text_encoder is not None
    assert enc.audio_encoder is not None


def test_audio_retriever_text_and_audio_queries():
    embed = DummyEmbedModel(
        {
            "text": [0.1, 0.2],
            "audio": [0.3, 0.4],
            "image": [0.0, 0.0],
            "video": [0.0, 0.0],
        }
    )
    index = DummyVectorStoreIndex(embed_model=embed)
    retriever = AudioRetriever(index=cast(VectorStoreIndex, index), top_k=5)

    result = asyncio.run(retriever.atext_to_audio_retrieve("hello world"))
    assert len(result) == 1
    assert index.vector_store.queries  # ensures vector store called

    audio_result = asyncio.run(
        retriever.aaudio_to_audio_retrieve("tests/data/audios/sample.wav")
    )
    assert len(audio_result) == 1


def test_audio_retriever_query_bundle_with_embedding():
    embed = DummyEmbedModel(
        {
            "text": [0.1, 0.2],
            "audio": [0.3, 0.4],
            "image": [0.0, 0.0],
            "video": [0.0, 0.0],
        }
    )
    index = DummyVectorStoreIndex(embed_model=embed)
    retriever = AudioRetriever(index=cast(VectorStoreIndex, index))

    bundle = QueryBundle(query_str="text bundle", embedding=[0.9, 0.1])
    res = asyncio.run(retriever.atext_to_audio_retrieve(bundle))
    assert res[0].node.node_id == "node-1"
    assert res[0].node.metadata["source"] == "docstore"


def test_video_retriever_all_queries():
    embed = DummyEmbedModel(
        {
            "text": [0.1, 0.2],
            "audio": [0.3, 0.4],
            "image": [0.5, 0.6],
            "video": [0.7, 0.8],
        }
    )
    index = DummyVectorStoreIndex(embed_model=embed)
    retriever = VideoRetriever(index=cast(VectorStoreIndex, index))

    asyncio.run(retriever.atext_to_video_retrieve("hello"))
    asyncio.run(retriever.aimage_to_video_retrieve("tests/data/images/sample.png"))
    asyncio.run(retriever.aaudio_to_video_retrieve("tests/data/audios/sample.wav"))
    asyncio.run(retriever.avideo_to_video_retrieve("tests/data/videos/sample.mp4"))

    assert len(index.vector_store.queries) == 4


def test_multimodal_bedrock_handles_media(monkeypatch):
    calls = setup_bedrock_mock(monkeypatch)
    embed = MultiModalBedrockEmbedding(
        model_name="amazon.nova-2-multimodal-embeddings-v1:0"
    )
    embed._get_audio_embeddings(["tests/data/audios/sample.mp3"])
    embed._get_video_embeddings(["tests/data/videos/sample.mp4"])
    assert len(calls) == 2


def test_clap_embedding_returns_embeddings(monkeypatch):
    setup_clap_mock(monkeypatch)
    embed = ClapEmbedding(model_name=ClapModels.EFFECT_VARLEN, device="cpu")
    vecs = embed._get_audio_embeddings(["tests/data/audios/sample.wav"])
    assert len(vecs) == 1
    assert vecs[0] == [0.3, 0.4]


def test_bedrock_delegates_to_super_for_non_nova(monkeypatch):
    setup_bedrock_mock(monkeypatch)
    texts = {"text": [], "query": []}

    def record(name, value):
        def _fn(self, *args, **kwargs):
            texts[name].append(args[0])
            return value

        return _fn

    monkeypatch.setattr(
        "raggify.llama_like.embeddings.bedrock.BedrockEmbedding._get_text_embedding",
        record("text", ["base"]),
    )
    monkeypatch.setattr(
        "raggify.llama_like.embeddings.bedrock.BedrockEmbedding._get_query_embedding",
        record("query", ["query-base"]),
    )
    embed = MultiModalBedrockEmbedding(model_name="amazon.titan-embed-text-v1")
    assert embed._get_text_embedding("hello") == ["base"]
    assert embed._get_query_embedding("hello") == ["query-base"]


def test_bedrock_text_routes_nova(monkeypatch):
    calls = setup_bedrock_mock(monkeypatch)
    embed = MultiModalBedrockEmbedding(
        model_name="amazon.nova-2-multimodal-embeddings-v1:0",
        text_payload_overrides={"foo": "bar"},
    )
    embed._get_text_embedding("hello")
    asyncio.run(embed._aget_text_embedding("hello"))
    embed._get_text_embeddings(["hello"])
    asyncio.run(embed._aget_text_embeddings(["hello"]))
    embed._get_query_embedding("hello")
    asyncio.run(embed._aget_query_embedding("hello"))
    assert len(calls) == 6


def test_bedrock_read_media_payload_and_format(monkeypatch, tmp_path):
    setup_bedrock_mock(monkeypatch, mock_read=False)
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"123")
    embed = MultiModalBedrockEmbedding(
        model_name="amazon.nova-2-multimodal-embeddings-v1:0"
    )
    encoded, fmt = embed._read_media_payload(
        str(audio), expected_exts={".mp3"}, fallback_format_key="audio_format"
    )
    assert fmt == "mp3"
    buf = io.BytesIO(b"bytes")
    buf.name = "clip.wav"
    encoded2, fmt2 = embed._read_media_payload(
        buf, expected_exts={".wav"}, fallback_format_key="audio_format"
    )
    assert fmt2 == "wav"
    assert encoded2


def test_bedrock_resolve_media_format_fallback(monkeypatch):
    setup_bedrock_mock(monkeypatch, mock_read=False)
    embed = MultiModalBedrockEmbedding(
        model_name="amazon.nova-2-multimodal-embeddings-v1:0",
        audio_format="ogg",
    )
    fmt = embed._resolve_media_format(
        file_name=None,
        expected_exts={".wav"},
        fallback_format_key="audio_format",
    )
    assert fmt == "ogg"
    assert embed._normalize_media_format("jpg") == "jpeg"


def test_bedrock_build_single_embedding_body(monkeypatch):
    setup_bedrock_mock(monkeypatch)
    embed = MultiModalBedrockEmbedding(
        model_name="amazon.nova-2-multimodal-embeddings-v1:0",
        embedding_dimension=512,
        audio_params_overrides={"foo": "bar"},
    )
    body = embed._build_single_embedding_body(
        media_field="audio",
        media_payload={"format": "mp3"},
        params_override_key="audio_params_overrides",
    )
    params = body["singleEmbeddingParams"]
    assert params["embeddingDimension"] == 512
    assert params["foo"] == "bar"


def test_bedrock_async_media_batches(monkeypatch):
    setup_bedrock_mock(monkeypatch)
    embed = MultiModalBedrockEmbedding(
        model_name="amazon.nova-2-multimodal-embeddings-v1:0",
    )

    events: dict[str, list[tuple[tuple, dict]]] = {"start": [], "end": []}

    def on_start(*args, **kwargs):
        events["start"].append((args, kwargs))
        return f"evt-{len(events['start'])}"

    def on_end(*args, **kwargs):
        events["end"].append((args, kwargs))

    object.__setattr__(
        embed,
        "callback_manager",
        SimpleNamespace(
            on_event_start=on_start,
            on_event_end=on_end,
        ),
    )
    object.__setattr__(embed, "embed_batch_size", 1)

    audio_files = ["tests/data/audios/sample.mp3", "tests/data/audios/sample.wav"]
    video_files = ["tests/data/videos/sample.mp4"]

    audio_vecs = asyncio.run(
        embed._aget_audio_embeddings(cast(list[AudioType], audio_files))
    )
    video_vecs = asyncio.run(
        embed._aget_video_embeddings(cast(list[VideoType], video_files))
    )
    assert len(audio_vecs) == len(audio_files)
    assert len(video_vecs) == len(video_files)

    asyncio.run(
        embed.aget_audio_embedding_batch(
            cast(list[AudioType], audio_files), show_progress=False
        )
    )
    asyncio.run(
        embed.aget_video_embedding_batch(
            cast(list[VideoType], video_files), show_progress=False
        )
    )

    assert (
        len(events["start"])
        == len(events["end"])
        == len(audio_files) + len(video_files)
    )


def test_bedrock_media_batch_handles_empty(monkeypatch):
    setup_bedrock_mock(monkeypatch)
    embed = MultiModalBedrockEmbedding(
        model_name="amazon.nova-2-multimodal-embeddings-v1:0",
    )

    async def worker(_: list[str]):
        raise AssertionError("worker must not run")

    result = asyncio.run(
        embed._aget_media_embedding_batch([], worker, show_progress=False)
    )
    assert result == []


def test_bedrock_embed_media_files_payload_overrides(monkeypatch):
    calls = setup_bedrock_mock(monkeypatch)
    embed = MultiModalBedrockEmbedding(
        model_name="amazon.nova-2-multimodal-embeddings-v1:0",
        audio_payload_overrides={"gain": "unit"},
    )
    embed._embed_media_files(
        ["tests/data/audios/sample.mp3"],
        expected_exts=Exts.AUDIO,
        fallback_format_key="audio_format",
        media_field="audio",
        payload_overrides_key="audio_payload_overrides",
        params_override_key="audio_params_overrides",
        payload_builder=lambda fmt, encoded: {
            "format": fmt,
            "source": {"bytes": encoded},
        },
    )
    params = calls[-1]["singleEmbeddingParams"]
    assert params["audio"]["gain"] == "unit"


def test_bedrock_invoke_embeddings_parses_stream(monkeypatch):
    setup_bedrock_mock(monkeypatch)
    embed = MultiModalBedrockEmbedding(
        model_name="amazon.nova-2-multimodal-embeddings-v1:0",
    )

    class DummyStream:
        def read(self):
            return json.dumps(
                {"embeddings": [{"embedding": [1, 2, 3]}, [4, 5, 6]]}
            ).encode("utf-8")

    embed._client = SimpleNamespace(
        invoke_model=lambda **kwargs: {"body": DummyStream()}
    )

    vecs = embed._invoke_embeddings({"foo": "bar"})
    assert vecs == [[1, 2, 3], [4, 5, 6]]


def test_bedrock_invoke_embeddings_raises_when_missing(monkeypatch):
    setup_bedrock_mock(monkeypatch)
    embed = MultiModalBedrockEmbedding(
        model_name="amazon.nova-2-multimodal-embeddings-v1:0",
    )
    embed._client = SimpleNamespace(
        invoke_model=lambda **kwargs: {"body": json.dumps({})}
    )

    with pytest.raises(RuntimeError):
        embed._invoke_embeddings({})


def test_bedrock_invoke_single_embedding_uses_first(monkeypatch):
    setup_bedrock_mock(monkeypatch, mock_single=False)

    def fake_invoke(self, body):
        return [[9, 9], [8, 8]]

    monkeypatch.setattr(
        "raggify.llama_like.embeddings.bedrock.MultiModalBedrockEmbedding._invoke_embeddings",
        fake_invoke,
    )

    embed = MultiModalBedrockEmbedding(
        model_name="amazon.nova-2-multimodal-embeddings-v1:0",
    )
    assert embed._invoke_single_embedding({"foo": "bar"}) == [9, 9]


def test_bedrock_aget_image_embedding(monkeypatch):
    setup_bedrock_mock(monkeypatch, mock_read=False)
    embed = MultiModalBedrockEmbedding(
        model_name="amazon.nova-2-multimodal-embeddings-v1:0",
    )
    vec = asyncio.run(embed._aget_image_embedding("tests/data/images/sample.png"))
    assert vec == [0.1, 0.2]


def test_clap_text_methods(monkeypatch):
    setup_clap_mock(monkeypatch)
    embed = ClapEmbedding(model_name=ClapModels.EFFECT_VARLEN, device="cpu")
    object.__setattr__(
        embed,
        "callback_manager",
        SimpleNamespace(
            on_event_start=lambda *a, **k: "evt", on_event_end=lambda *a, **k: None
        ),
    )
    assert embed._get_text_embedding("hello") == [0.1, 0.2]
    assert embed._get_text_embeddings(["hi"]) == [[0.1, 0.2]]
    assert embed._get_query_embedding("hi") == [0.1, 0.2]


def test_clap_async_audio_batch(monkeypatch):
    setup_clap_mock(monkeypatch)
    embed = ClapEmbedding(
        model_name=ClapModels.EFFECT_VARLEN, device="cpu", embed_batch_size=1
    )
    object.__setattr__(
        embed,
        "callback_manager",
        SimpleNamespace(
            on_event_start=lambda *a, **k: "evt",
            on_event_end=lambda *a, **k: None,
        ),
    )
    result = asyncio.run(
        embed.aget_audio_embedding_batch(
            ["tests/data/audios/sample.wav", "tests/data/audios/sample.mp3"]
        )
    )
    assert len(result) == 2


def test_clap_audio_helpers(monkeypatch):
    setup_clap_mock(monkeypatch)
    embed = ClapEmbedding(model_name=ClapModels.EFFECT_VARLEN, device="cpu")
    vecs = embed._get_audio_embeddings(["tests/data/audios/sample.wav"])
    assert vecs[0] == [0.3, 0.4]
    async_result = asyncio.run(
        embed._aget_audio_embeddings(
            cast(list[AudioType], ["tests/data/audios/sample.wav"])
        )
    )
    assert async_result[0] == [0.3, 0.4]


def test_clap_async_audio_batch_with_progress(monkeypatch):
    setup_clap_mock(monkeypatch)
    fake_module = SimpleNamespace(
        tqdm_asyncio=SimpleNamespace(
            gather=lambda *args, **kwargs: asyncio.gather(*args)
        )
    )
    monkeypatch.setitem(sys.modules, "tqdm.asyncio", fake_module)
    embed = ClapEmbedding(
        model_name=ClapModels.EFFECT_VARLEN, device="cpu", embed_batch_size=1
    )
    object.__setattr__(
        embed,
        "callback_manager",
        SimpleNamespace(
            on_event_start=lambda *a, **k: "evt",
            on_event_end=lambda *a, **k: None,
        ),
    )
    result = asyncio.run(
        embed.aget_audio_embedding_batch(
            ["tests/data/audios/sample.wav", "tests/data/audios/sample.mp3"],
            show_progress=True,
        )
    )
    assert len(result) == 2


def test_clap_effect_short_initialization(monkeypatch):
    fake_module = setup_clap_mock(monkeypatch)
    ClapEmbedding(model_name=ClapModels.EFFECT_SHORT, device="cpu")
    inst = fake_module.instances[-1]
    assert inst.kwargs["enable_fusion"] is False
    assert inst.kwargs["amodel"] == _AudioEncoderModel.HTSAT_TINY
    assert inst.model_id == 1


def test_clap_effect_varlen_initialization(monkeypatch):
    fake_module = setup_clap_mock(monkeypatch)
    ClapEmbedding(model_name=ClapModels.EFFECT_VARLEN, device="cpu")
    inst = fake_module.instances[-1]
    assert inst.kwargs["enable_fusion"] is True
    assert inst.kwargs["amodel"] == _AudioEncoderModel.HTSAT_TINY
    assert inst.model_id == 3


def test_clap_unsupported_models(monkeypatch):
    setup_clap_mock(monkeypatch)
    with pytest.raises(NotImplementedError):
        ClapEmbedding(model_name=ClapModels.MUSIC, device="cpu")
