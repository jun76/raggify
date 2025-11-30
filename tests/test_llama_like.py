from __future__ import annotations

import asyncio
from typing import cast

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.schema import QueryBundle

from raggify.llama_like.core.indices.multi_modal.retriever import (
    AudioEncoders,
    AudioRetriever,
    VideoRetriever,
)
from raggify.llama_like.embeddings.bedrock import MultiModalBedrockEmbedding
from raggify.llama_like.embeddings.clap import ClapEmbedding, ModelName
from tests.utils.mock_llama_like import (
    DummyEmbedModel,
    DummyVectorStoreIndex,
    setup_bedrock_mock,
    setup_clap_mock,
)


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
    embed = ClapEmbedding(model_name=ModelName.EFFECT_VARLEN, device="cpu")
    vecs = embed._get_audio_embeddings(["tests/data/audios/sample.wav"])
    assert len(vecs) == 1
    assert vecs[0] == [0.3, 0.4]
