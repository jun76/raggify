from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from llama_index.core.schema import (
    ImageNode,
    NodeRelationship,
    ObjectType,
    RelatedNodeInfo,
    TextNode,
)

from raggify.core.metadata import BasicMetaData
from raggify.core.metadata import MetaKeys as MK
from raggify.ingest.transform import (
    AddChunkIndexTransform,
    AudioSplitter,
    VideoSplitter,
    make_audio_embed_transform,
    make_image_embed_transform,
    make_text_embed_transform,
    make_video_embed_transform,
)
from raggify.llama.core.schema import AudioNode, Modality, VideoNode
from tests.utils.mock_embed import (
    DummyAudioBase,
    DummyAudioEmbedding,
    DummyImageEmbedding,
    DummyMultiModalBase,
    DummyTextEmbedding,
    DummyVideoBase,
    DummyVideoEmbedding,
    make_dummy_manager,
)
from tests.utils.node_factory import make_sample_text_node


@pytest.fixture(autouse=True)
def patch_embedding_bases(monkeypatch):
    monkeypatch.setattr(
        "llama_index.core.embeddings.multi_modal_base.MultiModalEmbedding",
        DummyMultiModalBase,
    )
    monkeypatch.setattr(
        "raggify.llama.embeddings.multi_modal_base.AudioEmbedding",
        DummyAudioBase,
    )
    monkeypatch.setattr(
        "raggify.llama.embeddings.multi_modal_base.VideoEmbedding",
        DummyVideoBase,
    )


def _assign_doc_id(node: TextNode, doc_id: str, node_type: ObjectType) -> None:
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
        node_id=doc_id,
        node_type=node_type,
        metadata={},
        hash="hash",
    )


def _make_media_node(node_cls, path: Path, ref_doc_id: str) -> TextNode:
    meta = BasicMetaData(file_path=str(path))
    node = node_cls(
        text="payload",
        id_=str(path),
        metadata=meta.to_dict(),
    )
    _assign_doc_id(node, ref_doc_id, ObjectType.TEXT)
    return node


def test_add_chunk_index_transform_assigns_per_doc():
    t = AddChunkIndexTransform()
    node1 = make_sample_text_node()
    node2 = make_sample_text_node()
    node3 = make_sample_text_node()
    _assign_doc_id(node1, "doc-a", ObjectType.TEXT)
    _assign_doc_id(node2, "doc-a", ObjectType.TEXT)
    _assign_doc_id(node3, "doc-b", ObjectType.TEXT)

    nodes = t([node1, node2, node3])
    assert nodes[0].metadata["chunk_no"] == 0
    assert nodes[1].metadata["chunk_no"] == 1
    assert nodes[2].metadata["chunk_no"] == 0

    async_nodes = asyncio.run(t.acall([node1, node2, node3]))
    assert async_nodes[0].metadata["chunk_no"] == 0


def test_audio_splitter_splits_and_rebuilds(monkeypatch, tmp_path):
    src = Path("tests/data/audios/sample.wav")
    local = tmp_path / "sample.wav"
    shutil.copy(src, local)

    chunk_dir = tmp_path / "chunks"

    def fake_probe(self, path):
        return 10.0

    def fake_create(self, path):
        chunk_paths = []
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            dest = chunk_dir / f"chunk_{i}.wav"
            shutil.copy(local, dest)
            chunk_paths.append(str(dest))
        return chunk_paths

    monkeypatch.setattr(
        "raggify.ingest.transform._BaseMediaSplitter._probe_duration", fake_probe
    )
    monkeypatch.setattr(
        "raggify.ingest.transform._BaseMediaSplitter._create_segments", fake_create
    )

    node = _make_media_node(AudioNode, local, ref_doc_id="audio-doc")

    splitter = AudioSplitter(chunk_seconds=2)
    result = splitter([node])

    assert len(result) == 2
    assert all(isinstance(n, AudioNode) for n in result)
    assert {n.metadata[MK.CHUNK_NO] for n in result} == {0, 1}


def test_video_splitter_splits(monkeypatch, tmp_path):
    src = Path("tests/data/videos/sample.mp4")
    local = tmp_path / "sample.mp4"
    shutil.copy(src, local)

    def fake_probe(self, path):
        return 12.0

    def fake_create(self, path):
        files = []
        for i in range(3):
            dest = tmp_path / f"segment_{i}.mp4"
            shutil.copy(local, dest)
            files.append(str(dest))
        return files

    monkeypatch.setattr(
        "raggify.ingest.transform._BaseMediaSplitter._probe_duration", fake_probe
    )
    monkeypatch.setattr(
        "raggify.ingest.transform._BaseMediaSplitter._create_segments", fake_create
    )

    node = _make_media_node(VideoNode, local, ref_doc_id="video-doc")

    splitter = VideoSplitter(chunk_seconds=3)
    nodes = splitter([node])

    assert len(nodes) == 3
    assert all(isinstance(n, VideoNode) for n in nodes)


def test_make_text_embed_transform_writes_embeddings():
    manager = make_dummy_manager({Modality.TEXT: DummyTextEmbedding()})
    transform = make_text_embed_transform(manager)

    node = make_sample_text_node()
    node.metadata[MK.TEMP_FILE_PATH] = "/tmp/temp.txt"
    node.metadata[MK.BASE_SOURCE] = "/orig/path.txt"
    blank = TextNode(text="  ", id_="blank")

    result = asyncio.run(transform.acall([node, blank]))

    assert result[0].embedding == [1.0, 1.0]
    assert result[0].metadata["file_path"] == "/orig/path.txt"
    assert result[1].embedding is None


def test_make_image_embed_transform():
    manager = make_dummy_manager({Modality.IMAGE: DummyImageEmbedding()})
    transform = make_image_embed_transform(manager)

    node = ImageNode(id_="img", metadata={"file_path": "tests/data/images/sample.png"})
    other = TextNode(text="text", id_="txt")

    nodes = asyncio.run(transform.acall([node, other]))
    assert nodes[0].embedding == [0.1, 0.1]
    assert nodes[1].embedding is None


def test_make_audio_embed_transform():
    manager = make_dummy_manager({Modality.AUDIO: DummyAudioEmbedding()})
    transform = make_audio_embed_transform(manager)

    audio = TextNode(
        text="audio",
        id_="audio",
        metadata={"file_path": "tests/data/audios/sample.mp3"},
    )
    skipped = TextNode(text="text", id_="skip")

    nodes = asyncio.run(transform.acall([audio, skipped]))
    assert nodes[0].embedding == [0.2, 0.2]
    assert nodes[1].embedding is None


def test_make_video_embed_transform():
    manager = make_dummy_manager({Modality.VIDEO: DummyVideoEmbedding()})
    transform = make_video_embed_transform(manager)

    video = TextNode(
        text="video",
        id_="video",
        metadata={"file_path": "tests/data/videos/sample.mp4"},
    )
    nodes = asyncio.run(transform.acall([video]))

    assert nodes[0].embedding == [0.3, 0.3]
