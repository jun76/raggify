from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import cast

import pytest
from llama_index.core.schema import (
    BaseNode,
    Document,
    ImageNode,
    NodeRelationship,
    ObjectType,
    RelatedNodeInfo,
    TextNode,
)

from raggify.config.ingest_config import IngestConfig
from raggify.core.metadata import BasicMetaData
from raggify.core.metadata import MetaKeys as MK
from raggify.ingest.transform import (
    AddChunkIndexTransform,
    DefaultSummarizeTransform,
    EmbedTransform,
    LLMSummarizeTransform,
    RemoveTempFileTransform,
    SplitTransform,
)
from raggify.llama_like.core.schema import AudioNode, Modality, VideoNode
from raggify.llm.llm_manager import LLMManager
from tests.utils.mock_embed import (
    DummyAudioEmbedding,
    DummyImageEmbedding,
    DummyTextEmbedding,
    DummyVideoEmbedding,
    make_dummy_manager,
)
from tests.utils.mock_transform import (
    DummyLLM,
    apply_patch_embedding_bases,
    make_dummy_runtime,
)
from tests.utils.node_factory import make_sample_text_node

from .config import configure_test_env

configure_test_env()


@pytest.fixture(autouse=True)
def patch_embedding_bases(monkeypatch):
    apply_patch_embedding_bases(monkeypatch)


def _make_ingest_cfg(**kwargs) -> IngestConfig:
    return IngestConfig(**kwargs)


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


def test_split_transform_splits_audio_and_rebuilds(monkeypatch, tmp_path):
    src = Path("tests/data/audios/sample.wav")
    local = tmp_path / "sample.wav"
    shutil.copy(src, local)

    def fake_probe(self, path):
        return 10.0

    def fake_split(self, src, dst, chunk_seconds):
        dst.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            dest = dst / f"chunk_{i}.wav"
            shutil.copy(local, dest)
        return dst

    monkeypatch.setattr(
        "raggify.ingest.util.MediaConverter.__init__",
        lambda self: None,
    )
    monkeypatch.setattr(
        "raggify.ingest.util.MediaConverter.probe_duration",
        fake_probe,
    )
    monkeypatch.setattr(
        "raggify.ingest.util.MediaConverter.split",
        fake_split,
    )

    node = _make_media_node(AudioNode, local, ref_doc_id="audio-doc")

    cfg = _make_ingest_cfg(audio_chunk_seconds=2)
    split_transform = SplitTransform(cfg)
    result = split_transform([node])

    assert len(result) == 2
    assert all(isinstance(n, AudioNode) for n in result)
    assert {n.metadata[MK.CHUNK_NO] for n in result} == {0, 1}


def test_split_transform_splits_video(monkeypatch, tmp_path):
    src = Path("tests/data/videos/sample.mp4")
    local = tmp_path / "sample.mp4"
    shutil.copy(src, local)

    def fake_probe(self, path):
        return 12.0

    def fake_split(self, src, dst, chunk_seconds):
        dst.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            dest = dst / f"segment_{i}.mp4"
            shutil.copy(local, dest)
        return dst

    monkeypatch.setattr(
        "raggify.ingest.util.MediaConverter.__init__",
        lambda self: None,
    )
    monkeypatch.setattr(
        "raggify.ingest.util.MediaConverter.probe_duration",
        fake_probe,
    )
    monkeypatch.setattr(
        "raggify.ingest.util.MediaConverter.split",
        fake_split,
    )

    node = _make_media_node(VideoNode, local, ref_doc_id="video-doc")

    cfg = _make_ingest_cfg(video_chunk_seconds=3)
    split_transform = SplitTransform(cfg)
    nodes = split_transform([node])

    assert len(nodes) == 3
    assert all(isinstance(n, VideoNode) for n in nodes)


def test_split_transform_dispatches_text(monkeypatch):
    node = make_sample_text_node()

    class DummySentenceSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, nodes: list[TextNode], **kwargs):
            first = make_sample_text_node()
            first.text = "part-1"
            second = make_sample_text_node()
            second.text = "part-2"
            return [first, second]

    monkeypatch.setattr(
        "llama_index.core.node_parser.SentenceSplitter",
        DummySentenceSplitter,
    )

    cfg = _make_ingest_cfg(text_chunk_size=100, text_chunk_overlap=10)
    split_transform = SplitTransform(cfg)
    result = split_transform([node])

    texts = [cast(TextNode, n).text for n in result]
    assert texts == ["part-1", "part-2"]


def test_embed_transform_writes_text_embeddings():
    manager = make_dummy_manager({Modality.TEXT: DummyTextEmbedding()})
    transform = EmbedTransform(manager)

    node = make_sample_text_node()
    node.metadata[MK.TEMP_FILE_PATH] = "/tmp/temp.txt"
    node.metadata[MK.BASE_SOURCE] = "/orig/path.txt"
    blank = TextNode(text="  ", id_="blank")

    result = asyncio.run(transform.acall([node, blank]))

    assert result[0].embedding == [1.0, 1.0]
    assert result[1].embedding is None


def test_embed_transform_handles_image_nodes():
    manager = make_dummy_manager({Modality.IMAGE: DummyImageEmbedding()})
    transform = EmbedTransform(manager)

    node = ImageNode(id_="img", metadata={"file_path": "tests/data/images/sample.png"})
    other = TextNode(text="text", id_="txt")

    nodes = asyncio.run(transform.acall([node, other]))
    assert nodes[0].embedding == [0.1, 0.1]
    assert isinstance(nodes[1], TextNode)


def test_embed_transform_handles_audio_nodes():
    manager = make_dummy_manager({Modality.AUDIO: DummyAudioEmbedding()})
    transform = EmbedTransform(manager)

    audio = AudioNode(
        text="audio",
        id_="audio",
        metadata={"file_path": "tests/data/audios/sample.mp3"},
    )
    skipped = TextNode(text="text", id_="skip")

    nodes = asyncio.run(transform.acall([audio, skipped]))
    assert nodes[0].embedding == [0.2, 0.2]
    assert nodes[1].embedding is None


def test_embed_transform_handles_video_nodes():
    manager = make_dummy_manager({Modality.VIDEO: DummyVideoEmbedding()})
    transform = EmbedTransform(manager)

    video = VideoNode(
        text="video",
        id_="video",
        metadata={"file_path": "tests/data/videos/sample.mp4"},
    )
    nodes = asyncio.run(transform.acall([video]))

    assert nodes[0].embedding == [0.3, 0.3]


def test_remove_temp_file_transform_removes_files(tmp_path):
    temp = tmp_path / "temp.txt"
    temp.write_text("payload")

    node = make_sample_text_node()
    node.metadata[MK.TEMP_FILE_PATH] = str(temp)
    node.metadata[MK.BASE_SOURCE] = "/orig/path.txt"
    node.metadata[MK.FILE_PATH] = str(temp)

    transform = RemoveTempFileTransform()
    result = transform([node])

    assert not temp.exists()
    assert result[0].metadata[MK.FILE_PATH] == "/orig/path.txt"


def test_default_summarize_transform_returns_nodes():
    summarize_transform = DefaultSummarizeTransform()
    node = make_sample_text_node()
    nodes: list[BaseNode] = [node]

    assert summarize_transform(nodes) is nodes
    assert asyncio.run(summarize_transform.acall(nodes)) is nodes


def test_llm_summarize_transform_summarizes_text():
    node = make_sample_text_node()
    node.text = "Original"
    text_llm = DummyLLM(" trimmed text ")
    runtime = make_dummy_runtime(text_llm=text_llm)

    summarize_transform = LLMSummarizeTransform(cast("LLMManager", runtime.llm_manager))
    result = asyncio.run(summarize_transform.acall([node]))
    text_node = cast(TextNode, result[0])

    assert text_node.text == "trimmed text"
    assert text_llm.calls


def test_llm_summarize_transform_handles_text_error():
    node = make_sample_text_node()
    node.text = "error text"
    text_llm = DummyLLM(error=RuntimeError("boom"))
    runtime = make_dummy_runtime(text_llm=text_llm)

    summarize_transform = LLMSummarizeTransform(cast("LLMManager", runtime.llm_manager))
    result = asyncio.run(summarize_transform.acall([node]))
    text_node = cast(TextNode, result[0])

    assert text_node.text == "error text"


def test_llm_summarize_transform_summarizes_image():
    image_node = ImageNode(
        id_="img",
        metadata={"file_path": "tests/data/images/sample.png"},
    )
    image_llm = DummyLLM(" caption ")
    runtime = make_dummy_runtime(image_llm=image_llm)

    summarize_transform = LLMSummarizeTransform(cast("LLMManager", runtime.llm_manager))
    result = asyncio.run(summarize_transform.acall([image_node]))
    img_node = cast(ImageNode, result[0])

    assert img_node.text == "caption"
    assert image_llm.calls


def test_llm_summarize_transform_summarizes_audio(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"\x00")
    audio = AudioNode(id_="audio", metadata={MK.FILE_PATH: str(audio_path)})
    audio_llm = DummyLLM(" transcribed audio ")
    runtime = make_dummy_runtime(audio_llm=audio_llm)
    summarize_transform = LLMSummarizeTransform(cast("LLMManager", runtime.llm_manager))

    result = asyncio.run(summarize_transform.acall([audio]))
    node = cast(AudioNode, result[0])

    assert node.text == "transcribed audio"
    assert audio_llm.calls


def test_llm_summarize_transform_logs_video_not_implemented(tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.write_text("")  # create placeholder file
    video = VideoNode(id_="video", metadata={MK.FILE_PATH: str(video_path)})
    runtime = make_dummy_runtime()
    summarize_transform = LLMSummarizeTransform(cast("LLMManager", runtime.llm_manager))

    result = asyncio.run(summarize_transform.acall([video]))
    node = cast(VideoNode, result[0])

    assert node.text == ""


def test_llm_summarize_transform_rejects_unknown_node():
    runtime = make_dummy_runtime()
    summarize_transform = LLMSummarizeTransform(cast("LLMManager", runtime.llm_manager))
    doc = Document(text="doc", id_="doc")

    with pytest.raises(ValueError):
        asyncio.run(summarize_transform.acall([doc]))
