from __future__ import annotations

from typing import Any

from llama_index.core.schema import (
    Document,
    MediaResource,
    NodeRelationship,
    NodeWithScore,
    ObjectType,
    RelatedNodeInfo,
    TextNode,
)

DEFAULT_PATH = "/workspaces/raggify/tests/data/texts/sample.c"


def make_sample_metadata(path: str = DEFAULT_PATH) -> dict[str, Any]:
    return {
        "file_path": path,
        "file_name": "sample.c",
        "file_type": "text/x-csrc",
        "file_size": 68,
        "creation_date": "2025-11-17",
        "last_modified_date": "2025-11-28",
        "chunk_no": 0,
    }


def make_sample_document(path: str = DEFAULT_PATH) -> Document:
    meta = make_sample_metadata(path)
    return Document(
        id_="doc-sample",
        metadata=meta,
        relationships={},
        text_resource=MediaResource(
            text='#include <stdio.h>\n\nint main(void)\n{\n    printf("hello world\\n");\n}\n'
        ),
    )


def make_sample_text_node(path: str = DEFAULT_PATH) -> TextNode:
    meta = make_sample_metadata(path)
    rel = RelatedNodeInfo(
        node_id=f"source:{path}",
        node_type=ObjectType.TEXT,
        metadata=meta.copy(),
        hash="dummy-hash",
    )
    return TextNode(
        id_="node-sample",
        text='#include <stdio.h>\nint main(void) { printf("hello world\\n"); }',
        metadata=meta,
        relationships={NodeRelationship.SOURCE: rel},
    )


def make_sample_nodes(path: str = DEFAULT_PATH) -> list[NodeWithScore]:
    primary = make_sample_text_node(path)
    secondary = TextNode(
        id_="node-secondary",
        text=primary.text + "\n",
        metadata=make_sample_metadata(path),
        relationships={},
    )
    return [
        NodeWithScore(node=primary, score=0.5),
        NodeWithScore(node=secondary, score=0.4),
    ]
