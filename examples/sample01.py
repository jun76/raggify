import json
from typing import Any

from llama_index.core.schema import NodeWithScore

from raggify import runtime
from raggify.ingest import ingest_url_list
from raggify.retrieve import query_text_text


def nodes_to_response(nodes: list[NodeWithScore]) -> list[dict[str, Any]]:
    return [
        {"text": node.text, "metadata": node.metadata, "score": node.score}
        for node in nodes
    ]


lst = [
    "https://developers.llamaindex.ai/python/examples/embeddings/openai/",
    "https://developers.llamaindex.ai/python/examples/embeddings/cohereai/",
    "https://developers.llamaindex.ai/python/examples/embeddings/voyageai/",
]

runtime.reload()

ingest_url_list(lst=lst)

nodes = query_text_text(query="voyage")

results = nodes_to_response(nodes)
for result in results:
    print(json.dumps(obj=result, indent=2))
