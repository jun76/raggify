import json

from raggify.ingest import ingest_url_list
from raggify.retrieve import query_text_image

urls = [
    "https://developers.llamaindex.ai/python/examples/multi_modal/multi_modal_retrieval/",
]


ingest_url_list(urls)
nodes = query_text_image(query="what is the main character in Batman", topk=3)

for node in nodes:
    print(
        json.dumps(
            obj={"text": node.text, "metadata": node.metadata, "score": node.score},
            indent=2,
        )
    )
