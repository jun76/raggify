import json

from raggify.ingest import ingest_url
from raggify.retrieve import query_text_image

url = "https://developers.llamaindex.ai/python/examples/multi_modal/multi_modal_retrieval/"

ingest_url(url)

nodes = query_text_image(query="what is the main character in Batman")

for node in nodes:
    print(
        json.dumps(
            obj={"text": node.text, "metadata": node.metadata, "score": node.score},
            indent=2,
        )
    )
