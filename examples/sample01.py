import asyncio
import json

from raggify.ingest import aingest_url_list
from raggify.retrieve import aquery_text_audio

urls = [
    "https://developers.llamaindex.ai/python/examples/embeddings/openai/",
    "https://developers.llamaindex.ai/python/examples/embeddings/cohereai/",
    "https://developers.llamaindex.ai/python/examples/embeddings/voyageai/",
]


async def func():
    await aingest_url_list(urls)
    nodes = await aquery_text_audio(query="voyage")

    for node in nodes:
        print(
            json.dumps(
                obj={"text": node.text, "metadata": node.metadata, "score": node.score},
                indent=2,
            )
        )


asyncio.run(func())
