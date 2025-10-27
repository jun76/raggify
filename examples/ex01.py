import json

from raggify.ingest import ingest_url_list
from raggify.retrieve import query_text_text

urls = [
    "https://en.wikipedia.org/wiki/Harry_Potter_(film_series)",
    "https://en.wikipedia.org/wiki/Star_Wars_(film)",
    "https://en.wikipedia.org/wiki/Forrest_Gump",
]

ingest_url_list(urls)

nodes = query_text_text(query="Half-Blood Prince")

for node in nodes:
    print(
        json.dumps(
            obj={"text": node.text, "metadata": node.metadata, "score": node.score},
            indent=2,
        )
    )
