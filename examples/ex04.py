import json

from raggify.ingest import ingest_path
from raggify.retrieve import query_image_video

knowledge_path = "/path/to/movies"

ingest_path(knowledge_path)

query_path = "/path/to/similar/image.png"

nodes = query_image_video(query_path)

for node in nodes:
    print(
        json.dumps(
            obj={"text": node.text, "metadata": node.metadata, "score": node.score},
            indent=2,
        )
    )
