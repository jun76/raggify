import json

from raggify.client import RestAPIClient

client = RestAPIClient("http://localhost:8000/v1")

print(json.dumps(client.health(), indent=2))

client.ingest_url("https://en.wikipedia.org/wiki/Artificial_intelligence")
