<img alt="Image" src="https://github.com/user-attachments/assets/ae21fe3a-0dff-4538-8e63-d92ebdbaa682" />
Raggify is a Python library for building multimodal retrieval-augmented generation systems that run locally or as a service. It bundles ingest pipelines for files, web pages, and URL lists, normalizes metadata, and persists fingerprints to avoid redundant upserts. Out of the box it prepares embeddings across text, image, and audio modalities and orchestrates vector stores through llama-index. A Typer CLI and REST client simplify ingestion and querying flows, while the FastAPI server exposes production-ready endpoints for applications. Async helpers keep pipelines responsive, and configuration dataclasses make it easy to tune providers, hardware targets, and rerankers for your deployment.

## 🔎Overview

## 💻How to Install

## 🚀Quick Start

### Python App

#### examples/ex01.py

Ingest from some web sites, then, search text documents by text query.

```python
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
```

#### examples/ex02.py

Ingest from llamaindex wiki, then, search image documents by text query.

```python
import json

from raggify.ingest import ingest_url
from raggify.retrieve import query_text_image

url = "https://developers.llamaindex.ai/python/examples/multi_modal/multi_modal_retrieval/"

ingest_url(url)

nodes = query_text_image(query="what is the main character in Batman")

...
```

#### examples/ex03.py

Ingest from some local files, then, search audio documents by text query.

```python
from raggify.ingest import ingest_path_list
from raggify.retrieve import query_text_audio

paths = [
    "/path/to/sound.mp3",
    "/path/to/sound.wav",
    "/path/to/sound.flac",
    "/path/to/sound.ogg",
]

ingest_path_list(paths)

nodes = query_text_audio(query="phone call")
```

#### examples/ex04.py

After initial startup according to the /etc/raggify/config.yaml, hot-reload the config values.

```python
from raggify.config.default_settings import EmbedProvider, VectorStoreProvider
from raggify.ingest import ingest_url
from raggify.runtime import get_runtime

rt = get_runtime()
rt.cfg.general.vector_store_provider = VectorStoreProvider.PGVECTOR
rt.cfg.general.audio_embed_provider = EmbedProvider.CLAP
rt.cfg.ingest.chunk_size = 300
rt.cfg.ingest.same_origin = False
rt.rebuild()

ingest_url("http://some.site.com")
```

### CLI

### Server

## 🛠️Configure

## See Also

- [マルチモーダルでローカルな RAG 基盤サーバを作ってみた](https://qiita.com/jun76/items/f2e392f530e24a6a8903)
-

## Specs
