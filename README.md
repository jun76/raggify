# Raggify

<img width="1024" height="1024" alt="Image" src="https://github.com/user-attachments/assets/054137c8-fb20-40bf-9fc2-fb31737a6d30" />

**Raggify** is a Python library for building multimodal retrieval-augmented generation systems that run locally or as a service. It bundles ingest pipelines for files, web pages, and URL lists, normalizes metadata, and persists fingerprints to avoid redundant upserts. Out of the box it prepares embeddings across text, **image, and audio modalities (not via text)** and orchestrates vector stores through llama-index.

# 🔎 Overview

<img width="1457" height="905" alt="Image" src="https://github.com/user-attachments/assets/26656f89-c6f8-4920-8bf1-cfe44974ebae" />

A Typer CLI and REST client simplify ingestion and querying flows, while the FastAPI server exposes production-ready endpoints for applications. Async helpers keep pipelines responsive, and configuration dataclasses make it easy to tune providers, hardware targets, and rerankers for your deployment.

# 🚀 How to Install

To install minimal, run:

```
pip install raggify
```

If you also use examples, run:

```
pip install raggify[exam]
```

Then, put your required API-KEY in .env file.

```
OPENAI_API_KEY="your-api-key"
COHERE_API_KEY="your-api-key"
VOYAGE_API_KEY="your-api-key"
```

Default providers (configured at /etc/raggify/config.yaml) are:

```
"vector_store_provider": "chroma",
"document_store_provider": "local",
"ingest_cache_provider": "local",
"text_embed_provider": "openai",
"image_embed_provider": "cohere",
"audio_embed_provider": null,
"rerank_provider": "cohere",
```

⚠️ To use the following features, additional installation from the Git repository is required.

- CLIP\
   `clip@git+https://github.com/openai/CLIP.git`
- CLAP\
   `openai-whisper@git+https://github.com/openai/whisper.git`

# 📚 as Library API

## examples/ex01.py

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

nodes = query_text_text("Half-Blood Prince")

for node in nodes:
    print(
        json.dumps(
            obj={"text": node.text, "metadata": node.metadata, "score": node.score},
            indent=2,
        )
    )
```

## examples/ex02.py

Ingest from llamaindex wiki, then, search image documents by text query.

```python
import json

from raggify.ingest import ingest_url
from raggify.retrieve import query_text_image

url = "https://developers.llamaindex.ai/python/examples/multi_modal/multi_modal_retrieval/"

ingest_url(url)

nodes = query_text_image("what is the main character in Batman")
```

## examples/ex03.py

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

nodes = query_text_audio("phone call")
```

⚠️ To use audio features (local CLAP), need to install openai-whisper:

```
pip install openai-whisper@git+https://github.com/openai/whisper.git
```

and set "audio_embed_provider" /etc/raggify/config.yaml:

```
"audio_embed_provider": CLAP
```

start/reload server

```
raggify reload
```

## examples/ex04.py

After initial startup according to the /etc/raggify/config.yaml, hot-reload the config values.

```python
from raggify.config.embed_config import EmbedProvider
from raggify.config.vector_store_config import VectorStoreProvider
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

⚠️ To use pgvector,

- start postgresql server
- exec examples/init_pgdb.sh
  - for the first time.
  - If you want to reset db (drop table), exec ./init_pgdb.sh --reset
- set `pgvector_password` at /etc/raggify/config.yaml
  - init_pgdb.sh set `raggify` as default password, so write it.

⚠️ To use redis,

- start redis server
- exec examples/init_redis.sh
  - for the first time. (Optional)
  - If you want to reset db (drop table), exec ./init_redis.sh --reset
- set `redis_password` at /etc/raggify/config.yaml
  - init_redis.sh set `raggify` as default password, so write it.

Using Docker containers is easy.

```bash
docker run --rm -p 6379:6379 --name redis-stack redis/redis-stack-server:latest
```

# 💻 as REST API Server

Before using almost functions of the CLI, please start the server as follows:

```
raggify server
```

<img width="1480" height="245" alt="Image" src="https://github.com/user-attachments/assets/860b2724-bfb2-4cc0-a67c-9ae1e5f8190e" />

<img width="2021" height="329" alt="Image" src="https://github.com/user-attachments/assets/e3628e55-9352-452c-ac97-c3879338ded7" />

Sample RAG system is examples/rag. which uses raggify server as backend.

<img width="540" height="468" alt="Image" src="https://github.com/user-attachments/assets/0c43ad96-c722-4f2d-9844-7b1526ff37c1" />

<!-- <img width="914" height="991" alt="Image" src="https://github.com/user-attachments/assets/c33fa027-4562-4e9a-b378-a79e38ec553e" /> -->

<img width="1009" height="1046" alt="Image" src="https://github.com/user-attachments/assets/0457a084-719a-461a-a436-558cd09ff55d" />

<!-- <img width="1134" height="863" alt="Image" src="https://github.com/user-attachments/assets/8ca99096-f0da-4273-afd8-7db9fde3ad62" /> -->

# ⌨️ as CLI

At first, run:

```
raggify --help
```

You can edit /etc/raggify/config.yaml to set default values, used by raggify runtime.

<img width="1177" height="1461" alt="Image" src="https://github.com/user-attachments/assets/e3676720-6cf5-4710-a4d1-fc948f7a6539" />

# 🤖️ as MCP Server

You can also specify --mcp option when you up server, or edit config.yaml.

<img width="675" height="270" alt="Image" src="https://github.com/user-attachments/assets/2a512c62-fc41-41ff-aef0-cd23c8fd1a87" />

For example, LM Studio mcp.json:

```
{
  "mcpServers": {
    "my_mcp_server": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

<img width="1259" height="1616" alt="Image" src="https://github.com/user-attachments/assets/08cfa62c-33a2-4b46-a21b-0bf95c258533" />

# 🛠️ Configure

## /etc/raggify/config.yaml

Generally, edit /etc/raggify/config.yaml before starting the server. You can also access the runtime to hot-reload configuration values, but this process is resource-intensive.

| Section        | Parameter                            | Description                                    | Default                         | Allowed values / examples                                                                                       |
| -------------- | ------------------------------------ | ---------------------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `general`      | `knowledgebase_name`                 | Identifier for the knowledge base.             | `default`                       | Any string (e.g., `default`, `project_a`).                                                                      |
|                | `host`                               | Hostname the FastAPI server binds to.          | `localhost`                     | Any hostname/IP (e.g., `localhost`, `0.0.0.0`).                                                                 |
|                | `port`                               | Port number for the FastAPI server.            | `8000`                          | Any integer port (e.g., `8000`, `8080`).                                                                        |
|                | `mcp`                                | Enable MCP server alongside FastAPI.           | `false`                         | `true` / `false`.                                                                                               |
|                | `vector_store_provider`              | Vector store backend.                          | `CHROMA`                        | `CHROMA`, `PGVECTOR`.                                                                                           |
|                | `text_embed_provider`                | Provider for text embeddings.                  | `OPENAI`                        | `OPENAI`, `COHERE`, `CLIP`(⚠️), `HUGGINGFACE`, `VOYAGE`, or `null`.                                             |
|                | `image_embed_provider`               | Provider for image embeddings.                 | `COHERE`                        | `COHERE`, `CLIP`(⚠️), `HUGGINGFACE`, `VOYAGE`, or `null`.                                                       |
|                | `audio_embed_provider`               | Provider for audio embeddings.                 | `null`                          | `CLAP`(⚠️) or `null`.                                                                                           |
|                | `rerank_provider`                    | Provider for reranking.                        | `COHERE`                        | `FLAGEMBEDDING`, `COHERE`, or `null`.                                                                           |
|                | `openai_base_url`                    | Custom OpenAI-compatible endpoint.             | `null`                          | Any URL string or `null`.                                                                                       |
|                | `device`                             | Target device for embedding models.            | `cpu`                           | `cpu`, `cuda`, `mps`.                                                                                           |
|                | `log_level`                          | Logging verbosity.                             | `DEBUG`                         | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.                                                                |
| `vector_store` | `cache_load_limit`                   | Upper bound for cache reload.                  | `10000`                         | Any integer (e.g., `10000`, `5000`).                                                                            |
|                | `check_update`                       | Recalculate embeddings on re-ingest.           | `false`                         | `true` / `false`.                                                                                               |
|                | `chroma_persist_dir`                 | Chroma persistence directory.                  | `/etc/raggify/raggify_db`       | Any filesystem path.                                                                                            |
|                | `chroma_host`                        | External Chroma hostname.                      | `null`                          | Any hostname or `null`.                                                                                         |
|                | `chroma_port`                        | External Chroma port.                          | `null`                          | Any integer port or `null`.                                                                                     |
|                | `chroma_tenant`                      | Chroma tenant name.                            | `null`                          | Any string or `null`.                                                                                           |
|                | `chroma_database`                    | Chroma database name.                          | `null`                          | Any string or `null`.                                                                                           |
|                | `pgvector_host`                      | PGVector hostname.                             | `localhost`                     | Any hostname.                                                                                                   |
|                | `pgvector_port`                      | PGVector port.                                 | `5432`                          | Any integer port (e.g., `5432`, `5433`).                                                                        |
|                | `pgvector_database`                  | PGVector database name.                        | `raggify`                       | Any string.                                                                                                     |
|                | `pgvector_user`                      | PGVector user.                                 | `raggify`                       | Any string.                                                                                                     |
|                | `pgvector_password`                  | PGVector password.                             | `null`                          | Any string or `null`.                                                                                           |
| `meta_store`   | `meta_store_path`                    | SQLite path for metadata store.                | `/etc/raggify/raggify_metas.db` | Any filesystem path.                                                                                            |
| `embed`        | `openai_embed_model_text.name`       | OpenAI text embed model.                       | `text-embedding-3-small`        | Fixed model name.                                                                                               |
|                | `openai_embed_model_text.dim`        | Dimension of OpenAI text embeddings.           | `1536`                          | Fixed value.                                                                                                    |
|                | `cohere_embed_model_text.name`       | Cohere text embed model.                       | `embed-v4.0`(⚠️)                | Fixed model name.                                                                                               |
|                | `cohere_embed_model_text.dim`        | Dimension of Cohere text embeddings.           | `1536`                          | Fixed value.                                                                                                    |
|                | `clip_embed_model_text.name`         | CLIP text embed model.                         | `ViT-B/32`                      | Fixed model name.                                                                                               |
|                | `clip_embed_model_text.dim`          | Dimension of CLIP text embeddings.             | `512`                           | Fixed value.                                                                                                    |
|                | `huggingface_embed_model_text.name`  | Hugging Face text embed model.                 | `intfloat/multilingual-e5-base` | Fixed model name.                                                                                               |
|                | `huggingface_embed_model_text.dim`   | Dimension of Hugging Face text embeddings.     | `768`                           | Fixed value.                                                                                                    |
|                | `voyage_embed_model_text.name`       | Voyage text embed model.                       | `voyage-3.5`                    | Fixed model name.                                                                                               |
|                | `voyage_embed_model_text.dim`        | Dimension of Voyage text embeddings.           | `2048`                          | Fixed value.                                                                                                    |
|                | `cohere_embed_model_image.name`      | Cohere image embed model.                      | `embed-v4.0`(⚠️)                | Fixed model name.                                                                                               |
|                | `cohere_embed_model_image.dim`       | Dimension of Cohere image embeddings.          | `1536`                          | Fixed value.                                                                                                    |
|                | `clip_embed_model_image.name`        | CLIP image embed model.                        | `ViT-B/32`                      | Fixed model name.                                                                                               |
|                | `clip_embed_model_image.dim`         | Dimension of CLIP image embeddings.            | `512`                           | Fixed value.                                                                                                    |
|                | `huggingface_embed_model_image.name` | Hugging Face image embed model.                | `llamaindex/vdr-2b-multi-v1`    | Fixed model name.                                                                                               |
|                | `huggingface_embed_model_image.dim`  | Dimension of Hugging Face image embeddings.    | `1536`                          | Fixed value.                                                                                                    |
|                | `voyage_embed_model_image.name`      | Voyage image embed model.                      | `voyage-multimodal-3`           | Fixed model name.                                                                                               |
|                | `voyage_embed_model_image.dim`       | Dimension of Voyage image embeddings.          | `1024`                          | Fixed value.                                                                                                    |
|                | `clap_embed_model_audio.name`        | CLAP audio embed model.                        | `effect_varlen`                 | `effect_short`, `effect_varlen`, `music`, `speech`, `general`. (`music`, `speech`, `general` are not impl yet.) |
|                | `clap_embed_model_audio.dim`         | Dimension of CLAP audio embeddings.            | `512`                           | Fixed value.                                                                                                    |
| `ingest`       | `chunk_size`                         | Chunk size for text splitting.                 | `500`                           | Any integer (e.g., `500`, `1024`).                                                                              |
|                | `chunk_overlap`                      | Overlap between adjacent chunks.               | `50`                            | Any integer (e.g., `50`, `100`).                                                                                |
|                | `upload_dir`                         | Directory for uploaded files.                  | `/etc/raggify/upload`           | Any filesystem path.                                                                                            |
|                | `user_agent`                         | User-Agent header for web ingestion.           | `raggify`                       | Any string.                                                                                                     |
|                | `load_asset`                         | Download linked assets during web ingestion.   | `true`                          | `true` / `false`.                                                                                               |
|                | `req_per_sec`                        | Request rate limit for web ingestion.          | `2`                             | Any integer (e.g., `2`, `5`).                                                                                   |
|                | `timeout_sec`                        | Timeout for web ingestion (seconds).           | `30`                            | Any integer (e.g., `30`, `60`).                                                                                 |
|                | `same_origin`                        | Restrict crawling to same origin.              | `true`                          | `true` / `false`.                                                                                               |
| `rerank`       | `flagembedding_rerank_model`         | FlagEmbedding reranker model name.             | `BAAI/bge-reranker-v2-m3`       | Fixed model name.                                                                                               |
|                | `cohere_rerank_model`                | Cohere reranker model name.                    | `rerank-multilingual-v3.0`      | Fixed model name.                                                                                               |
|                | `topk`                               | Number of candidates considered for reranking. | `10`                            | Any integer (e.g., `10`, `20`).                                                                                 |

⚠️: Need additional installation.

## Setting samples

Full Local

```
text_embed_provider: HUGGINGFACE
image_embed_provider: CLIP
audio_embed_provider: CLAP
rerank_provider: FLAGEMBEDDING
```

Adjust Web Scraping

```
user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
same_origin: false
```

Reduce logging

```
log_level: WARNING
```

## Main Modules

```python
# For reference
from raggify.config.default_settings import (
    DefaultSettings,
    EmbedProvider,
    RerankProvider,
    VectorStoreProvider,
)

# For ingestion
from raggify.ingest import ingest_path, ingest_path_list, ingest_url, ingest_url_list

# For retrieval
from raggify.retrieve import (
    query_audio_audio,
    query_image_image,
    query_text_audio,
    query_text_image,
    query_text_text,
)

# For hot reloading config
from raggify.runtime import get_runtime

# For REST API Call to the server
from raggify.client import RestAPIClient

# For logging
from raggify.logger import logger

# Retrievers return this structure
from llama_index.core.schema import NodeWithScore
```

# See Also

## Home Page

- [マルチモーダルでローカルな RAG 基盤サーバを作ってみた](https://qiita.com/jun76/items/f2e392f530e24a6a8903)

## Logo and Branding

The Raggify logo © 2025 Jun.  
You may use it to refer to the Raggify open-source project,  
but commercial or misleading usage is not allowed.
