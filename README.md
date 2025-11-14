# Raggify

<img width="1024" height="1024" alt="Image" src="https://github.com/user-attachments/assets/054137c8-fb20-40bf-9fc2-fb31737a6d30" />

**Raggify** is a Python library for building multimodal retrieval-augmented generation systems that run locally or as a service. It now ships with an asynchronous ingest pipeline for files, web pages, and URL lists, normalizes metadata, persists cache fingerprints to avoid redundant upserts, and keeps a document store in sync for BM25 / hybrid retrieval. Out of the box it prepares embeddings across text, **image, audio, and video modalities (not via text)** and orchestrates vector stores through llama-index.

# 🔎 Overview

<img width="1457" height="905" alt="Image" src="https://github.com/user-attachments/assets/26656f89-c6f8-4920-8bf1-cfe44974ebae" />

A Typer CLI and REST client simplify ingestion and querying flows, while the FastAPI server exposes production-ready endpoints for applications. Async helpers keep pipelines responsive, and configuration dataclasses make it easy to tune providers, hardware targets, and rerankers for your deployment.

Latest additions focus on:

- A persistable ingest pipeline (`ingest.pipe_persist_dir` / `ingest.batch_size`) that keeps document stores, vector stores, and ingest caches in lock-step per modality.
- Pluggable document-store and ingest-cache providers (Local, Redis, Postgres) powering BM25 as well as hybrid QueryFusion retrieval.
- Bedrock-backed video embeddings plus text/image/audio/video query endpoints across the Python API, REST server, and MCP server.

# 🚀 How to Install

To install minimal, run:

```
pip install raggify
```

If you also use examples, run:

```
pip install raggify[exam]
```

Then, put your required API-KEYs and credentials in .env file.

```bash
OPENAI_API_KEY="your-api-key"
COHERE_API_KEY="your-api-key"
VOYAGE_API_KEY="your-api-key"

AWS_ACCESS_KEY_ID="your-id"
AWS_SECRET_ACCESS_KEY="your-key"
# AWS_REGION="us-east-1" # (default)
# AWS_PROFILE="your-profile" # (optional)
# AWS_SESSION_TOKEN = "your-token" # (optional)
```

Default providers (configured at /etc/raggify/config.yaml) are:

```yaml
"vector_store_provider": "chroma",
"document_store_provider": "local",
"ingest_cache_provider": "local",
"text_embed_provider": "openai",
"image_embed_provider": "cohere",
"audio_embed_provider": null,
"video_embed_provider": null,
"use_modality_fallback": true,
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

```bash
pip install openai-whisper@git+https://github.com/openai/whisper.git
```

and set "audio_embed_provider" /etc/raggify/config.yaml:

```yaml
"audio_embed_provider": CLAP
```

start/reload server

```bash
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

<!--
# 🎬 Video Retrieval API

You can now ingest video assets and query them with text, image, audio, or video inputs. Provide AWS credentials (`AWS_PROFILE` or `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION`), set the video embed provider to `EmbedProvider.BEDROCK`, and rebuild the runtime once.

```python
from raggify.config.embed_config import EmbedProvider
from raggify.ingest import ingest_path_list
from raggify.retrieve import query_text_video, query_video_video
from raggify.runtime import get_runtime

rt = get_runtime()
rt.cfg.general.video_embed_provider = EmbedProvider.BEDROCK
rt.rebuild()

ingest_path_list(["./videos/clip01.mp4", "./videos/clip02.mov"])

video_hits = query_text_video("A slow pan through a museum exhibition")
reference_hits = query_video_video("./videos/query.mp4")
```

When `video_embed_provider` is unset but `use_modality_fallback` stays `true`, the loader automatically decomposes a video into representative frames plus audio so that image/audio pipelines can continue to work.

# 🧱 Ingest Pipeline & Document Store

The ingest stack now wires the vector store, document store, and ingest cache together per modality:

- `ingest.batch_size` controls asynchronous batches pushed through llama-index `IngestionPipeline` objects, avoiding long event-loop blocks even for thousands of nodes.
- `ingest.pipe_persist_dir` stores serialized pipelines so local deployments can warm-start caches and docstores while keeping remote providers (Redis / Postgres / pgvector) in sync.
- `document_store_provider` keeps a dedicated metadata store backed by Local disk, Redis, or Postgres so BM25-only or QueryFusion hybrid retrieval remains accurate even when vectors are refreshed.
- `ingest_cache_provider` (Local / Redis / Postgres) ensures deduplication via modality-specific cache namespaces that share the same knowledge-base identifier and embedding-space key.

To switch persistence targets at runtime:

```python
from pathlib import Path

from raggify.config.document_store_config import DocumentStoreProvider
from raggify.config.ingest_cache_config import IngestCacheProvider
from raggify.runtime import get_runtime

rt = get_runtime()
rt.cfg.general.document_store_provider = DocumentStoreProvider.POSTGRES
rt.cfg.general.ingest_cache_provider = IngestCacheProvider.POSTGRES
rt.cfg.ingest.pipe_persist_dir = Path("/etc/raggify/pipes/mykb")
rt.cfg.ingest.batch_size = 200
rt.rebuild()
``` -->

Hybrid retrieval is enabled by default (`retrieve.mode = FUSION`), so keeping the document store populated guarantees BM25 scoring as well as reranking access to clean text spans.

# 💻 as REST API Server

Before using almost functions of the CLI, please start the server as follows:

```bash
raggify server
```

New video-ready endpoints are published under the FastAPI server and the generated MCP schema:

- `POST /v1/query/text_video`
- `POST /v1/query/image_video`
- `POST /v1/query/audio_video`
- `POST /v1/query/video_video`

They pair with the REST client helpers (`RestAPIClient.query_*_video`) as well as the `retrieve.py` functions of the same names.

<img width="1480" height="245" alt="Image" src="https://github.com/user-attachments/assets/860b2724-bfb2-4cc0-a67c-9ae1e5f8190e" />

<img width="2021" height="329" alt="Image" src="https://github.com/user-attachments/assets/e3628e55-9352-452c-ac97-c3879338ded7" />

Sample RAG system is examples/rag. which uses raggify server as backend.

<img width="540" height="468" alt="Image" src="https://github.com/user-attachments/assets/0c43ad96-c722-4f2d-9844-7b1526ff37c1" />

<!-- <img width="914" height="991" alt="Image" src="https://github.com/user-attachments/assets/c33fa027-4562-4e9a-b378-a79e38ec553e" /> -->

<img width="1009" height="1046" alt="Image" src="https://github.com/user-attachments/assets/0457a084-719a-461a-a436-558cd09ff55d" />

<!-- <img width="1134" height="863" alt="Image" src="https://github.com/user-attachments/assets/8ca99096-f0da-4273-afd8-7db9fde3ad62" /> -->

# ⌨️ as CLI

At first, run:

```bash
raggify --help
```

You can edit /etc/raggify/config.yaml to set default values, used by raggify runtime.

<img width="1177" height="1461" alt="Image" src="https://github.com/user-attachments/assets/e3676720-6cf5-4710-a4d1-fc948f7a6539" />

# 🤖️ as MCP Server

You can also specify --mcp option when you up server, or edit config.yaml.

<img width="675" height="270" alt="Image" src="https://github.com/user-attachments/assets/2a512c62-fc41-41ff-aef0-cd23c8fd1a87" />

For example, LM Studio mcp.json:

```json
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

### General

| Parameter                 | Description                                        | Default      | Allowed values / examples                                                      |
| ------------------------- | -------------------------------------------------- | ------------ | ------------------------------------------------------------------------------ |
| `knowledgebase_name`      | Identifier for the knowledge base.                 | `default_kb` | Any string (e.g., `project_a`).                                                |
| `host`                    | Hostname the FastAPI server binds to.              | `localhost`  | Any hostname/IP (e.g., `0.0.0.0`).                                             |
| `port`                    | Port number for the FastAPI server.                | `8000`       | Any integer port.                                                              |
| `mcp`                     | Enable MCP server alongside FastAPI.               | `false`      | `true` / `false`.                                                              |
| `vector_store_provider`   | Vector store backend.                              | `CHROMA`     | `CHROMA`, `PGVECTOR`, `REDIS`.                                                 |
| `document_store_provider` | Document store backend.                            | `LOCAL`      | `LOCAL`, `REDIS`, `POSTGRES`.                                                  |
| `ingest_cache_provider`   | Ingest cache backend.                              | `LOCAL`      | `LOCAL`, `REDIS`, `POSTGRES`.                                                  |
| `text_embed_provider`     | Provider for text embeddings.                      | `OPENAI`     | `OPENAI`, `COHERE`, `CLIP`(⚠️), `HUGGINGFACE`, `VOYAGE`, `BEDROCK`, or `null`. |
| `image_embed_provider`    | Provider for image embeddings.                     | `COHERE`     | `COHERE`, `CLIP`(⚠️), `HUGGINGFACE`, `VOYAGE`, `BEDROCK`, or `null`.           |
| `audio_embed_provider`    | Provider for audio embeddings.                     | `null`       | `CLAP`(⚠️), `BEDROCK`, or `null`.                                              |
| `video_embed_provider`    | Provider for video embeddings.                     | `null`       | `BEDROCK` or `null`.                                                           |
| `use_modality_fallback`   | Decompose unsupported media into lower modalities. | `true`       | `true` / `false`.                                                              |
| `rerank_provider`         | Provider for reranking.                            | `COHERE`     | `FLAGEMBEDDING`, `COHERE`, or `null`.                                          |
| `openai_base_url`         | Custom OpenAI-compatible endpoint.                 | `null`       | Any URL string or `null`.                                                      |
| `device`                  | Target device for embedding models.                | `cpu`        | `cpu`, `cuda`, `mps`.                                                          |
| `log_level`               | Logging verbosity.                                 | `DEBUG`      | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.                               |

⚠️ Local CLIP / CLAP usage still needs the extra git-based pip installs listed in the install section above.

### Vector store

| Parameter            | Description                   | Default                            | Allowed values / examples                        |
| -------------------- | ----------------------------- | ---------------------------------- | ------------------------------------------------ |
| `chroma_persist_dir` | Chroma persistence directory. | `~/.local/share/raggify/chroma_db` | Any filesystem path.                             |
| `chroma_host`        | External Chroma hostname.     | `null`                             | Any hostname or `null`.                          |
| `chroma_port`        | External Chroma port.         | `null`                             | Any integer port or `null`.                      |
| `chroma_tenant`      | Chroma tenant name.           | `null`                             | Any string or `null`.                            |
| `chroma_database`    | Chroma database name.         | `null`                             | Any string or `null`.                            |
| `pgvector_host`      | PGVector hostname.            | `localhost`                        | Any hostname.                                    |
| `pgvector_port`      | PGVector port.                | `5432`                             | Any integer port.                                |
| `pgvector_database`  | PGVector database name.       | `raggify`                          | Any string.                                      |
| `pgvector_user`      | PGVector user.                | `raggify`                          | Any string.                                      |
| `pgvector_password`  | PGVector password.            | `null`                             | Any string (required when PGVECTOR is selected). |
| `redis_host`         | Redis host for vector search. | `localhost`                        | Any hostname.                                    |
| `redis_port`         | Redis port for vector search. | `6379`                             | Any integer port.                                |

### Document store

| Parameter           | Description                 | Default     | Allowed values / examples |
| ------------------- | --------------------------- | ----------- | ------------------------- |
| `redis_host`        | Redis host for docstore.    | `localhost` | Any hostname.             |
| `redis_port`        | Redis port for docstore.    | `6379`      | Any integer port.         |
| `postgres_host`     | Postgres host for docstore. | `localhost` | Any hostname.             |
| `postgres_port`     | Postgres port for docstore. | `5432`      | Any integer port.         |
| `postgres_database` | Postgres database name.     | `raggify`   | Any string.               |
| `postgres_user`     | Postgres user.              | `raggify`   | Any string.               |
| `postgres_password` | Postgres password.          | `null`      | Any string or `null`.     |

### Ingest cache

| Parameter           | Description                     | Default     | Allowed values / examples |
| ------------------- | ------------------------------- | ----------- | ------------------------- |
| `redis_host`        | Redis host for ingest cache.    | `localhost` | Any hostname.             |
| `redis_port`        | Redis port for ingest cache.    | `6379`      | Any integer port.         |
| `postgres_host`     | Postgres host for ingest cache. | `localhost` | Any hostname.             |
| `postgres_port`     | Postgres port for ingest cache. | `5432`      | Any integer port.         |
| `postgres_database` | Postgres database name.         | `raggify`   | Any string.               |
| `postgres_user`     | Postgres user.                  | `raggify`   | Any string.               |
| `postgres_password` | Postgres password.              | `null`      | Any string or `null`.     |

### Embed

| Parameter                            | Description                                 | Default                                    | Allowed values / examples                                                                      |
| ------------------------------------ | ------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `openai_embed_model_text.name`       | OpenAI text embed model.                    | `text-embedding-3-small`                   | Fixed model name.                                                                              |
| `openai_embed_model_text.dim`        | Dimension of OpenAI text embeddings.        | `1536`                                     | Fixed value.                                                                                   |
| `cohere_embed_model_text.name`       | Cohere text embed model.                    | `embed-v4.0`                               | Fixed model name.                                                                              |
| `cohere_embed_model_text.dim`        | Dimension of Cohere text embeddings.        | `1536`                                     | Fixed value.                                                                                   |
| `clip_embed_model_text.name`         | CLIP text embed model.                      | `ViT-B/32`                                 | Fixed model name.                                                                              |
| `clip_embed_model_text.dim`          | Dimension of CLIP text embeddings.          | `512`                                      | Fixed value.                                                                                   |
| `huggingface_embed_model_text.name`  | Hugging Face text embed model.              | `intfloat/multilingual-e5-base`            | Fixed model name.                                                                              |
| `huggingface_embed_model_text.dim`   | Dimension of Hugging Face text embeddings.  | `768`                                      | Fixed value.                                                                                   |
| `voyage_embed_model_text.name`       | Voyage text embed model.                    | `voyage-3.5`                               | Fixed model name.                                                                              |
| `voyage_embed_model_text.dim`        | Dimension of Voyage text embeddings.        | `2048`                                     | Fixed value.                                                                                   |
| `bedrock_embed_model_text.name`      | Bedrock text embed model.                   | `amazon.nova-2-multimodal-embeddings-v1:0` | Fixed model name.                                                                              |
| `bedrock_embed_model_text.dim`       | Dimension of Bedrock text embeddings.       | `1024`                                     | Fixed value.                                                                                   |
| `cohere_embed_model_image.name`      | Cohere image embed model.                   | `embed-v4.0`                               | Fixed model name.                                                                              |
| `cohere_embed_model_image.dim`       | Dimension of Cohere image embeddings.       | `1536`                                     | Fixed value.                                                                                   |
| `clip_embed_model_image.name`        | CLIP image embed model.                     | `ViT-B/32`                                 | Fixed model name.                                                                              |
| `clip_embed_model_image.dim`         | Dimension of CLIP image embeddings.         | `512`                                      | Fixed value.                                                                                   |
| `huggingface_embed_model_image.name` | Hugging Face image embed model.             | `llamaindex/vdr-2b-multi-v1`               | Fixed model name.                                                                              |
| `huggingface_embed_model_image.dim`  | Dimension of Hugging Face image embeddings. | `1536`                                     | Fixed value.                                                                                   |
| `voyage_embed_model_image.name`      | Voyage image embed model.                   | `voyage-multimodal-3`                      | Fixed model name.                                                                              |
| `voyage_embed_model_image.dim`       | Dimension of Voyage image embeddings.       | `1024`                                     | Fixed value.                                                                                   |
| `bedrock_embed_model_image.name`     | Bedrock image embed model.                  | `amazon.nova-2-multimodal-embeddings-v1:0` | Fixed model name.                                                                              |
| `bedrock_embed_model_image.dim`      | Dimension of Bedrock image embeddings.      | `1024`                                     | Fixed value.                                                                                   |
| `clap_embed_model_audio.name`        | CLAP audio embed model.                     | `effect_varlen`                            | `effect_short`, `effect_varlen`, `music`, `speech`, `general` (last 3 pending implementation). |
| `clap_embed_model_audio.dim`         | Dimension of CLAP audio embeddings.         | `512`                                      | Fixed value.                                                                                   |
| `bedrock_embed_model_audio.name`     | Bedrock audio embed model.                  | `amazon.nova-2-multimodal-embeddings-v1:0` | Fixed model name.                                                                              |
| `bedrock_embed_model_audio.dim`      | Dimension of Bedrock audio embeddings.      | `1024`                                     | Fixed value.                                                                                   |
| `bedrock_embed_model_video.name`     | Bedrock video embed model.                  | `amazon.nova-2-multimodal-embeddings-v1:0` | Fixed model name.                                                                              |
| `bedrock_embed_model_video.dim`      | Dimension of Bedrock video embeddings.      | `1024`                                     | Fixed value.                                                                                   |

### Ingest

| Parameter          | Description                                  | Default                             | Allowed values / examples          |
| ------------------ | -------------------------------------------- | ----------------------------------- | ---------------------------------- |
| `chunk_size`       | Chunk size for text splitting.               | `500`                               | Any integer (e.g., `500`, `1024`). |
| `chunk_overlap`    | Overlap between adjacent chunks.             | `50`                                | Any integer.                       |
| `upload_dir`       | Directory for uploaded files.                | `~/.local/share/raggify/upload`     | Any filesystem path.               |
| `pipe_persist_dir` | Pipeline persistence root per KB.            | `~/.local/share/raggify/default_kb` | Any filesystem path.               |
| `batch_size`       | Number of nodes processed per async batch.   | `100`                               | Any positive integer.              |
| `user_agent`       | User-Agent header for web ingestion.         | `raggify`                           | Any string.                        |
| `load_asset`       | Download linked assets during web ingestion. | `true`                              | `true` / `false`.                  |
| `req_per_sec`      | Request rate limit for web ingestion.        | `2`                                 | Any integer.                       |
| `timeout_sec`      | Timeout for web ingestion (seconds).         | `30`                                | Any integer.                       |
| `same_origin`      | Restrict crawling to same origin.            | `true`                              | `true` / `false`.                  |

### Rerank

| Parameter                    | Description                                    | Default                    | Allowed values / examples       |
| ---------------------------- | ---------------------------------------------- | -------------------------- | ------------------------------- |
| `flagembedding_rerank_model` | FlagEmbedding reranker model name.             | `BAAI/bge-reranker-v2-m3`  | Fixed model name.               |
| `cohere_rerank_model`        | Cohere reranker model name.                    | `rerank-multilingual-v3.0` | Fixed model name.               |
| `topk`                       | Number of candidates considered for reranking. | `20`                       | Any integer (e.g., `10`, `20`). |

### Retrieve

| Parameter              | Description                                 | Default  | Allowed values / examples             |
| ---------------------- | ------------------------------------------- | -------- | ------------------------------------- |
| `mode`                 | Retrieval strategy.                         | `FUSION` | `VECTOR_ONLY`, `BM25_ONLY`, `FUSION`. |
| `bm25_topk`            | Number of docstore hits when using BM25.    | `10`     | Any integer.                          |
| `fusion_lambda_vector` | Weight for vector retriever in QueryFusion. | `0.5`    | Float 0–1.                            |
| `fusion_lambda_bm25`   | Weight for BM25 retriever in QueryFusion.   | `0.5`    | Float 0–1.                            |

## Setting samples

Full Local

```yaml
vector_store_provider: CHROMA
document_store_provider: LOCAL
ingest_cache_provider: LOCAL
text_embed_provider: HUGGINGFACE
image_embed_provider: CLIP
audio_embed_provider: CLAP
video_embed_provider: null
rerank_provider: FLAGEMBEDDING
```

⚠️ Video embedding (native) is not supported yet in local. You can use `video_embed_provider: null` and `use_modality_fallback: True` to ingest videos as images + audio.

Disable Video Retrieval

```yaml
video_embed_provider: null
use_modality_fallback: false
```

Adjust Web Scraping

```yaml
user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
same_origin: false
```

Reduce logging

```yaml
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
    query_audio_video,
    query_image_image,
    query_image_video,
    query_text_audio,
    query_text_image,
    query_text_text,
    query_text_video,
    query_video_video,
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
