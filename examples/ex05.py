from raggify.config.embed_config import EmbedProvider
from raggify.config.vector_store_config import VectorStoreProvider
from raggify.ingest import ingest_url
from raggify.logger import configure_logging
from raggify.runtime import get_runtime

configure_logging()

rt = get_runtime()
rt.cfg.general.vector_store_provider = VectorStoreProvider.PGVECTOR
rt.cfg.general.audio_embed_provider = EmbedProvider.CLAP
rt.cfg.ingest.chunk_size = 300
rt.cfg.ingest.same_origin = False
rt.rebuild()

ingest_url("http://some.site.com")
