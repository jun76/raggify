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
