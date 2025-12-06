from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Iterable,
    Optional,
    Sequence,
    Union,
    cast,
)

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
    from llama_index.core.vector_stores.types import (
        MetadataFilters,
        VectorStoreQueryResult,
    )

Embeddings = Sequence[float]

__all__ = ["AudioEncoders", "VideoEncoders", "AudioRetriever", "VideoRetriever"]


@dataclass(kw_only=True)
class AudioEncoders:
    """Encoders for audio retrieval.

    Provide async functions that take lists of queries and return lists of embeddings
    for text_encoder and audio_encoder respectively.
    """

    text_encoder: Optional[Callable[[list[str]], Awaitable[list[Embeddings]]]] = None
    audio_encoder: Optional[Callable[[list[str]], Awaitable[list[Embeddings]]]] = None

    @classmethod
    def from_embed_model(cls, embed_model: Optional[BaseEmbedding]) -> AudioEncoders:
        """Create encoders available from the embedding model.

        Args:
            embed_model (Optional[BaseEmbedding]): Embedding model.

        Returns:
            AudioEncoders: Instance containing text and audio encoders.
        """
        if embed_model is None:
            return cls()

        text_encoder: Optional[Callable[[list[str]], Awaitable[list[Embeddings]]]] = (
            None
        )
        audio_encoder: Optional[Callable[[list[str]], Awaitable[list[Embeddings]]]] = (
            None
        )

        if hasattr(embed_model, "aget_text_embedding_batch"):

            async def encode_text(queries: list[str]) -> list[Embeddings]:
                return await embed_model.aget_text_embedding_batch(texts=queries)  # type: ignore[attr-defined]

            text_encoder = encode_text

        if hasattr(embed_model, "aget_audio_embedding_batch"):

            async def encode_audio(paths: list[str]) -> list[Embeddings]:
                return await embed_model.aget_audio_embedding_batch(  # type: ignore[attr-defined]
                    audio_file_paths=paths
                )

            audio_encoder = encode_audio

        return cls(text_encoder=text_encoder, audio_encoder=audio_encoder)

    async def aencode_text(self, queries: list[str]) -> list[Embeddings]:
        """Convert text queries to embeddings.

        Args:
            queries (list[str]): List of text queries.

        Returns:
            list[Embeddings]: List of text embeddings.
        """
        if self.text_encoder is None:
            raise RuntimeError("text encoder for audio retrieval is not available")

        return await self.text_encoder(queries)

    async def aencode_audio(self, paths: list[str]) -> list[Embeddings]:
        """Convert audio files to embeddings.

        Args:
            paths (list[str]): List of audio file paths.

        Returns:
            list[Embeddings]: List of audio embeddings.
        """
        if self.audio_encoder is None:
            raise RuntimeError("audio encoder for audio retrieval is not available")

        return await self.audio_encoder(paths)


@dataclass(kw_only=True)
class VideoEncoders(AudioEncoders):
    """Encoders for video retrieval."""

    image_encoder: Optional[Callable[[list[str]], Awaitable[list[Embeddings]]]] = None
    video_encoder: Optional[Callable[[list[str]], Awaitable[list[Embeddings]]]] = None

    @classmethod
    def from_embed_model(cls, embed_model: Optional[BaseEmbedding]) -> VideoEncoders:
        """Create encoders available from the embedding model.

        Args:
            embed_model (Optional[BaseEmbedding]): Embedding model.

        Returns:
            VideoEncoders: Instance containing text, audio, image, and video encoders.
        """
        base = super().from_embed_model(embed_model)
        image_encoder: Optional[Callable[[list[str]], Awaitable[list[Embeddings]]]] = (
            None
        )
        video_encoder: Optional[Callable[[list[str]], Awaitable[list[Embeddings]]]] = (
            None
        )

        if embed_model is not None and hasattr(
            embed_model, "aget_image_embedding_batch"
        ):

            async def encode_image(paths: list[str]) -> list[Embeddings]:
                return await embed_model.aget_image_embedding_batch(  # type: ignore[attr-defined]
                    img_file_paths=paths
                )

            image_encoder = encode_image

        if embed_model is not None and hasattr(
            embed_model, "aget_video_embedding_batch"
        ):

            async def encode_video(paths: list[str]) -> list[Embeddings]:
                return await embed_model.aget_video_embedding_batch(  # type: ignore[attr-defined]
                    video_file_paths=paths
                )

            video_encoder = encode_video

        return cls(
            text_encoder=base.text_encoder,
            audio_encoder=base.audio_encoder,
            image_encoder=image_encoder,
            video_encoder=video_encoder,
        )

    async def aencode_image(self, paths: list[str]) -> list[Embeddings]:
        """Convert image files to embeddings.

        Args:
            paths (list[str]): List of image file paths.

        Returns:
            list[Embeddings]: List of image embeddings.
        """
        if self.image_encoder is None:
            raise RuntimeError("image encoder for video retrieval is not available")

        return await self.image_encoder(paths)

    async def aencode_video(self, paths: list[str]) -> list[Embeddings]:
        """Convert video files to embeddings.

        Args:
            paths (list[str]): List of video file paths.

        Returns:
            list[Embeddings]: List of video embeddings.
        """
        if self.video_encoder is None:
            raise RuntimeError("video encoder for video retrieval is not available")

        return await self.video_encoder(paths)


class AudioRetriever(BaseRetriever):
    """Retriever specialized for the audio modality."""

    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = 10,
        encoders: Optional[AudioEncoders] = None,
        *,
        filters: Optional[MetadataFilters] = None,
        vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        node_ids: Optional[list[str]] = None,
        doc_ids: Optional[list[str]] = None,
        vector_store_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            index (VectorStoreIndex): Vector store index.
            top_k (int, optional): Maximum number of similar documents to fetch. Defaults to 10.
            encoders (Optional[AudioEncoders], optional): Pre-built encoders. Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filter conditions. Defaults to None.
            vector_store_query_mode (VectorStoreQueryMode, optional): Query mode. Defaults to VectorStoreQueryMode.DEFAULT.
            node_ids (Optional[list[str]], optional): Restrict target node IDs. Defaults to None.
            doc_ids (Optional[list[str]], optional): Restrict target document IDs. Defaults to None.
            vector_store_kwargs (Optional[dict], optional): Extra parameters passed to the vector store. Defaults to None.
        """
        self._index = index
        self._vector_store = index.vector_store
        self._docstore = index.docstore
        self._top_k = top_k
        self._filters = filters
        self._node_ids = node_ids
        self._doc_ids = doc_ids
        self._mode = VectorStoreQueryMode(vector_store_query_mode)
        self._kwargs = vector_store_kwargs or {}

        if encoders is None:
            # NOTE: VectorStoreIndex keeps the embedding model on _embed_model
            embed_model = getattr(index, "_embed_model", None)
            encoders = AudioEncoders.from_embed_model(embed_model)

        self._encoders = encoders

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Run synchronous search using a pre-embedded query.

        Args:
            query_bundle (QueryBundle): Query information.

        Raises:
            NotImplementedError: Not implemented.

        Returns:
            list[NodeWithScore]: Similar nodes.
        """
        raise NotImplementedError("AudioRetriever only supports async retrieval APIs")

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Run async search using a pre-embedded query.

        Args:
            query_bundle (QueryBundle): Query information.

        Returns:
            list[NodeWithScore]: Similar nodes.
        """
        if query_bundle.embedding is None:
            raise RuntimeError("embedding is required for async retrieval")

        return await self._aquery_with_embedding(
            embedding=query_bundle.embedding,
            query_str=query_bundle.query_str,
        )

    async def atext_to_audio_retrieve(
        self, query: Union[str, QueryBundle]
    ) -> list[NodeWithScore]:
        """Retrieve audio modality using a text query.

        Args:
            query (Union[str, QueryBundle]): Text query or QueryBundle.

        Returns:
            list[NodeWithScore]: Similar nodes.
        """
        from llama_index.core.schema import QueryBundle

        if isinstance(query, QueryBundle):
            query_str = query.query_str
            embedding = query.embedding
            if embedding is None:
                if query.embedding_strs:
                    texts = list(query.embedding_strs)
                else:
                    texts = [query.query_str]
                embedding = (await self._encoders.aencode_text(texts))[0]  # type: ignore

            return await self._aquery_with_embedding(
                embedding=embedding, query_str=query_str
            )

        embedding = (await self._encoders.aencode_text([query]))[0]  # type: ignore

        return await self._aquery_with_embedding(embedding=embedding, query_str=query)

    async def aaudio_to_audio_retrieve(self, audio_path: str) -> list[NodeWithScore]:
        """Search using an audio file as the query.

        Args:
            audio_path (str): Query audio file path.

        Returns:
            list[NodeWithScore]: Similar nodes.
        """
        embedding = (await self._encoders.aencode_audio([audio_path]))[0]  # type: ignore

        return await self._aquery_with_embedding(embedding=embedding, query_str="")

    async def _aquery_with_embedding(
        self,
        embedding: Sequence[float],
        query_str: str,
    ) -> list[NodeWithScore]:
        """Search the vector store using an embedding vector.

        Args:
            embedding (Sequence[float]): Query embedding vector.
            query_str (str): Query string.

        Returns:
            list[NodeWithScore]: Similar nodes.
        """
        from llama_index.core.vector_stores.types import VectorStoreQuery

        query = VectorStoreQuery(
            query_embedding=list(embedding),
            similarity_top_k=self._top_k,
            node_ids=self._node_ids,
            doc_ids=self._doc_ids,
            query_str=query_str,
            mode=self._mode,
            filters=self._filters,
        )

        query_result = await self._vector_store.aquery(query, **self._kwargs)

        return self._build_node_list_from_query_result(query_result)

    def _build_node_list_from_query_result(
        self, query_result: VectorStoreQueryResult
    ) -> list[NodeWithScore]:
        """Convert search results into a list of NodeWithScore.

        Args:
            query_result (VectorStoreQueryResult): Query result from the vector store.

        Returns:
            list[NodeWithScore]: Converted node list.
        """
        from llama_index.core.schema import NodeWithScore

        nodes: Iterable[BaseNode] = query_result.nodes or []
        nodes = list(nodes)

        # Re-fetch node from docstore when available.
        for idx, node in enumerate(nodes):
            if node is None:
                continue
            node_id = node.node_id
            if self._docstore.document_exists(node_id):
                nodes[idx] = self._docstore.get_node(node_id)  # type: ignore[assignment]

        node_with_scores: list[NodeWithScore] = []
        for idx, node in enumerate(nodes):
            score: Optional[float] = None
            if query_result.similarities is not None and idx < len(
                query_result.similarities
            ):
                score = query_result.similarities[idx]
            node_with_scores.append(NodeWithScore(node=node, score=score))

        return node_with_scores


class VideoRetriever(AudioRetriever):
    """Retriever specialized for the video modality."""

    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = 10,
        encoders: Optional[VideoEncoders] = None,
        *,
        filters: Optional["MetadataFilters"] = None,
        vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        node_ids: Optional[list[str]] = None,
        doc_ids: Optional[list[str]] = None,
        vector_store_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            index (VectorStoreIndex): Vector store index.
            top_k (int, optional): Maximum number of similar documents to fetch. Defaults to 10.
            encoders (Optional[VideoEncoders], optional): Pre-built encoders. Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filter conditions. Defaults to None.
            vector_store_query_mode (VectorStoreQueryMode, optional): Query mode. Defaults to VectorStoreQueryMode.DEFAULT.
            node_ids (Optional[list[str]], optional): Restrict target node IDs. Defaults to None.
            doc_ids (Optional[list[str]], optional): Restrict target document IDs. Defaults to None.
            vector_store_kwargs (Optional[dict], optional): Extra parameters passed to the vector store. Defaults to None.
        """
        if encoders is None:
            embed_model = getattr(index, "_embed_model", None)
            encoders = VideoEncoders.from_embed_model(embed_model)

        super().__init__(
            index=index,
            top_k=top_k,
            encoders=encoders,
            filters=filters,
            vector_store_query_mode=vector_store_query_mode,
            node_ids=node_ids,
            doc_ids=doc_ids,
            vector_store_kwargs=vector_store_kwargs,
        )

    @property
    def video_encoders(self) -> VideoEncoders:
        """Return the encoders for the video modality.

        Returns:
            VideoEncoders: Encoder set.
        """
        return cast(VideoEncoders, self._encoders)

    async def atext_to_video_retrieve(
        self, query: Union[str, QueryBundle]
    ) -> list[NodeWithScore]:
        """Retrieve video modality using a text query.

        Args:
            query (Union[str, QueryBundle]): Text query or QueryBundle.

        Returns:
            list[NodeWithScore]: Similar nodes.
        """
        from llama_index.core.schema import QueryBundle

        encoders = self.video_encoders
        if isinstance(query, QueryBundle):
            query_str = query.query_str
            embedding = query.embedding
            if embedding is None:
                if query.embedding_strs:
                    texts = list(query.embedding_strs)
                else:
                    texts = [query.query_str]
                embedding = (await encoders.aencode_text(texts))[0]

            return await self._aquery_with_embedding(
                embedding=embedding,
                query_str=query_str,
            )

        embedding = (await encoders.aencode_text([query]))[0]
        return await self._aquery_with_embedding(embedding=embedding, query_str=query)

    async def aimage_to_video_retrieve(self, image_path: str) -> list[NodeWithScore]:
        """Search using an image file as the query.

        Args:
            image_path (str): Query image file path.

        Returns:
            list[NodeWithScore]: Similar nodes.
        """
        encoders = self.video_encoders
        embedding = (await encoders.aencode_image([image_path]))[0]

        return await self._aquery_with_embedding(embedding=embedding, query_str="")

    async def aaudio_to_video_retrieve(self, audio_path: str) -> list[NodeWithScore]:
        """Search using an audio file as the query.

        Args:
            audio_path (str): Query audio file path.

        Returns:
            list[NodeWithScore]: Similar nodes.
        """
        encoders = self.video_encoders
        embedding = (await encoders.aencode_audio([audio_path]))[0]

        return await self._aquery_with_embedding(embedding=embedding, query_str="")

    async def avideo_to_video_retrieve(self, video_path: str) -> list[NodeWithScore]:
        """Search using a video file as the query.

        Args:
            video_path (str): Query video file path.

        Returns:
            list[NodeWithScore]: Similar nodes.
        """
        encoders = self.video_encoders
        embedding = (await encoders.aencode_video([video_path]))[0]

        return await self._aquery_with_embedding(embedding=embedding, query_str="")
