from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore


@dataclass
class RerankContainer:
    """リランク関連パラメータを集約"""

    provider_name: str
    rerank: BaseNodePostprocessor


class RerankManager:
    """リランクの管理クラス。"""

    def __init__(self, cont: Optional[RerankContainer] = None) -> None:
        """コンストラクタ

        Args:
            cont (RerankContainer): リランクコンテナ
        """
        self._cont = cont

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        return self._cont.provider_name if self._cont else "none"

    async def arerank(
        self, nodes: list[NodeWithScore], query: str, topk: int
    ) -> list[NodeWithScore]:
        """クエリに基づきリランカーで結果を並べ替える。

        Args:
            nodes (list[NodeWithScore]): 並べ替え対象ノード
            query (str): クエリ文字列
            topk (int): 取得件数

        Returns:
            list[NodeWithScore]: 並べ替え済みのノード

        Raises:
            RuntimeError: リランカーが処理に失敗した場合
        """
        if self._cont is None:
            logger.info("rerank provider is not specified")
            return nodes

        # reranker の top_n は本来インスタンス生成時に値を指定するが
        # 単発 retrieve 時に一時的に値を変更したい場合もあるので
        # 必ずここで書き換え・書き戻す運用とする。
        original_top_n: Optional[int] = None
        if hasattr(self._cont.rerank, "top_n"):
            original_top_n = getattr(self._cont.rerank, "top_n")
            setattr(self._cont.rerank, "top_n", topk)

        try:
            return await self._cont.rerank.apostprocess_nodes(
                nodes=nodes, query_str=query
            )
        except Exception as e:
            raise RuntimeError("failed to rerank documents") from e
        finally:
            if original_top_n:
                setattr(self._cont.rerank, "top_n", original_top_n)
