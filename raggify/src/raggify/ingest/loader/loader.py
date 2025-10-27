from __future__ import annotations


class Loader:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """ローダー基底クラス。

        Args:
            chunk_size (int): チャンクサイズ
            chunk_overlap (int): チャンク重複語数
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # 最上位の load_from_*_list() がループを回している間は一度もストアに書き出されないので
        # 同一ソースに対して何度もフェッチがかかる場合がある。それを避けるため、
        # Loader クラス内にも独自のキャッシュを持つ。
        self._source_cache: set[str] = set()
