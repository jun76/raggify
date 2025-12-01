from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from raggify.config.config_manager import ConfigManager
from raggify.config.general_config import GeneralConfig
from raggify.config.rerank_config import RerankConfig, RerankProvider
from raggify.rerank.rerank import create_rerank_manager
from raggify.rerank.rerank_manager import RerankContainer, RerankManager
from tests.utils.mock_rerank import FakeReranker, setup_rerank_mocks
from tests.utils.node_factory import make_sample_nodes


def _make_cfg(provider: RerankProvider | None) -> ConfigManager:
    general = GeneralConfig(rerank_provider=provider)
    rerank = RerankConfig()
    stub = SimpleNamespace(general=general, rerank=rerank)
    return cast(ConfigManager, stub)


@pytest.fixture(autouse=True)
def mock_rerankers(monkeypatch):
    return setup_rerank_mocks(monkeypatch)


@pytest.mark.parametrize(
    "provider,key",
    [
        (RerankProvider.COHERE, "cohere"),
        # (RerankProvider.FLAGEMBEDDING, "flag"),
        (RerankProvider.VOYAGE, "voyage"),
    ],
)
def test_create_rerank_manager_variants(provider, key, mock_rerankers):
    cfg = _make_cfg(provider)
    manager = create_rerank_manager(cfg)
    assert manager.name == provider
    assert mock_rerankers[key][0].model


def test_create_rerank_manager_without_provider():
    cfg = _make_cfg(None)
    manager = create_rerank_manager(cfg)
    assert manager.name == "none"


def test_rerank_manager_runs_rerank():
    nodes = make_sample_nodes()
    reranker = FakeReranker()
    reranker.top_n = 3
    cont = RerankContainer(
        provider_name="mock", rerank=cast(BaseNodePostprocessor, reranker)
    )
    manager = RerankManager(cont)

    result = asyncio.run(manager.arerank(nodes, "query", topk=1))
    assert len(result) == 1
    assert reranker.top_n == 3
    assert reranker.calls


def test_rerank_manager_returns_original_when_missing():
    nodes = make_sample_nodes()
    manager = RerankManager()
    result = asyncio.run(manager.arerank(nodes, "query", topk=2))
    assert result == nodes


def test_rerank_manager_raises_on_failure():
    class BadReranker(FakeReranker):
        async def apostprocess_nodes(self, *args, **kwargs):
            raise ValueError("boom")

    cont = RerankContainer(
        provider_name="bad", rerank=cast(BaseNodePostprocessor, BadReranker())
    )
    manager = RerankManager(cont)
    with pytest.raises(RuntimeError):
        asyncio.run(manager.arerank(make_sample_nodes(), "q", topk=1))
