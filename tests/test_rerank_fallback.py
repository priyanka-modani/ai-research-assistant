from unittest.mock import Mock, patch

import requests

from research_assistant.providers import OpenAICompatibleProvider


def test_rerank_404_falls_back_to_input_order():
    provider = object.__new__(OpenAICompatibleProvider)
    provider.rerank_model = "Qwen/Qwen3-Reranker-8B"
    provider.base_url = "https://api.tokenfactory.nebius.com/v1"
    provider.api_key = "test-key"
    provider.rerank_warning = None

    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("404 Client Error")
    with patch("research_assistant.providers.requests.post", return_value=response):
        result = provider.rerank("query", ["first", "second"])

    assert result == [(0, 0.0), (1, 0.0)]
    assert "Continuing with Reciprocal Rank Fusion" in provider.rerank_warning
