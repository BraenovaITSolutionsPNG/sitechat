"""
Unit tests for the OpenAI-compatible chat/embeddings service.

Covers OpenRouter, OpenAI and Azure endpoint resolution, chat completion
(non-streaming and streaming), and embeddings batching — the pieces that
let SiteChat run on hosted platforms (e.g. Render.com) without Ollama.
"""
import httpx
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.openai_compat import (
    OPENAI_BASE_URL,
    OPENROUTER_BASE_URL,
    OpenAICompatibleConfig,
    OpenAICompatibleEmbeddings,
    OpenAICompatibleService,
    client_error_message,
    format_http_error,
    get_openai_compat_service,
    openai_base_url,
    openrouter_base_url,
    openrouter_headers,
    resolve_chat_config,
)


# ==================== Helpers ====================

def _mock_settings(**overrides):
    """Settings stub with Ollama defaults overridden per test."""
    values = {
        "LLM_PROVIDER": "openrouter",
        "LLM_MODEL": "openai/gpt-4o-mini",
        "LLM_TEMPERATURE": 0.7,
        "LLM_MAX_TOKENS": 1000,
        "OPENAI_API_KEY": None,
        "OPENAI_BASE_URL": None,
        "OPENROUTER_API_KEY": "sk-or-test",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "OPENROUTER_SITE_URL": None,
        "OPENROUTER_APP_NAME": None,
        "AZURE_OPENAI_ENDPOINT": None,
        "AZURE_OPENAI_API_KEY": None,
        "AZURE_OPENAI_DEPLOYMENT": None,
        "AZURE_OPENAI_API_VERSION": "2024-10-21",
        "ANTHROPIC_API_KEY": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeStreamResponse:
    """Async context manager yielding SSE lines, mimicking httpx streaming."""

    def __init__(self, status_code=200, lines=()):
        self.status_code = status_code
        self._lines = list(lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line


# ==================== Config resolution ====================

class TestBaseUrlHelpers:
    """Tests for endpoint defaults and normalization."""

    def test_openai_base_url_default(self):
        with patch(
            "app.services.openai_compat.settings",
            SimpleNamespace(OPENAI_BASE_URL=None),
        ):
            assert openai_base_url() == OPENAI_BASE_URL == "https://api.openai.com/v1"

    def test_openai_base_url_strips_trailing_slash(self):
        with patch(
            "app.services.openai_compat.settings",
            SimpleNamespace(OPENAI_BASE_URL="https://gateway.example.com/v1/"),
        ):
            assert openai_base_url() == "https://gateway.example.com/v1"

    def test_openrouter_base_url_default(self):
        with patch(
            "app.services.openai_compat.settings",
            SimpleNamespace(OPENROUTER_BASE_URL=None),
        ):
            assert openrouter_base_url() == OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"

    def test_openrouter_base_url_strips_trailing_slash(self):
        with patch(
            "app.services.openai_compat.settings",
            SimpleNamespace(OPENROUTER_BASE_URL="https://proxy.example.com/api/v1/"),
        ):
            assert openrouter_base_url() == "https://proxy.example.com/api/v1"


class TestOpenRouterHeaders:
    """Tests for the shared attribution header builder."""

    def test_headers_empty_by_default(self):
        with patch(
            "app.services.openai_compat.settings",
            SimpleNamespace(OPENROUTER_SITE_URL=None, OPENROUTER_APP_NAME=None),
        ):
            assert openrouter_headers() == {}

    def test_headers_include_both_attribution_values(self):
        with patch(
            "app.services.openai_compat.settings",
            SimpleNamespace(
                OPENROUTER_SITE_URL="https://example.com",
                OPENROUTER_APP_NAME="SiteChat",
            ),
        ):
            assert openrouter_headers() == {
                "HTTP-Referer": "https://example.com",
                "X-Title": "SiteChat",
            }

    def test_openrouter_config_includes_attribution_headers(self):
        settings = _mock_settings(
            OPENROUTER_SITE_URL="https://example.com",
            OPENROUTER_APP_NAME="SiteChat",
        )
        with patch("app.services.openai_compat.settings", settings):
            config = resolve_chat_config()

        assert config.extra_headers == {
            "HTTP-Referer": "https://example.com",
            "X-Title": "SiteChat",
        }


class TestResolveChatConfig:
    """Tests for provider -> endpoint resolution."""

    def test_openrouter_config(self):
        with patch("app.services.openai_compat.settings", _mock_settings()):
            config = resolve_chat_config()

        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.api_key == "sk-or-test"
        assert config.model == "openai/gpt-4o-mini"
        assert config.extra_headers == {}
        assert config.auth_style == "bearer"

    def test_openrouter_attribution_headers(self):
        settings = _mock_settings(
            OPENROUTER_SITE_URL="https://example.com",
            OPENROUTER_APP_NAME="SiteChat",
        )
        with patch("app.services.openai_compat.settings", settings):
            config = resolve_chat_config()

        assert config.extra_headers == {
            "HTTP-Referer": "https://example.com",
            "X-Title": "SiteChat",
        }

    def test_openrouter_missing_api_key(self):
        with patch("app.services.openai_compat.settings", _mock_settings(OPENROUTER_API_KEY=None)):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
                resolve_chat_config()

    def test_openai_default_base_url(self):
        settings = _mock_settings(
            LLM_PROVIDER="openai",
            OPENAI_API_KEY="sk-test",
            LLM_MODEL="gpt-4o-mini",
        )
        with patch("app.services.openai_compat.settings", settings):
            config = resolve_chat_config()

        assert config.base_url == "https://api.openai.com/v1"
        assert config.api_key == "sk-test"

    def test_openai_custom_base_url(self):
        settings = _mock_settings(
            LLM_PROVIDER="openai",
            OPENAI_API_KEY="sk-test",
            OPENAI_BASE_URL="https://gateway.example.com/v1",
        )
        with patch("app.services.openai_compat.settings", settings):
            config = resolve_chat_config()

        assert config.base_url == "https://gateway.example.com/v1"

    def test_azure_config(self):
        settings = _mock_settings(
            LLM_PROVIDER="azure",
            AZURE_OPENAI_ENDPOINT="https://test.openai.azure.com/",
            AZURE_OPENAI_API_KEY="azure-key",
            AZURE_OPENAI_DEPLOYMENT="gpt-4-deployment",
            LLM_MODEL="gpt-4-deployment",
        )
        with patch("app.services.openai_compat.settings", settings):
            config = resolve_chat_config()

        assert config.base_url == "https://test.openai.azure.com/openai/deployments/gpt-4-deployment"
        assert config.query == {"api-version": "2024-10-21"}
        assert config.auth_style == "api-key"
        assert config.api_key == "azure-key"

    def test_azure_missing_credentials(self):
        settings = _mock_settings(LLM_PROVIDER="azure")
        with patch("app.services.openai_compat.settings", settings):
            with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"):
                resolve_chat_config()

    def test_azure_falls_back_to_llm_model_as_deployment(self):
        settings = _mock_settings(
            LLM_PROVIDER="azure",
            AZURE_OPENAI_ENDPOINT="https://test.openai.azure.com",
            AZURE_OPENAI_API_KEY="azure-key",
            AZURE_OPENAI_DEPLOYMENT=None,
            LLM_MODEL="gpt-4o-mini",
        )
        with patch("app.services.openai_compat.settings", settings):
            config = resolve_chat_config()

        assert config.base_url.endswith("/openai/deployments/gpt-4o-mini")

    def test_unsupported_provider(self):
        settings = _mock_settings(LLM_PROVIDER="anthropic")
        with patch("app.services.openai_compat.settings", settings):
            with pytest.raises(ValueError, match="not supported by the OpenAI-compatible chat service"):
                resolve_chat_config()


# ==================== Chat completions ====================

def _make_service(config: OpenAICompatibleConfig = None) -> OpenAICompatibleService:
    if config is None:
        config = OpenAICompatibleConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-test",
            model="openai/gpt-4o-mini",
        )
    return OpenAICompatibleService(config)


class TestOpenAICompatibleServiceGenerate:
    """Tests for non-streaming chat completion."""

    @pytest.mark.asyncio
    async def test_generate_returns_content(self):
        service = _make_service()
        response = MagicMock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "Hello there!"}}]}
        client = MagicMock()
        client.post = AsyncMock(return_value=response)

        with patch("app.services.openai_compat._get_http_client", return_value=client):
            result = await service.generate("Hi", "Be nice", temperature=0.5, max_tokens=100)

        assert result == "Hello there!"
        _, kwargs = client.post.call_args
        payload = kwargs["json"]
        assert payload["model"] == "openai/gpt-4o-mini"
        assert payload["messages"] == [
            {"role": "system", "content": "Be nice"},
            {"role": "user", "content": "Hi"},
        ]
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100
        assert payload["stream"] is False
        assert kwargs["headers"]["Authorization"] == "Bearer sk-or-test"

    @pytest.mark.asyncio
    async def test_generate_without_system_prompt(self):
        service = _make_service()
        response = MagicMock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        client = MagicMock()
        client.post = AsyncMock(return_value=response)

        with patch("app.services.openai_compat._get_http_client", return_value=client):
            await service.generate("Hi")

        assert client.post.call_args.kwargs["json"]["messages"] == [
            {"role": "user", "content": "Hi"}
        ]

    @pytest.mark.asyncio
    async def test_generate_error_status_raises_with_detail(self):
        service = _make_service()
        response = MagicMock(status_code=401)
        response.text = "invalid api key"
        client = MagicMock()
        client.post = AsyncMock(return_value=response)

        with patch("app.services.openai_compat._get_http_client", return_value=client):
            with pytest.raises(Exception, match="returned status 401"):
                await service.generate("Hi")

    @pytest.mark.asyncio
    async def test_generate_timeout_raises_friendly_error(self):
        service = _make_service()
        client = MagicMock()
        client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        with patch("app.services.openai_compat._get_http_client", return_value=client):
            with pytest.raises(Exception, match="timed out"):
                await service.generate("Hi")

    @pytest.mark.asyncio
    async def test_azure_uses_api_key_header_and_query(self):
        service = _make_service(OpenAICompatibleConfig(
            base_url="https://test.openai.azure.com/openai/deployments/gpt-4",
            api_key="azure-key",
            model="gpt-4",
            query={"api-version": "2024-10-21"},
            auth_style="api-key",
        ))
        response = MagicMock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        client = MagicMock()
        client.post = AsyncMock(return_value=response)

        with patch("app.services.openai_compat._get_http_client", return_value=client):
            await service.generate("Hi")

        _, kwargs = client.post.call_args
        assert kwargs["headers"]["api-key"] == "azure-key"
        assert "Authorization" not in kwargs["headers"]
        assert kwargs["params"] == {"api-version": "2024-10-21"}


class TestOpenAICompatibleServiceStream:
    """Tests for streaming chat completion (SSE)."""

    @pytest.mark.asyncio
    async def test_stream_yields_content_chunks(self):
        service = _make_service()
        lines = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]
        client = MagicMock()
        client.stream = MagicMock(return_value=_FakeStreamResponse(200, lines))

        with patch("app.services.openai_compat._get_http_client", return_value=client):
            chunks = [chunk async for chunk in service.generate_stream("Hi", "sys")]

        assert chunks == ["Hello", " world"]
        assert client.stream.call_args.args[1].endswith("/chat/completions")

    @pytest.mark.asyncio
    async def test_stream_stops_on_done(self):
        service = _make_service()
        lines = [
            'data: {"choices":[{"delta":{"content":"a"}}]}',
            "data: [DONE]",
            'data: {"choices":[{"delta":{"content":"ignored"}}]}',
        ]
        client = MagicMock()
        client.stream = MagicMock(return_value=_FakeStreamResponse(200, lines))

        with patch("app.services.openai_compat._get_http_client", return_value=client):
            chunks = [chunk async for chunk in service.generate_stream("Hi")]

        assert chunks == ["a"]

    @pytest.mark.asyncio
    async def test_stream_ignores_malformed_lines(self):
        service = _make_service()
        lines = [
            ": keep-alive comment",
            "data: not-json",
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "data: [DONE]",
        ]
        client = MagicMock()
        client.stream = MagicMock(return_value=_FakeStreamResponse(200, lines))

        with patch("app.services.openai_compat._get_http_client", return_value=client):
            chunks = [chunk async for chunk in service.generate_stream("Hi")]

        assert chunks == ["ok"]

    @pytest.mark.asyncio
    async def test_stream_error_status_raises(self):
        service = _make_service()

        class _ErrorResponse(_FakeStreamResponse):
            @property
            def text(self):
                return "model not found"

        client = MagicMock()
        client.stream = MagicMock(return_value=_ErrorResponse(404, []))

        with patch("app.services.openai_compat._get_http_client", return_value=client):
            with pytest.raises(Exception, match="returned status 404"):
                _ = [chunk async for chunk in service.generate_stream("Hi")]


class TestClientErrorMessages:
    """Tests that client-visible errors never leak endpoint or upstream body."""

    def test_format_http_error_includes_detail_for_logs(self):
        response = MagicMock(status_code=500)
        response.text = "Incorrect API key provided: sk-or-secret***"

        detail = format_http_error(
            response, "/chat/completions", "https://gateway.example.com/v1", "gpt-4o-mini"
        )

        # Log detail keeps the endpoint and upstream body for debugging.
        assert "https://gateway.example.com/v1/chat/completions" in detail
        assert "500" in detail
        assert "sk-or-secret" in detail

    def test_format_http_error_404_includes_endpoint(self):
        response = MagicMock(status_code=404)
        response.text = ""

        detail = format_http_error(
            response, "/chat/completions", "https://gateway.example.com/v1", "gpt-4o-mini"
        )

        assert 'Model "gpt-4o-mini" was not found at https://gateway.example.com/v1' in detail

    def test_client_error_message_hides_endpoint_and_body(self):
        message = client_error_message(401, "gpt-4o-mini")

        assert "401" in message
        assert "openrouter" not in message.lower()
        assert "gateway" not in message.lower()
        assert "API key is valid" in message

    def test_client_error_message_404_mentions_model_not_url(self):
        message = client_error_message(404, "gpt-4o-mini")

        assert "gpt-4o-mini" in message
        assert "https://" not in message

    def test_client_error_message_unknown_status_is_generic(self):
        message = client_error_message(503, "gpt-4o-mini")

        assert "503" in message
        assert "Please try again" in message

    @pytest.mark.asyncio
    async def test_generate_error_raises_sanitized_message(self):
        service = _make_service()
        response = MagicMock(status_code=401)
        response.text = '{"error": {"message": "Invalid key sk-or-abc123"}}'
        client = MagicMock()
        client.post = AsyncMock(return_value=response)

        with patch("app.services.openai_compat._get_http_client", return_value=client):
            with pytest.raises(Exception) as exc_info:
                await service.generate("Hi")

        message = str(exc_info.value)
        assert "401" in message
        assert "sk-or-abc123" not in message
        assert "openrouter.ai" not in message

    def test_embed_query_error_raises_sanitized_message(self):
        embeddings = OpenAICompatibleEmbeddings(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-test",
            model="openai/text-embedding-3-small",
        )
        client = MagicMock()
        client.post = MagicMock(
            return_value=MagicMock(status_code=402, text="insufficient credits: balance 0.00")
        )

        with patch("app.services.openai_compat._get_sync_client", return_value=client):
            with pytest.raises(Exception) as exc_info:
                embeddings.embed_query("hello")

        message = str(exc_info.value)
        assert "402" in message
        assert "insufficient credits: balance" not in message
        assert "openrouter.ai" not in message

    def test_embeddings_always_uses_bearer_header(self):
        embeddings = OpenAICompatibleEmbeddings(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-test",
            model="m",
        )
        client = MagicMock()
        client.post = MagicMock(return_value=_embeddings_response([[0.1]]))

        with patch("app.services.openai_compat._get_sync_client", return_value=client):
            embeddings.embed_query("hello")

        headers = client.post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-or-test"
        assert "api-key" not in headers


class TestOpenAICompatibleServiceHealth:
    """Tests for health check and model listing."""
    @pytest.mark.asyncio
    async def test_check_health_ok(self):
        service = _make_service()
        client = MagicMock()
        client.get = AsyncMock(return_value=MagicMock(status_code=200))

        with patch("app.services.openai_compat._get_http_client", return_value=client):
            assert await service.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_unauthorized(self):
        service = _make_service()
        client = MagicMock()
        client.get = AsyncMock(return_value=MagicMock(status_code=401))

        with patch("app.services.openai_compat._get_http_client", return_value=client):
            assert await service.check_health() is False

    @pytest.mark.asyncio
    async def test_list_models(self):
        service = _make_service()
        response = MagicMock(status_code=200)
        response.json.return_value = {"data": [{"id": "openai/gpt-4o-mini"}, {"id": "anthropic/claude-3.5-sonnet"}]}
        client = MagicMock()
        client.get = AsyncMock(return_value=response)

        with patch("app.services.openai_compat._get_http_client", return_value=client):
            models = await service.list_models()

        assert models == ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"]

    def test_get_openai_compat_service_singleton(self):
        import app.services.openai_compat as module

        module._service_instance = None
        with patch.object(module, "resolve_chat_config", return_value=OpenAICompatibleConfig(
            base_url="https://openrouter.ai/api/v1", api_key="k", model="m"
        )):
            first = get_openai_compat_service()
            second = get_openai_compat_service()

        assert first is second
        module._service_instance = None


# ==================== Embeddings ====================

def _embeddings_response(vectors):
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "data": [{"index": i, "embedding": vector} for i, vector in enumerate(vectors)]
    }
    return response


class TestOpenAICompatibleEmbeddings:
    """Tests for OpenAI-compatible embeddings."""

    def test_embed_query_sends_model_and_dimensions(self):
        embeddings = OpenAICompatibleEmbeddings(
            base_url="https://openrouter.ai/api/v1/",
            api_key="sk-or-test",
            model="openai/text-embedding-3-small",
            dimensions=1536,
        )
        client = MagicMock()
        client.post = MagicMock(return_value=_embeddings_response([[0.1, 0.2, 0.3]]))

        with patch("app.services.openai_compat._get_sync_client", return_value=client):
            vector = embeddings.embed_query("hello")

        assert vector == [0.1, 0.2, 0.3]
        _, kwargs = client.post.call_args
        assert kwargs["json"]["model"] == "openai/text-embedding-3-small"
        assert kwargs["json"]["input"] == ["hello"]
        assert kwargs["json"]["dimensions"] == 1536
        assert kwargs["headers"]["Authorization"] == "Bearer sk-or-test"
        assert client.post.call_args.args[0] == "https://openrouter.ai/api/v1/embeddings"

    def test_embed_query_omits_dimensions_when_unset(self):
        embeddings = OpenAICompatibleEmbeddings(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-test",
            model="openai/text-embedding-3-small",
        )
        client = MagicMock()
        client.post = MagicMock(return_value=_embeddings_response([[0.5]]))

        with patch("app.services.openai_compat._get_sync_client", return_value=client):
            embeddings.embed_query("hello")

        assert "dimensions" not in client.post.call_args.kwargs["json"]

    def test_embed_documents_batches(self):
        embeddings = OpenAICompatibleEmbeddings(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-test",
            model="m",
            batch_size=2,
        )
        client = MagicMock()
        client.post = MagicMock(side_effect=[
            _embeddings_response([[1.0], [2.0]]),
            _embeddings_response([[3.0], [4.0]]),
            _embeddings_response([[5.0]]),
        ])

        with patch("app.services.openai_compat._get_sync_client", return_value=client):
            vectors = embeddings.embed_documents(["a", "b", "c", "d", "e"])

        assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]
        assert client.post.call_count == 3

    def test_embed_documents_reorders_by_index(self):
        embeddings = OpenAICompatibleEmbeddings(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-test",
            model="m",
        )
        client = MagicMock()
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "data": [
                {"index": 1, "embedding": [2.0]},
                {"index": 0, "embedding": [1.0]},
            ]
        }
        client.post = MagicMock(return_value=response)

        with patch("app.services.openai_compat._get_sync_client", return_value=client):
            vectors = embeddings.embed_documents(["a", "b"])

        assert vectors == [[1.0], [2.0]]

    def test_embed_documents_empty_input(self):
        embeddings = OpenAICompatibleEmbeddings(
            base_url="https://openrouter.ai/api/v1", api_key="k", model="m"
        )
        assert embeddings.embed_documents([]) == []

    def test_embed_query_error_status(self):
        embeddings = OpenAICompatibleEmbeddings(
            base_url="https://openrouter.ai/api/v1", api_key="k", model="m"
        )
        client = MagicMock()
        client.post = MagicMock(return_value=MagicMock(status_code=402, text="insufficient credits"))

        with patch("app.services.openai_compat._get_sync_client", return_value=client):
            with pytest.raises(Exception, match="returned status 402"):
                embeddings.embed_query("hello")


# ==================== RAG engine dispatcher ====================

class TestGetChatService:
    """Tests that the RAG engine picks the right chat service per provider."""

    def test_ollama_provider_uses_ollama_service(self):
        sentinel = object()
        with patch("app.services.rag_engine.settings", SimpleNamespace(LLM_PROVIDER="ollama")), \
             patch("app.services.rag_engine.get_ollama_service", return_value=sentinel):
            from app.services.rag_engine import get_chat_service
            assert get_chat_service() is sentinel

    def test_openrouter_provider_uses_compat_service(self):
        sentinel = object()
        with patch("app.services.rag_engine.settings", SimpleNamespace(LLM_PROVIDER="openrouter")), \
             patch("app.services.rag_engine.get_openai_compat_service", return_value=sentinel):
            from app.services.rag_engine import get_chat_service
            assert get_chat_service() is sentinel

    def test_openai_provider_uses_compat_service(self):
        sentinel = object()
        with patch("app.services.rag_engine.settings", SimpleNamespace(LLM_PROVIDER="openai")), \
             patch("app.services.rag_engine.get_openai_compat_service", return_value=sentinel):
            from app.services.rag_engine import get_chat_service
            assert get_chat_service() is sentinel

    def test_anthropic_provider_raises_clear_error(self):
        from app.services.rag_engine import get_chat_service

        with patch("app.services.rag_engine.settings", SimpleNamespace(LLM_PROVIDER="anthropic")):
            with pytest.raises(ValueError, match="not supported by the OpenAI-compatible chat service"):
                get_chat_service()
