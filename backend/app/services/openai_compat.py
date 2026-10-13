"""
OpenAI-compatible LLM service (OpenRouter, OpenAI, Azure OpenAI).

The RAG engine talks to chat completions endpoints, which are wire-compatible
across these providers. This service exposes the same interface as
``OllamaService`` so ``RAGEngine`` can switch providers transparently.
"""
import json
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional

import httpx
from loguru import logger

from app.config import settings

# Reuse one HTTP client per process (avoids TCP/TLS handshake overhead per chat).
_http_client: Optional[httpx.AsyncClient] = None

# Embeddings run in worker threads, so they use a separate sync client.
_sync_client: Optional[httpx.Client] = None

# Embedding APIs cap the number of inputs per request; batch to stay under it.
EMBED_BATCH_SIZE = 32

# Single source of truth for OpenAI-compatible endpoint defaults.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )
    return _http_client


def _get_sync_client() -> httpx.Client:
    global _sync_client
    if _sync_client is None or _sync_client.is_closed:
        _sync_client = httpx.Client(
            timeout=httpx.Timeout(120.0),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )
    return _sync_client


@dataclass
class OpenAICompatibleConfig:
    """Resolved connection settings for an OpenAI-compatible endpoint."""

    base_url: str
    api_key: str
    model: str
    extra_headers: Dict[str, str] = field(default_factory=dict)
    query: Optional[Dict[str, str]] = None
    auth_style: str = "bearer"  # "bearer" (Authorization) or "api-key" (Azure)


def openrouter_headers() -> Dict[str, str]:
    """OpenRouter attribution headers, built once for chat and embeddings."""
    headers: Dict[str, str] = {}
    if settings.OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
    if settings.OPENROUTER_APP_NAME:
        headers["X-Title"] = settings.OPENROUTER_APP_NAME
    return headers


def openai_base_url() -> str:
    """Configured OpenAI base URL, normalized (trailing slash stripped)."""
    return str(settings.OPENAI_BASE_URL or OPENAI_BASE_URL).rstrip("/")


def openrouter_base_url() -> str:
    """Configured OpenRouter base URL, normalized (trailing slash stripped)."""
    return str(settings.OPENROUTER_BASE_URL or OPENROUTER_BASE_URL).rstrip("/")


def resolve_chat_config() -> OpenAICompatibleConfig:
    """
    Build the chat endpoint config for the configured LLM provider.

    Raises:
        ValueError: If the provider is not OpenAI-compatible or credentials are missing.
    """
    provider = settings.LLM_PROVIDER

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
        return OpenAICompatibleConfig(
            base_url=openai_base_url(),
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL,
        )

    if provider == "openrouter":
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required for the OpenRouter provider")
        return OpenAICompatibleConfig(
            base_url=openrouter_base_url(),
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.LLM_MODEL,
            extra_headers=openrouter_headers(),
        )

    if provider == "azure":
        if not settings.AZURE_OPENAI_ENDPOINT or not settings.AZURE_OPENAI_API_KEY:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are required for the Azure provider"
            )
        deployment = settings.AZURE_OPENAI_DEPLOYMENT or settings.LLM_MODEL
        return OpenAICompatibleConfig(
            base_url=f"{settings.AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{deployment}",
            api_key=settings.AZURE_OPENAI_API_KEY,
            model=settings.LLM_MODEL,
            query={"api-version": settings.AZURE_OPENAI_API_VERSION},
            auth_style="api-key",
        )

    raise ValueError(
        f"LLM provider '{provider}' is not supported by the OpenAI-compatible chat service. "
        "Use ollama, openai, openrouter, or azure."
    )


# Remediation hints per HTTP status, shared by chat and embeddings errors.
STATUS_HINTS = {
    401: "Check that the API key is valid and active.",
    402: "The account is out of credits — top up the provider balance.",
    429: "Rate limit reached — retry shortly or lower request volume.",
}


def format_http_error(
    response: httpx.Response,
    endpoint: str,
    base_url: str,
    model: Optional[str] = None,
    hints: Optional[Dict[int, str]] = None,
) -> str:
    """
    Build a detailed error message for logs.

    Includes the endpoint and upstream body — never send this to API clients,
    use ``client_error_message`` instead.
    """
    code = response.status_code
    body = (response.text or "").strip()
    if len(body) > 500:
        body = body[:500] + "…"

    effective_hints = dict(STATUS_HINTS)
    if hints:
        effective_hints.update(hints)

    hint = effective_hints.get(code)
    if hint is None:
        if code == 404 and model:
            hint = f'Model "{model}" was not found at {base_url}.'
        else:
            hint = f"Request: POST {base_url}{endpoint}."
    return f"Provider returned status {code}. {hint} Response: {body or '(empty)'}"


def client_error_message(
    status_code: int,
    model: Optional[str] = None,
    hints: Optional[Dict[int, str]] = None,
) -> str:
    """
    Build a client-safe error message: no endpoint URLs and no upstream body,
    so gateway credentials or internal hosts are never disclosed to callers.
    """
    effective_hints = dict(STATUS_HINTS)
    if hints:
        effective_hints.update(hints)

    hint = effective_hints.get(status_code)
    if hint is None:
        if status_code == 404 and model:
            hint = f'Model "{model}" was not found. Check LLM_MODEL.'
        else:
            hint = "Please try again later."
    return f"LLM provider returned status {status_code}. {hint}"


class OpenAICompatibleService:
    """Chat completions service for OpenAI-compatible providers."""

    def __init__(self, config: Optional[OpenAICompatibleConfig] = None):
        self.config = config or resolve_chat_config()
        self.model = self.config.model

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self.config.extra_headers)
        if self.config.auth_style == "api-key":
            headers["api-key"] = self.config.api_key
        else:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _extract_content(self, payload: dict) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        message = choice.get("message") or choice.get("delta") or {}
        return message.get("content") or ""

    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            The generated response text
        """
        try:
            client = _get_http_client()
            response = await client.post(
                f"{self.config.base_url}/chat/completions",
                headers=self._headers(),
                params=self.config.query,
                json={
                    "model": self.config.model,
                    "messages": self._build_messages(prompt, system_prompt),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )

            if response.status_code != 200:
                detail = format_http_error(
                    response, "/chat/completions", self.config.base_url, self.config.model
                )
                logger.error(detail)
                raise Exception(client_error_message(response.status_code, self.config.model))

            return self._extract_content(response.json())

        except httpx.TimeoutException:
            logger.error("LLM request timed out")
            raise Exception("LLM request timed out. Please try again.")
        except Exception as e:
            logger.error(f"OpenAI-compatible chat error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from the LLM (server-sent events).

        Yields:
            Chunks of the generated response
        """
        try:
            client = _get_http_client()
            async with client.stream(
                "POST",
                f"{self.config.base_url}/chat/completions",
                headers=self._headers(),
                params=self.config.query,
                json={
                    "model": self.config.model,
                    "messages": self._build_messages(prompt, system_prompt),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
            ) as response:
                if response.status_code != 200:
                    detail = format_http_error(
                        response, "/chat/completions", self.config.base_url, self.config.model
                    )
                    logger.error(detail)
                    raise Exception(client_error_message(response.status_code, self.config.model))

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data or data == "[DONE]":
                        if data == "[DONE]":
                            break
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    content = self._extract_content(event)
                    if content:
                        yield content

        except Exception as e:
            logger.error(f"OpenAI-compatible streaming error: {e}")
            raise

    async def check_health(self) -> bool:
        """Check if the chat endpoint is reachable and the key is accepted."""
        try:
            client = _get_http_client()
            response = await client.get(
                f"{self.config.base_url}/models",
                headers=self._headers(),
                params=self.config.query,
                timeout=10.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list:
        """List models available to the configured key (OpenAI-compatible /models)."""
        try:
            client = _get_http_client()
            response = await client.get(
                f"{self.config.base_url}/models",
                headers=self._headers(),
                params=self.config.query,
                timeout=10.0,
            )
            if response.status_code == 200:
                return [m["id"] for m in response.json().get("data", []) if m.get("id")]
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []


class OpenAICompatibleEmbeddings:
    """
    LangChain-compatible embeddings backed by an OpenAI-compatible endpoint.

    Implements the ``langchain_core.embeddings.Embeddings`` interface
    (``embed_documents`` / ``embed_query``) without extra dependencies, so
    deployments can skip local model downloads (e.g. no torch/HuggingFace).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        batch_size: int = EMBED_BATCH_SIZE,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.extra_headers = extra_headers or {}
        self.batch_size = max(1, batch_size)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self.extra_headers)
        headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        payload: Dict[str, object] = {"model": self.model, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        client = _get_sync_client()
        response = client.post(
            f"{self.base_url}/embeddings",
            headers=self._headers(),
            json=payload,
        )
        if response.status_code != 200:
            detail = format_http_error(
                response, "/embeddings", self.base_url, self.model
            )
            logger.error(detail)
            raise Exception(client_error_message(response.status_code, self.model))

        data = response.json().get("data", [])
        # Providers may return items out of order; index is authoritative.
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in ordered]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (batched)."""
        if not texts:
            return []
        results: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            results.extend(self._embed_batch(texts[start:start + self.batch_size]))
        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self._embed_batch([text])[0]


_service_instance: Optional[OpenAICompatibleService] = None


def get_openai_compat_service() -> OpenAICompatibleService:
    """Get or create the OpenAI-compatible chat service singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = OpenAICompatibleService()
    return _service_instance
