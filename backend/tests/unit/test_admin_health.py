"""
Unit tests for the admin health endpoint provider probe.

Verifies the endpoint is admin-authenticated and that the LLM provider probe
(which sends the configured API key to a metered endpoint) is cached so
anonymous or frequent callers cannot drive repeated API calls.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.routes.admin import _CHAT_PROBE_TTL_SECONDS, _chat_probe_cache, _cached_chat_probe
from app.routes.auth import require_admin


def _reset_probe_cache():
    _chat_probe_cache["status"] = None
    _chat_probe_cache["checked_at"] = 0.0


class TestHealthRouteAuth:
    """Tests that the health endpoint requires an admin."""

    def test_health_route_depends_on_require_admin(self):
        from app.routes.admin import router

        route = next(r for r in router.routes if r.path == "/api/admin/health")

        dependencies = [d.call for d in route.dependant.dependencies]
        assert require_admin in dependencies

    def test_other_admin_routes_checked_for_presence(self):
        """Guards the router prefix used by the health route lookup."""
        from app.routes.admin import router

        assert any(route.path == "/api/admin/health" for route in router.routes)


class TestCachedChatProbe:
    """Tests for the provider health probe cache."""

    @pytest.mark.asyncio
    async def test_probe_result_is_cached(self):
        chat_service = MagicMock()
        chat_service.check_health = AsyncMock(return_value=True)
        chat_service.list_models = AsyncMock(return_value=[{"id": "a"}, {"id": "b"}])

        _reset_probe_cache()
        try:
            with patch(
                "app.routes.admin.get_chat_service", return_value=chat_service
            ):
                first = await _cached_chat_probe()
                second = await _cached_chat_probe()

            assert first == second
            assert chat_service.check_health.await_count == 1
        finally:
            _reset_probe_cache()

    @pytest.mark.asyncio
    async def test_probe_error_is_reported_as_unhealthy(self):
        with patch(
            "app.routes.admin.get_chat_service",
            side_effect=ValueError("OPENROUTER_API_KEY is required for the OpenRouter provider"),
        ):
            _reset_probe_cache()
            try:
                status = await _cached_chat_probe()
            finally:
                _reset_probe_cache()

        assert status.startswith("unhealthy")
        assert "OPENROUTER_API_KEY" in status

    @pytest.mark.asyncio
    async def test_probe_unreachable_provider(self):
        chat_service = MagicMock()
        chat_service.check_health = AsyncMock(return_value=False)
        chat_service.list_models = AsyncMock(return_value=[])

        _reset_probe_cache()
        try:
            with patch("app.routes.admin.get_chat_service", return_value=chat_service):
                status = await _cached_chat_probe()
        finally:
            _reset_probe_cache()

        assert status.startswith("unhealthy")

    def test_probe_ttl_is_bounded(self):
        assert 0 < _CHAT_PROBE_TTL_SECONDS <= 60
