"""
Unit tests for the crawl_and_index_site background task.

Verifies that the site document is updated authoritatively alongside the
crawl job (status + error on failure, completed on success).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.asyncio
async def test_crawl_and_index_site_updates_site_on_success():
    """On success the site doc is marked completed with no error."""
    from app.routes.embed import crawl_and_index_site

    mongodb = MagicMock()
    mongodb.create_crawl_job = AsyncMock(return_value="job-1")
    mongodb.update_crawl_job = AsyncMock()
    mongodb.update_site = AsyncMock()

    pages = [
        {"url": "https://example.com", "title": "Home", "content": "hello world"},
    ]
    crawler = MagicMock()
    crawler.crawl = AsyncMock(return_value=pages)
    crawler.get_stats = MagicMock(return_value={"error_messages": []})
    indexer = MagicMock()
    indexer.index_pages = AsyncMock(return_value={"indexed_pages": 1, "total_chunks": 2})

    with patch("app.routes.embed.get_mongodb", AsyncMock(return_value=mongodb)), \
         patch("app.routes.embed.CrawlerService", return_value=crawler), \
         patch("app.routes.embed.IndexerService", return_value=indexer):
        await crawl_and_index_site("https://example.com", "site-123", max_pages=50)

    # Crawl job marked completed
    mongodb.update_crawl_job.assert_any_call(
        "job-1",
        status="completed",
        pages_crawled=1,
        pages_indexed=1,
    )
    # Site doc marked completed with no error
    mongodb.update_site.assert_any_call(
        "site-123", {"status": "completed", "error": None}
    )


@pytest.mark.asyncio
async def test_crawl_and_index_site_updates_site_on_indexing_error():
    """When indexing raises, the site doc is marked failed with the error."""
    from app.routes.embed import crawl_and_index_site

    mongodb = MagicMock()
    mongodb.create_crawl_job = AsyncMock(return_value="job-2")
    mongodb.update_crawl_job = AsyncMock()
    mongodb.update_site = AsyncMock()

    pages = [{"url": "https://example.com", "title": "Home", "content": "x"}]
    crawler = MagicMock()
    crawler.crawl = AsyncMock(return_value=pages)
    crawler.get_stats = MagicMock(return_value={"error_messages": []})
    indexer = MagicMock()
    indexer.index_pages = AsyncMock(side_effect=RuntimeError("embedding backend down"))

    with patch("app.routes.embed.get_mongodb", AsyncMock(return_value=mongodb)), \
         patch("app.routes.embed.CrawlerService", return_value=crawler), \
         patch("app.routes.embed.IndexerService", return_value=indexer):
        await crawl_and_index_site("https://example.com", "site-456", max_pages=50)

    # Crawl job marked failed with the error
    mongodb.update_crawl_job.assert_any_call(
        "job-2", status="failed", error="embedding backend down"
    )
    # Site doc marked failed with the error
    mongodb.update_site.assert_any_call(
        "site-456", {"status": "failed", "error": "embedding backend down"}
    )


@pytest.mark.asyncio
async def test_crawl_and_index_site_updates_site_on_no_pages():
    """When no pages are found the site doc is marked failed with the reason."""
    from app.routes.embed import crawl_and_index_site

    mongodb = MagicMock()
    mongodb.create_crawl_job = AsyncMock(return_value="job-3")
    mongodb.update_crawl_job = AsyncMock()
    mongodb.update_site = AsyncMock()

    crawler = MagicMock()
    crawler.crawl = AsyncMock(return_value=[])
    crawler.get_stats = MagicMock(return_value={"error_messages": ["403 Forbidden"]})
    indexer = MagicMock()
    indexer.index_pages = AsyncMock()

    with patch("app.routes.embed.get_mongodb", AsyncMock(return_value=mongodb)), \
         patch("app.routes.embed.CrawlerService", return_value=crawler), \
         patch("app.routes.embed.IndexerService", return_value=indexer):
        await crawl_and_index_site("https://example.com", "site-789", max_pages=50)

    # Crawl job marked failed with the derived error message
    failed_calls = [
        c for c in mongodb.update_crawl_job.call_args_list
        if c.kwargs.get("status") == "failed"
    ]
    assert failed_calls, "expected update_crawl_job to be marked failed"
    assert failed_calls[0].kwargs["error"] == "Site is blocking crawlers (403 Forbidden)"
    # Site doc marked failed with the same error
    mongodb.update_site.assert_any_call(
        "site-789",
        {"status": "failed", "error": "Site is blocking crawlers (403 Forbidden)"},
    )


@pytest.mark.asyncio
async def test_crawl_and_index_site_site_update_failure_does_not_mask_crawl_error():
    """If updating the site doc fails, the crawl job still records the crawl error.

    The site-update failure must not mask the original crawl error: the job is
    updated with the crawl error and the site-update failure is logged, not
    raised.
    """
    from app.routes.embed import crawl_and_index_site

    mongodb = MagicMock()
    mongodb.create_crawl_job = AsyncMock(return_value="job-4")
    mongodb.update_crawl_job = AsyncMock()
    mongodb.update_site = AsyncMock(side_effect=RuntimeError("site update exploded"))

    pages = [{"url": "https://example.com", "title": "Home", "content": "x"}]
    crawler = MagicMock()
    crawler.crawl = AsyncMock(return_value=pages)
    crawler.get_stats = MagicMock(return_value={"error_messages": []})
    indexer = MagicMock()
    indexer.index_pages = AsyncMock(side_effect=RuntimeError("crawl boom"))

    with patch("app.routes.embed.get_mongodb", AsyncMock(return_value=mongodb)), \
         patch("app.routes.embed.CrawlerService", return_value=crawler), \
         patch("app.routes.embed.IndexerService", return_value=indexer):
        # Must not raise: the site-update failure is swallowed/logged so the
        # original crawl error is what gets recorded on the job.
        await crawl_and_index_site("https://example.com", "site-000", max_pages=50)

    # Crawl job records the original crawl error, not the site-update error.
    mongodb.update_crawl_job.assert_any_call(
        "job-4", status="failed", error="crawl boom"
    )
    # The site update was attempted (and failed internally).
    mongodb.update_site.assert_called()