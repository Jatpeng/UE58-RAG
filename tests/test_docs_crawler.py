"""Tests for the bounded and resumable documentation downloader."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from pathlib import Path

import pytest

from ue_rag.crawler.docs import (
    CrawlRecord,
    CrawlStatus,
    DocumentationCrawler,
    DocumentationCrawlerConfig,
    HttpResponse,
    TopicConfig,
    load_docs_crawler_config,
)


class FakeHttpClient:
    def __init__(self) -> None:
        self.responses: dict[str, deque[HttpResponse | Exception]] = defaultdict(deque)
        self.calls: list[str] = []

    def add(self, url: str, *responses: HttpResponse | Exception) -> None:
        self.responses[url].extend(responses)

    def get(self, url: str, *, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        self.calls.append(url)
        if not self.responses[url]:
            raise AssertionError(f"Unexpected HTTP request: {url}")
        response = self.responses[url].popleft()
        if isinstance(response, Exception):
            raise response
        return response


class FixedRobotsPolicy:
    def __init__(self, allowed: bool) -> None:
        self.allowed = allowed

    def can_fetch(self, user_agent: str, url: str) -> bool:
        return self.allowed


@pytest.fixture
def crawler_config(tmp_path: Path) -> DocumentationCrawlerConfig:
    return DocumentationCrawlerConfig(
        engine_version="5.8",
        output_dir=tmp_path / "raw" / "docs",
        base_url="https://dev.epicgames.com",
        locale="en-us",
        robots_url="https://dev.epicgames.com/robots.txt",
        user_agent="UE58RAG-Test/0.1",
        timeout_seconds=5,
        delay_seconds=0,
        max_retries=2,
        retry_backoff_seconds=0,
        robots_cache_ttl_seconds=86400,
        max_pages=20,
        max_depth=1,
        allowed_hosts={"dev.epicgames.com"},
        allowed_path_prefixes={"/documentation/en-us/unreal-engine/"},
        topics={
            "gameplay": TopicConfig(
                seeds=["/documentation/unreal-engine/gameplay-root"],
                include_patterns=["gameplay"],
            )
        },
    )


def make_response(url: str, content: bytes, status: int = 200) -> HttpResponse:
    return HttpResponse(
        status_code=status,
        url=url,
        content=content,
        headers={"content-type": "text/html; charset=utf-8"},
    )


def test_config_loads_version_only_from_ue_config() -> None:
    config = load_docs_crawler_config()

    assert config.engine_version == "5.8"
    assert set(config.topics) == {
        "gameplay",
        "programming",
        "cpp",
        "blueprint",
        "networking",
        "rendering",
        "animation",
        "ai",
        "ui",
    }


def test_dry_run_has_no_network_or_filesystem_side_effects(
    crawler_config: DocumentationCrawlerConfig,
) -> None:
    client = FakeHttpClient()
    crawler = DocumentationCrawler(crawler_config, http_client=client)

    summary = crawler.crawl(["gameplay"], dry_run=True)

    assert len(summary.planned_urls) == 1
    assert summary.planned_urls[0].endswith("?application_version=5.8")
    assert client.calls == []
    assert not crawler_config.output_dir.exists()


def test_url_normalization_deduplicates_locale_query_and_fragment_variants(
    crawler_config: DocumentationCrawlerConfig,
) -> None:
    crawler = DocumentationCrawler(crawler_config)
    variants = {
        crawler.normalize_url(
            "/documentation/unreal-engine/gameplay-root?lang=en-US#overview"
        ),
        crawler.normalize_url(
            "https://dev.epicgames.com/documentation/fr-fr/unreal-engine/gameplay-root/"
        ),
        crawler.normalize_url(
            "/documentation/en-us/unreal-engine/gameplay-root?application_version=5.7"
        ),
    }

    assert variants == {
        "https://dev.epicgames.com/documentation/en-us/unreal-engine/gameplay-root"
        "?application_version=5.8"
    }


def test_download_writes_html_manifest_and_resumes_from_cache(
    crawler_config: DocumentationCrawlerConfig,
) -> None:
    client = FakeHttpClient()
    crawler = DocumentationCrawler(
        crawler_config,
        http_client=client,
        robots_policy=FixedRobotsPolicy(True),
    )
    seed = crawler.plan(["gameplay"])[0].url
    child = crawler.normalize_url("/documentation/unreal-engine/gameplay-child")
    html = (
        b'<html><a href="/documentation/unreal-engine/gameplay-child">child</a>'
        b'<a href="/documentation/en-us/unreal-engine/gameplay-child?lang=en-US">dupe</a>'
        b'<a href="https://example.com/gameplay-outside">outside</a></html>'
    )
    client.add(seed, make_response(seed, html))
    client.add(child, make_response(child, b"<html>child</html>"))

    first = crawler.crawl(["gameplay"])

    assert first.downloaded == 2
    assert client.calls == [seed, child]
    manifest_lines = crawler.manifest_path.read_text(encoding="utf-8").splitlines()
    records = [CrawlRecord.model_validate_json(line) for line in manifest_lines]
    assert len(records) == 2
    assert all(record.engine_version == "5.8" for record in records)
    assert all(record.fetched_at is not None for record in records)
    assert all(record.content_sha256 for record in records)
    assert all((crawler_config.output_dir / record.file_path).is_file() for record in records)

    resumed_client = FakeHttpClient()
    resumed = DocumentationCrawler(
        crawler_config,
        http_client=resumed_client,
        robots_policy=FixedRobotsPolicy(True),
    ).crawl(["gameplay"])

    assert resumed.cached == 2
    assert resumed.downloaded == 0
    assert resumed_client.calls == []
    assert len(crawler.manifest_path.read_text(encoding="utf-8").splitlines()) == 2


def test_retryable_failure_is_retried_then_downloaded(
    crawler_config: DocumentationCrawlerConfig,
) -> None:
    client = FakeHttpClient()
    crawler = DocumentationCrawler(
        crawler_config,
        http_client=client,
        robots_policy=FixedRobotsPolicy(True),
    )
    seed = crawler.plan()[0].url
    client.add(
        seed,
        make_response(seed, b"busy", status=503),
        make_response(seed, b"<html>ready</html>"),
    )

    summary = crawler.crawl(max_depth=0)
    record = CrawlRecord.model_validate_json(
        crawler.manifest_path.read_text(encoding="utf-8").strip()
    )

    assert summary.downloaded == 1
    assert len(client.calls) == 2
    assert record.attempts == 2
    assert record.status is CrawlStatus.DOWNLOADED


def test_robots_disallow_is_recorded_without_fetching_page(
    crawler_config: DocumentationCrawlerConfig,
) -> None:
    client = FakeHttpClient()
    crawler = DocumentationCrawler(
        crawler_config,
        http_client=client,
        robots_policy=FixedRobotsPolicy(False),
    )

    summary = crawler.crawl(max_depth=0)
    record = CrawlRecord.model_validate_json(
        crawler.manifest_path.read_text(encoding="utf-8").strip()
    )

    assert summary.blocked == 1
    assert client.calls == []
    assert record.status is CrawlStatus.BLOCKED
    assert record.attempts == 0


def test_successful_robots_rules_are_cached_between_runs(
    crawler_config: DocumentationCrawlerConfig,
) -> None:
    first_client = FakeHttpClient()
    first_crawler = DocumentationCrawler(crawler_config, http_client=first_client)
    seed = first_crawler.plan()[0].url
    first_client.add(
        crawler_config.robots_url,
        HttpResponse(
            status_code=200,
            url=crawler_config.robots_url,
            content=b"User-agent: *\nAllow: /documentation/\n",
            headers={"content-type": "text/plain"},
        ),
    )
    first_client.add(seed, make_response(seed, b"<html>cached page</html>"))

    first = first_crawler.crawl(max_depth=0)
    assert first.downloaded == 1
    assert (crawler_config.output_dir / "robots.txt").is_file()

    resumed_client = FakeHttpClient()
    resumed = DocumentationCrawler(
        crawler_config, http_client=resumed_client
    ).crawl(max_depth=0)

    assert resumed.cached == 1
    assert resumed_client.calls == []


def test_unknown_topic_is_rejected_before_network_access(
    crawler_config: DocumentationCrawlerConfig,
) -> None:
    client = FakeHttpClient()
    crawler = DocumentationCrawler(crawler_config, http_client=client)

    with pytest.raises(ValueError, match="Unknown documentation topics"):
        crawler.crawl(["audio"])

    assert client.calls == []


def test_seed_outside_allowed_documentation_scope_is_rejected(
    crawler_config: DocumentationCrawlerConfig,
) -> None:
    crawler_config.topics["gameplay"].seeds = ["https://example.com/gameplay"]
    client = FakeHttpClient()
    crawler = DocumentationCrawler(crawler_config, http_client=client)

    with pytest.raises(ValueError, match="outside the configured documentation scope"):
        crawler.crawl(["gameplay"])

    assert client.calls == []
