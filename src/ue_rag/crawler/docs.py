"""Polite, resumable downloader for selected Unreal Engine documentation."""

from __future__ import annotations

import hashlib
import re
import time
import urllib.robotparser
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TopicConfig(BaseModel):
    """Seeds and path patterns that define one bounded crawl topic."""

    model_config = ConfigDict(extra="forbid")

    seeds: list[str] = Field(min_length=1)
    include_patterns: list[str] = Field(default_factory=list)


class DocumentationCrawlerConfig(BaseModel):
    """Validated crawler settings assembled from crawler and UE config files."""

    model_config = ConfigDict(extra="forbid")

    engine_version: str = Field(min_length=1)
    output_dir: Path
    base_url: str
    locale: str = "en-us"
    robots_url: str
    user_agent: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0)
    delay_seconds: float = Field(ge=0)
    max_retries: int = Field(ge=0)
    retry_backoff_seconds: float = Field(ge=0)
    robots_cache_ttl_seconds: float = Field(ge=0)
    max_pages: int = Field(gt=0)
    max_depth: int = Field(ge=0)
    allowed_hosts: set[str] = Field(min_length=1)
    allowed_path_prefixes: list[str] = Field(min_length=1)
    topics: dict[str, TopicConfig] = Field(min_length=1)


def load_docs_crawler_config(
    ue_config_path: str | Path = "config/ue58.yaml",
    crawler_config_path: str | Path = "config/docs_crawler.yaml",
) -> DocumentationCrawlerConfig:
    """Load crawler settings while sourcing the UE version only from ue58.yaml."""

    ue_path = Path(ue_config_path).resolve()
    crawler_path = Path(crawler_config_path).resolve()
    with ue_path.open(encoding="utf-8") as config_file:
        ue_config = yaml.safe_load(config_file)
    with crawler_path.open(encoding="utf-8") as config_file:
        crawler_config = yaml.safe_load(config_file)

    project_root = ue_path.parent.parent
    raw_dir = Path(ue_config["data"]["raw"])
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir

    request_config = crawler_config.pop("request")
    limit_config = crawler_config.pop("limits")
    return DocumentationCrawlerConfig(
        engine_version=str(ue_config["engine"]["version"]),
        output_dir=raw_dir / "docs",
        timeout_seconds=request_config["timeout_seconds"],
        delay_seconds=request_config["delay_seconds"],
        max_retries=request_config["max_retries"],
        retry_backoff_seconds=request_config["retry_backoff_seconds"],
        robots_cache_ttl_seconds=request_config["robots_cache_ttl_seconds"],
        max_pages=limit_config["max_pages"],
        max_depth=limit_config["max_depth"],
        **crawler_config,
    )


@dataclass(frozen=True)
class HttpResponse:
    """Minimal HTTP response used by the crawler and its tests."""

    status_code: int
    url: str
    content: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


class HttpClient(Protocol):
    """Injectable HTTP transport."""

    def get(self, url: str, *, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        """Fetch one URL."""


class UrllibHttpClient:
    """Standard-library HTTP transport with redirect support."""

    def get(self, url: str, *, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        request = Request(url, headers=dict(headers), method="GET")
        try:
            with urlopen(request, timeout=timeout) as response:
                return HttpResponse(
                    status_code=response.status,
                    url=response.geturl(),
                    content=response.read(),
                    headers={key.lower(): value for key, value in response.headers.items()},
                )
        except HTTPError as error:
            return HttpResponse(
                status_code=error.code,
                url=error.geturl(),
                content=error.read(),
                headers={key.lower(): value for key, value in error.headers.items()},
            )


class RobotsPolicy(Protocol):
    """Policy used to authorize a URL before downloading it."""

    def can_fetch(self, user_agent: str, url: str) -> bool:
        """Return whether the user agent may fetch the URL."""


class RobotsRules:
    """robots.txt rules backed by Python's standard parser."""

    def __init__(self, robots_url: str, text: str) -> None:
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(text.splitlines())
        self._parser = parser

    def can_fetch(self, user_agent: str, url: str) -> bool:
        return self._parser.can_fetch(user_agent, url)


class CrawlStatus(str, Enum):
    """Persisted state for a crawl attempt."""

    DOWNLOADED = "downloaded"
    FAILED = "failed"
    BLOCKED = "blocked_by_robots"


class CrawlRecord(BaseModel):
    """One append-only manifest event for a documentation URL."""

    model_config = ConfigDict(extra="forbid")

    url: str
    final_url: str | None = None
    topic: str
    engine_version: str
    status: CrawlStatus
    recorded_at: datetime
    fetched_at: datetime | None = None
    file_path: str | None = None
    content_sha256: str | None = None
    size_bytes: int | None = None
    content_type: str | None = None
    http_status: int | None = None
    attempts: int = Field(ge=0)
    error: str | None = None


@dataclass(frozen=True)
class CrawlTask:
    url: str
    topic: str
    depth: int


@dataclass
class CrawlSummary:
    """Counters returned to the CLI and tests."""

    planned_urls: list[str] = field(default_factory=list)
    downloaded: int = 0
    cached: int = 0
    failed: int = 0
    blocked: int = 0
    processed: int = 0


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.links.append(value)
                break


class DocumentationCrawler:
    """Download a bounded subset of Epic documentation into a local raw cache."""

    RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        config: DocumentationCrawlerConfig,
        *,
        http_client: HttpClient | None = None,
        robots_policy: RobotsPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.http_client = http_client or UrllibHttpClient()
        self.robots_policy = robots_policy
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_at: float | None = None

    @property
    def manifest_path(self) -> Path:
        return self.config.output_dir / "manifest.jsonl"

    def plan(self, topics: Sequence[str] | None = None) -> list[CrawlTask]:
        """Return deduplicated seed tasks without performing network access."""

        selected_topics = self._select_topics(topics)
        tasks: list[CrawlTask] = []
        seen: set[str] = set()
        for topic_name in selected_topics:
            for seed in self.config.topics[topic_name].seeds:
                url = self.normalize_url(seed)
                if not self._is_allowed_url(url):
                    raise ValueError(
                        f"Seed URL is outside the configured documentation scope: {url}"
                    )
                if url not in seen:
                    tasks.append(CrawlTask(url=url, topic=topic_name, depth=0))
                    seen.add(url)
        return tasks

    def crawl(
        self,
        topics: Sequence[str] | None = None,
        *,
        dry_run: bool = False,
        max_pages: int | None = None,
        max_depth: int | None = None,
    ) -> CrawlSummary:
        """Run a resumable crawl or return its seed plan when ``dry_run`` is set."""

        initial_tasks = self.plan(topics)
        summary = CrawlSummary(planned_urls=[task.url for task in initial_tasks])
        if dry_run:
            return summary

        page_limit = self.config.max_pages if max_pages is None else max_pages
        depth_limit = self.config.max_depth if max_depth is None else max_depth
        if page_limit <= 0:
            raise ValueError("max_pages must be greater than zero")
        if depth_limit < 0:
            raise ValueError("max_depth cannot be negative")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        latest_records = self._load_latest_manifest_records()
        robots = self.robots_policy or self._load_robots_rules()
        queue = deque(initial_tasks)
        queued = {task.url for task in initial_tasks}
        processed_urls: set[str] = set()

        while queue and summary.processed < page_limit:
            task = queue.popleft()
            if task.url in processed_urls:
                continue
            processed_urls.add(task.url)
            summary.processed += 1

            cached_path = self._cached_path(task.url, latest_records)
            if cached_path is not None:
                summary.cached += 1
                if task.depth < depth_limit:
                    self._enqueue_links(cached_path.read_bytes(), task, queue, queued)
                continue

            if not robots.can_fetch(self.config.user_agent, task.url):
                summary.blocked += 1
                self._append_manifest(
                    CrawlRecord(
                        url=task.url,
                        topic=task.topic,
                        engine_version=self.config.engine_version,
                        status=CrawlStatus.BLOCKED,
                        recorded_at=self._now(),
                        attempts=0,
                        error="Disallowed by robots.txt",
                    )
                )
                continue

            response, attempts, request_error = self._request_with_retries(task.url)
            if response is None or response.status_code != 200:
                summary.failed += 1
                self._append_manifest(
                    CrawlRecord(
                        url=task.url,
                        final_url=response.url if response else None,
                        topic=task.topic,
                        engine_version=self.config.engine_version,
                        status=CrawlStatus.FAILED,
                        recorded_at=self._now(),
                        http_status=response.status_code if response else None,
                        attempts=attempts,
                        error=request_error
                        or f"Unexpected HTTP status {response.status_code}",
                    )
                )
                continue

            content_type = response.headers.get("content-type", "")
            if content_type and "text/html" not in content_type.lower():
                summary.failed += 1
                self._append_manifest(
                    CrawlRecord(
                        url=task.url,
                        final_url=response.url,
                        topic=task.topic,
                        engine_version=self.config.engine_version,
                        status=CrawlStatus.FAILED,
                        recorded_at=self._now(),
                        http_status=response.status_code,
                        content_type=content_type,
                        attempts=attempts,
                        error="Response was not HTML",
                    )
                )
                continue

            relative_path = self._relative_page_path(task.url)
            output_path = self.config.output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(response.content)
            fetched_at = self._now()
            record = CrawlRecord(
                url=task.url,
                final_url=response.url,
                topic=task.topic,
                engine_version=self.config.engine_version,
                status=CrawlStatus.DOWNLOADED,
                recorded_at=fetched_at,
                fetched_at=fetched_at,
                file_path=relative_path.as_posix(),
                content_sha256=hashlib.sha256(response.content).hexdigest(),
                size_bytes=len(response.content),
                content_type=content_type or None,
                http_status=response.status_code,
                attempts=attempts,
            )
            self._append_manifest(record)
            latest_records[task.url] = record
            summary.downloaded += 1

            if task.depth < depth_limit:
                self._enqueue_links(response.content, task, queue, queued)

        return summary

    def normalize_url(self, href: str) -> str:
        """Canonicalize locale and version so equivalent links deduplicate."""

        base = self.config.base_url.rstrip("/") + "/"
        absolute = urljoin(base, href.replace("\\", "/"))
        parsed = urlsplit(absolute)
        path = re.sub(r"/{2,}", "/", parsed.path)
        segments = [segment for segment in path.split("/") if segment]
        if len(segments) >= 2 and segments[:2] == ["documentation", "unreal-engine"]:
            segments.insert(1, self.config.locale.lower())
        elif (
            len(segments) >= 3
            and segments[0] == "documentation"
            and segments[2] == "unreal-engine"
        ):
            segments[1] = self.config.locale.lower()
        path = "/" + "/".join(segments)
        if path != "/":
            path = path.rstrip("/")
        query = urlencode({"application_version": self.config.engine_version})
        return urlunsplit(("https", parsed.netloc.lower(), path, query, ""))

    def _select_topics(self, topics: Sequence[str] | None) -> list[str]:
        selected = list(self.config.topics) if not topics else list(dict.fromkeys(topics))
        unknown = sorted(set(selected) - set(self.config.topics))
        if unknown:
            raise ValueError(f"Unknown documentation topics: {', '.join(unknown)}")
        return selected

    def _load_robots_rules(self) -> RobotsRules:
        cache_path = self.config.output_dir / "robots.txt"
        if cache_path.is_file():
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age <= self.config.robots_cache_ttl_seconds:
                return RobotsRules(
                    self.config.robots_url,
                    cache_path.read_text(encoding="utf-8", errors="replace"),
                )

        response, _, error = self._request_with_retries(self.config.robots_url)
        if response is None or response.status_code != 200:
            # A stale robots cache is safer than silently ignoring robots.txt.
            # Some CDNs intermittently reject a standalone refresh even though
            # documentation pages remain available, so retain the last known
            # policy and try refreshing it again on a later run.
            if cache_path.is_file():
                return RobotsRules(
                    self.config.robots_url,
                    cache_path.read_text(encoding="utf-8", errors="replace"),
                )
            detail = error or (f"HTTP {response.status_code}" if response else "no response")
            raise RuntimeError(f"Unable to load robots.txt; crawl stopped: {detail}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(response.content)
        return RobotsRules(
            self.config.robots_url,
            response.content.decode("utf-8", errors="replace"),
        )

    def _request_with_retries(
        self, url: str
    ) -> tuple[HttpResponse | None, int, str | None]:
        attempts = self.config.max_retries + 1
        last_response: HttpResponse | None = None
        last_error: str | None = None
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            "Accept-Language": "en-US,en;q=0.8",
            "Accept-Encoding": "identity",
        }
        for attempt in range(1, attempts + 1):
            self._wait_for_rate_limit()
            try:
                response = self.http_client.get(
                    url, headers=headers, timeout=self.config.timeout_seconds
                )
                last_response = response
                last_error = None
            except Exception as error:  # Network transports expose several exception types.
                response = None
                last_error = f"{type(error).__name__}: {error}"

            should_retry = response is None or response.status_code in self.RETRYABLE_STATUSES
            if not should_retry or attempt == attempts:
                return response, attempt, last_error

            retry_delay = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
            if response is not None:
                retry_after = response.headers.get("retry-after")
                if retry_after:
                    try:
                        retry_delay = max(retry_delay, min(float(retry_after), 60.0))
                    except ValueError:
                        pass
            if retry_delay:
                self._sleep(retry_delay)

        return last_response, attempts, last_error

    def _wait_for_rate_limit(self) -> None:
        now = self._monotonic()
        if self._last_request_at is not None:
            remaining = self.config.delay_seconds - (now - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = self._monotonic()

    def _enqueue_links(
        self,
        content: bytes,
        task: CrawlTask,
        queue: deque[CrawlTask],
        queued: set[str],
    ) -> None:
        extractor = _LinkExtractor()
        extractor.feed(content.decode("utf-8", errors="replace"))
        topic_config = self.config.topics[task.topic]
        patterns = [re.compile(pattern, re.IGNORECASE) for pattern in topic_config.include_patterns]
        for href in extractor.links:
            if href.startswith(("#", "mailto:", "javascript:", "tel:")):
                continue
            url = self.normalize_url(urljoin(task.url, href))
            if url in queued or not self._is_allowed_url(url):
                continue
            path = urlsplit(url).path
            if patterns and not any(pattern.search(path) for pattern in patterns):
                continue
            queue.append(CrawlTask(url=url, topic=task.topic, depth=task.depth + 1))
            queued.add(url)

    def _is_allowed_url(self, url: str) -> bool:
        parsed = urlsplit(url)
        return (
            parsed.scheme == "https"
            and (parsed.hostname or "").lower()
            in {host.lower() for host in self.config.allowed_hosts}
            and any(parsed.path.startswith(prefix) for prefix in self.config.allowed_path_prefixes)
        )

    def _relative_page_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return Path("pages") / digest[:2] / f"{digest}.html"

    def _load_latest_manifest_records(self) -> dict[str, CrawlRecord]:
        if not self.manifest_path.exists():
            return {}
        latest: dict[str, CrawlRecord] = {}
        with self.manifest_path.open(encoding="utf-8") as manifest:
            for line_number, line in enumerate(manifest, start=1):
                if not line.strip():
                    continue
                try:
                    record = CrawlRecord.model_validate_json(line)
                except Exception as error:
                    raise ValueError(
                        f"Invalid crawl manifest at line {line_number}: {self.manifest_path}"
                    ) from error
                latest[record.url] = record
        return latest

    def _cached_path(
        self, url: str, latest_records: Mapping[str, CrawlRecord]
    ) -> Path | None:
        record = latest_records.get(url)
        if not record or record.status is not CrawlStatus.DOWNLOADED or not record.file_path:
            return None
        candidate = self.config.output_dir / record.file_path
        return candidate if candidate.is_file() else None

    def _append_manifest(self, record: CrawlRecord) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8", newline="\n") as manifest:
            manifest.write(record.model_dump_json())
            manifest.write("\n")

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
