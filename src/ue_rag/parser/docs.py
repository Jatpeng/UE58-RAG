"""Parse cached Epic documentation HTML into unified UE documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import yaml
from bs4 import BeautifulSoup, NavigableString, Tag
from pydantic import BaseModel, ConfigDict

from ue_rag.crawler.docs import CrawlRecord, CrawlStatus
from ue_rag.jsonl import save_jsonl
from ue_rag.schema import SourceScope, SourceType, UEDocument


PARSER_VERSION = "1"


class DocumentationParserConfig(BaseModel):
    """Input and output locations for documentation parsing."""

    model_config = ConfigDict(extra="forbid")

    engine_version: str
    raw_docs_dir: Path
    output_path: Path


class DocumentationParseIssue(BaseModel):
    """Auditable reason that one downloaded page could not become a document."""

    model_config = ConfigDict(extra="forbid")

    url: str
    file_path: str | None
    engine_version: str
    error: str


class DocumentParseError(ValueError):
    """Raised when downloaded HTML has no usable documentation content."""


@dataclass(frozen=True)
class ParseSummary:
    """Counters produced by a complete manifest parse."""

    documents: int
    headings: int
    code_blocks: int
    skipped: int
    output_path: Path
    issues_path: Path


def load_docs_parser_config(
    ue_config_path: str | Path = "config/ue58.yaml",
) -> DocumentationParserConfig:
    """Resolve parser paths and the engine version from ue58.yaml."""

    config_path = Path(ue_config_path).resolve()
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    project_root = config_path.parent.parent
    raw_dir = Path(config["data"]["raw"])
    parsed_dir = Path(config["data"]["parsed"])
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir
    if not parsed_dir.is_absolute():
        parsed_dir = project_root / parsed_dir

    return DocumentationParserConfig(
        engine_version=str(config["engine"]["version"]),
        raw_docs_dir=raw_dir / "docs",
        output_path=parsed_dir / "docs" / "documents.jsonl",
    )


class DocumentationParser:
    """Convert downloaded Epic pages to clean Markdown-backed UEDocuments."""

    NOISE_SELECTORS = (
        "nav",
        "footer",
        "script",
        "style",
        "noscript",
        "svg",
        "table-of-contents",
        ".document-table-of-content-block",
        ".document-header",
        ".block-code-snippet-actions",
        ".pre-overlay",
        "textarea",
        "button",
        "[class*='cookie']",
        "[id*='cookie']",
    )

    def parse_html(self, html: str | bytes, record: CrawlRecord) -> UEDocument:
        """Parse one downloaded manifest record and its raw HTML."""

        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("main") or soup.body or soup
        page_header = main.select_one("header.section-page-header") if isinstance(main, Tag) else None
        title = self._extract_title(soup, main, page_header)
        description = self._extract_description(page_header)
        article = main.find("article") if isinstance(main, Tag) else None
        content_root = article or main
        if article is None and page_header is not None:
            page_header.decompose()

        for selector in self.NOISE_SELECTORS:
            for node in content_root.select(selector):
                node.decompose()

        renderer = _MarkdownRenderer(record.url)
        article_markdown = renderer.render_children(content_root)
        if not article_markdown:
            raise DocumentParseError("Documentation page has no article content")
        parts = [f"# {title}"]
        if description:
            parts.append(description)
        if article_markdown:
            parts.append(article_markdown)
        markdown = _clean_markdown("\n\n".join(parts))

        heading_paths = _extract_heading_paths(content_root, title)
        code_languages = renderer.code_languages
        document_id = _document_id(record.engine_version, record.url)
        return UEDocument(
            id=document_id,
            engine_version=record.engine_version,
            source_scope=SourceScope.GLOBAL,
            source_type=SourceType.DOCS,
            content=markdown,
            title=title,
            file_path=record.file_path,
            metadata={
                "url": record.url,
                "final_url": record.final_url,
                "topic": record.topic,
                "fetched_at": record.fetched_at.isoformat() if record.fetched_at else None,
                "raw_content_sha256": record.content_sha256,
                "heading_paths": heading_paths,
                "code_block_count": renderer.code_block_count,
                "code_languages": code_languages,
                "parser_version": PARSER_VERSION,
            },
        )

    def parse_manifest(
        self,
        config: DocumentationParserConfig,
        *,
        manifest_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> ParseSummary:
        """Parse every latest successful manifest record into deterministic JSONL."""

        manifest = Path(manifest_path) if manifest_path else config.raw_docs_dir / "manifest.jsonl"
        output = Path(output_path) if output_path else config.output_path
        issues_path = output.with_name("issues.jsonl")
        records = _load_latest_downloaded_records(manifest)
        documents: list[UEDocument] = []
        issues: list[DocumentationParseIssue] = []
        heading_count = 0
        code_block_count = 0

        for record in records:
            if record.engine_version != config.engine_version:
                raise ValueError(
                    f"Manifest engine version {record.engine_version!r} does not match "
                    f"configured version {config.engine_version!r}: {record.url}"
                )
            if not record.file_path:
                raise ValueError(f"Downloaded manifest record has no file path: {record.url}")
            raw_path = config.raw_docs_dir / record.file_path
            if not raw_path.is_file():
                raise FileNotFoundError(f"Cached HTML is missing: {raw_path}")
            try:
                document = self.parse_html(raw_path.read_bytes(), record)
            except DocumentParseError as error:
                issues.append(
                    DocumentationParseIssue(
                        url=record.url,
                        file_path=record.file_path,
                        engine_version=record.engine_version,
                        error=str(error),
                    )
                )
                continue
            documents.append(document)
            heading_count += len(document.metadata["heading_paths"])
            code_block_count += document.metadata["code_block_count"]

        save_jsonl(output, documents)
        save_jsonl(issues_path, issues)
        return ParseSummary(
            documents=len(documents),
            headings=heading_count,
            code_blocks=code_block_count,
            skipped=len(issues),
            output_path=output,
            issues_path=issues_path,
        )

    @staticmethod
    def _extract_title(
        soup: BeautifulSoup, main: Tag, page_header: Tag | None
    ) -> str:
        title_node = page_header.find("h1") if page_header else main.find("h1")
        if title_node:
            title = _plain_text(title_node)
            if title:
                return title
        if soup.title:
            fallback = _plain_text(soup.title).split("|", maxsplit=1)[0].strip()
            if fallback:
                return fallback
        raise DocumentParseError("Documentation page has no usable title")

    @staticmethod
    def _extract_description(page_header: Tag | None) -> str | None:
        if not page_header:
            return None
        description = page_header.find("p")
        text = _plain_text(description) if description else ""
        return text or None


class _MarkdownRenderer:
    """Small deterministic renderer for the semantic elements used by Epic docs."""

    def __init__(self, page_url: str) -> None:
        self.page_url = page_url
        self.code_block_count = 0
        self.code_languages: list[str] = []

    def render_children(self, node: Tag) -> str:
        return _clean_markdown("".join(self.render(child) for child in node.children))

    def render(self, node: object) -> str:
        if isinstance(node, NavigableString):
            return _normalize_literal_markdown_links(str(node))
        if not isinstance(node, Tag):
            return ""

        name = node.name.lower()
        if name in {"script", "style", "noscript", "svg", "button", "textarea"}:
            return ""
        if name in {f"h{level}" for level in range(1, 7)}:
            level = int(name[1])
            return f"\n\n{'#' * level} {self.render_inline(node)}\n\n"
        if name == "p":
            text = self.render_inline(node)
            return f"\n\n{text}\n\n" if text else ""
        if name == "pre":
            return self._render_code_block(node)
        if name == "block-code-snippet":
            pre = node.find("pre")
            return self._render_code_block(pre) if pre else ""
        if name == "code":
            return self._render_inline_code(node)
        if name in {"ul", "ol"}:
            return self._render_list(node)
        if name == "table":
            return self._render_table(node)
        if name == "blockquote":
            return self._render_quote(self.render_children(node))
        if name == "block-callout-md":
            label = (node.get("callout-type") or "note").replace("-", " ").title()
            body = self.render_children(node)
            return self._render_quote(f"**{label}:** {body}")
        if name == "block-callout":
            label = self._callout_label(node)
            content_node = node.select_one(".block-callout-content") or node
            body = self.render_children(content_node)
            return self._render_quote(f"**{label}:** {body}")
        if name == "block-dir-item-md":
            return self._render_directory_item(node)
        if name == "br":
            return "\n"
        if name == "hr":
            return "\n\n---\n\n"
        if name == "img":
            return self._render_image(node)
        return "".join(self.render(child) for child in node.children)

    def render_inline(self, node: Tag) -> str:
        return _normalize_inline("".join(self._render_inline_node(child) for child in node.children))

    def _render_inline_node(self, node: object) -> str:
        if isinstance(node, NavigableString):
            return _normalize_literal_markdown_links(str(node))
        if not isinstance(node, Tag):
            return ""
        name = node.name.lower()
        if name == "code":
            return self._render_inline_code(node)
        if name == "a":
            text = self.render_inline(node) or (node.get("href") or "")
            href = node.get("href")
            normalized_href = href.replace("\\", "/") if href else None
            rendered = (
                f"[{text}]({urljoin(self.page_url, normalized_href)})"
                if normalized_href
                else text
            )
            return self._preserve_boundary_spaces(node, rendered)
        if name in {"strong", "b"}:
            return self._preserve_boundary_spaces(node, f"**{self.render_inline(node)}**")
        if name in {"em", "i"}:
            return self._preserve_boundary_spaces(node, f"*{self.render_inline(node)}*")
        if name in {"s", "del"}:
            return self._preserve_boundary_spaces(node, f"~~{self.render_inline(node)}~~")
        if name == "br":
            return "\n"
        if name == "img":
            return self._render_image(node)
        if name in {"ul", "ol", "table", "pre"}:
            return self.render(node)
        return "".join(self._render_inline_node(child) for child in node.children)

    def _render_inline_code(self, node: Tag) -> str:
        content = node.get_text("", strip=False).strip()
        if not content:
            return ""
        fence = "``" if "`" in content else "`"
        return self._preserve_boundary_spaces(node, f"{fence}{content}{fence}")

    @staticmethod
    def _preserve_boundary_spaces(node: Tag, rendered: str) -> str:
        raw_text = node.get_text("", strip=False)
        prefix = " " if raw_text[:1].isspace() else ""
        suffix = " " if raw_text[-1:].isspace() else ""
        return f"{prefix}{rendered}{suffix}"

    def _render_code_block(self, node: Tag) -> str:
        code_node = node.find("code")
        content = (code_node or node).get_text("", strip=False).strip("\n")
        language = self._code_language(node)
        longest_fence = max((len(match) for match in re.findall(r"`+", content)), default=0)
        fence = "`" * max(3, longest_fence + 1)
        self.code_block_count += 1
        if language and language not in self.code_languages:
            self.code_languages.append(language)
        return f"\n\n{fence}{language}\n{content}\n{fence}\n\n"

    @staticmethod
    def _code_language(node: Tag) -> str:
        container = node.find_parent("block-code-snippet")
        if not container:
            return ""
        label = container.select_one(".block-code-snippet-header-type")
        if not label:
            return ""
        language = _plain_text(label).lower()
        aliases = {"c++": "cpp", "c#": "csharp", "command line": "shell"}
        return aliases.get(language, re.sub(r"[^a-z0-9_+-]", "", language))

    def _render_list(self, node: Tag) -> str:
        ordered = node.name.lower() == "ol"
        lines: list[str] = []
        for index, item in enumerate(node.find_all("li", recursive=False), start=1):
            prefix = f"{index}." if ordered else "-"
            inline_parts: list[str] = []
            nested_parts: list[str] = []
            for child in item.children:
                if isinstance(child, Tag) and child.name.lower() in {"ul", "ol"}:
                    nested_parts.append(self._render_list(child).strip())
                elif isinstance(child, Tag) and child.name.lower() == "p":
                    inline_parts.append(self.render_inline(child))
                else:
                    inline_parts.append(self._render_inline_node(child))
            text = _normalize_inline("".join(inline_parts))
            lines.append(f"{prefix} {text}".rstrip())
            for nested in nested_parts:
                lines.extend(f"  {line}" for line in nested.splitlines())
        joined_lines = "\n".join(lines)
        return f"\n\n{joined_lines}\n\n" if lines else ""

    def _render_table(self, node: Tag) -> str:
        rows: list[list[str]] = []
        for row in node.find_all("tr"):
            cells = row.find_all(["th", "td"], recursive=False)
            if not cells:
                continue
            rows.append([self.render_inline(cell).replace("|", "\\|") for cell in cells])
        if not rows:
            return ""
        width = max(len(row) for row in rows)
        normalized = [row + [""] * (width - len(row)) for row in rows]
        header = normalized[0]
        body = normalized[1:]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in range(width)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        joined_lines = "\n".join(lines)
        return f"\n\n{joined_lines}\n\n"

    def _render_directory_item(self, node: Tag) -> str:
        title = (node.get("page-name") or "Related documentation").strip()
        href = node.get("href")
        description = (node.get("description") or "").strip()
        normalized_href = href.replace("\\", "/") if href else None
        label = (
            f"[{title}]({urljoin(self.page_url, normalized_href)})"
            if normalized_href
            else title
        )
        suffix = f" — {description}" if description else ""
        return f"\n\n- {label}{suffix}\n\n"

    @staticmethod
    def _render_quote(content: str) -> str:
        clean = _clean_markdown(content)
        if not clean:
            return ""
        quoted = "\n".join(f"> {line}" if line else ">" for line in clean.splitlines())
        return f"\n\n{quoted}\n\n"

    @staticmethod
    def _callout_label(node: Tag) -> str:
        for class_name in node.get("class", []):
            if class_name.startswith("is-"):
                return class_name.removeprefix("is-").replace("-", " ").title()
        return "Note"

    @staticmethod
    def _render_image(node: Tag) -> str:
        src = node.get("src")
        if not src:
            return ""
        normalized_src = src.replace("\\", "/")
        return f"![{(node.get('alt') or '').strip()}]({normalized_src})"


def _load_latest_downloaded_records(manifest_path: Path) -> list[CrawlRecord]:
    latest: dict[str, CrawlRecord] = {}
    with manifest_path.open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                record = CrawlRecord.model_validate_json(line)
            except Exception as error:
                raise ValueError(
                    f"Invalid crawl manifest at line {line_number}: {manifest_path}"
                ) from error
            latest[record.url] = record
    return [record for record in latest.values() if record.status is CrawlStatus.DOWNLOADED]


def _extract_heading_paths(root: Tag, page_title: str) -> list[dict[str, object]]:
    stack: list[tuple[int, str]] = [(1, page_title)]
    paths: list[dict[str, object]] = [
        {"level": 1, "title": page_title, "path": [page_title]}
    ]
    for heading in root.find_all(re.compile(r"^h[1-6]$")):
        level = int(heading.name[1])
        title = _plain_text(heading)
        if not title:
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        paths.append(
            {"level": level, "title": title, "path": [item[1] for item in stack]}
        )
    return paths


def _document_id(engine_version: str, url: str) -> str:
    digest = hashlib.sha256(f"docs:{engine_version}:{url}".encode("utf-8")).hexdigest()
    return f"docs-{digest}"


def _plain_text(node: Tag | None) -> str:
    if not node:
        return ""
    return _normalize_inline(_normalize_literal_markdown_links(node.get_text(" ", strip=True)))


def _normalize_inline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_literal_markdown_links(text: str) -> str:
    return re.sub(
        r"(\]\()([^\n)]+)(\))",
        lambda match: f"{match.group(1)}{match.group(2).replace(chr(92), '/')}{match.group(3)}",
        text,
    )


def _clean_markdown(markdown: str) -> str:
    lines = [line.rstrip() for line in markdown.replace("\r\n", "\n").split("\n")]
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned
