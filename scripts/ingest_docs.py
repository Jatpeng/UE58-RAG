"""Download selected Unreal Engine documentation into the raw cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.crawler import DocumentationCrawler, load_docs_crawler_config


def build_parser() -> argparse.ArgumentParser:
    """Create the documentation crawler command-line interface."""

    parser = argparse.ArgumentParser(
        description="Download a bounded, resumable UE documentation corpus."
    )
    parser.add_argument(
        "--ue-config",
        type=Path,
        default=Path("config/ue58.yaml"),
        help="UE configuration containing the required engine version.",
    )
    parser.add_argument(
        "--crawler-config",
        type=Path,
        default=Path("config/docs_crawler.yaml"),
        help="Crawler topics, access rules, limits, and retry settings.",
    )
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        help="Topic to crawl; repeat this option to select several (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print seed URLs without accessing the network or writing files.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Override the configured page cap for this run.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Override the configured link-discovery depth for this run.",
    )
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="List configured topics and exit without network access.",
    )
    return parser


def main() -> int:
    """Run the configured crawler and report a compact summary."""

    args = build_parser().parse_args()
    try:
        config = load_docs_crawler_config(args.ue_config, args.crawler_config)
        if args.list_topics:
            for topic in config.topics:
                print(topic)
            return 0

        crawler = DocumentationCrawler(config)
        summary = crawler.crawl(
            args.topics,
            dry_run=args.dry_run,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
        )
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 2

    if args.dry_run:
        print("Planned seed URLs (link discovery requires a real crawl):")
        for url in summary.planned_urls:
            print(url)
        print(f"Seeds: {len(summary.planned_urls)}")
        return 0

    print(f"Output: {config.output_dir}")
    print(f"Processed: {summary.processed}")
    print(f"Downloaded: {summary.downloaded}")
    print(f"Cached: {summary.cached}")
    print(f"Blocked by robots.txt: {summary.blocked}")
    print(f"Failed: {summary.failed}")
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
