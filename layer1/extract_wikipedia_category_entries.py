#!/usr/bin/env python3
"""Resolve Wikipedia article pages to their main category and export unique page members recursively."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
CATEGORY_NS = 14
PAGE_NS = 0
CHECKPOINT_VERSION = 1
DEFAULT_URLS = [
    "https://en.wikipedia.org/wiki/Academy_of_sciences",
    "https://en.wikipedia.org/wiki/National_academy",
    "https://en.wikipedia.org/wiki/Learned_society",
]
OUTPUT_FIELDS = [
    "member_title",
    "member_url",
    "pageid",
    "source_articles",
    "source_article_urls",
    "root_categories",
    "root_category_urls",
    "categories_found_in",
]


def page_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    title = parsed.path.removeprefix("/wiki/")
    return unquote(title).replace("_", " ")


def log(message: str, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


def checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + '.checkpoint.json')


def get_with_retry(session: requests.Session, url: str, *, params: dict | None = None, timeout: int = 30) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 429:
                wait_seconds = min(2 ** attempt, 30)
                print(f"[retry] 429 from {url}; sleeping {wait_seconds}s", flush=True)
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt == 4:
                break
            wait_seconds = min(2 ** attempt, 30)
            print(f"[retry] {type(exc).__name__} for {url}; sleeping {wait_seconds}s", flush=True)
            time.sleep(wait_seconds)
    assert last_error is not None
    raise last_error


def fetch_main_category(session: requests.Session, article_url: str) -> tuple[str, str]:
    response = get_with_retry(session, article_url, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")

    main_category_link = None
    for node in soup.find_all(["p", "div", "td"]):
        if "Main category:" not in node.get_text(" ", strip=True):
            continue
        candidate = node.select_one('a[title^="Category:"]')
        if candidate is not None:
            main_category_link = candidate
            break

    if main_category_link is None:
        raise ValueError(f"Could not resolve main category from {article_url}")

    category_title = main_category_link["title"]
    category_url = urljoin(article_url, main_category_link["href"])
    return category_title, category_url


def iter_category_members(session: requests.Session, category_title: str):
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category_title,
        "cmlimit": "max",
    }

    while True:
        response = get_with_retry(session, WIKIPEDIA_API, params=params, timeout=30)
        payload = response.json()

        for member in payload.get("query", {}).get("categorymembers", []):
            title = member["title"]
            yield {
                "pageid": str(member.get("pageid", "")),
                "ns": int(member.get("ns", -1)),
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            }

        continuation = payload.get("continue")
        if not continuation:
            break
        params.update(continuation)


def join_sorted(values: set[str]) -> str:
    return " | ".join(sorted(v for v in values if v))


def new_root_state(article_url: str) -> dict[str, object]:
    return {
        "article_url": article_url,
        "article_title": page_title_from_url(article_url),
        "root_category_title": None,
        "root_category_url": None,
        "queue": [],
        "seen_categories": [],
        "categories_processed": 0,
        "page_hits": 0,
        "completed": False,
    }


def serialize_deduped(deduped: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    serialized: dict[str, dict[str, object]] = {}
    for key, row in deduped.items():
        serialized[key] = {
            "member_title": row["member_title"],
            "member_url": row["member_url"],
            "pageid": row["pageid"],
            "source_articles": sorted(row["source_articles"]),
            "source_article_urls": sorted(row["source_article_urls"]),
            "root_categories": sorted(row["root_categories"]),
            "root_category_urls": sorted(row["root_category_urls"]),
            "categories_found_in": sorted(row["categories_found_in"]),
        }
    return serialized


def deserialize_deduped(payload: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    deduped: dict[str, dict[str, object]] = {}
    for key, row in payload.items():
        deduped[key] = {
            "member_title": row["member_title"],
            "member_url": row["member_url"],
            "pageid": row["pageid"],
            "source_articles": set(row["source_articles"]),
            "source_article_urls": set(row["source_article_urls"]),
            "root_categories": set(row["root_categories"]),
            "root_category_urls": set(row["root_category_urls"]),
            "categories_found_in": set(row["categories_found_in"]),
        }
    return deduped


def save_checkpoint(
    checkpoint_path: Path,
    *,
    urls: list[str],
    max_depth: int,
    roots: list[dict[str, object]],
    deduped: dict[str, dict[str, object]],
) -> None:
    payload = {
        "version": CHECKPOINT_VERSION,
        "urls": urls,
        "max_depth": max_depth,
        "roots": roots,
        "deduped": serialize_deduped(deduped),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def load_checkpoint(checkpoint_path: Path, *, urls: list[str], max_depth: int):
    if not checkpoint_path.exists():
        return None
    payload = json.loads(checkpoint_path.read_text(encoding='utf-8'))
    if payload.get('version') != CHECKPOINT_VERSION:
        return None
    if payload.get('urls') != urls or payload.get('max_depth') != max_depth:
        return None
    return payload


def write_output(output_path: Path, deduped: dict[str, dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in sorted(deduped.values(), key=lambda item: str(item['member_title']).lower()):
            writer.writerow(
                {
                    'member_title': row['member_title'],
                    'member_url': row['member_url'],
                    'pageid': row['pageid'],
                    'source_articles': join_sorted(row['source_articles']),
                    'source_article_urls': join_sorted(row['source_article_urls']),
                    'root_categories': join_sorted(row['root_categories']),
                    'root_category_urls': join_sorted(row['root_category_urls']),
                    'categories_found_in': join_sorted(row['categories_found_in']),
                }
            )


def export_rows(urls: list[str], output_path: Path, *, max_depth: int, progress: bool, progress_every: int) -> int:
    session = requests.Session()
    session.headers.update({"User-Agent": "codex-wikipedia-category-extractor/1.0"})
    checkpoint_path = checkpoint_path_for(output_path)

    checkpoint = load_checkpoint(checkpoint_path, urls=urls, max_depth=max_depth)
    if checkpoint is None:
        roots = [new_root_state(url) for url in urls]
        deduped: dict[str, dict[str, object]] = {}
        save_checkpoint(checkpoint_path, urls=urls, max_depth=max_depth, roots=roots, deduped=deduped)
        log(f"[resume] starting new crawl; checkpoint={checkpoint_path}", progress)
    else:
        roots = checkpoint['roots']
        deduped = deserialize_deduped(checkpoint['deduped'])
        log(f"[resume] loaded checkpoint={checkpoint_path} unique_pages={len(deduped)}", progress)

    for root in roots:
        if root['completed']:
            continue

        article_url = str(root['article_url'])
        article_title = str(root['article_title'])
        seen_categories = set(root['seen_categories'])
        queue = deque((str(item[0]), int(item[1])) for item in root['queue'])
        categories_processed = int(root['categories_processed'])
        page_hits = int(root['page_hits'])

        if root['root_category_title'] is None:
            root_category_title, root_category_url = fetch_main_category(session, article_url)
            root['root_category_title'] = root_category_title
            root['root_category_url'] = root_category_url
            if not queue:
                queue.append((root_category_title, 0))
            root['queue'] = [[name, depth] for name, depth in queue]
            save_checkpoint(checkpoint_path, urls=urls, max_depth=max_depth, roots=roots, deduped=deduped)
        else:
            root_category_title = str(root['root_category_title'])
            root_category_url = str(root['root_category_url'])
            if not queue:
                queue.append((root_category_title, 0))

        log(
            f"[root] article={article_title} root_category={root_category_title} max_depth={max_depth}",
            progress,
        )

        while queue:
            category_title, depth = queue[0]
            if category_title in seen_categories:
                queue.popleft()
                root['queue'] = [[name, depth_value] for name, depth_value in queue]
                save_checkpoint(checkpoint_path, urls=urls, max_depth=max_depth, roots=roots, deduped=deduped)
                continue

            log(
                f"[crawl] category={category_title} depth={depth} processed={categories_processed} pages={page_hits}",
                progress,
            )

            try:
                for member in iter_category_members(session, category_title):
                    if member['ns'] == PAGE_NS:
                        page_hits += 1
                        if progress and progress_every > 0 and page_hits % progress_every == 0:
                            log(
                                f"[crawl] category={category_title} depth={depth} pages={page_hits} seen_categories={len(seen_categories)} unique_pages={len(deduped)}",
                                True,
                            )
                        key = member['pageid'] or member['title']
                        row = deduped.setdefault(
                            key,
                            {
                                'member_title': member['title'],
                                'member_url': member['url'],
                                'pageid': member['pageid'],
                                'source_articles': set(),
                                'source_article_urls': set(),
                                'root_categories': set(),
                                'root_category_urls': set(),
                                'categories_found_in': set(),
                            },
                        )
                        row['source_articles'].add(article_title)
                        row['source_article_urls'].add(article_url)
                        row['root_categories'].add(root_category_title)
                        row['root_category_urls'].add(root_category_url)
                        row['categories_found_in'].add(category_title)
                    elif member['ns'] == CATEGORY_NS and (max_depth is None or depth < max_depth):
                        if member['title'] not in seen_categories and member['title'] not in {name for name, _ in queue}:
                            queue.append((member['title'], depth + 1))
            except requests.exceptions.RequestException as exc:
                root['queue'] = [[name, depth_value] for name, depth_value in queue]
                root['seen_categories'] = sorted(seen_categories)
                root['categories_processed'] = categories_processed
                root['page_hits'] = page_hits
                save_checkpoint(checkpoint_path, urls=urls, max_depth=max_depth, roots=roots, deduped=deduped)
                raise SystemExit(
                    f"Network error after checkpointing progress to {checkpoint_path}: {type(exc).__name__}: {exc}"
                ) from exc

            queue.popleft()
            seen_categories.add(category_title)
            categories_processed += 1
            root['queue'] = [[name, depth_value] for name, depth_value in queue]
            root['seen_categories'] = sorted(seen_categories)
            root['categories_processed'] = categories_processed
            root['page_hits'] = page_hits
            save_checkpoint(checkpoint_path, urls=urls, max_depth=max_depth, roots=roots, deduped=deduped)

        root['completed'] = True
        root['queue'] = []
        root['seen_categories'] = sorted(seen_categories)
        root['categories_processed'] = categories_processed
        root['page_hits'] = page_hits
        save_checkpoint(checkpoint_path, urls=urls, max_depth=max_depth, roots=roots, deduped=deduped)
        log(f"[root] article={article_title} unique_pages_so_far={len(deduped)}", progress)

    write_output(output_path, deduped)
    checkpoint_path.unlink(missing_ok=True)
    return len(deduped)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export unique page members from the main Wikipedia categories and all subcategories."
    )
    parser.add_argument(
        "urls",
        nargs="*",
        default=DEFAULT_URLS,
        help="Wikipedia article URLs to resolve to their main category",
    )
    parser.add_argument(
        "--output",
        help="CSV output path",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum subcategory depth to traverse. 0 means root only, 1 includes direct subcategories.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logging.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Log an extra progress message after every N page hits during crawling.",
    )
    args = parser.parse_args()

    if args.max_depth < 0:
        raise SystemExit("--max-depth must be 0 or greater")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be 1 or greater")

    output_path = Path(args.output).expanduser().resolve()
    count = export_rows(
        args.urls,
        output_path,
        max_depth=args.max_depth,
        progress=not args.quiet,
        progress_every=args.progress_every,
    )
    print(f"Exported {count} unique rows to {output_path}")


if __name__ == "__main__":
    main()
