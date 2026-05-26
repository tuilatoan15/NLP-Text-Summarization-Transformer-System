#!/usr/bin/env python3
"""Vietnamese article crawler and dataset generator for research benchmarking."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from newspaper import Article, ArticleException
except ImportError:
    Article = None
    ArticleException = Exception

from src.preprocess import clean_text, tokenize_words
from src.utils import logger

# Predefined RSS feeds for dynamic article discovery
RSS_FEEDS = {
    "vnexpress_tin_noi_bat": "https://vnexpress.net/rss/tin-noi-bat.rss",
    "vnexpress_the_gioi": "https://vnexpress.net/rss/the-gioi.rss",
    "vnexpress_thoi_su": "https://vnexpress.net/rss/thoi-su.rss",
    "vnexpress_khoa_hoc": "https://vnexpress.net/rss/khoa-hoc.rss",
    "vnexpress_so_hoa": "https://vnexpress.net/rss/so-hoa.rss"
}


def fetch_urls_from_rss(feed_url: str, limit: int = 10) -> list[str]:
    """Discover article URLs from a given RSS feed."""
    logger.info(f"Fetching RSS feed: {feed_url}")
    urls = []
    try:
        req = urllib.request.Request(
            feed_url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        for item in root.findall(".//item")[:limit]:
            link = item.find("link")
            if link is not None and link.text:
                urls.append(link.text.strip())
    except Exception as exc:
        logger.error(f"Failed to fetch RSS feed {feed_url}: {exc}")
    return urls


def crawl_article(url: str, timeout: int = 10) -> Optional[dict]:
    """Crawl a single URL and return raw/cleaned text."""
    if Article is None:
        logger.error("newspaper3k is not installed. Please install it to use the crawler.")
        return None

    try:
        logger.info(f"Crawling: {url}")
        article = Article(url, language="vi", request_timeout=timeout)
        article.download()
        article.parse()
        
        raw_text = article.text.strip()
        title = article.title.strip()
        
        if not raw_text:
            logger.warning(f"No text extracted from: {url}")
            return None

        # Clean text using our advanced modular preprocessor
        cleaned_text = clean_text(raw_text, aggressive=True)
        cleaned_title = clean_text(title, aggressive=True)
        
        word_count = len(cleaned_text.split())
        if word_count < 40:
            logger.warning(f"Skipping article {url} (too short: {word_count} words)")
            return None

        logger.info(f"✅ Crawl success: '{title[:50]}...' ({word_count} words)")
        return {
            "url": url,
            "title": cleaned_title,
            "raw_text": raw_text,
            "article": cleaned_text,
            "word_count": word_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

    except Exception as exc:
        logger.error(f"Failed to crawl {url}: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Vietnamese article crawler for research benchmarking.")
    parser.add_argument("--urls", nargs="+", help="Explicit URLs to crawl.")
    parser.add_argument("--feed", default="vnexpress_tin_noi_bat", choices=list(RSS_FEEDS.keys()) + ["all"], help="RSS feed name to auto-discover URLs.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum URLs to discover.")
    parser.add_argument("--output", default="data/raw/crawled_articles.jsonl", help="Output file path (.jsonl).")
    parser.add_argument("--delay", type=float, default=1.0, help="Polite delay between requests (seconds).")
    args = parser.parse_args()

    # Create target directory
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    urls_to_crawl = []
    if args.urls:
        urls_to_crawl = args.urls
    else:
        # Discover via RSS feeds
        if args.feed == "all":
            for feed_name, feed_url in RSS_FEEDS.items():
                urls_to_crawl.extend(fetch_urls_from_rss(feed_url, limit=args.limit // len(RSS_FEEDS) + 1))
        else:
            urls_to_crawl = fetch_urls_from_rss(RSS_FEEDS[args.feed], limit=args.limit)

    # Deduplicate URLs
    urls_to_crawl = list(dict.fromkeys(urls_to_crawl))[:args.limit]
    
    if not urls_to_crawl:
        logger.error("No URLs found to crawl. Exiting.")
        sys.exit(1)

    logger.info(f"Starting sequential crawl of {len(urls_to_crawl)} URLs...")
    success_count = 0

    with output_file.open("w", encoding="utf-8") as writer:
        for idx, url in enumerate(urls_to_crawl, start=1):
            logger.info(f"[{idx}/{len(urls_to_crawl)}] Processing...")
            result = crawl_article(url)
            if result:
                writer.write(json.dumps(result, ensure_ascii=False) + "\n")
                success_count += 1
            
            if idx < len(urls_to_crawl):
                time.sleep(args.delay)

    logger.info(f"🏁 Crawl complete! Saved {success_count} articles to {args.output}")


if __name__ == "__main__":
    main()
