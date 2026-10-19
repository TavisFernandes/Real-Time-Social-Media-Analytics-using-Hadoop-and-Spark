"""Collect real public Bluesky posts and save them as JSONL."""

import json
import logging
import os
import re
import sys
import argparse
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.config import (
    KEYWORDS,
    MAX_POSTS_PER_KEYWORD,
    COLLECTION_LIMIT,
    HISTORY_DATA_DIR,
    HISTORY_POSTS_PER_WEEK,
    HISTORY_TOPICS,
    RAW_DATA_DIR,
    OUTPUT_FILE_PREFIX,
    OUTPUT_FILE_EXTENSION,
    LOG_FILE,
    LOG_LEVEL,
    BLUESKY_SEARCH_URL,
    REQUEST_TIMEOUT_SECONDS,
    LIVE_REFRESH_TIMEOUT_SECONDS,
    LIVE_REFRESH_LIMIT_PER_KEYWORD,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Setup logging
LOG_PATH = PROJECT_ROOT / LOG_FILE
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BlueskyCollector:
    """Collect public posts through Bluesky's unauthenticated search API."""
    
    def __init__(self):
        self.seen_post_ids = set()
        
    def extract_post_data(self, post, keyword):
        """Map a Bluesky API post to the pipeline's existing JSONL schema."""
        record = post.get("record", {})
        text = record.get("text", "")
        hashtags = set(re.findall(r"(?<!\w)#([\w]+)", text, flags=re.UNICODE))
        
        for facet in record.get("facets", []):
            for feature in facet.get("features", []):
                if feature.get("$type") == "app.bsky.richtext.facet#tag":
                    tag = feature.get("tag")
                    if tag:
                        hashtags.add(tag)

        return {
            "tweet_id": post.get("uri"),
            "created_at": record.get("createdAt") or post.get("indexedAt"),
            "username": post.get("author", {}).get("handle", "unknown"),
            "text": text,
            "likes": post.get("likeCount", 0),
            "retweets": post.get("repostCount", 0),
            "replies": post.get("replyCount", 0),
            "hashtags": sorted(hashtags),
            "keyword_used": keyword,
        }

    def collect_posts(self, keyword, limit=MAX_POSTS_PER_KEYWORD):
        """Fetch real public posts for a keyword, paging through API results."""
        logger.info("Collecting Bluesky posts for keyword: %s", keyword)
        posts_collected = []
        cursor = None

        try:
            while len(posts_collected) < limit:
                page_size = min(100, limit - len(posts_collected))
                params = {"q": keyword, "limit": page_size, "sort": "latest"}
                if cursor:
                    params["cursor"] = cursor

                payload = self.fetch_search_page(params)
                page = payload.get("posts", [])
                if not page:
                    break

                for post in page:
                    post_data = self.extract_post_data(post, keyword)
                    post_id = post_data["tweet_id"]
                    if not post_id or post_id in self.seen_post_ids:
                        continue

                    posts_collected.append(post_data)
                    self.seen_post_ids.add(post_id)
                    if len(posts_collected) >= limit:
                        break

                next_cursor = payload.get("cursor")
                if not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor

        except Exception as e:
            logger.error("Error collecting Bluesky posts for %s: %s", keyword, e)

        logger.info("Collected %s posts for %s", len(posts_collected), keyword)
        return posts_collected

    def collect_latest_posts(self, keywords=KEYWORDS, limit_per_keyword=LIVE_REFRESH_LIMIT_PER_KEYWORD):
        """Fetch every topic independently so one blocked search cannot abort a refresh."""
        collected = []
        failures = []
        seen_topic_posts = set()
        for keyword in keywords:
            try:
                payload = self.fetch_search_page(
                    {"q": keyword, "limit": min(100, limit_per_keyword), "sort": "latest"},
                    timeout=LIVE_REFRESH_TIMEOUT_SECONDS,
                    max_attempts=1,
                    bypass_cache=True,
                )
            except requests.RequestException as error:
                logger.warning("Live refresh failed for %s: %s", keyword, error)
                failures.append(f"{keyword}: {error}")
                continue

            for post in payload.get("posts", []):
                post_data = self.extract_post_data(post, keyword)
                key = (post_data["tweet_id"], keyword)
                if post_data["tweet_id"] and key not in seen_topic_posts:
                    seen_topic_posts.add(key)
                    collected.append(post_data)
        return collected, failures

    @staticmethod
    def save_latest_posts(posts):
        """Persist a live refresh so it remains available on restricted networks."""
        history_dir = PROJECT_ROOT / HISTORY_DATA_DIR
        history_dir.mkdir(parents=True, exist_ok=True)
        filename = f"bluesky_latest_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl"
        history_file = history_dir / filename
        with history_file.open("x", encoding="utf-8") as output:
            for post in posts:
                output.write(json.dumps(post, ensure_ascii=False) + "\n")
        return history_file

    @staticmethod
    def fetch_search_page(params, timeout=REQUEST_TIMEOUT_SECONDS, max_attempts=4, bypass_cache=False):
        """Fetch a Bluesky search page with backoff for temporary API blocks."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        retry_statuses = {403, 429, 500, 502, 503, 504}
        for attempt in range(max_attempts):
            try:
                request_options = {"params": params, "timeout": timeout}
                if bypass_cache:
                    request_options["headers"] = {
                        "Cache-Control": "no-cache",
                        "Pragma": "no-cache",
                    }
                response = requests.get(
                    BLUESKY_SEARCH_URL,
                    **request_options,
                )
                if response.status_code in retry_statuses and attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2 ** attempt)

    def collect_historical_posts(self, start_date, end_date, topics, limit_per_week):
        """Collect a bounded sample from each topic in weekly historical windows."""
        collected = []
        seen_by_topic = {topic: set() for topic in topics}

        for topic in topics:
            window_start = start_date
            while window_start < end_date:
                window_end = min(window_start + timedelta(days=7), end_date)
                params = {
                    "q": topic,
                    "limit": min(100, limit_per_week),
                    "sort": "latest",
                    "since": window_start.isoformat(),
                    "until": window_end.isoformat(),
                }
                try:
                    payload = self.fetch_search_page(params)
                    for post in payload.get("posts", []):
                        post_data = self.extract_post_data(post, topic)
                        created_at = post_data.get("created_at")
                        try:
                            post_date = datetime.fromisoformat(created_at.replace("Z", "+00:00")).date()
                        except (AttributeError, ValueError):
                            continue
                        if not window_start <= post_date < window_end:
                            continue

                        post_id = post_data["tweet_id"]
                        if post_id and post_id not in seen_by_topic[topic]:
                            seen_by_topic[topic].add(post_id)
                            collected.append(post_data)
                except requests.RequestException as error:
                    logger.warning("Skipping %s history window %s to %s: %s", topic, window_start, window_end, error)

                window_start = window_end

        return collected

    def run_history(self, years=3):
        """Backfill and persist a real Bluesky post sample for offline analysis."""
        end_date = date.today() + timedelta(days=1)
        try:
            start_date = end_date.replace(year=end_date.year - years)
        except ValueError:
            start_date = end_date.replace(year=end_date.year - years, day=28)

        logger.info("Backfilling %s years for topics: %s", years, ", ".join(HISTORY_TOPICS))
        posts = self.collect_historical_posts(
            start_date,
            end_date,
            HISTORY_TOPICS,
            HISTORY_POSTS_PER_WEEK,
        )
        history_dir = PROJECT_ROOT / HISTORY_DATA_DIR
        history_dir.mkdir(parents=True, exist_ok=True)
        filename = f"bluesky_history_{start_date:%Y%m%d}_{end_date - timedelta(days=1):%Y%m%d}.jsonl"
        history_file = history_dir / filename
        with history_file.open("w", encoding="utf-8") as output:
            for post in posts:
                output.write(json.dumps(post, ensure_ascii=False) + "\n")

        logger.info("Saved %s historical posts to %s", len(posts), history_file)
        return history_file
    
    def save_to_jsonl(self, posts, filename):
        """Save posts to JSONL file"""
        raw_dir = PROJECT_ROOT / RAW_DATA_DIR
        raw_dir.mkdir(parents=True, exist_ok=True)
        filepath = raw_dir / filename
        
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                for post in posts:
                    f.write(json.dumps(post, ensure_ascii=False) + '\n')
            logger.info("Saved %s posts to %s", len(posts), filepath)
            return filepath
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            return None
    
    def run(self):
        """Main collection loop"""
        logger.info("Starting Bluesky public-post collection")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_FILE_PREFIX}{timestamp}{OUTPUT_FILE_EXTENSION}"
        
        all_posts = []
        
        for keyword in KEYWORDS:
            remaining = COLLECTION_LIMIT - len(all_posts)
            if remaining <= 0:
                break
            posts = self.collect_posts(keyword, min(MAX_POSTS_PER_KEYWORD, remaining))
            all_posts.extend(posts)

        logger.info("Total unique posts collected: %s", len(all_posts))
        
        # Save to file
        if all_posts:
            self.save_to_jsonl(all_posts, filename)

        return filename


def main():
    """Run live collection or an explicit multi-year history backfill."""
    parser = argparse.ArgumentParser(description="Collect Bluesky posts for analytics.")
    parser.add_argument("--history-years", type=int, choices=(2, 3))
    args = parser.parse_args()
    collector = BlueskyCollector()
    if args.history_years:
        collector.run_history(args.history_years)
    else:
        collector.run()


if __name__ == "__main__":
    main()
