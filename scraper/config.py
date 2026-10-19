"""
Configuration file for Bluesky public data collector
"""

import os

# Keywords to search for
KEYWORDS = [
    "AI",
    "ChatGPT",
    "OpenAI",
    "IPL",
    "iPhone"
]

# Collection settings
MAX_POSTS_PER_KEYWORD = 100
COLLECTION_LIMIT = 500
BLUESKY_SEARCH_URL = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
REQUEST_TIMEOUT_SECONDS = 30
LIVE_REFRESH_TIMEOUT_SECONDS = 8
LIVE_REFRESH_LIMIT_PER_KEYWORD = 25
HISTORY_TOPICS = ["AI", "ChatGPT", "OpenAI", "IPL", "iPhone"]
HISTORY_POSTS_PER_WEEK = 50

# Output paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
EXPORTS_DIR = "data/exports"
HISTORY_DATA_DIR = "data/history"
LOGS_DIR = "logs"

# HDFS paths
HDFS_NAMENODE = os.getenv("HDFS_NAMENODE", "http://localhost:9870")
HDFS_RAW_PATH = "/twitter/raw"
HDFS_PROCESSED_PATH = "/twitter/processed"

# File naming
OUTPUT_FILE_PREFIX = "tweets_"
OUTPUT_FILE_EXTENSION = ".jsonl"

# Logging
LOG_FILE = "logs/scraper.log"
LOG_LEVEL = "INFO"
