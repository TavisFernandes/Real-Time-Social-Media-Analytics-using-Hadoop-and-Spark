"""
Configuration file for Twitter/X data collector
"""

# Keywords to search for
KEYWORDS = [
    "AI",
    "ChatGPT",
    "OpenAI",
    "IPL",
    "iPhone"
]

# Collection settings
MAX_TWEETS_PER_KEYWORD = 100
COLLECTION_LIMIT = 500

# Output paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
EXPORTS_DIR = "data/exports"
LOGS_DIR = "logs"

# HDFS paths
HDFS_NAMENODE = "http://localhost:9870"
HDFS_RAW_PATH = "/twitter/raw"
HDFS_PROCESSED_PATH = "/twitter/processed"

# File naming
OUTPUT_FILE_PREFIX = "tweets_"
OUTPUT_FILE_EXTENSION = ".jsonl"

# Logging
LOG_FILE = "logs/scraper.log"
LOG_LEVEL = "INFO"
