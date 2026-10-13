"""
Twitter/X Data Collector using twscrape
Collects tweets based on configured keywords and saves to JSONL format
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import twscrape

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    KEYWORDS,
    MAX_TWEETS_PER_KEYWORD,
    COLLECTION_LIMIT,
    RAW_DATA_DIR,
    OUTPUT_FILE_PREFIX,
    OUTPUT_FILE_EXTENSION,
    LOG_FILE,
    LOG_LEVEL
)

# Setup logging
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TwitterCollector:
    """Collects tweets using twscrape"""
    
    def __init__(self):
        self.api = twscrape.API()
        self.seen_tweet_ids = set()
        
    async def collect_tweets(self, keyword, limit=MAX_TWEETS_PER_KEYWORD):
        """Collect tweets for a specific keyword"""
        logger.info(f"Collecting tweets for keyword: {keyword}")
        tweets_collected = []
        
        try:
            async for tweet in self.api.search(f"{keyword} -filter:retweets", limit=limit):
                if tweet.id in self.seen_tweet_ids:
                    continue
                    
                tweet_data = self.extract_tweet_data(tweet, keyword)
                if tweet_data:
                    tweets_collected.append(tweet_data)
                    self.seen_tweet_ids.add(tweet.id)
                    
                    if len(tweets_collected) >= limit:
                        break
                        
        except Exception as e:
            logger.error(f"Error collecting tweets for {keyword}: {e}")
            
        logger.info(f"Collected {len(tweets_collected)} tweets for {keyword}")
        return tweets_collected
    
    def save_to_jsonl(self, tweets, filename):
        """Save tweets to JSONL file"""
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DATA_DIR, filename)
        
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                for tweet in tweets:
                    f.write(json.dumps(tweet, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(tweets)} tweets to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            return None
    
    async def run(self):
        """Main collection loop"""
        logger.info("Starting Twitter data collection")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_FILE_PREFIX}{timestamp}{OUTPUT_FILE_EXTENSION}"
        
        all_tweets = []
        
        for keyword in KEYWORDS:
            tweets = await self.collect_tweets(keyword)
            all_tweets.extend(tweets)
            
            if len(all_tweets) >= COLLECTION_LIMIT:
                logger.info(f"Reached collection limit of {COLLECTION_LIMIT}")
                break
        
        # Deduplicate
        unique_tweets = {t['tweet_id']: t for t in all_tweets}.values()
        unique_tweets = list(unique_tweets)
        
        logger.info(f"Total unique tweets collected: {len(unique_tweets)}")
        
        # Save to file
        if unique_tweets:
            self.save_to_jsonl(unique_tweets, filename)
        
        return filename


async def main():
    """Main entry point"""
    collector = TwitterCollector()
    await collector.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
