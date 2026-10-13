"""
Spark analytics module for Twitter sentiment data
Generates time series aggregates and analytics
"""

import logging
import os
import sys
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, hour, count, avg, sum as _sum,
    date_trunc, window, when, lit
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.config import (
    EXPORTS_DIR,
    PROCESSED_DATA_DIR,
    LOG_FILE
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SparkAnalytics:
    """Performs analytics on Twitter sentiment data"""
    
    def __init__(self, app_name="TwitterAnalytics"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        
    def load_data(self, input_path):
        """Load data with sentiment"""
        logger.info(f"Loading data from {input_path}")
        
        df = self.spark.read.json(input_path)
        
        # Convert timestamp if needed
        if "created_at_ts" in df.columns:
            df = df.withColumn("created_at_ts", col("created_at_ts"))
        else:
            df = df.withColumn("created_at_ts", to_timestamp(col("created_at")))
            
        logger.info(f"Loaded {df.count()} records")
        return df
    
    def hourly_sentiment_aggregates(self, df):
        """Generate hourly sentiment aggregates"""
        logger.info("Generating hourly sentiment aggregates")
        
        hourly_df = df.groupBy(
            to_date(col("created_at_ts")).alias("date"),
            hour(col("created_at_ts")).alias("hour"),
            col("sentiment")
        ).agg(
            count("*").alias("tweet_count"),
            avg("sentiment_score").alias("avg_sentiment_score")
        ).orderBy("date", "hour", "sentiment")
        
        return hourly_df
    
    def daily_sentiment_aggregates(self, df):
        """Generate daily sentiment aggregates"""
        logger.info("Generating daily sentiment aggregates")
        
        daily_df = df.groupBy(
            to_date(col("created_at_ts")).alias("date"),
            col("sentiment")
        ).agg(
            count("*").alias("tweet_count"),
            avg("sentiment_score").alias("avg_sentiment_score")
        ).orderBy("date", "sentiment")
        
        return daily_df
    
    def tweet_volume_over_time(self, df):
        """Generate tweet volume over time (hourly)"""
        logger.info("Generating tweet volume over time")
        
        volume_df = df.groupBy(
            to_date(col("created_at_ts")).alias("date"),
            hour(col("created_at_ts")).alias("hour")
        ).agg(
            count("*").alias("tweet_count")
        ).orderBy("date", "hour")
        
        return volume_df
    
    def sentiment_by_keyword(self, df):
        """Generate sentiment breakdown by keyword"""
        logger.info("Generating sentiment by keyword")
        
        keyword_df = df.groupBy(
            col("keyword_used"),
            col("sentiment")
        ).agg(
            count("*").alias("tweet_count"),
            avg("sentiment_score").alias("avg_sentiment_score")
        ).orderBy("keyword_used", "tweet_count")
        
        return keyword_df
    
    def save_to_csv(self, df, output_path):
        """Save DataFrame to CSV"""
        logger.info(f"Saving to {output_path}")
        
        df.coalesce(1).write.mode('overwrite').csv(
            output_path,
            header=True
        )
        logger.info("CSV saved successfully")
    
    def run_all_analytics(self):
        """Run all analytics and save results"""
        logger.info("Starting analytics pipeline")
        
        # Load data
        df = self.load_data(EXPORTS_DIR)
        
        # Generate analytics
        hourly_sentiment = self.hourly_sentiment_aggregates(df)
        daily_sentiment = self.daily_sentiment_aggregates(df)
        tweet_volume = self.tweet_volume_over_time(df)
        keyword_sentiment = self.sentiment_by_keyword(df)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.save_to_csv(
            hourly_sentiment,
            f"{PROCESSED_DATA_DIR}/hourly_sentiment_{timestamp}"
        )
        self.save_to_csv(
            daily_sentiment,
            f"{PROCESSED_DATA_DIR}/daily_sentiment_{timestamp}"
        )
        self.save_to_csv(
            tweet_volume,
            f"{PROCESSED_DATA_DIR}/tweet_volume_{timestamp}"
        )
        self.save_to_csv(
            keyword_sentiment,
            f"{PROCESSED_DATA_DIR}/keyword_sentiment_{timestamp}"
        )
        
        logger.info("Analytics pipeline complete")
    
    def stop(self):
        """Stop Spark session"""
        self.spark.stop()


def main():
    """Main analytics pipeline"""
    analytics = SparkAnalytics()
    analytics.run_all_analytics()
    analytics.stop()


if __name__ == "__main__":
    main()
