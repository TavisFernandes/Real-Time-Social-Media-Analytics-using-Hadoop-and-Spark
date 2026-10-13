"""
Spark preprocessing module for Twitter data
Cleans and prepares data for sentiment analysis
"""

import logging
import os
import re
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, udf, to_timestamp, regexp_replace, lower, trim,
    when, coalesce, size, explode, array, lit
)
from pyspark.sql.types import StringType, ArrayType

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.config import (
    RAW_DATA_DIR,
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


class SparkPreprocessor:
    """Preprocesses Twitter data using PySpark"""
    
    def __init__(self, app_name="TwitterPreprocessor"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        
    def load_data(self, input_path):
        """Load JSONL data from local file"""
        logger.info(f"Loading data from {input_path}")
        
        df = self.spark.read.json(input_path)
        logger.info(f"Loaded {df.count()} records")
        
        return df
    
    def clean_text(self, text):
        """Clean tweet text"""
        if text is None:
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        # Remove special characters but keep emojis
        text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_hashtags_udf(self, hashtags_array):
        """Extract hashtags from array"""
        if hashtags_array is None:
            return []
        return hashtags_array
    
    def preprocess(self, df):
        """Apply preprocessing transformations"""
        logger.info("Starting preprocessing")
        
        # Register UDF for text cleaning
        clean_text_udf = udf(self.clean_text, StringType())
        
        # Handle nulls
        df = df.na.fill({
            'text': '',
            'username': 'unknown',
            'likes': 0,
            'retweets': 0,
            'replies': 0,
            'hashtags': []
        })
        
        # Clean text
        df = df.withColumn("cleaned_text", clean_text_udf(col("text")))
        
        # Convert timestamp
        df = df.withColumn("created_at_ts", to_timestamp(col("created_at")))
        
        # Ensure hashtags is array type
        df = df.withColumn(
            "hashtags",
            when(col("hashtags").isNull(), array()).otherwise(col("hashtags"))
        )
        
        # Remove duplicates based on tweet_id
        df = df.dropDuplicates(["tweet_id"])
        
        logger.info(f"Preprocessing complete. Records: {df.count()}")
        
        return df
    
    def save_processed(self, df, output_path):
        """Save processed data"""
        logger.info(f"Saving processed data to {output_path}")
        
        df.write.mode('overwrite').json(output_path)
        logger.info("Data saved successfully")
    
    def stop(self):
        """Stop Spark session"""
        self.spark.stop()


def main():
    """Main preprocessing pipeline"""
    preprocessor = SparkPreprocessor()
    
    # Load raw data
    df = preprocessor.load_data(RAW_DATA_DIR)
    
    # Preprocess
    processed_df = preprocessor.preprocess(df)
    
    # Save processed data
    preprocessor.save_processed(processed_df, PROCESSED_DATA_DIR)
    
    preprocessor.stop()


if __name__ == "__main__":
    main()
