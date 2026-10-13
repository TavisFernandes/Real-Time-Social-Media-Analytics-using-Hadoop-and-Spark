"""
Spark sentiment analysis pipeline using VADER
Analyzes sentiment of preprocessed Twitter data
"""

import logging
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, lit
from pyspark.sql.types import StringType, DoubleType

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.config import (
    PROCESSED_DATA_DIR,
    EXPORTS_DIR,
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


class SentimentAnalyzer:
    """Performs sentiment analysis using VADER"""
    
    def __init__(self, app_name="SentimentAnalyzer"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        self.vader = SentimentIntensityAnalyzer()
        
    def get_sentiment(self, text):
        """Get sentiment label from text using VADER"""
        if not text or text == "":
            return "Neutral", 0.0
            
        scores = self.vader.polarity_scores(text)
        compound_score = scores['compound']
        
        if compound_score >= 0.05:
            return "Positive", compound_score
        elif compound_score <= -0.05:
            return "Negative", compound_score
        else:
            return "Neutral", compound_score
    
    def analyze_sentiment(self, df):
        """Apply sentiment analysis to DataFrame"""
        logger.info("Starting sentiment analysis")
        
        # Register UDFs
        sentiment_label_udf = udf(
            lambda x: self.get_sentiment(x)[0], 
            StringType()
        )
        sentiment_score_udf = udf(
            lambda x: self.get_sentiment(x)[1], 
            DoubleType()
        )
        
        # Apply sentiment analysis on cleaned text
        df = df.withColumn(
            "sentiment",
            sentiment_label_udf(col("cleaned_text"))
        )
        df = df.withColumn(
            "sentiment_score",
            sentiment_score_udf(col("cleaned_text"))
        )
        
        logger.info("Sentiment analysis complete")
        
        return df
    
    def load_processed_data(self, input_path):
        """Load preprocessed data"""
        logger.info(f"Loading processed data from {input_path}")
        
        df = self.spark.read.json(input_path)
        logger.info(f"Loaded {df.count()} records")
        
        return df
    
    def save_with_sentiment(self, df, output_path):
        """Save data with sentiment analysis"""
        logger.info(f"Saving data with sentiment to {output_path}")
        
        # Select relevant columns
        columns_to_save = [
            "tweet_id", "created_at", "username", "text", 
            "likes", "retweets", "replies", "hashtags", 
            "keyword_used", "sentiment", "sentiment_score"
        ]
        
        df = df.select([c for c in columns_to_save if c in df.columns])
        
        df.write.mode('overwrite').json(output_path)
        logger.info("Data saved successfully")
    
    def stop(self):
        """Stop Spark session"""
        self.spark.stop()


def main():
    """Main sentiment analysis pipeline"""
    analyzer = SentimentAnalyzer()
    
    # Load preprocessed data
    df = analyzer.load_processed_data(PROCESSED_DATA_DIR)
    
    # Analyze sentiment
    df_with_sentiment = analyzer.analyze_sentiment(df)
    
    # Save results
    analyzer.save_with_sentiment(df_with_sentiment, EXPORTS_DIR)
    
    analyzer.stop()


if __name__ == "__main__":
    main()
