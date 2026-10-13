-- Hive table creation for Twitter sentiment analytics
-- Run this after uploading data to HDFS

-- Create database
CREATE DATABASE IF NOT EXISTS twitter_analytics;
USE twitter_analytics;

-- Drop table if exists
DROP TABLE IF EXISTS tweets;

-- Create external table for raw tweets
CREATE EXTERNAL TABLE IF NOT EXISTS tweets (
    tweet_id BIGINT,
    created_at STRING,
    username STRING,
    text STRING,
    likes INT,
    retweets INT,
    replies INT,
    hashtags ARRAY<STRING>,
    keyword_used STRING
)
ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe'
STORED AS TEXTFILE
LOCATION '/twitter/raw'
TBLPROPERTIES ("skip.header.line.count"="1");

-- Create external table for processed tweets with sentiment
DROP TABLE IF EXISTS tweets_processed;
CREATE EXTERNAL TABLE IF NOT EXISTS tweets_processed (
    tweet_id BIGINT,
    created_at STRING,
    username STRING,
    text STRING,
    likes INT,
    retweets INT,
    replies INT,
    hashtags ARRAY<STRING>,
    keyword_used STRING,
    sentiment STRING,
    sentiment_score DOUBLE
)
ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe'
STORED AS TEXTFILE
LOCATION '/twitter/processed'
TBLPROPERTIES ("skip.header.line.count"="1");

-- Show tables
SHOW TABLES;
