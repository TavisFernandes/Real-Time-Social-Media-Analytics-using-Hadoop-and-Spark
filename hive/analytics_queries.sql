-- Hive analytics queries for Twitter sentiment analytics
-- Run these queries after creating tables and loading data

USE twitter_analytics;

-- Query 1: Total tweets
SELECT COUNT(*) AS total_tweets
FROM tweets;

-- Query 2: Tweets by keyword
SELECT keyword_used, COUNT(*) AS tweet_count
FROM tweets
GROUP BY keyword_used
ORDER BY tweet_count DESC;

-- Query 3: Top hashtags
SELECT 
    hashtag,
    COUNT(*) AS frequency
FROM tweets LATERAL VIEW EXPLODE(hashtags) exploded_hashtags AS hashtag
GROUP BY hashtag
ORDER BY frequency DESC
LIMIT 20;

-- Query 4: Most active users
SELECT 
    username,
    COUNT(*) AS tweet_count,
    SUM(likes) AS total_likes,
    SUM(retweets) AS total_retweets
FROM tweets
GROUP BY username
ORDER BY tweet_count DESC
LIMIT 20;

-- Query 5: Most active hour
SELECT 
    HOUR(created_at) AS hour_of_day,
    COUNT(*) AS tweet_count
FROM tweets
GROUP BY HOUR(created_at)
ORDER BY tweet_count DESC;

-- Query 6: Daily tweet count
SELECT 
    TO_DATE(created_at) AS date,
    COUNT(*) AS tweet_count
FROM tweets
GROUP BY TO_DATE(created_at)
ORDER BY date DESC;

-- Query 7: Average likes by keyword
SELECT 
    keyword_used,
    AVG(likes) AS avg_likes,
    AVG(retweets) AS avg_retweets,
    AVG(replies) AS avg_replies
FROM tweets
GROUP BY keyword_used
ORDER BY avg_likes DESC;

-- Query 8: Top engagement tweets
SELECT 
    tweet_id,
    username,
    text,
    likes,
    retweets,
    replies,
    (likes + retweets + replies) AS total_engagement
FROM tweets
ORDER BY total_engagement DESC
LIMIT 20;

-- Query 9: Sentiment distribution (from processed table)
SELECT 
    sentiment,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM tweets_processed
GROUP BY sentiment
ORDER BY count DESC;

-- Query 10: Sentiment by keyword
SELECT 
    keyword_used,
    sentiment,
    COUNT(*) AS count
FROM tweets_processed
GROUP BY keyword_used, sentiment
ORDER BY keyword_used, count DESC;

-- Query 11: Sentiment over time (daily)
SELECT 
    TO_DATE(created_at) AS date,
    sentiment,
    COUNT(*) AS count
FROM tweets_processed
GROUP BY TO_DATE(created_at), sentiment
ORDER BY date DESC, sentiment;

-- Query 12: Sentiment over time (hourly)
SELECT 
    TO_DATE(created_at) AS date,
    HOUR(created_at) AS hour,
    sentiment,
    COUNT(*) AS count
FROM tweets_processed
GROUP BY TO_DATE(created_at), HOUR(created_at), sentiment
ORDER BY date DESC, hour, sentiment;
