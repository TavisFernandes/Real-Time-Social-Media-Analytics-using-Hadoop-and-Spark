"""
Streamlit Dashboard for Twitter Sentiment Analytics
Multi-page dashboard with Overview, Sentiment, Trends, Hashtags, and Recent Tweets pages
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analytics",
    page_icon="🐦",
    layout="wide"
)

# Configuration
EXPORTS_DIR = "data/exports"
PROCESSED_DATA_DIR = "data/processed"


@st.cache_data
def load_data():
    """Load processed data with sentiment"""
    data_files = list(Path(EXPORTS_DIR).glob("*.json"))
    
    if not data_files:
        return pd.DataFrame()
    
    all_data = []
    for file in data_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    all_data.append(json.loads(line))
                except:
                    continue
    
    df = pd.DataFrame(all_data)
    
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    return df


@st.cache_data
def load_analytics():
    """Load analytics CSV files"""
    analytics_files = {
        'hourly': list(Path(PROCESSED_DATA_DIR).glob("hourly_sentiment_*.csv")),
        'daily': list(Path(PROCESSED_DATA_DIR).glob("daily_sentiment_*.csv")),
        'volume': list(Path(PROCESSED_DATA_DIR).glob("tweet_volume_*.csv")),
        'keyword': list(Path(PROCESSED_DATA_DIR).glob("keyword_sentiment_*.csv"))
    }
    
    data = {}
    for key, files in analytics_files.items():
        if files:
            try:
                data[key] = pd.read_csv(files[-1])
            except:
                data[key] = pd.DataFrame()
    
    return data


def overview_page(df):
    """Page 1: Overview"""
    st.title("📊 Overview")
    
    if df.empty:
        st.warning("No data available. Please run the data collection and processing pipeline first.")
        return
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_tweets = len(df)
    positive_pct = (df['sentiment'] == 'Positive').sum() / total_tweets * 100
    negative_pct = (df['sentiment'] == 'Negative').sum() / total_tweets * 100
    neutral_pct = (df['sentiment'] == 'Neutral').sum() / total_tweets * 100
    total_keywords = df['keyword_used'].nunique()
    
    col1.metric("Total Tweets", f"{total_tweets:,}")
    col2.metric("Positive %", f"{positive_pct:.1f}%")
    col3.metric("Negative %", f"{negative_pct:.1f}%")
    col4.metric("Neutral %", f"{neutral_pct:.1f}%")
    col5.metric("Keywords", total_keywords)
    
    # Recent activity
    st.subheader("Recent Activity")
    recent_df = df.sort_values('created_at', ascending=False).head(10)
    
    for _, row in recent_df.iterrows():
        with st.expander(f"@{row['username']} - {row['created_at'].strftime('%Y-%m-%d %H:%M')}"):
            st.write(f"**Text:** {row['text']}")
            st.write(f"**Sentiment:** {row['sentiment']}")
            st.write(f"**Engagement:** {row['likes']} likes, {row['retweets']} retweets")


def sentiment_page(df):
    """Page 2: Sentiment Analysis"""
    st.title("😊 Sentiment Analysis")
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution (Pie)")
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Sentiment Counts (Bar)")
        fig_bar = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
        )
        fig_bar.update_layout(xaxis_title="Sentiment", yaxis_title="Count")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Sentiment by keyword
    st.subheader("Sentiment by Keyword")
    keyword_sentiment = df.groupby(['keyword_used', 'sentiment']).size().unstack(fill_value=0)
    st.bar_chart(keyword_sentiment)


def trends_page(df, analytics):
    """Page 3: Trends"""
    st.title("📈 Trends")
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Tweets over time
    st.subheader("Tweet Volume Over Time")
    
    if 'volume' in analytics and not analytics['volume'].empty:
        volume_df = analytics['volume']
        volume_df['datetime'] = pd.to_datetime(volume_df['date']) + pd.to_timedelta(volume_df['hour'], unit='h')
        fig_volume = px.line(volume_df, x='datetime', y='tweet_count', title='Tweet Volume Over Time')
        st.plotly_chart(fig_volume, use_container_width=True)
    else:
        df['hour'] = df['created_at'].dt.floor('H')
        hourly_volume = df.groupby('hour').size().reset_index(name='count')
        fig_volume = px.line(hourly_volume, x='hour', y='count', title='Tweet Volume Over Time')
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Sentiment over time
    st.subheader("Sentiment Over Time")
    
    if 'daily' in analytics and not analytics['daily'].empty:
        daily_df = analytics['daily']
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        fig_sentiment = px.line(
            daily_df, 
            x='date', 
            y='tweet_count', 
            color='sentiment',
            title='Sentiment Over Time',
            color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)


def hashtags_page(df):
    """Page 4: Hashtags"""
    st.title("#️⃣ Hashtags")
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Extract all hashtags
    all_hashtags = []
    for hashtags in df['hashtags']:
        if isinstance(hashtags, list):
            all_hashtags.extend(hashtags)
        elif isinstance(hashtags, str):
            all_hashtags.append(hashtags)
    
    if not all_hashtags:
        st.info("No hashtags found in the data.")
        return
    
    # Count hashtags
    from collections import Counter
    hashtag_counts = Counter(all_hashtags)
    top_hashtags = hashtag_counts.most_common(20)
    
    # Display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 20 Hashtags")
        hashtag_df = pd.DataFrame(top_hashtags, columns=['Hashtag', 'Count'])
        st.dataframe(hashtag_df, use_container_width=True)
    
    with col2:
        st.subheader("Hashtag Frequency")
        fig_hashtags = px.bar(
            hashtag_df.head(10),
            x='Hashtag',
            y='Count',
            title='Top 10 Hashtags'
        )
        st.plotly_chart(fig_hashtags, use_container_width=True)


def recent_tweets_page(df):
    """Page 5: Recent Tweets"""
    st.title("🐦 Recent Tweets")
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Search
    search_term = st.text_input("Search tweets", "")
    
    # Filter
    filtered_df = df.copy()
    if search_term:
        filtered_df = filtered_df[
            filtered_df['text'].str.contains(search_term, case=False, na=False) |
            filtered_df['username'].str.contains(search_term, case=False, na=False)
        ]
    
    # Sort by date
    filtered_df = filtered_df.sort_values('created_at', ascending=False)
    
    # Display
    st.subheader(f"Showing {len(filtered_df)} tweets")
    
    for _, row in filtered_df.head(50).iterrows():
        sentiment_color = {
            'Positive': '🟢',
            'Negative': '🔴',
            'Neutral': '⚪'
        }.get(row['sentiment'], '⚪')
        
        with st.container():
            st.markdown(f"**{sentiment_color} @{row['username']}** - {row['created_at'].strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"*{row['text']}*")
            st.caption(f"Sentiment: {row['sentiment']} | Likes: {row['likes']} | Retweets: {row['retweets']} | Keyword: {row['keyword_used']}")
            st.divider()


def main():
    """Main app"""
    # Load data
    df = load_data()
    analytics = load_analytics()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Sentiment", "Trends", "Hashtags", "Recent Tweets"]
    )
    
    # Render selected page
    if page == "Overview":
        overview_page(df)
    elif page == "Sentiment":
        sentiment_page(df)
    elif page == "Trends":
        trends_page(df, analytics)
    elif page == "Hashtags":
        hashtags_page(df)
    elif page == "Recent Tweets":
        recent_tweets_page(df)


if __name__ == "__main__":
    main()
