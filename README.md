# Real-Time X (Twitter) Big Data Sentiment Analytics Platform

A production-quality academic project that collects live X/Twitter posts, stores data in Hadoop HDFS, analyzes using Hive and Spark, performs sentiment analysis, and visualizes results in a Streamlit dashboard.

## Architecture

```
X/Twitter
    ↓
twscrape (Data Collection)
    ↓
JSONL (Raw Data)
    ↓
HDFS (Storage)
    ↙       ↘
Hive       Spark
(Query)   (Processing)
    ↘       ↙
   Analytics
       ↓
Streamlit Dashboard
```

## Tech Stack

- **Python 3.11**
- **twscrape** - Twitter/X data collection
- **Hadoop HDFS** - Distributed storage (Docker)
- **Hive** - Data warehousing (Docker)
- **Apache Spark** - Big data processing (Docker)
- **Streamlit** - Interactive dashboard
- **VADER** - Sentiment analysis
- **Docker Compose** - Infrastructure orchestration

## Project Structure

```
twitter-bigdata/
│
├── docker/
│   ├── docker-compose.yml
│   └── configs/
│
├── scraper/
│   ├── collector.py          # twscrape data collector
│   └── config.py            # Configuration settings
│
├── data/
│   ├── raw/                 # Raw JSONL files
│   ├── processed/           # Spark processed data
│   └── exports/             # Final data with sentiment
│
├── hdfs/
│   └── hdfs_manager.py      # HDFS operations
│
├── hive/
│   ├── create_tables.sql    # Hive table definitions
│   └── analytics_queries.sql # Analytics queries
│
├── spark/
│   ├── sentiment_pipeline.py # Sentiment analysis
│   ├── preprocess.py        # Data preprocessing
│   └── analytics.py         # Time series analytics
│
├── dashboard/
│   └── app.py               # Streamlit dashboard
│
├── logs/                    # Application logs
├── requirements.txt
└── README.md
```

## Prerequisites

1. **Docker Desktop** installed and running
2. **Python 3.11+**
3. **Twitter/X API credentials** (for twscrape)

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Twitter/X Accounts for twscrape

twscrape requires Twitter accounts to collect data. Add accounts using:

```bash
# Add accounts interactively
twscrape add_accounts

# Or add from file
twscrape add_accounts accounts.txt
```

Follow the prompts to authenticate each account.

### 3. Start Docker Infrastructure

Navigate to the docker directory and start the services:

```bash
cd docker
docker-compose up -d
```

This will start:
- HDFS Namenode (port 9870)
- HDFS Datanode (port 9864)
- Hive Metastore (port 9083)
- Hive Server (port 10000)
- Spark Master (port 8080)
- Spark Worker (port 8081)

Wait for all services to be healthy (2-3 minutes).

Check status:
```bash
docker-compose ps
```

### 4. Initialize HDFS Directories

```bash
cd ..
python hdfs/hdfs_manager.py
```

This creates `/twitter/raw` and `/twitter/processed` directories in HDFS.

### 5. Create Hive Tables

Connect to Hive server and run the table creation script:

```bash
# Connect to Hive container
docker exec -it hive-server bash

# Run Hive CLI
hive

# Execute table creation
source /hive/create_tables.sql
```

Or use beeline:
```bash
docker exec -it hive-server beeline -u jdbc:hive2://localhost:10000 -f hive/create_tables.sql
```

## Running the Project

### Step 1: Collect Twitter Data

```bash
cd scraper
python collector.py
```

This collects tweets for configured keywords (AI, ChatGPT, OpenAI, IPL, iPhone) and saves to `data/raw/`.

### Step 2: Upload to HDFS

```bash
python hdfs/hdfs_manager.py
```

Modify `hdfs_manager.py` to upload specific files or directories.

### Step 3: Preprocess with Spark

```bash
cd spark
python preprocess.py
```

This cleans data, removes duplicates, and prepares for sentiment analysis.

### Step 4: Run Sentiment Analysis

```bash
python sentiment_pipeline.py
```

This applies VADER sentiment analysis and saves results to `data/exports/`.

### Step 5: Generate Analytics

```bash
python analytics.py
```

This generates time series aggregates (hourly/daily sentiment, tweet volume).

### Step 6: Launch Dashboard

```bash
cd dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## Dashboard Pages

1. **Overview** - Total tweets, sentiment percentages, keyword count, recent activity
2. **Sentiment** - Pie chart and bar chart of sentiment distribution
3. **Trends** - Tweet volume over time, sentiment trends
4. **Hashtags** - Top hashtags and frequency analysis
5. **Recent Tweets** - Searchable table of recent tweets with sentiment labels

## Configuration

Edit `scraper/config.py` to customize:

- **Keywords** - Add/remove search keywords
- **Collection limits** - Max tweets per keyword
- **Output paths** - Data storage locations
- **HDFS paths** - HDFS directory structure

## Hive Queries

Run analytics queries from `hive/analytics_queries.sql`:

```bash
docker exec -it hive-server beeline -u jdbc:hive2://localhost:10000 -f hive/analytics_queries.sql
```

Available queries:
- Total tweets
- Tweets by keyword
- Top hashtags
- Most active users
- Most active hour
- Daily tweet count
- Average likes by keyword
- Top engagement tweets
- Sentiment distribution
- Sentiment by keyword
- Sentiment over time

## Stopping the Infrastructure

```bash
cd docker
docker-compose down
```

To remove volumes:
```bash
docker-compose down -v
```

## Data Flow

1. **Collection**: twscrape collects tweets → JSONL files in `data/raw/`
2. **Storage**: JSONL files uploaded to HDFS `/twitter/raw/`
3. **Preprocessing**: Spark reads from HDFS/local, cleans data → `data/processed/`
4. **Sentiment Analysis**: VADER analyzes sentiment → `data/exports/`
5. **Analytics**: Spark generates aggregates → `data/processed/`
6. **Visualization**: Streamlit dashboard reads from `data/exports/`

## Troubleshooting

### Docker Services Not Starting

Check logs:
```bash
docker-compose logs namenode
docker-compose logs hive-server
```

Ensure Docker Desktop has sufficient resources (4GB+ RAM recommended).

### HDFS Connection Issues

Verify HDFS is running:
```bash
docker exec namenode hdfs dfs -ls /
```

### Hive Connection Issues

Check Hive metastore:
```bash
docker exec hive-metastore hive --service metastore
```

### Spark Issues

Ensure Spark can connect to HDFS:
```bash
docker exec spark-master spark-shell
```

### twscrape Authentication

Ensure accounts are properly authenticated:
```bash
twscrape accounts
```

## Adding New Keywords

Edit `scraper/config.py`:

```python
KEYWORDS = [
    "AI",
    "ChatGPT",
    "OpenAI",
    "IPL",
    "iPhone",
    "YourNewKeyword"  # Add here
]
```

## Logs

Application logs are stored in `logs/`:
- `logs/scraper.log` - Data collection logs
- Spark logs available in Spark UI (http://localhost:8081)

## Web UIs

- **HDFS Namenode**: http://localhost:9870
- **Spark Master**: http://localhost:8080
- **Spark Worker**: http://localhost:8081
- **Streamlit Dashboard**: http://localhost:8501

## Future Enhancements

- Real-time streaming with Spark Structured Streaming
- Advanced sentiment models (BERT, RoBERTa)
- Time series forecasting (ARIMA, Prophet)
- Topic modeling (LDA)
- User clustering and segmentation
- Geospatial analysis

## License

This is an academic project for educational purposes.

## Acknowledgments

- twscrape library for Twitter data collection
- VADER sentiment analysis
- Apache Hadoop, Hive, and Spark communities
- Streamlit for dashboard framework
