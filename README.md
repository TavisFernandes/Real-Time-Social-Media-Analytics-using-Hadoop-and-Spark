# Real-Time Bluesky Big Data Sentiment Analytics Platform

A project that collects real public Bluesky posts without account credentials, stores data in Hadoop HDFS, analyzes using Hive and Spark, performs sentiment analysis, and visualizes results in a Streamlit dashboard.

## Architecture

```
Bluesky public search API
    ↓
requests (No login required)
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
- **Bluesky public API** - Real public post search without account credentials
- **Hadoop HDFS** - Distributed storage (Docker)
- **Hive** - Data warehousing (Docker)
- **Apache Spark** - Big data processing (Docker)
- **Streamlit** - Interactive dashboard
- **VADER** - Sentiment analysis
- **Docker Compose** - Infrastructure orchestration

## Project Structure

```
bluesky-bigdata/
│
├── docker/
│   ├── docker-compose.yml
│   └── configs/
│
├── scraper/
│   ├── collector.py          # Public Bluesky API collector
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
3. Internet access to `public.api.bsky.app` (no Bluesky account required)

## Setup Instructions

### Run Everything with Docker

From the project root, run:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Compose builds the Python/Java application image, starts HDFS and Hive, runs the Bluesky collection and Spark pipeline, uploads results to HDFS, initializes the Hive tables, and starts the dashboard. Open `http://localhost:8501` for the dashboard. The first run may take several minutes while images download and Hadoop initializes.

Data and logs persist in the project folders. To stop the services, press `Ctrl+C`; to stop and remove the containers later, run `docker compose -f docker/docker-compose.yml down`. The collector requires internet access to Bluesky's public API. If no posts can be fetched, the pipeline logs that condition and leaves the dashboard available without creating synthetic data.

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Docker Infrastructure

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

### 3. Initialize HDFS Directories

```bash
cd ..
python hdfs/hdfs_manager.py
```

This creates `/twitter/raw` and `/twitter/processed` directories in HDFS.

### 4. Create Hive Tables

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

### Step 1: Collect Real Bluesky Data

```bash
cd scraper
python collector.py
```

This searches Bluesky's public API for configured keywords (AI, ChatGPT, OpenAI, IPL, iPhone) and saves real public posts to `data/raw/`. No account credentials or generated sample data are used. X/Twitter's official API is a separate service and requires X developer credentials for search access.

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

This generates time series aggregates (hourly/daily sentiment, post volume).

### Step 6: Launch Dashboard

```bash
cd dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## Dashboard Pages

1. **Overview** - Total posts, sentiment percentages, keyword count, recent activity
2. **Sentiment** - Pie chart and bar chart of sentiment distribution
3. **Trends** - Post volume over time, sentiment trends
4. **Hashtags** - Top hashtags and frequency analysis
5. **Recent Posts** - Searchable table of recent posts with sentiment labels
6. **Topic Forecasts** - A two-topic monthly comparison with ADF stationarity checks, ACF/PACF plots, AR/MA/ARMA/ARIMA/SARIMA forecasts, holdout errors, residual ACF, and Ljung-Box checks. ChatGPT, OpenAI, and iPhone also use archived text mentions to build longer topic samples; limited samples receive a clearly labeled baseline forecast.

The dashboard reads Spark's nested JSON/CSV output folders and the local history archive from the project data directory, regardless of the directory used to launch Streamlit. **Reload and fetch latest posts** makes a short-timeout live search; results are saved locally and appear in Recent Posts and Trends. If the network blocks Bluesky's API, the dashboard stays usable with the newest posts already in the local archive. Forecasts describe the collected sample, not total Bluesky activity.

### Build a Local History Archive

With Docker and internet access available, run a one-time three-year backfill for all monitored topics:

```bash
docker compose -f docker/docker-compose.yml run --build --rm pipeline python scraper/collector.py --history-years 3
```

Use `--history-years 2` for a two-year archive. The collector stores up to 50 real posts per topic per week in `data/history/`. The dashboard loads this file locally, so the saved time series remains available without internet access on campus. Historical forecasts for ChatGPT, OpenAI, and iPhone include matching archived post text alongside directly searched posts; fit comparisons are shown when enough monthly data is available, with a labeled average baseline for sparse samples. Rebuild the dashboard after pulling code changes with `docker compose -f docker/docker-compose.yml up -d --build dashboard`.

## Configuration

Edit `scraper/config.py` to customize:

- **Keywords** - Add/remove search keywords
- **Collection limits** - Max posts per keyword
- **Output paths** - Data storage locations
- **HDFS paths** - HDFS directory structure

## Hive Queries

Run analytics queries from `hive/analytics_queries.sql`:

```bash
docker exec -it hive-server beeline -u jdbc:hive2://localhost:10000 -f hive/analytics_queries.sql
```

Available queries:
- Total posts
- Posts by keyword
- Top hashtags
- Most active users
- Most active hour
- Daily post count
- Average likes by keyword
- Top engagement posts
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

1. **Collection**: Bluesky public search API returns real posts → JSONL files in `data/raw/`
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

### Bluesky API Connectivity

The collector needs internet access to `public.api.bsky.app`. If collection fails, check the connection and try again; the API may also enforce request limits.

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

- Bluesky public API for real post data
- VADER sentiment analysis
- Apache Hadoop, Hive, and Spark communities
- Streamlit for dashboard framework
