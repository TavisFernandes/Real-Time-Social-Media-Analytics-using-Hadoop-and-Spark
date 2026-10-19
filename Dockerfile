FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --timeout=300 --retries=10 -r requirements.txt
RUN pip install --no-cache-dir --timeout=300 --retries=10 pyarrow==14.0.2

COPY pipeline.py ./pipeline.py
COPY scraper ./scraper
COPY spark ./spark
COPY dashboard ./dashboard

RUN mkdir -p data/raw data/processed data/exports logs

EXPOSE 8501