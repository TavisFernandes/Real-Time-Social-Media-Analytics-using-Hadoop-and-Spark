"""Run collection, Spark processing, and HDFS publication in sequence."""

import logging
import subprocess
import sys
import time
from pathlib import Path

from hdfs import InsecureClient
from hdfs.util import HdfsError

from scraper import config


PROJECT_ROOT = Path(__file__).resolve().parent
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_script(relative_path):
    """Run one pipeline stage with the current Python interpreter."""
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / relative_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def wait_for_hdfs(client, timeout_seconds=180):
    """Wait for WebHDFS to become available during Hadoop startup."""
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
            try:
                client.status("/")
                client.makedirs(config.HDFS_RAW_PATH)
                client.makedirs(config.HDFS_PROCESSED_PATH)
                return
            except HdfsError as error:
                if "safe mode" not in str(error).casefold():
                    raise
                last_error = error
                logger.info("Waiting for HDFS safe mode to end: %s", error)
            except Exception as error:
                last_error = error
                logger.info("Waiting for HDFS: %s", error)
            time.sleep(5)
    raise TimeoutError("HDFS did not become ready before the timeout") from last_error


def upload_files(client, local_root, hdfs_root, patterns):
    """Upload matching local files into an HDFS directory."""
    client.makedirs(hdfs_root)
    for pattern in patterns:
        for local_file in Path(local_root).rglob(pattern):
            if local_file.is_file():
                remote_path = f"{hdfs_root.rstrip('/')}/{local_file.name}"
                client.upload(remote_path, str(local_file), overwrite=True)
                logger.info("Uploaded %s to HDFS %s", local_file, remote_path)


def main():
    for directory in (config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR, config.EXPORTS_DIR, config.LOGS_DIR):
        Path(directory).mkdir(parents=True, exist_ok=True)

    hdfs_client = InsecureClient(config.HDFS_NAMENODE, user="root")
    wait_for_hdfs(hdfs_client)

    run_script("scraper/collector.py")
    raw_files = [path for path in Path(config.RAW_DATA_DIR).glob("*.jsonl") if path.stat().st_size]
    if not raw_files:
        logger.warning("No real posts were collected; skipping Spark and HDFS data uploads")
        return

    run_script("spark/preprocess.py")
    run_script("spark/sentiment_pipeline.py")
    run_script("spark/analytics.py")

    upload_files(hdfs_client, config.RAW_DATA_DIR, config.HDFS_RAW_PATH, ["*.jsonl"])
    upload_files(hdfs_client, config.EXPORTS_DIR, config.HDFS_PROCESSED_PATH, ["*.json"])
    logger.info("Collection and analytics pipeline completed")


if __name__ == "__main__":
    main()