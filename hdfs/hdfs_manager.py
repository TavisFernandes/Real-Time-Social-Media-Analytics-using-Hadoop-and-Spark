"""
HDFS Manager for uploading and managing data in Hadoop HDFS
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.config import (
    HDFS_NAMENODE,
    HDFS_RAW_PATH,
    HDFS_PROCESSED_PATH,
    RAW_DATA_DIR,
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


class HDFSManager:
    """Manages HDFS operations"""
    
    def __init__(self, namenode_url=HDFS_NAMENODE):
        self.namenode_url = namenode_url
        self.raw_path = HDFS_RAW_PATH
        self.processed_path = HDFS_PROCESSED_PATH
        
    def run_hdfs_command(self, command):
        """Execute HDFS command inside namenode container"""
        try:
            full_command = f"docker exec namenode {command}"
            result = subprocess.run(
                full_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                logger.error(f"HDFS command failed: {result.stderr}")
                return False
            
            logger.info(f"HDFS command success: {command}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("HDFS command timed out")
            return False
        except Exception as e:
            logger.error(f"Error running HDFS command: {e}")
            return False
    
    def create_directories(self):
        """Create HDFS directories for raw and processed data"""
        logger.info("Creating HDFS directories")
        
        commands = [
            f"hdfs dfs -mkdir -p {self.raw_path}",
            f"hdfs dfs -mkdir -p {self.processed_path}"
        ]
        
        for cmd in commands:
            self.run_hdfs_command(cmd)
    
    def upload_file(self, local_path, hdfs_path):
        """Upload a file to HDFS"""
        logger.info(f"Uploading {local_path} to {hdfs_path}")
        
        # Get absolute path
        abs_local_path = os.path.abspath(local_path)
        
        command = f"hdfs dfs -put -f {abs_local_path} {hdfs_path}"
        success = self.run_hdfs_command(command)
        
        if success:
            logger.info(f"Successfully uploaded {local_path}")
        return success
    
    def upload_directory(self, local_dir, hdfs_path):
        """Upload entire directory to HDFS"""
        logger.info(f"Uploading directory {local_dir} to {hdfs_path}")
        
        abs_local_dir = os.path.abspath(local_dir)
        command = f"hdfs dfs -put -f {abs_local_dir}/* {hdfs_path}"
        success = self.run_hdfs_command(command)
        
        if success:
            logger.info(f"Successfully uploaded directory {local_dir}")
        return success
    
    def list_files(self, hdfs_path):
        """List files in HDFS directory"""
        logger.info(f"Listing files in {hdfs_path}")
        
        command = f"hdfs dfs -ls {hdfs_path}"
        self.run_hdfs_command(command)
    
    def verify_upload(self, hdfs_path, expected_files=1):
        """Verify files were uploaded successfully"""
        logger.info(f"Verifying upload to {hdfs_path}")
        
        command = f"hdfs dfs -count {hdfs_path}"
        result = subprocess.run(
            f"docker exec namenode {command}",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Upload verification: {result.stdout}")
            return True
        return False
    
    def delete_file(self, hdfs_path):
        """Delete a file from HDFS"""
        logger.info(f"Deleting {hdfs_path}")
        
        command = f"hdfs dfs -rm {hdfs_path}"
        return self.run_hdfs_command(command)


def main():
    """Test HDFS operations"""
    manager = HDFSManager()
    
    # Create directories
    manager.create_directories()
    
    # List directories
    manager.list_files("/")
    manager.list_files(HDFS_RAW_PATH)
    manager.list_files(HDFS_PROCESSED_PATH)


if __name__ == "__main__":
    main()
