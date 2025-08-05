# s3_handler.py
import os
import tempfile
import logging
from pathlib import Path
from fastapi import HTTPException
from datetime import datetime
import boto3
from data.data_ingestion.storage_handler import DatasetStorageHandler, TEMP_DIR
from data.data_ingestion.models.metadata import DatasetMetadata
from data.data_ingestion.handlers.utils import summarize_dataset
from data.data_ingestion.handlers.conditions import DatasetConditionChecker

logger = logging.getLogger(__name__)

class S3Handler:
    def __init__(self, *, access_key: str, secret_key: str, endpoint_url: str, bucket_name: str):
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name
        self.storage_handler = DatasetStorageHandler()
        logger.info(f"S3Handler initialized for endpoint '{endpoint_url}' and bucket '{bucket_name}'.")

    def download_file(self, remote_path: str, local_path: Path, s3_client) -> None:
        if not s3_client:
            raise HTTPException(status_code=503, detail="S3 service is not available.")
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3_client.head_object(Bucket=self.bucket_name, Key=remote_path)  
            except s3_client.exceptions.NoSuchKey:
                raise ValueError(f"The file or folder '{remote_path}' does not exist in the S3 bucket '{self.bucket_name}'.")
            
            s3_client.download_file(self.bucket_name, remote_path, str(local_path))
            logger.info(f"S3 Download: '{remote_path}' -> '{local_path}' (bucket='{self.bucket_name}')")
        except Exception as e:
            logger.error(f"S3 download failed for '{remote_path}' in bucket '{self.bucket_name}': {e}")
            raise HTTPException(status_code=500, detail=f"S3 download error: {e}")

    def process_s3_dataset(self, s3_file_path: str, dataset_name: str, user_name: str, private: bool , dataset_tag : str) -> dict:
        try:
            # Create S3 client inside the method
            DatasetConditionChecker().check_s3_size(
                    access_key=self.access_key,
                    secret_key=self.secret_key,
                    endpoint_url=self.endpoint_url,
                    bucket_name=self.bucket_name,
                    s3_path=s3_file_path
                )
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                endpoint_url=self.endpoint_url
            )
            with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
                temp_path = Path(temp_dir)
                logger.info(f"Created temporary directory for S3 download: {temp_path}")
                local_file = temp_path / Path(s3_file_path).name
                self.download_file(s3_file_path, local_file, s3_client)

                dataset_id = self.storage_handler.generate_dataset_id()
                metadata = DatasetMetadata(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name or local_file.stem,
                    dataset_config=None,
                    last_commit=None,
                    last_modified=None,
                    user_name=user_name,
                    private=private,
                    source="s3",
                    created_at=datetime.now().isoformat(),
                    file_path="",
                    summary=summarize_dataset(local_file),
                    dataset_tag = dataset_tag
                )
                stored_path = self.storage_handler.store_dataset(local_file, metadata)
                return {
                    "status": "success",
                    "message": "Dataset from S3 stored successfully.",
                    "dataset_id": dataset_id,
                    "stored_path": stored_path
                }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing S3 dataset: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal error processing S3 dataset: {e}")