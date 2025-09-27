import logging
from datetime import datetime
from pathlib import Path
import os

import boto3
from fastapi import HTTPException

from data.data_ingestion.handlers.conditions import DatasetConditionChecker
from data.data_ingestion.handlers.utils import summarize_dataset
from data.data_ingestion.models.metadata import DatasetMetadata
from data.data_ingestion.storage_handler import DatasetStorageHandler
from data.data_validation.validation import DirectoryValidator

logger = logging.getLogger(__name__)


class S3Handler:
    config = [
        {
            "name": "Validate User CSVs",
            "path_pattern": "**/*.csv",
            "validator": "tabular",
        },
        {
            "name": "Validate Transaction Parquet Files",
            "path_pattern": "**/*.parquet",
            "validator": "tabular",
        },
        {
            "name": "Validate for Image files",
            "path_pattern": "**/*.jpg",
            "validator": "image",
        },
        {
            "name": "Validate for Image files(png)",
            "path_pattern": "**/*.png",
            "validator": "image",
        },
    ]
    def __init__(
        self, *, access_key: str, secret_key: str, endpoint_url: str, bucket_name: str
    ):
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name
        logger.info(
            f"S3Handler initialized for endpoint '{endpoint_url}' and bucket '{bucket_name}'."
        )

    def download_file(self, remote_path: str, local_path: Path, s3_client) -> None:
        if not s3_client:
            raise HTTPException(status_code=503, detail="S3 service is not available.")
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3_client.head_object(Bucket=self.bucket_name, Key=remote_path)
            except s3_client.exceptions.NoSuchKey:
                raise ValueError(
                    f"The file or folder '{remote_path}' does not exist in the S3 bucket '{self.bucket_name}'."
                )

            s3_client.download_file(self.bucket_name, remote_path, str(local_path))
            logger.info(
                f"S3 Download: '{remote_path}' -> '{local_path}' (bucket='{self.bucket_name}')"
            )
        except Exception as e:
            logger.error(
                f"S3 download failed for '{remote_path}' in bucket '{self.bucket_name}': {e}"
            )
            raise HTTPException(status_code=500, detail=f"S3 download error: {e}")

    def process_s3_dataset(
        self,
        s3_file_path: str,
        dataset_name: str,
        user_name: str,
        private: bool,
        dataset_tag: str,
        s3_config_target: dict = None,
        clearml_config: dict = None,
    ) -> dict:
        """
        Download an S3 dataset, persist it to S3-mounted temp directory, and return metadata.
        """
        try:
            mount_dataset_name = dataset_name or Path(s3_file_path).stem
            storage_handler = DatasetStorageHandler(mount_dataset_name)
            print(f"storage_handler:{storage_handler}")
            DatasetConditionChecker().check_s3_size(
                access_key=self.access_key,
                secret_key=self.secret_key,
                endpoint_url=self.endpoint_url,
                bucket_name=self.bucket_name,
                s3_path=s3_file_path,
            )
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                endpoint_url=self.endpoint_url,
            )

            temp_dir = storage_handler.temp_dir
            logger.info(f"Using S3-mounted temp directory: {temp_dir}")
            s3_filename = Path(s3_file_path).name
            local_file = temp_dir / s3_filename  # Combine directory with filename

            logger.info(f"temp_dir: {temp_dir}")

            print(s3_file_path)

            from aipmodel.model_registry import CephS3Manager

            manager = CephS3Manager(
                endpoint_url=self.endpoint_url,
                access_key=self.access_key,
                secret_key=self.secret_key,
                bucket_name=self.bucket_name,
            )
            print("0" * 10)
            manager.download(s3_file_path, local_file)
            # self.download_file(s3_file_path, local_file, s3_client)

            dataset_id = storage_handler.generate_dataset_id()
            metadata = DatasetMetadata(
                dataset_id=dataset_id,
                dataset_name=dataset_name or local_file.stem,
                revision="main",
                last_commit=None,
                last_modified=None,
                user_name=user_name,
                private=private,
                source="s3",
                created_at=datetime.now().isoformat(),
                file_path="",
                summary=summarize_dataset(local_file),
                dataset_tag=dataset_tag or "ML",
            )

            
            logger.info(f"Starting validation for dataset at: {local_file}")
            # Ensure the validator is configured to unzip archives, which is the default
            validator = DirectoryValidator(config=self.config, unzip_first=True)
            report = validator.validate(str(local_file))
            logger.info(f"Validation report: {report}")

            error_details = []
            final_status = "success"
            final_message = "Dataset from S3 stored successfully."

            if report.has_failures():
                final_status = "completed_with_errors"
                final_message = "Dataset stored, but some files failed validation and were excluded."
                logger.warning(
                    f"Validation failed for some files in dataset from S3 path '{s3_file_path}'. "
                    "These files will be excluded."
                )

                invalid_file_paths = set()
                for result in report.results:
                    if not result.is_valid:
                        error_details.append({
                            "file": result.file_path,
                            "errors": result.errors,
                            "warnings": result.warnings,
                        })
                        invalid_file_paths.add(result.file_path)

                # NEW: Remove the invalid files from the temporary directory
                logger.info(f"Removing {len(invalid_file_paths)} invalid file(s)...")
                for file_path in invalid_file_paths:
                    try:
                        os.remove(file_path)
                        logger.debug(f"Removed invalid file: {file_path}")
                    except OSError as e:
                        logger.error(f"Error removing invalid file {file_path}: {e}")


            # check if all files are not valid 
            files = [f for f in os.listdir(local_file) if os.path.isfile(os.path.join(local_file, f))]
            if not files:
                final_status = "failed"
                final_message = "All files in the dataset failed validation. No files were stored."
                logger.error(
                    f"All files in dataset from S3 path '{s3_file_path}' failed validation. Nothing to store."
                )
                raise HTTPException(
                    status_code=400,
                    detail="All files in the dataset failed validation. No files were stored.",
                )
            
            logger.info(f"uploading {len(files)} files: {files}")
            stored_path = storage_handler.store_dataset(
                local_file, 
                metadata,
                s3_config=s3_config_target,
                clearml_config=clearml_config,
            )
            
            # --- MODIFIED: Construct final response ---
            response = {
                "status": final_status,
                "message": final_message,
                "dataset_id": dataset_id,
                "stored_path": stored_path,
            }
            if error_details:
                response["validation_errors"] = error_details
            
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing S3 dataset: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Internal error processing S3 dataset: {e}"
            )
        finally:
            pass
            # storage_handler.unmount()
