import json
import logging
import os
import shutil
import subprocess
import traceback
import uuid
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig # Import TransferConfig
from clearml import Task, Dataset as ClearMLDataset
from fastapi import HTTPException

# Assuming this model is defined in your project
from data.data_ingestion.models.metadata import DatasetMetadata

logger = logging.getLogger(__name__)

# --- These paths remain unchanged as they are specific to your environment ---
PVC_BASE_DIR = Path(__file__).resolve().parent.parent / "PV_Datasets"
METADATA_DIR = PVC_BASE_DIR / "metadata"
TEMP_BASE_DIR = PVC_BASE_DIR / "temp"

class DatasetStorageHandler:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.main_dir = PVC_BASE_DIR / dataset_name
        self.base_dir = PVC_BASE_DIR / dataset_name / "main"
        self.metadata_dir = self.base_dir / "metadata"
        self.temp_dir = TEMP_BASE_DIR / dataset_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    # --- Helper methods remain unchanged ---
    def unmount(self):
        try:
            subprocess.run(["fusermount", "-u", str(self.mount_point)], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to unmount {self.mount_point}: {e}")

    def cleanup_temp(self):
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir {self.temp_dir}: {e}")

    def generate_dataset_id(self) -> str:
        return str(uuid.uuid4())

    # --- REFACTORED store_dataset METHOD ---
    def store_dataset(self, source_path: Path, metadata: DatasetMetadata, s3_config: dict = None, clearml_config: dict = None) -> str:
        dataset_dir = self.base_dir / metadata.dataset_name
        try:
            # 1. Local File Management (No changes needed here)
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            dataset_dir.mkdir(exist_ok=True)

            if source_path.is_dir():
                stored_path = dataset_dir / "data"
                shutil.move(str(source_path), str(stored_path))
            elif source_path.is_file():
                stored_path = dataset_dir / source_path.name
                shutil.move(str(source_path), str(stored_path))
            else:
                raise FileNotFoundError(f"Source path {source_path} does not exist or is not a file/directory.")

            metadata.file_path = str(stored_path)
            self.save_metadata(metadata)
            logger.info(f"Dataset {metadata.dataset_id} stored locally at {stored_path}")

            # 2. Resilient S3 Upload using Boto3 (This is the new logic)
            if s3_config:
                logger.info("S3 configuration provided, starting resilient upload...")
                try:
                    s3_client = boto3.client(
                        's3',
                        endpoint_url=s3_config["endpoint_url"],
                        aws_access_key_id=s3_config["access_key"],
                        aws_secret_access_key=s3_config["secret_key"]
                    )

                    transfer_config = TransferConfig(
                        multipart_threshold=100 * 1024 * 1024,  # 100MB threshold for multipart
                        max_concurrency=10,                      # Up to 10 parallel uploads
                        multipart_chunksize=25 * 1024 * 1024,    # 25MB chunks
                        use_threads=True
                    )
                    
                    # The remote path in S3 where the dataset will be stored
                    s3_remote_path = f"Datasets/{metadata.dataset_name}"
                    
                    # NOTE: We upload `self.main_dir` to include the metadata file and the data in its structure.
                    # If you only want to upload the data itself, change `self.main_dir` to `stored_path`.
                    upload_source_dir = self.main_dir
                    
                    logger.info(f"Uploading directory '{upload_source_dir}' to 's3://{s3_config['bucket_name']}/{s3_remote_path}'")

                    for root, _, files in os.walk(upload_source_dir):
                        for filename in files:
                            local_file_path = Path(root) / filename
                            relative_path = local_file_path.relative_to(upload_source_dir)
                            s3_key = str(Path(s3_remote_path) / relative_path)

                            logger.debug(f"Uploading file {local_file_path} to {s3_key}...")
                            s3_client.upload_file(
                                Filename=str(local_file_path),
                                Bucket=s3_config['bucket_name'],
                                Key=s3_key,
                                Config=transfer_config
                            )

                    logger.info(f"✅ Uploaded dataset {metadata.dataset_name} to S3 successfully.")

                    # 3. Register with ClearML
                    s3_main_path = f's3://{s3_config["bucket_name"]}/{s3_remote_path}'
                    self.register_with_clearml(metadata, str(self.base_dir), s3_config, s3_main_path, clearml_config)

                except Exception as e:
                    logger.error(f"Failed to upload dataset to S3: {e}", exc_info=True)
                    # Re-raise the exception to be caught by the outer block
                    raise

            return str(stored_path)
        
        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Failed to store dataset {metadata.dataset_id}: {e}")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            raise HTTPException(status_code=500, detail=f"Error storing dataset file: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during dataset processing: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during processing: {e}")
        finally:
            self.cleanup_temp()

    # --- Metadata methods remain unchanged ---
    def save_metadata(self, metadata: DatasetMetadata):
        metadata_file = self.metadata_dir / f"{metadata.dataset_id}.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata.dict(), f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save metadata for {metadata.dataset_id}: {e}")
            raise HTTPException(status_code=500, detail="Error saving dataset metadata.")

    def load_metadata(self, dataset_id: str) -> DatasetMetadata:
        metadata_file = self.metadata_dir / f"{dataset_id}.json"
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            return DatasetMetadata(**data)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Metadata not found for this dataset.")
        except Exception as e:
            logger.error(f"Failed to load metadata for {dataset_id}: {e}")
            raise HTTPException(status_code=500, detail="Error reading dataset metadata.")

    # --- list_s3_files already used boto3, so it remains unchanged ---
    def list_s3_files(self, s3_config, bucket, prefix):
        files = []
        s3_client = boto3.client(
            's3',
            endpoint_url=s3_config['endpoint_url'],
            aws_access_key_id=s3_config['access_key'],
            aws_secret_access_key=s3_config['secret_key']
        )
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                files.append(f"s3://{bucket}/{obj['Key']}")
        return files

    # --- REFACTORED register_with_clearml METHOD (Best Practice) ---
    def register_with_clearml(self, metadata: DatasetMetadata, stored_path: str, s3_config: dict, s3_main_path: str, clearml_config: dict):
        try:
            # Best Practice: Use the SDK to set credentials instead of environment variables.
            # This is safer, especially in multi-threaded environments.
            Task.set_credentials(
                # You might need to add your api_host here if not default
                # api_host="https://api.clear.ml", 
                api_key=clearml_config['access_key'],
                api_secret=clearml_config['secret_key']
            )

            logger.info("Creating ClearML dataset...")
            dataset = ClearMLDataset.create(
                dataset_name=metadata.dataset_name,
                dataset_project=f"DatasetIngestion_{metadata.user_name}",
                description=f"Dataset ID: {metadata.dataset_id}\nSource S3 Path: {s3_main_path}"
            )
            
            # Use ClearML's S3 configuration to point to the external data
            dataset.add_external_files(
                source_url=s3_main_path,
                aws_key=s3_config['access_key'],
                aws_secret=s3_config['secret_key'],
                aws_endpoint=s3_config['endpoint_url']
            )

            ds_id = dataset.id
            logger.info(f"ClearML dataset created with ID: {ds_id}")

            dataset.add_tags([
                f"s3_path:{s3_main_path}",
                f"dataset_tag:{metadata.dataset_tag}"
            ])
            
            all_metadata = metadata.dict()
            all_metadata.update({
                "s3_path": s3_main_path,
                "clearml_dataset_id": ds_id
            })
            dataset.set_metadata(all_metadata)

            # Uploads only the metadata and file pointers to ClearML server. The data remains in your S3.
            dataset.upload()
            dataset.finalize()
            logger.info(f"Dataset {metadata.dataset_name} successfully registered with ClearML. ID={ds_id}")

        except Exception as e:
            logger.error(f"Failed to register dataset with ClearML: {e}", exc_info=True)
            # It's better to re-raise here so the main function knows about the failure.
            raise