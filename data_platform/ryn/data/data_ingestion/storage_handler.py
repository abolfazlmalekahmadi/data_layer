import os
import shutil
import uuid
import json
from pathlib import Path
from fastapi import HTTPException
import logging
from data.data_ingestion.models.metadata import DatasetMetadata
from data.data_ingestion.lakefs.lakefs_handler import lakefs_handler
import subprocess
from botocore.config import Config
import boto3

from clearml import Dataset as ClearMLDataset
import traceback
logger = logging.getLogger(__name__)
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
    
    def store_dataset(self, source_path: Path, metadata: DatasetMetadata, s3_config: dict = None) -> str:
        dataset_dir = self.base_dir / metadata.dataset_name
        print(source_path)
        try:
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
            logger.info(f"Dataset {metadata.dataset_id} stored at {stored_path}")
            
            s3_remote_path = None
            clearml_path = str(stored_path)
            if s3_config:
                try:
                    from aipmodel.model_registry import CephS3Manager
                    print("llll"*100)
                    print(s3_config)
                    manager = CephS3Manager(
                        endpoint_url=s3_config["endpoint_url"],
                        access_key=s3_config["access_key"],
                        secret_key=s3_config["secret_key"],
                        bucket_name=s3_config["bucket_name"]
                    )

                    # bucket = s3_config.get("bucket")
                    # remote_path = f"Datasets/{metadata.dataset_name}"
                    # clearml_path = f"s3://{s3_config['bucket_name']}/{remote_path}"
                    # print("he"*100)
                    # print(clearml_path)
                    s3_remote_path = f"Datasets/{metadata.dataset_name}"

                    manager.upload(str(self.main_dir), s3_remote_path)

                    logger.info(f"Uploaded dataset {metadata.dataset_name} to S3 at {s3_config['bucket_name']}/{s3_remote_path}")

                    # Register with ClearML using S3 path

                    s3_main_path = f's3://{s3_config["bucket_name"]}/{s3_remote_path}'

                    print("Register to clearml")
                    print("*"*50)
                    self.register_with_clearml(metadata, str(self.base_dir), s3_config, s3_main_path)
                    
                    
                    # manager.upload(str(self.main_dir), remote_path)
                    logger.info(f"Uploaded dataset {metadata.dataset_name} to S3 at {s3_config['bucket_name']}/{metadata.dataset_name}")
                except Exception as e:
                    logger.error(f"Failed to upload dataset to S3: {e}")
                    print(traceback.print_exc())

            # register with ClearML (non-blocking ideally)
            # clearml_path = f's3://{s3_config["bucket_name"]}/{remote_path}'
            # self.register_with_clearml(metadata, clearml_path)
            return str(stored_path)
        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Failed to store dataset {metadata.dataset_id}: {e}")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            raise HTTPException(status_code=500, detail=f"Error storing dataset file: {e}")
        finally:
            # clean up temp to save PVC space
            try:
                self.cleanup_temp()
            except Exception:
                pass
    
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
        
    def register_with_clearml(self, metadata: DatasetMetadata, stored_path: str, s3_config, s3_main_path: str):
        print(metadata)
        print(stored_path)
        print(s3_config)
        print(s3_main_path)
        try:
            os.environ['AWS_ACCESS_KEY_ID'] = s3_config['access_key']
            os.environ['AWS_SECRET_ACCESS_KEY'] = s3_config['secret_key']
            os.environ['AWS_S3_ENDPOINT_URL'] = s3_config['endpoint_url']
            
            dataset = ClearMLDataset.create(
                dataset_name=metadata.dataset_name,
                dataset_project="importing",
                description=f"Dataset ID: {metadata.dataset_id}\nStored path: {stored_path}"
            )
            dataset.add_external_files(
                source_url=stored_path,
                dataset_path=metadata.dataset_name
            )
            
            dataset.add_tags([
                f"s3_path:{s3_main_path}",
                f"dataset_tag:{metadata.dataset_tag}"
                ])
            dataset.set_metadata({"s3_path": s3_main_path})
            all_metadata = metadata.dict()
            dataset.set_metadata(all_metadata)

            dataset.upload()
            dataset.finalize()
            logger.info(f"Dataset {metadata.dataset_name} registered with ClearML.")
        except Exception as e:
            logger.error(f"Failed to register dataset with ClearML: {e}")