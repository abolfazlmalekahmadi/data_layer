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
                    from aipmodel.CephS3Manager import CephS3Manager
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

                    clearml_s3_path = f's3://{s3_config["bucket_name"]}/{s3_remote_path}'

                    self.register_with_clearml(metadata, clearml_s3_path, use_external=True)
                    
                    
                    # manager.upload(str(self.main_dir), remote_path)
                    logger.info(f"Uploaded dataset {metadata.dataset_name} to S3 at {s3_config['bucket_name']}/{metadata.dataset_name}")
                except Exception as e:
                    logger.error(f"Failed to upload dataset to S3: {e}")
                    print(traceback.print_exc())

            # register with ClearML (non-blocking ideally)
            # clearml_path = f's3://{s3_config["bucket_name"]}/{remote_path}'
            self.register_with_clearml(metadata, clearml_path)
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
        
    def register_with_clearml(self, metadata: DatasetMetadata, stored_path: str, use_external: bool = True):
        """
        Register dataset with ClearML.
        
        Args:
            metadata: Dataset metadata
            stored_path: Path to the stored dataset (local or S3 URL)
            use_external: If True, use add_external_files for S3 URLs. If False, upload local files.
        """
        try:
            # Create unique dataset name with timestamp to avoid conflicts
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            clearml_dataset_name = f"{metadata.dataset_name}_{timestamp}"
            
            dataset = ClearMLDataset.create(
                dataset_name=clearml_dataset_name,
                dataset_project="DatasetIngestion",  # Use consistent project name
                description=f"Dataset ID: {metadata.dataset_id}\nSource: {metadata.source}\nStored path: {stored_path}\nUser: {metadata.user_name}"
            )
            
            logger.info(f"Created ClearML dataset: {clearml_dataset_name}")
            logger.info(f"Stored path: {stored_path}")
            
            if use_external and stored_path.startswith('s3://'):
                # For S3 URLs, use external files (no upload needed)
                
                dataset.add_external_files(
                    source_url=stored_path,
                    dataset_path="/"  # Root path in the dataset
                )
                logger.info(f"Added external S3 files to ClearML dataset from: {stored_path}")
            # else:
            #     # For local paths, upload files to ClearML
            #     if Path(stored_path).exists():
            #         dataset.add_files(path=stored_path)
            #         dataset.upload()
            #         logger.info(f"Uploaded local files to ClearML dataset from: {stored_path}")
            #     else:
            #         logger.warning(f"Local path does not exist: {stored_path}")
            #         return
            
            # Add metadata as tags and custom properties
            dataset.add_tags([
                metadata.source,
                f"user:{metadata.user_name}",
                f"private:{metadata.private}"
            ])
            
            # Set custom properties
            dataset.set_metadata({
                'dataset_id': metadata.dataset_id,
                'original_name': metadata.dataset_name,
                'source': metadata.source,
                'user_name': metadata.user_name,
                'private': str(metadata.private),
                'dataset_config': metadata.dataset_config or '',
                'dataset_tag': metadata.dataset_tag or '',
                'created_at': metadata.created_at
            })
            
            dataset.finalize()
            logger.info(f"Dataset {clearml_dataset_name} successfully registered with ClearML (ID: {dataset.id})")
            
        except Exception as e:
            logger.error(f"Failed to register dataset with ClearML: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            # Don't raise the exception to avoid failing the main process
            pass