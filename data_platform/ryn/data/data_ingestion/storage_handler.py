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
    
    def configure_clearml_s3(self, s3_config: dict):
        """Configure ClearML to work with custom S3 endpoint"""
        try:
            # Set environment variables for ClearML S3 access
            os.environ['AWS_ACCESS_KEY_ID'] = s3_config['access_key']
            os.environ['AWS_SECRET_ACCESS_KEY'] = s3_config['secret_key']
            os.environ['AWS_DEFAULT_REGION'] = s3_config.get('region', 'us-east-1')
            
            # For custom S3 endpoints, you might need to set additional configs
            if 'endpoint_url' in s3_config:
                os.environ['AWS_S3_ENDPOINT_URL'] = s3_config['endpoint_url']
                # For non-AWS S3 services, disable SSL verification if needed
                os.environ['AWS_S3_USE_SSL'] = 'false' if 'http://' in s3_config['endpoint_url'] else 'true'
            
            logger.info("Configured ClearML S3 credentials")
        except Exception as e:
            logger.error(f"Failed to configure ClearML S3 credentials: {e}")
    
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

                    s3_remote_path = f"Datasets/{metadata.dataset_name}"
                    manager.upload(str(self.main_dir), s3_remote_path)
                    logger.info(f"Uploaded dataset {metadata.dataset_name} to S3 at {s3_config['bucket_name']}/{s3_remote_path}")

                    # Configure ClearML with S3 credentials
                    self.configure_clearml_s3(s3_config)
                    
                    # Register with ClearML using S3 path
                    clearml_s3_path = f's3://{s3_config["bucket_name"]}/{s3_remote_path}'
                    print("Register to clearml")
                    print("*"*50)
                    print(f"ClearML S3 path: {clearml_s3_path}")
                    
                    self.register_with_clearml(metadata, clearml_s3_path, s3_config)
                    
                except Exception as e:
                    logger.error(f"Failed to upload dataset to S3: {e}")
                    print(traceback.print_exc())
                    # Fallback to local path registration
                    self.register_with_clearml(metadata, str(stored_path))

            else:
                # Register with local path if no S3 config
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
    
    def register_with_clearml(self, metadata: DatasetMetadata, stored_path: str, s3_config: dict = None):
        try:
            dataset = ClearMLDataset.create(
                dataset_name=metadata.dataset_name,
                dataset_project="importing",
                description=f"Dataset ID: {metadata.dataset_id}\nStored path: {stored_path}"
            )
            
            if s3_config and stored_path.startswith('s3://'):
                # Method 1: Try adding external files with S3 URL
                try:
                    dataset.add_external_files(
                        source_url=stored_path,
                        dataset_path="."  # Use root path instead of dataset name
                    )
                    logger.info(f"Added S3 external files for dataset {metadata.dataset_name}")
                except Exception as e:
                    logger.warning(f"Failed to add S3 external files, trying alternative method: {e}")
                    
                    # Method 2: Try adding files using wildcard pattern
                    try:
                        # Add files with wildcard to include all files in the S3 directory
                        dataset.add_external_files(
                            source_url=f"{stored_path}/*",
                            dataset_path="."
                        )
                        logger.info(f"Added S3 external files with wildcard for dataset {metadata.dataset_name}")
                    except Exception as e2:
                        logger.warning(f"Wildcard method also failed: {e2}")
                        
                        # Method 3: Add metadata to indicate S3 location without trying to sync files
                        dataset.add_tags([f"s3_location:{stored_path}"])
                        logger.info(f"Added S3 location as tag for dataset {metadata.dataset_name}")
            else:
                # For local paths, add files normally
                if Path(stored_path).is_file():
                    dataset.add_files(path=stored_path)
                elif Path(stored_path).is_dir():
                    dataset.add_files(path=stored_path, wildcard="*")
                logger.info(f"Added local files for dataset {metadata.dataset_name}")
            
            # Upload and finalize
            dataset.upload()
            dataset.finalize()
            logger.info(f"Dataset {metadata.dataset_name} successfully registered with ClearML.")
            
        except Exception as e:
            logger.error(f"Failed to register dataset with ClearML: {e}")
            print(f"ClearML registration error: {traceback.format_exc()}")
            # Don't raise the exception to avoid breaking the main flow
            logger.warning("Dataset stored successfully but ClearML registration failed")