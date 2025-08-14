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
            
            # For custom S3 endpoints
            if 'endpoint_url' in s3_config:
                os.environ['AWS_S3_ENDPOINT_URL'] = s3_config['endpoint_url']
                os.environ['AWS_S3_USE_SSL'] = 'false' if 'http://' in s3_config['endpoint_url'] else 'true'
                # Additional settings for custom S3
                os.environ['AWS_S3_ADDRESSING_STYLE'] = 'path'  # Use path-style addressing
                os.environ['AWS_S3_REGION'] = s3_config.get('region', 'us-east-1')
            
            logger.info("Configured ClearML S3 credentials")
        except Exception as e:
            logger.error(f"Failed to configure ClearML S3 credentials: {e}")
    
    def test_s3_access(self, s3_config: dict, s3_path: str) -> bool:
        """Test if ClearML can access the S3 path"""
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ClientError
            
            # Create boto3 client with the same config
            client = boto3.client(
                's3',
                endpoint_url=s3_config['endpoint_url'],
                aws_access_key_id=s3_config['access_key'],
                aws_secret_access_key=s3_config['secret_key'],
                region_name=s3_config.get('region', 'us-east-1')
            )
            
            # Parse S3 path
            bucket_name = s3_config['bucket_name']
            prefix = s3_path.replace(f's3://{bucket_name}/', '')
            
            # Try to list objects
            response = client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                MaxKeys=1
            )
            
            return True
        except Exception as e:
            logger.warning(f"S3 access test failed: {e}")
            return False
    
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
                # Test S3 access first
                can_access_s3 = self.test_s3_access(s3_config, stored_path)
                
                if can_access_s3:
                    # Method 1: Try adding external files with S3 URL
                    try:
                        dataset.add_external_files(
                            source_url=stored_path,
                            dataset_path="."
                        )
                        logger.info(f"Added S3 external files for dataset {metadata.dataset_name}")
                    except Exception as e:
                        logger.warning(f"Failed to add S3 external files: {e}")
                        # Fallback to local registration
                        self._register_local_with_s3_metadata(dataset, metadata, stored_path)
                else:
                    logger.warning("ClearML cannot access S3 endpoint, registering with metadata only")
                    self._register_local_with_s3_metadata(dataset, metadata, stored_path)
                    
            else:
                # For local paths, add files normally
                local_path = stored_path.replace('s3://', '').split('/', 1)[1] if stored_path.startswith('s3://') else stored_path
                local_full_path = str(self.base_dir / metadata.dataset_name / "data")
                
                if Path(local_full_path).exists():
                    if Path(local_full_path).is_file():
                        dataset.add_files(path=local_full_path)
                    elif Path(local_full_path).is_dir():
                        dataset.add_files(path=local_full_path, wildcard="*")
                    logger.info(f"Added local files for dataset {metadata.dataset_name}")
                else:
                    logger.warning(f"Local path {local_full_path} does not exist")
            
            # Upload and finalize
            dataset.upload()
            dataset.finalize()
            logger.info(f"Dataset {metadata.dataset_name} successfully registered with ClearML.")
            
        except Exception as e:
            logger.error(f"Failed to register dataset with ClearML: {e}")
            print(f"ClearML registration error: {traceback.format_exc()}")
            # Don't raise the exception to avoid breaking the main flow
            logger.warning("Dataset stored successfully but ClearML registration failed")
    
    def _register_local_with_s3_metadata(self, dataset, metadata: DatasetMetadata, s3_path: str):
        """Register dataset with local files but include S3 metadata"""
        try:
            # Add local files
            local_data_path = str(self.base_dir / metadata.dataset_name / "data")
            if Path(local_data_path).exists():
                if Path(local_data_path).is_file():
                    dataset.add_files(path=local_data_path)
                elif Path(local_data_path).is_dir():
                    dataset.add_files(path=local_data_path, wildcard="*")
            
            # Add S3 metadata
            dataset.add_tags([f"s3_backup:{s3_path}"])
            dataset.set_metadata("s3_location", s3_path)
            dataset.set_metadata("storage_type", "hybrid_local_s3")
            
            logger.info(f"Registered dataset {metadata.dataset_name} with local files and S3 metadata")
            
        except Exception as e:
            logger.error(f"Failed to register with local files and S3 metadata: {e}")