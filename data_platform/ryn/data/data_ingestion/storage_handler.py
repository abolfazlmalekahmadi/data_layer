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
from clearml import Dataset as ClearMLDataset

logger = logging.getLogger(__name__)
PVC_BASE_DIR = Path(__file__).resolve().parent.parent / "PV_Datasets"
METADATA_DIR = PVC_BASE_DIR / "metadata"
TEMP_DIR = PVC_BASE_DIR 
ACCESS_KEY="AKIAJYO3GANY5GC44BUQ"
SECRET_KEY="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"
class DatasetStorageHandler:
    
    def __init__(self):
        self.metadata_dir = METADATA_DIR
        self.temp_dir = TEMP_DIR
        self.access_key=ACCESS_KEY
        self.secret_key=SECRET_KEY
    def generate_dataset_id(self) -> str:
        return str(uuid.uuid4())

    def store_dataset(self, source_path: Path, metadata: DatasetMetadata) -> str:
        print(metadata.dataset_name)
        self.mount_point = Path(f"/mnt/s3/{metadata.dataset_name}")
        self.base_dir = self.mount_point / "main"
        self.metadata_dir = self.base_dir / "metadata"
        self.temp_dir = self.mount_point / "main"
        # create_repo()
        lakefs_clt = lakefs_handler(metadata.dataset_name,self.access_key,self.secret_key)
        lakefs_clt.mount_repo(str(self.mount_point))
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Storage directories initialized. Persistent storage: {self.base_dir}, Temp storage: {self.temp_dir}")
        except OSError as e:
            logger.critical(f"CRITICAL: Could not create base directories: {e}")

        dataset_dir = self.base_dir / metadata.dataset_name
        try:
            dataset_dir.mkdir(exist_ok=True)
            if source_path.is_dir():
                stored_path = dataset_dir / "data"
                if stored_path.exists():
                    shutil.rmtree(stored_path)
                shutil.copytree(source_path, stored_path)
            elif source_path.is_file():
                stored_path = dataset_dir / source_path.name
                shutil.copy2(source_path, stored_path)
            else:
                raise FileNotFoundError(f"Source path {source_path} does not exist or is not a file/directory.")

            metadata.file_path = str(stored_path)
            # raise ValueError(metadata)
            self.save_metadata(metadata)
            logger.info(f"Dataset {metadata.dataset_id} stored at {stored_path}")
            
            clearml_path = f"s3://{metadata.dataset_name}/main"
            self.register_with_clearml(metadata, clearml_path)
            result = subprocess.run([f"fusermount","-u",str(self.mount_point)], check=True, capture_output=True, text=True)

            return str(stored_path)

        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Failed to store dataset {metadata.dataset_id}: {e}")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            raise HTTPException(status_code=500, detail=f"Error storing dataset file: {e}")

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
        
        
    def register_with_clearml(self, metadata: DatasetMetadata, stored_path: Path):
        try:
            dataset = ClearMLDataset.create(
                dataset_name=metadata.dataset_name,
                dataset_project="importing",
                # dataset_tags=[metadata.dataset_type] if metadata.dataset_type else [],
                description=f"Dataset ID: {metadata.dataset_id}\nStored path: {stored_path}"
            )

            
            dataset.add_external_files(
                source_url=stored_path,
                dataset_path=metadata.dataset_name
            )


            # Also attach metadata JSON
            metadata_json_path = self.metadata_dir / f"{metadata.dataset_id}.json"
            if metadata_json_path.exists():
                dataset.add_files(path=str(metadata_json_path))

            dataset.upload()
            dataset.finalize()

            logger.info(f"Dataset {metadata.dataset_name} registered with ClearML.")
        except Exception as e:
            logger.error(f"Failed to register dataset with ClearML: {e}")