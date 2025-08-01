import openml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import os
import shutil
import kagglehub
from pathlib import Path
import pandas as pd
from datasets import load_dataset , get_dataset_infos
from huggingface_hub import HfApi
import json
from typing import Optional, List
import logging
from datetime import datetime
from fastapi.responses import FileResponse
import uuid
from datetime import datetime
import tempfile
import zipfile


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Ingestion API",
    description="API to download and store datasets from S3, Kaggle, Hugging Face, and OpenML into a persistent volume using a centralized temporary directory.",
    version="0.0.2"
)

PVC_BASE_DIR = Path("PV_Datasets")
METADATA_DIR = PVC_BASE_DIR / "metadata"
TEMP_DIR = PVC_BASE_DIR / "temp"

try:
    PVC_BASE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Storage directories initialized. Persistent storage: {PVC_BASE_DIR}, Temp storage: {TEMP_DIR}")
except OSError as e:
    logger.critical(f"CRITICAL: Could not create base directories: {e}")


class DatasetMetadata(BaseModel):
    """Schema for the final, persisted metadata of a dataset."""
    dataset_id: str
    dataset_name: str
    dataset_config: Optional[str] = None
    last_commit: Optional[str] = None
    last_modified: Optional[str] = None 
    user_name: str
    private: bool = False
    version: int = 0
    temp: bool = True
    source: str
    created_at: str
    file_path: str
    summary: Optional[str] = None

class S3DownloadRequest(BaseModel):
    s3_file_path: str
    user_name: str
    dataset_name: Optional[str] = None
    access_key:str
    secret_key:str
    endpoint_url:str
    bucket_name:str
    private: bool = False

    

class KaggleDownloadRequest(BaseModel):
    dataset_id: str
    user_name: str 
    dataset_name: Optional[str] = None
    private: bool = False
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None


class HuggingFaceDownloadRequest(BaseModel):
    dataset_name: str
    dataset_config: Optional[str] = None
    user_name: str
    dataset_name_override: Optional[str] = None
    private: bool = False

class OpenMLDownloadRequest(BaseModel):
    dataset_id: int
    user_name: str
    dataset_name: Optional[str] = None
    private: bool = False

class ListDatasetsRequest(BaseModel):
    user_name: Optional[str] = None
    private_only: bool = False

class DownloadDatasetRequest(BaseModel):
    dataset_id: str
    user_name: str



def _save_request_info_to_temp(temp_path: Path, request_info: dict):
    try:
        with open(temp_path / "_request_info.json", 'w') as f:
            request_info["timestamp"] = datetime.now().isoformat()
            json.dump(request_info, f, indent=4)
    except (IOError, TypeError) as e:
        logger.warning(f"Could not save request info to temp directory {temp_path}: {e}")

def summarize_dataset(path: Path) -> str:
    summary_parts = []
    try:
        if path.is_file():
             if path.suffix == '.csv':
                df = pd.read_csv(path, nrows=5)
                return f"CSV file: {path.name} with columns: {list(df.columns)}"
             else:
                return f"File: {path.name}"
        csv_files = list(path.rglob("*.csv"))
        for csv in csv_files[:3]:
            try:
                df = pd.read_csv(csv, nrows=5)
                summary_parts.append(f"- CSV: {csv.relative_to(path)} (Cols: {list(df.columns)})")
            except Exception as e:
                summary_parts.append(f"- Could not read CSV '{csv.name}': {e}")
    except Exception as e:
        logger.warning(f"Could not generate summary for {path}: {e}")
        return f"Could not generate summary: {e}"
    return "\n".join(summary_parts) if summary_parts else "No summary available."

class DatasetStorageHandler:
    def __init__(self):
        self.base_dir = PVC_BASE_DIR
        self.metadata_dir = METADATA_DIR
    
    def generate_dataset_id(self) -> str:
        return str(uuid.uuid4())
    
    def save_metadata(self, metadata: DatasetMetadata) -> None:
        metadata_file = self.metadata_dir / f"{metadata.dataset_id}.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata.dict(), f, indent=4)
            logger.info(f"Metadata saved for dataset {metadata.dataset_id}")
        except (IOError, TypeError) as e:
            logger.error(f"Error saving metadata for dataset {metadata.dataset_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save metadata: {e}")

    def load_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        metadata_file = self.metadata_dir / f"{dataset_id}.json"
        if not metadata_file.exists():
            return None
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            return DatasetMetadata(**data)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading metadata for dataset {dataset_id}: {e}")
            return None
    
    def list_datasets(self, user_name: Optional[str] = None, private_only: bool = False) -> List[DatasetMetadata]:
        datasets = []
        try:
            for filename in os.listdir(self.metadata_dir):
                if filename.endswith('.json'):
                    dataset_id = filename[:-5]
                    metadata = self.load_metadata(dataset_id)
                    if metadata:
                        if user_name and metadata.user_name != user_name:
                            continue
                        if private_only and not metadata.private:
                            continue
                        datasets.append(metadata)
        except OSError as e:
            logger.error(f"Could not list files in metadata directory: {e}")
            raise HTTPException(status_code=500, detail="Error accessing dataset metadata storage.")
        return datasets
    
    def store_dataset(self, source_path: Path, metadata: DatasetMetadata) -> str:
        dataset_dir = self.base_dir / metadata.dataset_id
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
            self.save_metadata(metadata)
            logger.info(f"Dataset {metadata.dataset_id} stored at {stored_path}")
            return str(stored_path)
        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Failed to store dataset {metadata.dataset_id}: {e}")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            raise HTTPException(status_code=500, detail=f"Error storing dataset file: {e}")

    def get_dataset_path(self, dataset_id: str) -> Optional[str]:
        metadata = self.load_metadata(dataset_id)
        if metadata and metadata.file_path and Path(metadata.file_path).exists():
            return metadata.file_path
        return None

storage_handler = DatasetStorageHandler()

class S3Handler:
    def __init__(self, *, access_key: str, secret_key: str, endpoint_url: str, bucket_name: str):
        self.bucket_name = bucket_name
        try:
            # Build a client for this request
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                endpoint_url=endpoint_url
            )
            # Do NOT log secrets
            logger.info(f"S3 client initialized for endpoint '{endpoint_url}' and bucket '{bucket_name}'.")
        except Exception as e:
            logger.critical(f"Failed to initialize S3 client: {e}")
            self.s3_client = None

    def download_file(self, remote_path: str, local_path: Path) -> None:
        if not self.s3_client:
            raise HTTPException(status_code=503, detail="S3 service is not available.")
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket_name, remote_path, str(local_path))
            logger.info(f"S3 Download: '{remote_path}' -> '{local_path}' (bucket='{self.bucket_name}')")
        except Exception as e:
            logger.error(f"S3 download failed for '{remote_path}' in bucket '{self.bucket_name}': {e}")
            raise HTTPException(status_code=500, detail=f"S3 download error: {e}")

    def process_s3_dataset(self, s3_file_path: str, dataset_name: Optional[str], user_name: str, private: bool) -> dict:
        try:
            with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
                temp_path = Path(temp_dir)
                logger.info(f"Created temporary directory for S3 download: {temp_path}")

                # Save only non-sensitive request info
                _save_request_info_to_temp(
                    temp_path,
                    {
                        "source": "s3",
                        "s3_file_path": s3_file_path,
                        "user_name": user_name,
                        "bucket_name": self.bucket_name
                    }
                )

                local_file = temp_path / Path(s3_file_path).name
                self.download_file(s3_file_path, local_file)

                dataset_id = storage_handler.generate_dataset_id()
                metadata = DatasetMetadata(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name or local_file.stem,
                    dataset_config=None,
                    last_commit = None,
                    last_modified = None ,
                    user_name=user_name,
                    private=private,
                    source="s3",
                    created_at=datetime.now().isoformat(),
                    file_path="",
                    summary=summarize_dataset(local_file)
                )
                stored_path = storage_handler.store_dataset(local_file, metadata)
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

def process_kaggle_dataset(request: KaggleDownloadRequest) -> dict:
    """
    Download from Kaggle into a temp cache and then store permanently.
    Handles authentication by prioritizing credentials passed in the request.
    """
    original_kaggle_user = os.environ.get("KAGGLE_USERNAME")
    original_kaggle_key = os.environ.get("KAGGLE_KEY")

    try:
        if request.kaggle_username and request.kaggle_key:
            logger.info("Using Kaggle credentials provided in the API request.")
            os.environ["KAGGLE_USERNAME"] = request.kaggle_username
            os.environ["KAGGLE_KEY"] = request.kaggle_key
        elif original_kaggle_user and original_kaggle_key:
            logger.info("Using Kaggle credentials from server environment variables.")
        else:
            raise HTTPException(
                status_code=400,
                detail="Kaggle credentials not provided. Please include 'kaggle_username' and 'kaggle_key' in your request, or set them as server environment variables."
            )

        with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Created temporary directory for Kaggle download: {temp_path}")
            
            _save_request_info_to_temp(temp_path, {"source": "kaggle", "dataset_id": request.dataset_id, "user_name": request.user_name})

            logger.info(f"Downloading Kaggle dataset: {request.dataset_id}")
            download_path = Path(kagglehub.dataset_download(request.dataset_id))
            
            ds_id = storage_handler.generate_dataset_id()
            metadata = DatasetMetadata(
                dataset_id=ds_id,
                dataset_name=request.dataset_name or request.dataset_id.replace("/", "_"),
                dataset_config=None,
                last_commit = None,
                last_modified = None ,
                user_name=request.user_name,
                private=request.private,
                source="kaggle",
                created_at=datetime.now().isoformat(),
                file_path="",
                summary=summarize_dataset(download_path)
            )
            
            stored_path = storage_handler.store_dataset(download_path, metadata)
            
            return {
                "status": "success",
                "message": f"Kaggle dataset '{request.dataset_id}' stored.",
                "dataset_id": ds_id,
                "stored_path": stored_path,
            }
    except Exception as e:
        logger.error(f"Error processing Kaggle dataset '{request.dataset_id}': {e}", exc_info=True)
        if "403" in str(e) or "404" in str(e) or "401" in str(e):
            raise HTTPException(
                status_code=403,
                detail=f"Access denied for Kaggle dataset '{request.dataset_id}'. Please check the following: 1) Your 'kaggle_username' and 'kaggle_key' are correct. 2) You have accepted the dataset's rules on the Kaggle website. 3) The dataset is public or you have permission to access it."
            )
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred with the Kaggle dataset: {e}")
    finally:
        if original_kaggle_user:
            os.environ["KAGGLE_USERNAME"] = original_kaggle_user
        elif "KAGGLE_USERNAME" in os.environ:
            del os.environ["KAGGLE_USERNAME"]

        if original_kaggle_key:
            os.environ["KAGGLE_KEY"] = original_kaggle_key
        elif "KAGGLE_KEY" in os.environ:
            del os.environ["KAGGLE_KEY"]

def process_huggingface_dataset(dataset_name: str, dataset_config: Optional[str], dataset_name_override: Optional[str], user_name: str, private: bool) -> dict:
    try:
        DatasetConditionChecker().check_huggingface_size(dataset_name, dataset_config)
        with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Created temporary directory for Hugging Face download: {temp_path}")
            _save_request_info_to_temp(temp_path, {"source": "huggingface", "dataset_name": dataset_name, "user_name": user_name})
            logger.info(f"Downloading Hugging Face dataset '{dataset_name}'")
            ds = load_dataset(dataset_name, dataset_config, cache_dir=str(temp_path / "cache"))
            api = HfApi()

            # Latest commit on the default branch (usually "main")
            info = api.repo_info(dataset_name, repo_type="dataset")
            local_dataset_dir = temp_path / "huggingface_dataset"
            ds.save_to_disk(str(local_dataset_dir))
            ds_id = storage_handler.generate_dataset_id()
            metadata = DatasetMetadata(
                dataset_id=ds_id,
                # dataset_name=dataset_name_override or dataset_name.replace('/', '_'),
                dataset_name=dataset_name.replace('/', '_'),
                dataset_config=dataset_config,
                last_commit = info.sha,
                last_modified = info.last_modified.isoformat(),
                user_name=user_name,
                private=private,
                source="huggingface",
                created_at=datetime.now().isoformat(),
                file_path="",
                summary=f"Hugging Face dataset with splits: {list(ds.keys())}"
            )
            stored_path = storage_handler.store_dataset(local_dataset_dir, metadata)
            return {
                "status": "success",
                "message": f"Hugging Face dataset '{dataset_name}' stored.",
                "dataset_id": ds_id,
                "stored_path": stored_path
            }
    except Exception as e:
        logger.error(f"Error processing Hugging Face dataset '{dataset_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error with Hugging Face dataset: {e}")

def process_openml_dataset(dataset_id: int, dataset_name: Optional[str], user_name: str, private: bool) -> dict:
    try:
        DatasetConditionChecker().check_openml_size(dataset_id)
        with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Created temporary directory for OpenML download: {temp_path}")
            _save_request_info_to_temp(temp_path, {"source": "openml", "dataset_id": dataset_id, "user_name": user_name})
            logger.info(f"Downloading OpenML dataset ID={dataset_id}")
            dataset = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=True)
            X, y, _, _ = dataset.get_data(dataset_format="dataframe")
            df = pd.concat([X, y], axis=1) if y is not None else X
            local_path = temp_path / f"{dataset.name}.csv"
            df.to_csv(local_path, index=False)
            ds_id = storage_handler.generate_dataset_id()
            metadata = DatasetMetadata(
                dataset_id=ds_id,
                dataset_name=dataset_name or dataset.name,
                dataset_config= str(dataset.version),
                last_commit = str(dataset.version_label),
                last_modified = dataset.upload_date,
                user_name=user_name,
                private=private,
                source="openml",
                created_at=datetime.now().isoformat(),
                file_path="",
                summary=f"Name: {dataset.name}. Instances: {dataset.qualities.get('NumberOfInstances')}. Features: {dataset.qualities.get('NumberOfFeatures')}."
            )
            stored_path = storage_handler.store_dataset(local_path, metadata)
            return {
                "status": "success",
                "message": f"OpenML dataset '{dataset_id}' stored.",
                "dataset_id": ds_id,
                "stored_path": stored_path,
            }
    except Exception as e:
        logger.error(f"Error processing OpenML dataset '{dataset_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error with OpenML dataset: {e}")

@app.post("/s3-to-pvc", summary="Ingest from S3", tags=["Ingestion"])
async def s3_to_pvc(request: S3DownloadRequest):
    handler = S3Handler(
        access_key=request.access_key,
        secret_key=request.secret_key,
        endpoint_url=request.endpoint_url,
        bucket_name=request.bucket_name,
    )
    return handler.process_s3_dataset(
        request.s3_file_path,
        request.dataset_name,
        request.user_name,
        request.private
    )



class DatasetConditionChecker:
    def __init__(self, temp_dir: Path = TEMP_DIR):
        self.temp_dir = temp_dir

    def has_enough_space(self, required_bytes: int, buffer_ratio: float = 1.1) -> bool:
        total, used, free = shutil.disk_usage(self.temp_dir)
        logger.info(f"storage total : {total} , used : {used} , free : {free}")
        return free > required_bytes * buffer_ratio

    def check_huggingface_size(self, dataset_name: str, dataset_config: Optional[str] = None) -> None:
        try:
            infos = get_dataset_infos(dataset_name)
            for config, info in infos.items():
                if dataset_config is None or config == dataset_config:
                    estimated_size = 0
                    if hasattr(info, "size_in_bytes") and info.size_in_bytes:
                        estimated_size = info.size_in_bytes
                    elif hasattr(info, "dataset_size") and info.dataset_size:
                        estimated_size = info.dataset_size
                    elif hasattr(info, "download_size") and info.download_size:
                        estimated_size = info.download_size

                    logger.info(f"estimated size of dataset is {estimated_size}")
                    if estimated_size > 0 and not self.has_enough_space(estimated_size):
                        raise HTTPException(status_code=507, detail="Not enough disk space for Hugging Face dataset.")
        except Exception as e:
            logger.warning(f"Could not estimate Hugging Face dataset size: {e}")

    def check_openml_size(self, dataset_id: int) -> None:
        try:
            dataset = openml.datasets.get_dataset(dataset_id, download_all_files=False)
            n_instances = dataset.qualities.get("NumberOfInstances")
            n_features = dataset.qualities.get("NumberOfFeatures")
            if n_instances and n_features:
                estimated_size = int(n_instances) * int(n_features) * 100  # worst case (bytes)
                logger.info(f"estimated size of dataset is {estimated_size}")
                if not self.has_enough_space(estimated_size):
                    raise HTTPException(status_code=507, detail="Not enough disk space for OpenML dataset.")
        except Exception as e:
            logger.warning(f"Could not estimate OpenML dataset size: {e}")








@app.post("/kaggle-to-pvc", summary="Ingest from Kaggle", tags=["Ingestion"])
async def kaggle_to_pvc(request: KaggleDownloadRequest):
    """
    Downloads a dataset from Kaggle.
    You can provide 'kaggle_username' and 'kaggle_key' in the request body for authentication.
    """
    return process_kaggle_dataset(request)

@app.post("/huggingface-to-pvc", summary="Ingest from Hugging Face", tags=["Ingestion"])
async def huggingface_to_pvc(request: HuggingFaceDownloadRequest):
    return process_huggingface_dataset(
        request.dataset_name, request.dataset_config, request.dataset_name_override, 
        request.user_name, request.private
    )

@app.post("/openml-to-pvc", summary="Ingest from OpenML", tags=["Ingestion"])
async def openml_to_pvc(request: OpenMLDownloadRequest):
    return process_openml_dataset(
        request.dataset_id, request.dataset_name, request.user_name, request.private
    )

@app.post("/list-datasets", summary="List Available Datasets", tags=["Management"])
async def list_datasets(request: ListDatasetsRequest):
    datasets = storage_handler.list_datasets(request.user_name, request.private_only)
    return {
        "status": "success",
        "count": len(datasets),
        "datasets": [ds.dict() for ds in datasets]
    }

@app.post("/download-dataset", summary="Download a Stored Dataset", tags=["Management"])
async def download_dataset(request: DownloadDatasetRequest):
    try:
        metadata = storage_handler.load_metadata(request.dataset_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found.")
        if metadata.private and metadata.user_name != request.user_name:
            raise HTTPException(status_code=403, detail="Access denied to private dataset.")
        dataset_path_str = storage_handler.get_dataset_path(request.dataset_id)
        if not dataset_path_str:
             raise HTTPException(status_code=404, detail="Dataset files are missing from storage (path not found in metadata).")
        dataset_path = Path(dataset_path_str)
        if not dataset_path.exists():
            logger.error(f"Metadata points to a non-existent path: {dataset_path}")
            raise HTTPException(status_code=404, detail="Dataset files are missing from storage (file not found).")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            zip_path = Path(tmp_zip.name)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if dataset_path.is_dir():
                    for file_path in dataset_path.rglob('*'):
                        zipf.write(file_path, file_path.relative_to(dataset_path))
                else:
                    zipf.write(dataset_path, dataset_path.name)
            zip_filename = f"{metadata.dataset_name.replace(' ', '_')}_{metadata.dataset_id[:8]}.zip"
            return FileResponse(zip_path, filename=zip_filename, media_type="application/zip", background=lambda: os.unlink(zip_path))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during dataset download for '{request.dataset_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not process download request: {e}")

@app.get("/dataset-info/{dataset_id}", summary="Get Dataset Metadata", tags=["Management"])
async def get_dataset_info(dataset_id: str):
    metadata = storage_handler.load_metadata(dataset_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return {"status": "success", "metadata": metadata.dict()}

@app.get("/health", summary="Health Check", tags=["System"])
async def health_check():
    return {"status": "healthy", "storage_path": str(PVC_BASE_DIR), "temp_path": str(TEMP_DIR)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1235)