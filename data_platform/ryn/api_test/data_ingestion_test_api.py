from fastapi import FastAPI, HTTPException
from data.data_ingestion.models.requests import (
    HuggingFaceDownloadRequest,
    KaggleDownloadRequest,
    OpenMLDownloadRequest,
    S3DownloadRequest,
    ListDatasetsRequest,
    DownloadDatasetRequest,
)
from data.data_ingestion.storage_handler import DatasetStorageHandler
from data.data_ingestion.handlers.huggingface_handler import process_huggingface_dataset
from data.data_ingestion.handlers.kaggle_handler import process_kaggle_dataset
from data.data_ingestion.handlers.openml_handler import process_openml_dataset
from data.data_ingestion.handlers.s3_handler import S3Handler

app = FastAPI(
    title="Dataset Ingestion API",
    description="API to download datasets from various sources like HuggingFace, Kaggle, S3, and OpenML.",
    version="1.0.0",
)

storage_handler = DatasetStorageHandler()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/huggingface-to-pvc")
def ingest_huggingface(request: HuggingFaceDownloadRequest):
    return process_huggingface_dataset(
        dataset_name=request.dataset_name,
        dataset_config=request.dataset_config,
        dataset_name_override=request.dataset_name_override,
        user_name=request.user_name,
        private=request.private,
    )


@app.post("/kaggle-to-pvc")
def ingest_kaggle(request: KaggleDownloadRequest):
    return process_kaggle_dataset(request)


@app.post("/openml-to-pvc")
def ingest_openml(request: OpenMLDownloadRequest):
    return process_openml_dataset(
        dataset_id=request.dataset_id,
        dataset_name=request.dataset_name,
        user_name=request.user_name,
        private=request.private,
    )


@app.post("/s3-to-pvc")
def ingest_s3(request: S3DownloadRequest):
    s3_handler = S3Handler(
        access_key=request.access_key,
        secret_key=request.secret_key,
        endpoint_url=request.endpoint_url,
        bucket_name=request.bucket_name,
    )
    return s3_handler.process_s3_dataset(
        s3_file_path=request.s3_file_path,
        dataset_name=request.dataset_name,
        user_name=request.user_name,
        private=request.private,
    )


@app.post("/list-datasets")
def list_datasets(request: ListDatasetsRequest):
    try:
        datasets = []
        for metadata_file in storage_handler.metadata_dir.glob("*.json"):
            metadata = storage_handler.load_metadata(metadata_file.stem)
            if request.user_name and metadata.user_name != request.user_name:
                continue
            if request.private_only and not metadata.private:
                continue
            datasets.append(metadata)
        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {e}")


@app.post("/download-dataset")
def download_dataset(request: DownloadDatasetRequest):
    try:
        metadata = storage_handler.load_metadata(request.dataset_id)
        if metadata.private and metadata.user_name != request.user_name:
            raise HTTPException(status_code=403, detail="Access denied.")
        return {
            "dataset_id": metadata.dataset_id,
            "dataset_name": metadata.dataset_name,
            "file_path": metadata.file_path,
            "summary": metadata.summary,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")