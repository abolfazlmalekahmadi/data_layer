from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import tempfile
import os
import zipfile

# Import all data ingestion logic and models from the new ingestion_core.py
from ingestion_core import (
    DatasetMetadata,
    S3DownloadRequest,
    KaggleDownloadRequest,
    HuggingFaceDownloadRequest,
    OpenMLDownloadRequest,
    ListDatasetsRequest,
    DownloadDatasetRequest,
    storage_handler,
    S3Handler,
    process_kaggle_dataset,
    process_huggingface_dataset,
    process_openml_dataset,
    PVC_BASE_DIR,
    TEMP_DIR,
)

app = FastAPI(
    title="Data Ingestion API",
    description="API to download and store datasets from S3, Kaggle, Hugging Face, and OpenML into a persistent volume using a centralized temporary directory.",
    version="0.0.2"
)


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