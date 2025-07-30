# import openml
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import boto3
# import os
# import shutil
# import kagglehub
# from pathlib import Path
# import pandas as pd
# from clearml import Task, Dataset as ClearMLDataset
# from datasets import load_dataset, DatasetDict
# import random
# import json
# import numpy as np
# from typing import Optional, List
# import logging
# from fastapi.responses import FileResponse
# import shutil
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI(title="Data Ingestion API", description="API to download and upload datasets to ClearML from S3, Kaggle, and Hugging Face")

# # Pydantic models for input validation
# class S3DownloadRequest(BaseModel):
#     s3_file_path: str
#     local_file_path: str
#     task_name: Optional[str] = "S3_to_ClearML"
#     dataset_name: Optional[str] = "S3_Dataset"
#     project_name: Optional[str] = "S3_Integration"
#     upload_as: str = "artifact"  # Options: "artifact" or "dataset"

# class KaggleDownloadRequest(BaseModel):
#     dataset_id: str
#     task_name: Optional[str] = None
#     dataset_name: Optional[str] = None
#     project_name: Optional[str] = "Kaggle Data Exploration"
#     upload_as: str = "dataset"  # Options: "artifact" or "dataset"

# class HuggingFaceDownloadRequest(BaseModel):
#     dataset_name: str
#     dataset_config: Optional[str] = None

#     task_name: Optional[str] = None
#     dataset_name: Optional[str] = None
#     project_name: Optional[str] = "HuggingFace Pipeline"
#     upload_as: str = "dataset"  # Options: "artifact" or "dataset"

# # S3 Configuration
# S3_CONFIG = {
#     "access_key": "OAF0MC26UA7DV9WS11X5",
#     "secret_key": "6SY2dTxhcIVEsjbfpjRUBhe3k7mMJIjZpccwvw3d",
#     "endpoint_url": "https://s3.cloud-ai.ir",
#     "bucket_name": "mlops"
# }

# # S3 Client
# s3_client = boto3.client(
#     's3',
#     aws_access_key_id=S3_CONFIG["access_key"],
#     aws_secret_access_key=S3_CONFIG["secret_key"],
#     endpoint_url=S3_CONFIG["endpoint_url"]
# )

# # S3 to ClearML Handler
# class S3ClearMLHandler:
#     def download_file(self, remote_path: str, local_path: str) -> str:
#         """Download a single file from S3."""
#         os.makedirs(os.path.dirname(local_path), exist_ok=True)
#         try:
#             s3_client.download_file(S3_CONFIG["bucket_name"], remote_path, local_path)
#             logger.info(f"Downloaded '{remote_path}' to '{local_path}'")
#             return local_path
#         except Exception as e:
#             logger.error(f"Error downloading file '{remote_path}': {e}")
#             raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

#     def upload_to_clearml_artifact(self, remote_path: str, local_path: str, task_name: str, project_name: str) -> dict:
#         """Download from S3 and upload as ClearML artifact."""
#         task = Task.init(project_name=project_name, task_name=task_name)
#         self.download_file(remote_path, local_path)
#         if os.path.exists(local_path):
#             task.upload_artifact(name=os.path.basename(local_path), artifact_object=local_path)
#             logger.info(f"Uploaded '{local_path}' to ClearML task '{task_name}' as artifact")
#             task.close()
#             return {"status": "success", "message": f"Uploaded '{local_path}' to ClearML artifact"}
#         else:
#             task.close()
#             raise HTTPException(status_code=404, detail=f"File '{local_path}' not found for upload")

#     def upload_to_clearml_dataset(self, remote_path: str, local_path: str, dataset_name: str, project_name: str) -> dict:
#         """Download from S3 and add to ClearML dataset."""
#         dataset_name = dataset_name or os.path.basename(remote_path).split('.')[0]  # Use file name without extension if dataset_name is None
#         self.download_file(remote_path, local_path)
#         dataset = ClearMLDataset.create(dataset_name=dataset_name, dataset_project=project_name)
#         if os.path.exists(local_path):
#             dataset.add_files(local_path)
#             dataset.upload()
#             dataset.finalize()
#             logger.info(f"Uploaded '{local_path}' to ClearML dataset '{dataset_name}'")
#             return {"status": "success", "message": f"Uploaded '{local_path}' to ClearML dataset '{dataset_name}'"}
#         else:
#             raise HTTPException(status_code=404, detail=f"File '{local_path}' not found for dataset upload")
# # Kaggle to ClearML Handler
# def summarize_dataset(path: Path) -> str:
#     """Summarize dataset contents."""
#     summary = []
#     items = list(path.rglob("*"))
#     for subfolder in ["train", "test", "valid"]:
#         subpath = path / subfolder
#         if subpath.exists() and subpath.is_dir():
#             files = list(subpath.glob("*.jpg"))
#             summary.append(f"{subfolder.capitalize()} folder: {len(files)} image(s)")
#     csv_files = list(path.rglob("*.csv"))
#     for csv in csv_files:
#         try:
#             df = pd.read_csv(csv, nrows=5)
#             summary.append(f"CSV file: {csv.name} with columns: {list(df.columns)}")
#         except Exception as e:
#             summary.append(f"CSV file: {csv.name} could not be read: {str(e)}")
#     all_dirs = {p.parent.name for p in items if p.is_file()}
#     all_dirs.discard("train")
#     all_dirs.discard("test")
#     all_dirs.discard("valid")
#     if all_dirs:
#         summary.append(f"Additional folders: {', '.join(all_dirs)}")
#     return "\n".join(summary)

# def download_and_upload_kaggle(dataset_id: str, task_name: Optional[str], dataset_name: Optional[str], project_name: str, upload_as: str) -> dict:
#     """Download dataset from Kaggle and upload to ClearML as artifact or dataset."""
#     try:
#         # Set Kaggle API credentials in environment variables
#         os.environ["KAGGLE_USERNAME"] = "abolfazlmalekahmadi"
#         os.environ["KAGGLE_KEY"] = "3c6c648164bf3e848f60de422d84a1cb"
#         logger.info("Kaggle API credentials set in environment variables.")

#         # Download dataset from Kaggle
#         logger.info(f"Downloading Kaggle dataset: {dataset_id}")
#         path = kagglehub.dataset_download(dataset_id)
#         path = Path(path)
#         logger.info(f"Dataset downloaded to: {path}")

#         # Summarize dataset contents
#         summary = summarize_dataset(path)
#         logger.info(f"Dataset Summary:\n{summary}")
        
#         # Set task and dataset names
#         task_name = task_name or dataset_id.replace("/", "_")
#         dataset_name = dataset_name or dataset_id.replace("/", "_")  # Use dataset_id if dataset_name is None
        
#         # Initialize ClearML task
#         task = Task.init(project_name=project_name, task_name=task_name)
        
#         if upload_as == "artifact":
#             task.upload_artifact(name=dataset_name, artifact_object=str(path))
#             logger.info(f"Uploaded '{path}' to ClearML task '{task_name}' as artifact")
#             task.connect_configuration({'dataset_summary': summary})
#             task.close()
#             return {
#                 "status": "success",
#                 "message": f"Dataset '{dataset_id}' uploaded to ClearML as artifact",
#                 "summary": summary
#             }
#         elif upload_as == "dataset":
#             dataset = ClearMLDataset.create(dataset_name=dataset_name, dataset_project=project_name)
#             dataset.add_files(str(path))
#             dataset.upload()
#             dataset.finalize()
#             task.connect_configuration({'dataset_summary': summary})
#             logger.info(f"Uploaded '{path}' to ClearML dataset '{dataset_name}'")
#             task.close()
#             return {
#                 "status": "success",
#                 "message": f"Dataset '{dataset_id}' uploaded to ClearML as dataset",
#                 "summary": summary
#             }
#         else:
#             task.close()
#             raise HTTPException(status_code=400, detail="Invalid 'upload_as' value. Use 'artifact' or 'dataset'.")
#     except Exception as e:
#         logger.error(f"Error processing Kaggle dataset '{dataset_id}': {e}")
#         task.close() if 'task' in locals() else None
#         # Check if the error is related to permissions or dataset terms
#         if "403 Client Error" in str(e):
#             raise HTTPException(
#                 status_code=403,
#                 detail=(
#                     f"Failed to download Kaggle dataset '{dataset_id}'. "
#                     "This dataset may be private or require acceptance of terms. "
#                     "Please visit https://www.kaggle.com/datasets/{dataset_id}, "
#                     "ensure you are logged in, and accept any dataset terms if required. "
#                     "Then try again."
#                 )
#             )
#         raise HTTPException(status_code=500, detail=f"Error processing Kaggle dataset: {str(e)}")
# # Hugging Face to ClearML Handler
# def split_dataset_if_needed(dataset: DatasetDict, split_name: str = "train", train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> DatasetDict:
#     """Split dataset if only one split exists."""
#     if len(dataset.keys()) == 1 and split_name in dataset:
#         logger.info(f"Single split detected ({split_name}). Splitting into train, validation, and test...")
#         assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1"
#         train_test_split = dataset[split_name].train_test_split(test_size=(val_ratio + test_ratio), seed=seed)
#         train_ds = train_test_split["train"]
#         temp_ds = train_test_split["test"]
#         val_test_ratio = test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0
#         val_test_split = temp_ds.train_test_split(test_size=val_test_ratio, seed=seed)
#         return DatasetDict({
#             "train": train_ds,
#             "validation": val_test_split["train"],
#             "test": val_test_split["test"]
#         })
#     return dataset

# def log_sample(task: Task, sample: dict, sample_idx: int, split_name: str):
#     """Log a sample to ClearML."""
#     task.get_logger().report_text(f"Sample {sample_idx+1} ({split_name}):")
#     for feature, value in sample.items():
#         if isinstance(value, (list, np.ndarray)):
#             try:
#                 shape = np.shape(value)
#                 display_value = f"Array with shape {shape}"
#             except Exception:
#                 display_value = "Complex array/list data"
#         elif isinstance(value, dict):
#             display_value = f"Dict with keys: {list(value.keys())}"
#         elif isinstance(value, str):
#             display_value = value[:200] + "..." if len(value) > 200 else value
#         else:
#             display_value = str(value)
#         task.get_logger().report_text(f"{feature}: {display_value}")
#     try:
#         df_sample = pd.DataFrame([{k: (str(v) if isinstance(v, (dict, list, np.ndarray)) else v) for k, v in sample.items()}])
#         task.get_logger().report_table(
#             title=f"Sample {sample_idx+1} ({split_name})",
#             series=f"Random Samples ({split_name})",
#             iteration=0,
#             table_plot=df_sample
#         )
#     except Exception as e:
#         task.get_logger().report_text(f"Warning: could not log table for sample {sample_idx+1}: {str(e)}")

# def upload_and_replace_image_paths(samples: List[dict], artifact_dir: str) -> List[dict]:
#     """Upload images and replace paths in samples."""
#     os.makedirs(artifact_dir, exist_ok=True)
#     for sample in samples:
#         if "image" in sample:
#             images = sample["image"]
#             if isinstance(images, list):
#                 new_images = []
#                 for img_info in images:
#                     if isinstance(img_info, dict) and "path" in img_info and img_info["path"]:
#                         old_path = img_info["path"]
#                         filename = os.path.basename(old_path)
#                         new_path = os.path.join(artifact_dir, filename)
#                         shutil.copy2(old_path, new_path)
#                         new_images.append({"path": new_path, "bytes": None})
#                     else:
#                         new_images.append(img_info)
#                 sample["image"] = new_images
#             else:
#                 if isinstance(sample["image"], dict) and "path" in sample["image"]:
#                     old_path = sample["image"]["path"]
#                     filename = os.path.basename(old_path)
#                     new_path = os.path.join(artifact_dir, filename)
#                     shutil.copy2(old_path, new_path)
#                     sample["image"]["path"] = new_path
#                 elif isinstance(sample["image"], str):
#                     old_path = sample["image"]
#                     filename = os.path.basename(old_path)
#                     new_path = os.path.join(artifact_dir, filename)
#                     shutil.copy2(old_path, new_path)
#                     sample["image"] = new_path
#     return samples
# def process_huggingface_dataset(dataset_name: str, dataset_config: Optional[str], task_name: Optional[str], clearml_dataset_name: Optional[str], project_name: str, upload_as: str) -> dict:
#     """Download a Hugging Face dataset in its entirety and upload to ClearML as artifact or dataset."""
#     try:
#         task_name = task_name or "HuggingFace_Dataset_Processing"
#         clearml_dataset_name = clearml_dataset_name or dataset_name.replace('/', '_')  # Use dataset_name if clearml_dataset_name is None
        
#         # Initialize ClearML task
#         task = Task.init(project_name=project_name, task_name=task_name)
#         logger.info(f"Starting download for Hugging Face dataset '{dataset_name}' (config: {dataset_config})...")

#         # Download the dataset
#         ds = load_dataset(dataset_name, dataset_config)
#         logger.info(f"Dataset '{dataset_name}' downloaded successfully with splits: {list(ds.keys())}")

#         # Save the dataset to a local directory in its native format
#         local_dataset_dir = f"./huggingface_{clearml_dataset_name}"
#         os.makedirs(local_dataset_dir, exist_ok=True)
#         ds.save_to_disk(local_dataset_dir)
#         logger.info(f"Dataset saved to local directory: {local_dataset_dir}")

#         if upload_as == "artifact":
#             # Upload the entire dataset directory as a ClearML artifact
#             task.upload_artifact(name=clearml_dataset_name, artifact_object=local_dataset_dir)
#             logger.info(f"Uploaded '{local_dataset_dir}' to ClearML task '{task_name}' as artifact")
#             task.close()
#             return {
#                 "status": "success",
#                 "message": f"Dataset '{dataset_name}' uploaded to ClearML as artifact",
#                 "local_path": local_dataset_dir
#             }
        
#         elif upload_as == "dataset":
#             # Create a ClearML dataset and upload the entire dataset directory
#             clearml_dataset = ClearMLDataset.create(
#                 dataset_name=clearml_dataset_name,
#                 dataset_project=project_name,
#                 dataset_tags=[dataset_name.split('/')[-1], "huggingface"]
#             )
#             clearml_dataset.add_files(local_dataset_dir)
#             clearml_dataset.upload()
#             clearml_dataset.finalize()
#             logger.info(f"Uploaded '{local_dataset_dir}' to ClearML dataset '{clearml_dataset_name}'")
#             task.close()
#             return {
#                 "status": "success",
#                 "message": f"Dataset '{dataset_name}' uploaded to ClearML as dataset",
#                 "local_path": local_dataset_dir
#             }
        
#         else:
#             task.close()
#             raise HTTPException(status_code=400, detail="Invalid 'upload_as' value. Use 'artifact' or 'dataset'.")
    
#     except Exception as e:
#         logger.error(f"Error processing Hugging Face dataset '{dataset_name}': {e}")
#         task.close() if 'task' in locals() else None
#         raise HTTPException(status_code=500, detail=f"Error processing Hugging Face dataset: {str(e)}")
# # API Endpoints
# @app.post("/s3-to-clearml")
# async def s3_to_clearml(request: S3DownloadRequest):
#     """Download from S3 and upload to ClearML as artifact or dataset."""
#     handler = S3ClearMLHandler()
#     if request.upload_as == "artifact":
#         return handler.upload_to_clearml_artifact(
#             request.s3_file_path, request.local_file_path, request.task_name, request.project_name
#         )
#     elif request.upload_as == "dataset":
#         return handler.upload_to_clearml_dataset(
#             request.s3_file_path, request.local_file_path, request.dataset_name, request.project_name
#         )
#     else:
#         raise HTTPException(status_code=400, detail="Invalid 'upload_as' value. Use 'artifact' or 'dataset'.")

# @app.post("/kaggle-to-clearml")
# async def kaggle_to_clearml(request: KaggleDownloadRequest):
#     """Download from Kaggle and upload to ClearML."""
#     return download_and_upload_kaggle(
#         request.dataset_id, request.task_name, request.dataset_name, request.project_name, request.upload_as
#     )

# @app.post("/huggingface-to-clearml")
# async def huggingface_to_clearml(request: HuggingFaceDownloadRequest):
#     """Download from Hugging Face and upload to ClearML."""
#     return process_huggingface_dataset(
#         request.dataset_name, request.dataset_config,
#         request.task_name, request.dataset_name, request.project_name, request.upload_as
#     )

# # Add new Pydantic model for OpenML download requests
# class OpenMLDownloadRequest(BaseModel):
#     dataset_id: int  # OpenML datasets are identified by integer IDs
#     task_name: Optional[str] = None
#     dataset_name: Optional[str] = None
#     project_name: Optional[str] = "OpenML Data Exploration"
#     upload_as: str = "dataset"  # Options: "artifact" or "dataset"

# # OpenML to ClearML Handler
# def summarize_openml_dataset(dataset: openml.datasets.OpenMLDataset, local_path: Path) -> str:
#     """Summarize OpenML dataset contents."""
#     summary = []
#     summary.append(f"Dataset Name: {dataset.name}")
#     summary.append(f"Description: {dataset.description[:200] + '...' if len(dataset.description) > 200 else dataset.description}")
#     summary.append(f"Number of Instances: {dataset.qualities.get('NumberOfInstances', 'Unknown')}")
#     summary.append(f"Number of Features: {dataset.qualities.get('NumberOfFeatures', 'Unknown')}")
    
#     # If the dataset is stored locally as a CSV or ARFF file, summarize its contents
#     if local_path.exists():
#         try:
#             df = pd.read_csv(local_path, nrows=5) if local_path.suffix == '.csv' else pd.read_table(local_path, nrows=5)
#             summary.append(f"File: {local_path.name} with columns: {list(df.columns)}")
#         except Exception as e:
#             summary.append(f"File: {local_path.name} could not be read: {str(e)}")
#     return "\n".join(summary)

# def download_and_upload_openml(dataset_id: int, task_name: Optional[str], dataset_name: Optional[str], project_name: str, upload_as: str) -> dict:
#     """Download dataset from OpenML and upload to ClearML as artifact or dataset."""
#     try:
#         logger.info(f"Downloading OpenML dataset: ID={dataset_id}, Requested Name={dataset_name}")
        
#         # If dataset_name is provided, check if it exists in OpenML
#         final_dataset_name = None
#         if dataset_name:
#             available_datasets = openml.datasets.list_datasets(output_format="dataframe")
#             matching_datasets = available_datasets[available_datasets["name"] == dataset_name]
#             if not matching_datasets.empty:
#                 # If dataset_name exists in OpenML, use it and verify if the ID matches
#                 matched_dataset_id = matching_datasets.iloc[0]["did"]
#                 if matched_dataset_id == dataset_id:
#                     final_dataset_name = dataset_name
#                     logger.info(f"Dataset with name '{dataset_name}' found in OpenML with ID {dataset_id}")
#                 else:
#                     logger.warning(f"Dataset name '{dataset_name}' found in OpenML but with ID {matched_dataset_id}, not {dataset_id}. Using dataset_id instead.")
#             else:
#                 logger.warning(f"Dataset name '{dataset_name}' not found in OpenML. Using dataset_id {dataset_id}.")
        
#         # Download dataset using dataset_id
#         dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
#         final_dataset_name = final_dataset_name or dataset.name  # Use dataset.name if no valid dataset_name provided
#         task_name = task_name or f"openml_{dataset_id}"
        
#         # Get the dataset as a pandas DataFrame and save it locally
#         X, y, _, _ = dataset.get_data(dataset_format="dataframe")
#         if y is not None:
#             df = pd.concat([X, y], axis=1)
#         else:
#             df = X
#         local_path = Path(f"openml_{dataset_id}.csv")
#         df.to_csv(local_path, index=False)
        
#         summary = summarize_openml_dataset(dataset, local_path)
#         logger.info(f"Dataset Summary:\n{summary}")
        
#         task = Task.init(project_name=project_name, task_name=task_name)
        
#         if upload_as == "artifact":
#             task.upload_artifact(name=final_dataset_name, artifact_object=str(local_path))
#             logger.info(f"Uploaded '{local_path}' to ClearML task '{task_name}' as artifact")
#             task.connect_configuration({'dataset_summary': summary})
#             task.close()
#             return {
#                 "status": "success",
#                 "message": f"Dataset '{dataset_id}' uploaded to ClearML as artifact",
#                 "dataset_name": final_dataset_name,  # Include final dataset name in response
#                 "summary": summary
#             }
#         elif upload_as == "dataset":
#             clearml_dataset = ClearMLDataset.create(dataset_name=final_dataset_name, dataset_project=project_name, dataset_tags=[f"openml_{dataset_id}", "openml"])
#             clearml_dataset.add_files(str(local_path))
#             clearml_dataset.upload()
#             clearml_dataset.finalize()
#             task.connect_configuration({'dataset_summary': summary})
#             logger.info(f"Uploaded '{local_path}' to ClearML dataset '{final_dataset_name}'")
#             task.close()
#             return {
#                 "status": "success",
#                 "message": f"Dataset '{dataset_id}' uploaded to ClearML as dataset",
#                 "dataset_name": final_dataset_name,  # Include final dataset name in response
#                 "summary": summary
#             }
#         else:
#             task.close()
#             raise HTTPException(status_code=400, detail="Invalid 'upload_as' value. Use 'artifact' or 'dataset'.")
#     except Exception as e:
#         logger.error(f"Error processing OpenML dataset '{dataset_id}': {e}")
#         task.close() if 'task' in locals() else None
#         raise HTTPException(status_code=500, detail=f"Error processing OpenML dataset: {str(e)}")
    
# class ClearMLDownloadDatasetRequest(BaseModel):
#     dataset_name: str                    # Name of the dataset to search
#     local_path: str                     # Local path to save the dataset
#     task_name: Optional[str] = "Download_ClearML_Dataset"
#     project_name: Optional[str] = "ClearML_Dataset_Download"
#     source: Optional[str] = None         # Source of dataset if not found: 'openml', 'kaggle', 'huggingface', 's3'
#     source_dataset_id: Optional[str] = None  # Dataset ID for Kaggle, Hugging Face, or S3 file path
#     source_dataset_config: Optional[str] = None  # Dataset config for Hugging Face
#     source_dataset_id_int: Optional[int] = None  # Dataset ID for OpenML (integer)
    
# # Add new API endpoint
# @app.post("/openml-to-clearml")
# async def openml_to_clearml(request: OpenMLDownloadRequest):
#     """Download from OpenML and upload to ClearML."""
#     return download_and_upload_openml(
#         request.dataset_id, request.task_name, request.dataset_name, request.project_name, request.upload_as
#     )
    
# # Pydantic models for ClearML dataset operations
# class ClearMLListDatasetsRequest(BaseModel):
#     project_name: Optional[str] = None  # Filter by project name (optional)
#     tags: Optional[List[str]] = None    # Filter by tags (optional)

# # ClearML Dataset Handler
# class ClearMLDatasetHandler:
#     def list_datasets(self, project_name: Optional[str] = None, tags: Optional[List[str]] = None) -> List[dict]:
#         """List datasets in ClearML with optional filters."""
#         try:
#             logger.info(f"Listing ClearML datasets (project: {project_name}, tags: {tags})")
#             # Use Dataset.list_datasets to query datasets
#             query_filters = {}
#             if project_name:
#                 query_filters["project_name"] = project_name
#             if tags:
#                 query_filters["tags"] = tags
#             datasets = ClearMLDataset.list_datasets(**query_filters)
#             dataset_list = [
#                 {
#                     "id": dataset["id"],
#                     "name": dataset["name"],
#                     "project": dataset.get("project", "Unknown"),
#                     "tags": dataset.get("tags", []),
#                     "created": dataset.get("created", ""),
#                     "version": dataset.get("version", "Unknown")
#                 }
#                 for dataset in datasets
#             ]
#             logger.info(f"Found {len(dataset_list)} datasets")
#             return dataset_list
#         except Exception as e:
#             logger.error(f"Error listing ClearML datasets: {e}")
#             raise HTTPException(status_code=500, detail=f"Error listing ClearML datasets: {str(e)}")

#     def download_dataset(self, dataset_id: str, local_path: str, task_name: str, project_name: str) -> dict:
#         """Download a dataset from ClearML to a local path."""
#         try:
#             task = Task.init(project_name=project_name, task_name=task_name)
#             logger.info(f"Downloading ClearML dataset '{dataset_id}' to '{local_path}'")
#             dataset = ClearMLDataset.get(dataset_id=dataset_id)
#             dataset_path = dataset.get_local_copy(local_path=local_path)
#             if os.path.exists(dataset_path):
#                 logger.info(f"Dataset '{dataset_id}' downloaded to '{dataset_path}'")
#                 task.upload_artifact(name=f"downloaded_dataset_{dataset_id}", artifact_object=dataset_path)
#                 task.get_logger().report_text(f"Dataset '{dataset_id}' downloaded to '{dataset_path}'")
#                 task.close()
#                 return {
#                     "status": "success",
#                     "message": f"Dataset '{dataset_id}' downloaded to '{dataset_path}'",
#                     "local_path": dataset_path
#                 }
#             else:
#                 task.close()
#                 raise HTTPException(status_code=404, detail=f"Failed to download dataset '{dataset_id}'")
#         except Exception as e:
#             logger.error(f"Error downloading ClearML dataset '{dataset_id}': {e}")
#             task.close()
#             raise HTTPException(status_code=500, detail=f"Error downloading ClearML dataset: {str(e)}")
# # API Endpoints for ClearML Dataset Operations
# @app.post("/list-clearml-datasets")
# async def list_clearml_datasets(request: ClearMLListDatasetsRequest):
#     """List datasets in ClearML with optional filters."""
#     handler = ClearMLDatasetHandler()
#     return handler.list_datasets(project_name=request.project_name, tags=request.tags)

# @app.post("/download_dataset/")
# async def download_dataset(request: ClearMLDownloadDatasetRequest):
#     """Download a dataset from ClearML by name, or fetch from specified source if not found."""
#     try:
#         # Search for dataset in ClearML by name
#         datasets = ClearMLDataset.list_datasets(partial_name=request.dataset_name, only_completed=True)
#         if datasets:
#             dataset_id = datasets[0]["id"]
#             dataset = ClearMLDataset.get(dataset_id=dataset_id)
#             logger.info(f"Found dataset '{request.dataset_name}' in ClearML with ID '{dataset_id}'")
            
#             # Download the dataset to a temporary location
#             dataset_path = dataset.get_local_copy()
#             if not dataset_path or not os.path.exists(dataset_path):
#                 raise HTTPException(
#                     status_code=404,
#                     detail=f"Failed to download dataset '{request.dataset_name}' from ClearML. Dataset path not found."
#                 )

#             # Move or copy to the requested local_path
#             os.makedirs(os.path.dirname(request.local_path), exist_ok=True)
#             if os.path.isdir(dataset_path):
#                 # If dataset_path is a directory, copy its contents to the requested local_path
#                 if os.path.exists(request.local_path):
#                     shutil.rmtree(request.local_path, ignore_errors=True)
#                 shutil.copytree(dataset_path, request.local_path)
#             else:
#                 # If dataset_path is a single file, copy it to the requested local_path
#                 shutil.copy2(dataset_path, request.local_path)

#             # Create zip file for download
#             DATASETS_DIR = "downloaded_datasets"
#             os.makedirs(DATASETS_DIR, exist_ok=True)
#             zip_filename = f"{dataset.name.replace('/', '_')}_{dataset.id}.zip"
#             zip_path = os.path.join(DATASETS_DIR, zip_filename)

#             if not os.path.exists(zip_path):
#                 shutil.make_archive(base_name=zip_path.replace('.zip', ''), format="zip", root_dir=request.local_path)

#             return FileResponse(zip_path, filename=zip_filename, media_type="application/zip")
#         else:
#             # If dataset not found and source is specified, download from source
#             if not request.source:
#                 raise HTTPException(
#                     status_code=404,
#                     detail=f"Dataset '{request.dataset_name}' not found in ClearML and no source specified"
#                 )
            
#             logger.info(f"Dataset '{request.dataset_name}' not found in ClearML. Downloading from {request.source}...")
            
#             if request.source == "openml":
#                 if not request.source_dataset_id_int:
#                     raise HTTPException(status_code=400, detail="OpenML dataset ID (integer) is required")
#                 response = download_and_upload_openml(
#                     dataset_id=request.source_dataset_id_int,
#                     task_name=request.task_name,
#                     dataset_name=request.dataset_name,
#                     project_name=request.project_name,
#                     upload_as="dataset"
#                 )
#                 datasets = ClearMLDataset.list_datasets(partial_name=request.dataset_name, only_completed=True)
#                 if not datasets:
#                     raise HTTPException(
#                         status_code=500,
#                         detail=f"Failed to download and upload dataset '{request.dataset_name}' from OpenML"
#                     )
#                 dataset_id = datasets[0]["id"]
#                 dataset = ClearMLDataset.get(dataset_id=dataset_id)
            
#             elif request.source == "kaggle":
#                 if not request.source_dataset_id:
#                     raise HTTPException(status_code=400, detail="Kaggle dataset ID is required")
#                 response = download_and_upload_kaggle(
#                     dataset_id=request.source_dataset_id,
#                     task_name=request.task_name,
#                     dataset_name=request.dataset_name,
#                     project_name=request.project_name,
#                     upload_as="dataset"
#                 )
#                 datasets = ClearMLDataset.list_datasets(partial_name=request.dataset_name, only_completed=True)
#                 if not datasets:
#                     raise HTTPException(
#                         status_code=500,
#                         detail=f"Failed to download and upload dataset '{request.dataset_name}' from Kaggle"
#                     )
#                 dataset_id = datasets[0]["id"]
#                 dataset = ClearMLDataset.get(dataset_id=dataset_id)
            
#             elif request.source == "huggingface":
#                 if not request.source_dataset_id:
#                     raise HTTPException(status_code=400, detail="Hugging Face dataset ID is required")
#                 response = process_huggingface_dataset(
#                     dataset_name=request.source_dataset_id,
#                     dataset_config=request.source_dataset_config,
             
#                     task_name=request.task_name,
#                     clearml_dataset_name=request.dataset_name,
#                     project_name=request.project_name,
#                     upload_as="dataset"
#                 )
#                 datasets = ClearMLDataset.list_datasets(partial_name=request.dataset_name, only_completed=True)
#                 if not datasets:
#                     raise HTTPException(
#                         status_code=500,
#                         detail=f"Failed to download and upload dataset '{request.dataset_name}' from Hugging Face"
#                     )
#                 dataset_id = datasets[0]["id"]
#                 dataset = ClearMLDataset.get(dataset_id=dataset_id)
            
#             elif request.source == "s3":
#                 if not request.source_dataset_id:
#                     raise HTTPException(status_code=400, detail="S3 file path is required")
#                 handler = S3ClearMLHandler()
#                 response = handler.upload_to_clearml_dataset(
#                     remote_path=request.source_dataset_id,
#                     local_path=request.local_path,
#                     dataset_name=request.dataset_name,
#                     project_name=request.project_name
#                 )
#                 datasets = ClearMLDataset.list_datasets(partial_name=request.dataset_name, only_completed=True)
#                 if not datasets:
#                     raise HTTPException(
#                         status_code=500,
#                         detail=f"Failed to download and upload dataset '{request.dataset_name}' from S3"
#                     )
#                 dataset_id = datasets[0]["id"]
#                 dataset = ClearMLDataset.get(dataset_id=dataset_id)
            
#             else:
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Invalid source: {request.source}. Use 'openml', 'kaggle', 'huggingface', or 's3'"
#                 )

#             # Download the newly uploaded dataset
#             dataset_path = dataset.get_local_copy()
#             if not dataset_path or not os.path.exists(dataset_path):
#                 raise HTTPException(
#                     status_code=404,
#                     detail=f"Failed to download dataset '{request.dataset_name}' from ClearML after uploading from source"
#                 )

#             # Move or copy to the requested local_path
#             os.makedirs(os.path.dirname(request.local_path), exist_ok=True)
#             if os.path.isdir(dataset_path):
#                 if os.path.exists(request.local_path):
#                     shutil.rmtree(request.local_path, ignore_errors=True)
#                 shutil.copytree(dataset_path, request.local_path)
#             else:
#                 shutil.copy2(dataset_path, request.local_path)

#             # Create zip file for download
#             DATASETS_DIR = "downloaded_datasets"
#             os.makedirs(DATASETS_DIR, exist_ok=True)
#             zip_filename = f"{dataset.name.replace('/', '_')}_{dataset.id}.zip"
#             zip_path = os.path.join(DATASETS_DIR, zip_filename)

#             if not os.path.exists(zip_path):
#                 shutil.make_archive(base_name=zip_path.replace('.zip', ''), format="zip", root_dir=request.local_path)

#             return FileResponse(zip_path, filename=zip_filename, media_type="application/zip")
    
#     except Exception as e:
#         logger.error(f"Error downloading dataset '{request.dataset_name}': {e}")
#         raise HTTPException(status_code=500, detail=f"Error downloading dataset: {str(e)}")
# # Run the API (for local testing)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=6990)



import os
import boto3
from tqdm import tqdm

S3_CONFIG = {
    "access_key": os.getenv("S3_ACCESS_KEY", "OAF0MC26UA7DV9WS11X5"),
    "secret_key": os.getenv("S3_SECRET_KEY", "6SY2dTxhcIVEsjbfpjRUBhe3k7mMJIjZpccwvw3d"),
    "endpoint_url": os.getenv("S3_ENDPOINT_URL", "https://s3.cloud-ai.ir"),
    "bucket_name": os.getenv("S3_BUCKET_NAME", "mlops")
}

def get_s3_folder_size(bucket_name, prefix, s3_config):
    s3 = boto3.client(
        's3',
        aws_access_key_id=s3_config["access_key"],
        aws_secret_access_key=s3_config["secret_key"],
        endpoint_url=s3_config["endpoint_url"],
    )

    total_size = 0
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    found = False
    for page in tqdm(pages, desc="Scanning S3"):
        contents = page.get('Contents', [])
        if contents:
            found = True
            for obj in contents:
                total_size += obj['Size']

    if not found:
        print(f"No objects found at prefix '{prefix}' in bucket '{bucket_name}'.")

    size_mb = total_size / (1024 ** 2)
    return size_mb

# Correct S3 object key or prefix — NOT a URL
prefix = "datasets/"  # or "folder_name/"
folder_size_mb = get_s3_folder_size(S3_CONFIG["bucket_name"], prefix, S3_CONFIG)
print(f"Folder size: {folder_size_mb:.2f} MB")
