import asyncio
import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from clearml import Task, Dataset
from data.data_ingestion.models.requests import (
    HuggingFaceDownloadRequest,
    KaggleDownloadRequest,
    OpenMLDownloadRequest,
    S3DownloadRequest,
    ListDatasetsRequest,
    DownloadDatasetRequest,
)
from data.data_ingestion.storage_handler import DatasetStorageHandler
from data.data_ingestion.handlers.s3_handler import S3Handler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dataset Ingestion API with ClearML",
    description="API to download datasets from various sources using ClearML agents for task management.",
    version="2.0.0",
)

# ClearML configuration
CLEARML_PROJECT_NAME = "DatasetIngestion"
DEFAULT_QUEUE_NAME = "dataset_ingestion"

# ClearML imports
from clearml.backend_api.session.client import APIClient
from clearml import Task
import time

class ClearMLTaskManager:
    """Manages ClearML task creation and enqueueing for dataset ingestion operations."""

    @staticmethod
    def create_task(task_name: str, script_name: str, parameters: Dict[str, Any]) -> Task:
        unique_id = uuid.uuid4().hex[:8]
        full_task_name = f"{task_name}_{unique_id}"

        task = Task.create(
            project_name=CLEARML_PROJECT_NAME,
            task_name=full_task_name,
            task_type=Task.TaskTypes.data_processing,
            script="data_platform/ryn/api_test/processors/" + script_name,
            repo="git@github.com:AIP-MLOPS/tools.git",
            branch="data_layer",
            working_directory="data_platform/ryn/",
            packages=["datasets", "kaggle", "openml", "boto3", "pandas", "numpy", "fastapi", "logging", "kagglehub", "python-dotenv", "aipmodel==0.2.28"],
            docker=None
        )

        # for key, value in parameters.items():
        #     task.set_parameter(key, value)
        task.connect(parameters)

        logger.info(f"Created ClearML task: {full_task_name} with ID: {task.id}")
        return task

    @staticmethod
    def _ensure_queue_exists(queue_name: str, max_retries: int = 5, retry_delay: float = 1.0) -> bool:
        """
        Ensure the queue exists on the ClearML server.
        Returns True if queue exists or was successfully created.
        """
        try:
            client = APIClient()
            
            # First, try to get existing queue
            for attempt in range(max_retries):
                try:
                    resp = client.queues.get_all(name=queue_name)
                    if hasattr(resp, 'queues') and resp.queues:
                        queue_id = resp.queues[0].id
                        logger.info(f"Queue '{queue_name}' found (id: {queue_id})")
                        return True
                except Exception as e:
                    logger.debug(f"Attempt {attempt+1} - Error checking for queue: {e}")
                    time.sleep(retry_delay)
            
            # Queue doesn't exist, try to create it
            logger.info(f"Queue '{queue_name}' not found, attempting to create...")
            
            for attempt in range(max_retries):
                try:
                    # Try using the create endpoint
                    create_resp = client.queues.create(
                        name=queue_name,
                        system_tags=[queue_name]
                    )
                    logger.info(f"Queue creation response: {create_resp}")
                    time.sleep(retry_delay)
                    
                    # Verify it was created
                    resp = client.queues.get_all(name=queue_name)
                    if hasattr(resp, 'queues') and resp.queues:
                        queue_id = resp.queues[0].id
                        logger.info(f"Successfully created queue '{queue_name}' (id: {queue_id})")
                        return True
                        
                except Exception as e:
                    logger.debug(f"Attempt {attempt+1} - Error creating queue: {e}")
                    
                    # If creation failed, try the update method as fallback
                    try:
                        payload = {"name": queue_name, "system_tags": [queue_name]}
                        client.queues.update(payload)
                        time.sleep(retry_delay)
                        
                        # Check if it exists now
                        resp = client.queues.get_all(name=queue_name)
                        if hasattr(resp, 'queues') and resp.queues:
                            queue_id = resp.queues[0].id
                            logger.info(f"Successfully created queue '{queue_name}' via update (id: {queue_id})")
                            return True
                    except Exception as update_e:
                        logger.debug(f"Update method also failed: {update_e}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to ensure queue '{queue_name}' exists: {e}")
            return False

    @staticmethod
    def enqueue_task(task: Task, queue_name: str = DEFAULT_QUEUE_NAME) -> str:
        """Ensure queue exists and enqueue the task to the specified ClearML agent queue."""
        
        # Try to ensure queue exists
        if not ClearMLTaskManager._ensure_queue_exists(queue_name):
            logger.warning(f"Could not ensure queue '{queue_name}' exists, but attempting to enqueue anyway")
        
        try:
            # Try to enqueue the task
            Task.enqueue(task=task, queue_name=queue_name)
            logger.info(f"Successfully enqueued task {task.name} to queue '{queue_name}' with ID: {task.id}")
            return task.id
            
        except Exception as enqueue_error:
            logger.error(f"Failed to enqueue task to queue '{queue_name}': {enqueue_error}")
            
            # Try with default queue as fallback
            try:
                logger.info("Attempting to enqueue to 'default' queue as fallback")
                Task.enqueue(task=task, queue_name="default")
                logger.info(f"Successfully enqueued task {task.name} to 'default' queue with ID: {task.id}")
                return task.id
            except Exception as default_error:
                logger.error(f"Failed to enqueue to default queue: {default_error}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to enqueue task to any queue. Original error: {enqueue_error}. Default queue error: {default_error}"
                )

    @staticmethod
    def get_available_queues() -> list:
        """Get list of available queues."""
        try:
            client = APIClient()
            resp = client.queues.get_all()
            if hasattr(resp, 'queues'):
                return [{"name": q.name, "id": q.id} for q in resp.queues]
            return []
        except Exception as e:
            logger.error(f"Error getting available queues: {e}")
            return []


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0", "queue_system": "ClearML"}

@app.get("/queues")
async def list_queues():
    """List available ClearML queues."""
    try:
        queues = ClearMLTaskManager.get_available_queues()
        return {"queues": queues, "total": len(queues)}
    except Exception as e:
        logger.error(f"Error listing queues: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list queues: {str(e)}")

@app.post("/create-queue")
async def create_queue(queue_name: str):
    """Create a new ClearML queue."""
    try:
        success = ClearMLTaskManager._ensure_queue_exists(queue_name)
        if success:
            return {"status": "success", "message": f"Queue '{queue_name}' is available"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to create queue '{queue_name}'")
    except Exception as e:
        logger.error(f"Error creating queue: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create queue: {str(e)}")

@app.post("/huggingface")
async def ingest_huggingface(hf_req: HuggingFaceDownloadRequest):
    """Create and enqueue a ClearML task for HuggingFace dataset ingestion."""
    try:
        # Prepare task parameters
        parameters = {
            "task_type": "huggingface",
            "dataset_name": hf_req.dataset_name,
            "dataset_config": hf_req.dataset_config or "",
            "user_name": hf_req.user_name,
            "private": hf_req.private,
            "dataset_tag": hf_req.dataset_tag or "",
            "s3_config": {
                "endpoint_url": hf_req.endpoint_url,
                "access_key": hf_req.access_key,
                "secret_key": hf_req.secret_key,
                "bucket_name": hf_req.bucket_name,
            }
        }
        
        # Create ClearML task
        task = ClearMLTaskManager.create_task(
            task_name=f"HuggingFace_{hf_req.dataset_name.replace('/', '-')}",
            script_name="huggingface_processor.py",
            parameters=parameters
        )
        
        # Enqueue the task
        task_id = ClearMLTaskManager.enqueue_task(task)
        
        return {
            "status": "success",
            "message": f"HuggingFace dataset ingestion task created and enqueued",
            "task_id": task_id,
            "dataset_name": hf_req.dataset_name,
            "queue": DEFAULT_QUEUE_NAME
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating HuggingFace ingestion task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create ingestion task: {str(e)}")

@app.post("/kaggle")
async def ingest_kaggle(request: KaggleDownloadRequest):
    """Create and enqueue a ClearML task for Kaggle dataset ingestion."""
    try:
        # Prepare task parameters
        parameters = {
            "task_type": "kaggle",
            "dataset_id": request.dataset_id,
            "user_name": request.user_name,
            "kaggle_username": request.kaggle_username,
            "kaggle_key": request.kaggle_key,
            "private": request.private
        }
        
        # Create ClearML task
        task = ClearMLTaskManager.create_task(
            task_name=f"Kaggle_{request.dataset_id}",
            script_name="kaggle_processor.py",
            parameters=parameters
        )
        
        # Enqueue the task
        task_id = ClearMLTaskManager.enqueue_task(task)
        
        return {
            "status": "success",
            "message": f"Kaggle dataset ingestion task created and enqueued",
            "task_id": task_id,
            "dataset_id": request.dataset_id,
            "queue": DEFAULT_QUEUE_NAME
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating Kaggle ingestion task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create ingestion task: {str(e)}")

@app.post("/openml")
async def ingest_openml(request: OpenMLDownloadRequest):
    """Create and enqueue a ClearML task for OpenML dataset ingestion."""
    try:
        # Prepare task parameters
        parameters = {
            "task_type": "openml",
            "dataset_id": request.dataset_id,
            "dataset_name": request.dataset_name,
            "user_name": request.user_name,
            "private": request.private
        }
        
        # Create ClearML task
        task = ClearMLTaskManager.create_task(
            task_name=f"OpenML_{request.dataset_name or request.dataset_id}",
            script_name="openml_processor.py",
            parameters=parameters
        )
        
        # Enqueue the task
        task_id = ClearMLTaskManager.enqueue_task(task)
        
        return {
            "status": "success",
            "message": f"OpenML dataset ingestion task created and enqueued",
            "task_id": task_id,
            "dataset_id": request.dataset_id,
            "queue": DEFAULT_QUEUE_NAME
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating OpenML ingestion task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create ingestion task: {str(e)}")

@app.post("/s3")
async def ingest_s3(request: Request, s3_request: S3DownloadRequest):
    """Create and enqueue a ClearML task for S3 dataset ingestion."""
    try:
        raw_body = await request.body()
        logger.info(f"Raw request body: {raw_body}")
        
        # Prepare task parameters
        parameters = {
            "task_type": "s3",
            "access_key": s3_request.access_key,
            "secret_key": s3_request.secret_key,
            "endpoint_url": s3_request.endpoint_url,
            "bucket_name": s3_request.bucket_name,
            "s3_file_path": s3_request.s3_file_path,
            "dataset_name": s3_request.dataset_name,
            "user_name": s3_request.user_name,
            "private": s3_request.private,
            "dataset_tag":s3_request.dataset_tag,
            "s3_config_target": {
                    "endpoint_url_target": s3_request.endpoint_url_target,
                    "access_key_target": s3_request.access_key_target,
                    "secret_key_target": s3_request.secret_key_target,
                    "bucket_name_target": s3_request.bucket_name_target,
                }
        }
        # raise ValueError(parameters)
        # Create ClearML task
        task = ClearMLTaskManager.create_task(
            task_name=f"S3_{s3_request.dataset_name}",
            script_name="s3_processor.py",
            parameters=parameters
        )
        
        # Enqueue the task
        task_id = ClearMLTaskManager.enqueue_task(task)
        
        return {
            "status": "success",
            "message": f"S3 dataset ingestion task created and enqueued",
            "task_id": task_id,
            "dataset_name": s3_request.dataset_name,
            "queue": DEFAULT_QUEUE_NAME
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating S3 ingestion task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create ingestion task: {str(e)}")
    
    


@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a ClearML task."""
    try:
        task = Task.get_task(task_id=task_id)
        return {
            "task_id": task_id,
            "name": task.name,
            "status": task.get_status(),
            "project": task.get_project_name(),
            "created": task.data.created,
            "last_update": task.data.last_update
        }
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=404, detail=f"Task not found or error retrieving status: {str(e)}")

@app.get("/tasks")
async def list_tasks(project_name: str = CLEARML_PROJECT_NAME, limit: int = 50):
    """List recent ClearML tasks in the project."""
    try:
        from clearml.backend_api.session.client import APIClient
        client = APIClient()
        
        # Get project ID
        projects = client.projects.get_all(name=project_name)
        if not projects.projects:
            return {"tasks": [], "message": f"No project found with name: {project_name}"}
        
        project_id = projects.projects[0].id
        
        # Get tasks for the project
        tasks_response = client.tasks.get_all(
            project=[project_id],
            page_size=limit,
            order_by=["-created"]
        )
        
        tasks = []
        for task in tasks_response.tasks:
            tasks.append({
                "id": task.id,
                "name": task.name,
                "status": task.status,
                "created": task.created,
                "last_update": task.last_update,
                "type": task.type
            })
        
        return {"tasks": tasks, "total": len(tasks)}
        
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

@app.post("/list-datasets")
async def list_datasets(request: ListDatasetsRequest):
    """List available datasets (unchanged from original)."""
    storage_handler = None
    try:
        storage_handler = DatasetStorageHandler("temp_mount")
        datasets = []
        for metadata_file in storage_handler.metadata_dir.glob("*.json"):
            metadata = storage_handler.load_metadata(metadata_file.stem)
            if request.user_name and metadata.user_name != request.user_name:
                continue
            if request.private_only and not metadata.private:
                continue
            datasets.append(metadata)
        return {"datasets": datasets, "total": len(datasets)}
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {e}")
    finally:
        if storage_handler:
            storage_handler.unmount()
@app.post("/download-dataset")
async def download_dataset(request: DownloadDatasetRequest):
    """
    Download a dataset from ClearML and S3.
    Finds the dataset by name and user_name, gets S3 path from metadata, and downloads from S3.
    """
    try:
        logger.info(f"Starting download for dataset: {request.dataset_name}, user: {request.user_name}")
        
        # Step 1: Find the ClearML dataset by name
        datasets = Dataset.list_datasets(
            # Your project name for datasets
            partial_name=request.dataset_name,
            only_completed=True
        )
        
        if not datasets:
            raise HTTPException(
                status_code=404, 
                detail=f"No completed datasets found with name containing '{request.dataset_name}'"
            )
        
        # Step 2: Find the exact match by name and user from metadata
        target_dataset = None
        target_dataset_obj = None
        
        for dataset_dict in datasets:
            try:
                # Get the actual Dataset object to access metadata
                dataset_obj = Dataset.get(dataset_id=dataset_dict["id"])
                dataset_metadata = dataset_obj.get_metadata()
                
                logger.info(f"Checking dataset: {dataset_dict.get('name')}")
                logger.info(f"Metadata: {dataset_metadata}")
                
                # Check if this matches our criteria (check both metadata and dataset name)
                metadata_name_match = dataset_metadata.get("dataset_name") == request.dataset_name
                dict_name_match = dataset_dict.get("name") == request.dataset_name
                user_match = dataset_metadata.get("user_name") == request.user_name
                
                # Check if dataset has s3_path in metadata
                has_s3_path = dataset_metadata.get("s3_path") is not None
                
                logger.info(f"Name matches: metadata={metadata_name_match}, dict={dict_name_match}")
                logger.info(f"User match: {user_match}, Has S3 path: {has_s3_path}")
                
                if (metadata_name_match or dict_name_match) and user_match and has_s3_path:
                    target_dataset = dataset_dict
                    target_dataset_obj = dataset_obj
                    logger.info(f"Found matching dataset: {dataset_dict['name']} (ID: {dataset_dict['id']})")
                    break
                    
            except Exception as e:
                logger.warning(f"Could not get metadata for dataset {dataset_dict['id']}: {e}")
                continue
        
        if not target_dataset or not target_dataset_obj:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{request.dataset_name}' not found for user '{request.user_name}'"
            )
        
        logger.info(f"Found target dataset: {target_dataset['id']}")
        
        # Step 3: Extract S3 information from dataset metadata
        dataset_metadata = target_dataset_obj.get_metadata()
        s3_path = dataset_metadata.get("s3_path")
        
        if not s3_path:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset does not have S3 path information in metadata. Available metadata keys: {list(dataset_metadata.keys())}"
            )
        
        logger.info(f"Found S3 path: {s3_path}")
        
        # Step 4: Parse S3 path and extract S3 config from request
        # Expected format: s3://bucket-name/path/to/dataset
        if not s3_path.startswith("s3://"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid S3 path format: {s3_path}"
            )
        
        # Parse S3 path
        s3_path_parts = s3_path[5:].split("/", 1)  # Remove 's3://' and split
        if len(s3_path_parts) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid S3 path format: {s3_path}"
            )
        
        s3_bucket = s3_path_parts[0]
        s3_key = s3_path_parts[1]
        
        # Step 5: Initialize S3 handler with provided credentials
        s3_handler = S3Handler(
            access_key=request.access_key,
            secret_key=request.secret_key,
            endpoint_url=request.endpoint_url,
            bucket_name=s3_bucket
        )
        
        # Step 6: Create local download directory
        import tempfile
        from pathlib import Path
        
        download_base_dir = Path(tempfile.gettempdir()) / "dataset_downloads"
        download_base_dir.mkdir(exist_ok=True)
        
        local_download_path = download_base_dir / f"{request.dataset_name}_{request.user_name}"
        local_download_path.mkdir(exist_ok=True)
        
        # Step 7: Download the dataset from S3
        try:
            # Initialize S3 client
            import boto3
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=request.access_key,
                aws_secret_access_key=request.secret_key,
                endpoint_url=request.endpoint_url
            )
            
            # List all objects under the S3 key (in case it's a directory)
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=s3_bucket, Prefix=s3_key)
            
            downloaded_files = []
            total_size = 0
            
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        obj_key = obj['Key']
                        obj_size = obj['Size']
                        
                        # Create local file path maintaining directory structure
                        relative_path = obj_key[len(s3_key):].lstrip('/')
                        if not relative_path:  # This is the exact file
                            relative_path = Path(obj_key).name
                        
                        local_file_path = local_download_path / relative_path
                        local_file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Download the file
                        logger.info(f"Downloading {obj_key} to {local_file_path}")
                        s3_client.download_file(s3_bucket, obj_key, str(local_file_path))
                        
                        downloaded_files.append({
                            "s3_key": obj_key,
                            "local_path": str(local_file_path),
                            "size": obj_size
                        })
                        total_size += obj_size
            
            if not downloaded_files:
                raise HTTPException(
                    status_code=404,
                    detail=f"No files found at S3 path: {s3_path}"
                )
            
            logger.info(f"Successfully downloaded {len(downloaded_files)} files, total size: {total_size} bytes")
            
            return {
                "status": "success",
                "message": "Dataset downloaded successfully from S3",
                "dataset_info": {
                    "dataset_id": target_dataset["id"],
                    "dataset_name": request.dataset_name,
                    "user_name": request.user_name,
                    "clearml_project": target_dataset["project"],
                    "created": target_dataset["created"]
                },
                "s3_info": {
                    "bucket": s3_bucket,
                    "key": s3_key,
                    "full_path": s3_path
                },
                "download_info": {
                    "local_download_path": str(local_download_path),
                    "total_files": len(downloaded_files),
                    "total_size_bytes": total_size,
                    "files": downloaded_files
                }
            }
            
        except Exception as s3_error:
            logger.error(f"S3 download failed: {s3_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download from S3: {str(s3_error)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in download_dataset: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during dataset download: {str(e)}"
        )
@app.get("/clearml-datasets")
async def list_clearml_datasets(project_name: str = CLEARML_PROJECT_NAME):
    """List ClearML datasets in the project."""
    try:
        datasets = Dataset.list_datasets(
            dataset_project=project_name,
            partial_name="",
            only_completed=True
        )
        
        dataset_list = []
        for dataset_dict in datasets:
            dataset_list.append({
                "id": dataset_dict.get("id"),
                "name": dataset_dict.get("name"),
                "project": dataset_dict.get("project"),
                "created": dataset_dict.get("created"),
                "status": dataset_dict.get("status"),
                "size": dataset_dict.get("size", 0)
            })
        
        return {"datasets": dataset_list, "total": len(dataset_list)}
        
    except Exception as e:
        logger.error(f"Error listing ClearML datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list ClearML datasets: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7099)