import asyncio
import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
import tempfile
from pathlib import Path

# Import your existing models
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
    title="Multi-User Dataset Ingestion API with ClearML",
    description="API to download datasets from various sources using ClearML agents with per-user credentials.",
    version="3.0.0",
)

# Security
security = HTTPBearer()

# New models for user credentials and setup
class ClearMLCredentials(BaseModel):
    # web_host: str
    # api_host: str 
    # files_host: str
    access_key: str
    secret_key: str

class S3Credentials(BaseModel):
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    bucket_name: Optional[str] = None
    

class UserSetupRequest(BaseModel):
    user_id: str
    clearml_credentials: ClearMLCredentials
    # default_s3_config: S3Credentials

class UserCredentialsManager:
    """Manages per-user ClearML credentials in memory."""
    
    def __init__(self):
        self._user_credentials: Dict[str, ClearMLCredentials] = {}
        self._user_tokens: Dict[str, str] = {}  # token -> user_id mapping
    
    def setup_user(self, user_id: str, credentials: ClearMLCredentials) -> str:
        """Setup credentials for a user and return an access token."""
        token = f"user_{user_id}_{uuid.uuid4().hex[:16]}"
        self._user_credentials[user_id] = credentials
        self._user_tokens[token] = user_id
        logger.info(f"Setup credentials for user: {user_id}")
        return token
    
    def get_user_from_token(self, token: str) -> Optional[str]:
        """Get user_id from token."""
        return self._user_tokens.get(token)
    
    def get_user_credentials(self, user_id: str) -> Optional[ClearMLCredentials]:
        """Get credentials for a user."""
        return self._user_credentials.get(user_id)
    
    def remove_user(self, user_id: str):
        """Remove user credentials and invalidate all their tokens."""
        if user_id in self._user_credentials:
            del self._user_credentials[user_id]
        
        # Remove all tokens for this user
        tokens_to_remove = [token for token, uid in self._user_tokens.items() if uid == user_id]
        for token in tokens_to_remove:
            del self._user_tokens[token]
        
        logger.info(f"Removed credentials for user: {user_id}")

# Global credentials manager
credentials_manager = UserCredentialsManager()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract user from authorization token."""
    token = credentials.credentials
    user_id = credentials_manager.get_user_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id

class ClearMLTaskManager:
    """Manages ClearML task creation and enqueueing for dataset ingestion operations."""

    @staticmethod
    def create_task_with_user_credentials(
        task_name: str, 
        script_name: str, 
        parameters: Dict[str, Any],
        user_credentials: ClearMLCredentials
    ) -> 'Task':
        """Create a ClearML task with specific user credentials."""
        
        # Set environment variables for this specific task creation
        original_env = {}
        env_vars = {
            # 'CLEARML_WEB_HOST': user_credentials.web_host,
            # 'CLEARML_API_HOST': user_credentials.api_host,
            # 'CLEARML_FILES_HOST': user_credentials.files_host,
            'CLEARML_API_ACCESS_KEY': user_credentials.access_key,
            'CLEARML_API_SECRET_KEY': user_credentials.secret_key
        }
        
        # Backup original environment
        for key in env_vars:
            original_env[key] = os.environ.get(key)
            os.environ[key] = env_vars[key]
        
        try:
            # Import here to ensure it uses the updated environment
            from clearml import Task
            
            unique_id = uuid.uuid4().hex[:8]
            full_task_name = f"{task_name}_{unique_id}"

            task = Task.create(
                project_name="DatasetIngestion",
                task_name=full_task_name,
                task_type=Task.TaskTypes.data_processing,
                script="data_platform/ryn/api_test/processors/" + script_name,
                repo="https://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@github.com/AIP-MLOPS/tools.git",
                branch="data_layer",
                working_directory="data_platform/ryn/",
                packages=["datasets", "kaggle", "openml", "boto3", "pandas", "numpy", "fastapi", "kagglehub", "python-dotenv", "aipmodel==0.2.28"],
                docker=None
            )

            # Add user credentials to task parameters so the processor can use them
            parameters["clearml_credentials"] = user_credentials.dict()
            task.connect(parameters)

            logger.info(f"Created ClearML task: {full_task_name} with ID: {task.id}")
            return task
            
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    @staticmethod
    def enqueue_task_with_credentials(
        task: 'Task', 
        user_credentials: ClearMLCredentials,
        queue_name: str = "default"
    ) -> str:
        """Enqueue task with user-specific credentials."""
        
        # Set environment for enqueueing
        original_env = {}
        env_vars = {
            # 'CLEARML_WEB_HOST': user_credentials.web_host,
            # 'CLEARML_API_HOST': user_credentials.api_host,
            # 'CLEARML_FILES_HOST': user_credentials.files_host,
            'CLEARML_API_ACCESS_KEY': user_credentials.access_key,
            'CLEARML_API_SECRET_KEY': user_credentials.secret_key
        }
        
        for key in env_vars:
            original_env[key] = os.environ.get(key)
            os.environ[key] = env_vars[key]
        
        try:
            from clearml import Task
            Task.enqueue(task=task, queue_name=queue_name)
            logger.info(f"Successfully enqueued task {task.name} to queue '{queue_name}' with ID: {task.id}")
            return task.id
            
        except Exception as e:
            logger.error(f"Failed to enqueue task: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to enqueue task: {str(e)}")
        finally:
            # Restore environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

# API Endpoints

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "3.0.0", "queue_system": "ClearML", "multi_user": True}

@app.post("/setup-user")
async def setup_user(request: UserSetupRequest):
    """Setup ClearML credentials for a user and return access token."""
    try:
        token = credentials_manager.setup_user(request.user_id, request.clearml_credentials)
        return {
            "status": "success",
            "message": f"User '{request.user_id}' setup successfully",
            "access_token": token,
            "token_type": "bearer"
        }
    except Exception as e:
        logger.error(f"Error setting up user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to setup user: {str(e)}")

@app.delete("/remove-user")
async def remove_user(user_id: str = Depends(get_current_user)):
    """Remove user credentials and invalidate all tokens."""
    try:
        credentials_manager.remove_user(user_id)
        return {
            "status": "success",
            "message": f"User '{user_id}' removed successfully"
        }
    except Exception as e:
        logger.error(f"Error removing user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove user: {str(e)}")

@app.get("/queues")
async def list_queues(user_id: str = Depends(get_current_user)):
    """List available ClearML queues for the authenticated user."""
    try:
        user_credentials = credentials_manager.get_user_credentials(user_id)
        if not user_credentials:
            raise HTTPException(status_code=404, detail="User credentials not found")
        
        # Set environment temporarily to list queues
        original_env = {}
        env_vars = {
            # 'CLEARML_WEB_HOST': user_credentials.web_host,
            # 'CLEARML_API_HOST': user_credentials.api_host,
            # 'CLEARML_FILES_HOST': user_credentials.files_host,
            'CLEARML_API_ACCESS_KEY': user_credentials.access_key,
            'CLEARML_API_SECRET_KEY': user_credentials.secret_key
        }
        
        for key in env_vars:
            original_env[key] = os.environ.get(key)
            os.environ[key] = env_vars[key]
        
        try:
            from clearml.backend_api.session.client import APIClient
            client = APIClient()
            resp = client.queues.get_all()
            queues = []
            if hasattr(resp, 'queues'):
                queues = [{"name": q.name, "id": q.id} for q in resp.queues]
            
            return {"queues": queues, "total": len(queues)}
        finally:
            # Restore environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing queues: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list queues: {str(e)}")

@app.post("/huggingface")
async def ingest_huggingface(hf_req: HuggingFaceDownloadRequest, user_id: str = Depends(get_current_user)):
    """Create and enqueue a ClearML task for HuggingFace dataset ingestion."""
    try:
        user_credentials = credentials_manager.get_user_credentials(user_id)
        if not user_credentials:
            raise HTTPException(status_code=404, detail="User credentials not found")
        
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
        
        # Create ClearML task with user-specific credentials
        task = ClearMLTaskManager.create_task_with_user_credentials(
            task_name=f"HuggingFace_{hf_req.dataset_name.replace('/', '-')}",
            script_name="huggingface_processor.py",
            parameters=parameters,
            user_credentials=user_credentials
        )
        
        # Enqueue the task with user credentials
        task_id = ClearMLTaskManager.enqueue_task_with_credentials(task, user_credentials)
        
        return {
            "status": "success",
            "message": f"HuggingFace dataset ingestion task created and enqueued",
            "task_id": task_id,
            "dataset_name": hf_req.dataset_name,
            "user_id": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating HuggingFace ingestion task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create ingestion task: {str(e)}")

@app.post("/kaggle")
async def ingest_kaggle(request: KaggleDownloadRequest, user_id: str = Depends(get_current_user)):
    """Create and enqueue a ClearML task for Kaggle dataset ingestion."""
    try:
        user_credentials = credentials_manager.get_user_credentials(user_id)
        if not user_credentials:
            raise HTTPException(status_code=404, detail="User credentials not found")
        
        # Prepare task parameters
        parameters = {
            "task_type": "kaggle",
            "dataset_id": request.dataset_id,
            "user_name": request.user_name,
            "kaggle_username": request.kaggle_username,
            "kaggle_key": request.kaggle_key,
            "private": request.private
        }
        
        # Create ClearML task with user-specific credentials
        task = ClearMLTaskManager.create_task_with_user_credentials(
            task_name=f"Kaggle_{request.dataset_id}",
            script_name="kaggle_processor.py",
            parameters=parameters,
            user_credentials=user_credentials
        )
        
        # Enqueue the task with user credentials
        task_id = ClearMLTaskManager.enqueue_task_with_credentials(task, user_credentials)
        
        return {
            "status": "success",
            "message": f"Kaggle dataset ingestion task created and enqueued",
            "task_id": task_id,
            "dataset_id": request.dataset_id,
            "user_id": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating Kaggle ingestion task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create ingestion task: {str(e)}")

@app.post("/openml")
async def ingest_openml(request: OpenMLDownloadRequest, user_id: str = Depends(get_current_user)):
    """Create and enqueue a ClearML task for OpenML dataset ingestion."""
    try:
        user_credentials = credentials_manager.get_user_credentials(user_id)
        if not user_credentials:
            raise HTTPException(status_code=404, detail="User credentials not found")
        
        # Prepare task parameters
        parameters = {
            "task_type": "openml",
            "dataset_id": request.dataset_id,
            "dataset_name": request.dataset_name,
            "user_name": request.user_name,
            "private": request.private
        }
        
        # Create ClearML task with user-specific credentials
        task = ClearMLTaskManager.create_task_with_user_credentials(
            task_name=f"OpenML_{request.dataset_name or request.dataset_id}",
            script_name="openml_processor.py",
            parameters=parameters,
            user_credentials=user_credentials
        )
        
        # Enqueue the task with user credentials
        task_id = ClearMLTaskManager.enqueue_task_with_credentials(task, user_credentials)
        
        return {
            "status": "success",
            "message": f"OpenML dataset ingestion task created and enqueued",
            "task_id": task_id,
            "dataset_id": request.dataset_id,
            "user_id": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating OpenML ingestion task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create ingestion task: {str(e)}")

@app.post("/s3")
async def ingest_s3(request: Request, s3_request: S3DownloadRequest, user_id: str = Depends(get_current_user)):
    """Create and enqueue a ClearML task for S3 dataset ingestion."""
    try:
        user_credentials = credentials_manager.get_user_credentials(user_id)
        if not user_credentials:
            raise HTTPException(status_code=404, detail="User credentials not found")
        
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
            "dataset_tag": s3_request.dataset_tag,
            "s3_config_target": {
                "endpoint_url_target": s3_request.endpoint_url_target,
                "access_key_target": s3_request.access_key_target,
                "secret_key_target": s3_request.secret_key_target,
                "bucket_name_target": s3_request.bucket_name_target,
            }
        }
        
        # Create ClearML task with user-specific credentials
        task = ClearMLTaskManager.create_task_with_user_credentials(
            task_name=f"S3_{s3_request.dataset_name}",
            script_name="s3_processor.py",
            parameters=parameters,
            user_credentials=user_credentials
        )
        
        # Enqueue the task with user credentials
        task_id = ClearMLTaskManager.enqueue_task_with_credentials(task, user_credentials)
        
        return {
            "status": "success",
            "message": f"S3 dataset ingestion task created and enqueued",
            "task_id": task_id,
            "dataset_name": s3_request.dataset_name,
            "user_id": user_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating S3 ingestion task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create ingestion task: {str(e)}")

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str, user_id: str = Depends(get_current_user)):
    """Get the status of a ClearML task."""
    try:
        user_credentials = credentials_manager.get_user_credentials(user_id)
        if not user_credentials:
            raise HTTPException(status_code=404, detail="User credentials not found")
        
        # Set environment for task status check
        original_env = {}
        env_vars = {
            'CLEARML_WEB_HOST': user_credentials.web_host,
            'CLEARML_API_HOST': user_credentials.api_host,
            'CLEARML_FILES_HOST': user_credentials.files_host,
            'CLEARML_API_ACCESS_KEY': user_credentials.access_key,
            'CLEARML_API_SECRET_KEY': user_credentials.secret_key
        }
        
        for key in env_vars:
            original_env[key] = os.environ.get(key)
            os.environ[key] = env_vars[key]
        
        try:
            from clearml import Task
            task = Task.get_task(task_id=task_id)
            return {
                "task_id": task_id,
                "name": task.name,
                "status": task.get_status(),
                "project": task.get_project_name(),
                "created": task.data.created,
                "last_update": task.data.last_update,
                "user_id": user_id
            }
        finally:
            # Restore environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
                    
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=404, detail=f"Task not found or error retrieving status: {str(e)}")

@app.get("/tasks")
async def list_tasks(user_id: str = Depends(get_current_user), limit: int = 50):
    """List recent ClearML tasks for the authenticated user."""
    try:
        user_credentials = credentials_manager.get_user_credentials(user_id)
        if not user_credentials:
            raise HTTPException(status_code=404, detail="User credentials not found")
        
        # Set environment temporarily
        original_env = {}
        env_vars = {
            'CLEARML_WEB_HOST': user_credentials.web_host,
            'CLEARML_API_HOST': user_credentials.api_host,
            'CLEARML_FILES_HOST': user_credentials.files_host,
            'CLEARML_API_ACCESS_KEY': user_credentials.access_key,
            'CLEARML_API_SECRET_KEY': user_credentials.secret_key
        }
        
        for key in env_vars:
            original_env[key] = os.environ.get(key)
            os.environ[key] = env_vars[key]
        
        try:
            from clearml.backend_api.session.client import APIClient
            client = APIClient()
            
            # Get project ID
            projects = client.projects.get_all(name="DatasetIngestion")
            if not projects.projects:
                return {"tasks": [], "message": "No project found with name: DatasetIngestion"}
            
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
            
            return {"tasks": tasks, "total": len(tasks), "user_id": user_id}
        finally:
            # Restore environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

@app.post("/list-datasets")
async def list_datasets(request: ListDatasetsRequest, user_id: str = Depends(get_current_user)):
    """List available datasets for the authenticated user."""
    storage_handler = None
    try:
        storage_handler = DatasetStorageHandler("temp_mount")
        datasets = []
        for metadata_file in storage_handler.metadata_dir.glob("*.json"):
            metadata = storage_handler.load_metadata(metadata_file.stem)
            # Filter by authenticated user
            if metadata.user_name != user_id:
                continue
            if request.user_name and metadata.user_name != request.user_name:
                continue
            if request.private_only and not metadata.private:
                continue
            datasets.append(metadata)
        return {"datasets": datasets, "total": len(datasets), "user_id": user_id}
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {e}")
    finally:
        if storage_handler:
            storage_handler.unmount()

@app.post("/download-dataset")
async def download_dataset(request: DownloadDatasetRequest, user_id: str = Depends(get_current_user)):
    """Download a dataset from ClearML and S3 for the authenticated user."""
    try:
        user_credentials = credentials_manager.get_user_credentials(user_id)
        if not user_credentials:
            raise HTTPException(status_code=404, detail="User credentials not found")
        
        logger.info(f"Starting download for dataset: {request.dataset_name}, user: {user_id}")
        
        # Set environment for ClearML operations
        original_env = {}
        env_vars = {
            'CLEARML_WEB_HOST': user_credentials.web_host,
            'CLEARML_API_HOST': user_credentials.api_host,
            'CLEARML_FILES_HOST': user_credentials.files_host,
            'CLEARML_API_ACCESS_KEY': user_credentials.access_key,
            'CLEARML_API_SECRET_KEY': user_credentials.secret_key
        }
        
        for key in env_vars:
            original_env[key] = os.environ.get(key)
            os.environ[key] = env_vars[key]
        
        try:
            from clearml import Dataset
            
            # Find the ClearML dataset by name
            datasets = Dataset.list_datasets(
                partial_name=request.dataset_name,
                only_completed=True
            )
            
            if not datasets:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No completed datasets found with name containing '{request.dataset_name}'"
                )
            
            # Find exact match by name and user
            target_dataset = None
            target_dataset_obj = None
            
            for dataset_dict in datasets:
                try:
                    dataset_obj = Dataset.get(dataset_id=dataset_dict["id"])
                    dataset_metadata = dataset_obj.get_metadata()
                    
                    metadata_name_match = dataset_metadata.get("dataset_name") == request.dataset_name
                    dict_name_match = dataset_dict.get("name") == request.dataset_name
                    user_match = dataset_metadata.get("user_name") == user_id  # Use authenticated user
                    has_s3_path = dataset_metadata.get("s3_path") is not None
                    
                    if (metadata_name_match or dict_name_match) and user_match and has_s3_path:
                        target_dataset = dataset_dict
                        target_dataset_obj = dataset_obj
                        break
                        
                except Exception as e:
                    logger.warning(f"Could not get metadata for dataset {dataset_dict['id']}: {e}")
                    continue
            
            if not target_dataset or not target_dataset_obj:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset '{request.dataset_name}' not found for user '{user_id}'"
                )
            
            # Extract S3 information and proceed with download
            dataset_metadata = target_dataset_obj.get_metadata()
            s3_path = dataset_metadata.get("s3_path")
            
            if not s3_path:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset does not have S3 path information in metadata"
                )
            
            # Parse S3 path
            if not s3_path.startswith("s3://"):
                raise HTTPException(status_code=400, detail=f"Invalid S3 path format: {s3_path}")
            
            s3_path_parts = s3_path[5:].split("/", 1)
            if len(s3_path_parts) < 2:
                raise HTTPException(status_code=400, detail=f"Invalid S3 path format: {s3_path}")
            
            s3_bucket = s3_path_parts[0]
            s3_key = s3_path_parts[1]
            
            # Initialize S3 handler with provided credentials
            s3_handler = S3Handler(
                access_key=request.access_key,
                secret_key=request.secret_key,
                endpoint_url=request.endpoint_url,
                bucket_name=s3_bucket
            )
            
            # Create local download directory
            download_base_dir = Path(tempfile.gettempdir()) / "dataset_downloads"
            download_base_dir.mkdir(exist_ok=True)
            
            local_download_path = download_base_dir / f"{request.dataset_name}_{user_id}"
            local_download_path.mkdir(exist_ok=True)
            
            # Download logic (same as your original code)
            import boto3
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=request.access_key,
                aws_secret_access_key=request.secret_key,
                endpoint_url=request.endpoint_url
            )
            
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=s3_bucket, Prefix=s3_key)
            
            downloaded_files = []
            total_size = 0
            
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        obj_key = obj['Key']
                        obj_size = obj['Size']
                        
                        relative_path = obj_key[len(s3_key):].lstrip('/')
                        if not relative_path:
                            relative_path = Path(obj_key).name
                        
                        local_file_path = local_download_path / relative_path
                        local_file_path.parent.mkdir(parents=True, exist_ok=True)
                        
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
            
            return {
                "status": "success",
                "message": "Dataset downloaded successfully from S3",
                "dataset_info": {
                    "dataset_id": target_dataset["id"],
                    "dataset_name": request.dataset_name,
                    "user_name": user_id,
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
        finally:
            # Restore environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in download_dataset: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during dataset download: {str(e)}"
        )

@app.get("/clearml-datasets")
async def list_clearml_datasets(user_id: str = Depends(get_current_user)):
    """List ClearML datasets for the authenticated user."""
    try:
        user_credentials = credentials_manager.get_user_credentials(user_id)
        if not user_credentials:
            raise HTTPException(status_code=404, detail="User credentials not found")
        
        # Set environment temporarily
        original_env = {}
        env_vars = {
            'CLEARML_WEB_HOST': user_credentials.web_host,
            'CLEARML_API_HOST': user_credentials.api_host,
            'CLEARML_FILES_HOST': user_credentials.files_host,
            'CLEARML_API_ACCESS_KEY': user_credentials.access_key,
            'CLEARML_API_SECRET_KEY': user_credentials.secret_key
        }
        
        for key in env_vars:
            original_env[key] = os.environ.get(key)
            os.environ[key] = env_vars[key]
        
        try:
            from clearml import Dataset
            datasets = Dataset.list_datasets(
                dataset_project="DatasetIngestion",
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
            
            return {"datasets": dataset_list, "total": len(dataset_list), "user_id": user_id}
        finally:
            # Restore environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing ClearML datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list ClearML datasets: {str(e)}")

@app.get("/user-info")
async def get_user_info(user_id: str = Depends(get_current_user)):
    """Get information about the authenticated user."""
    try:
        user_credentials = credentials_manager.get_user_credentials(user_id)
        if not user_credentials:
            raise HTTPException(status_code=404, detail="User credentials not found")
        
        return {
            "user_id": user_id,
            "clearml_web_host": user_credentials.web_host,
            "clearml_api_host": user_credentials.api_host,
            "clearml_files_host": user_credentials.files_host,
            "has_credentials": True
        }
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8179)