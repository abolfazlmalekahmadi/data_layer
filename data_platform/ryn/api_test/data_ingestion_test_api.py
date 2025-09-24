import io
import logging
import os
import uuid
import zipfile
from typing import Any, Dict
from urllib.parse import quote
import boto3
from botocore.exceptions import ClientError
from threading import Lock

from clearml.backend_api.session import Session

from contextlib import contextmanager


from data.data_ingestion.models.requests import (
    DownloadDatasetRequest,
    HuggingFaceDownloadRequest,
    KaggleDownloadRequest,
    OpenMLDownloadRequest,
    S3DownloadRequest,
    TaskStatusRequest,
    ListClearMLDatasetsRequest,
    ListTasksRequest,
    S3Credentials,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-User Dataset Ingestion API with ClearML",
    description="API to download datasets from various sources using ClearML agents with direct credentials.",
    version="3.0.0",
)

# Global lock for environment variable access
env_lock = Lock()

# main.py (add this utility at the top or in a helpers.py file)

@contextmanager
def clearml_client_session(credentials: dict):
    """
    A context manager that creates an explicit, isolated ClearML Session and APIClient
    based on the provided user credentials.
    """
    # 1. Create the explicit Session object with credentials
    session = Session(
        api_key=credentials.get("access_key"),
        secret_key=credentials.get("secret_key"),
        # You might need to specify the api_server if not using the default
        # host="https://api.clear.ml"
    )
    
    try:
        # 3. Yield the fully configured client for the 'with' block to use
        yield
    finally:
        logger.info("ClearML session restored and client cleaned up.")


class ClearMLTaskManager:
    """Manages ClearML task creation and enqueueing for dataset ingestion operations."""

    @staticmethod
    def create_and_enqueue_task(
        task_name: str,
        script_name: str,
        parameters: Dict[str, Any],
        user_credentials: Dict[str, str],
        queue_name: str = "default",
    ) -> str:  # Return the task_id (string)
        """
        Creates and immediately enqueues a ClearML task within a single, consistent context.
        """
        # Enter the context ONCE for both operations
        with clearml_client_session(user_credentials):
            logging.info(
                f"Creating and enqueuing with credentials: KEY={user_credentials.get('access_key')}"
            )

            # Import here to ensure it uses the updated environment
            from clearml import Task

            Task.set_credentials(key=user_credentials.get("access_key"),secret=user_credentials.get("secret_key"))

            unique_id = uuid.uuid4().hex[:8]
            full_task_name = f"{task_name}_{unique_id}"
            git_token = "https://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"
            task = Task.create(
                project_name=f"DatasetIngestion_{parameters.get('user_name')}",
                task_name=full_task_name,
                task_type=Task.TaskTypes.data_processing,
                script="data_platform/ryn/api_test/processors/" + script_name,
                repo=f"{git_token}@github.com/AIP-MLOPS/tools.git",
                branch="data_layer_dev",
                working_directory="data_platform/ryn/",
                packages=[
                    "datasets",
                    "kaggle",
                    "openml",
                    "boto3",
                    "pandas",
                    "numpy",
                    "fastapi",
                    "kagglehub",
                    "python-dotenv",
                    "aipmodel==0.2.28",
                    "opencv-python",
                    "opencv-python-headless",
                    "polars"
                ],
                docker=None,
            )

            parameters["clearml_credentials"] = user_credentials
            task.connect(parameters)
            logger.info(f"Created ClearML task: {full_task_name} with ID: {task.id}")

            # --- 2. ENQUEUE THE TASK (using the same Task class) ---
            try:
                Task.enqueue(task=task, queue_name=queue_name)
                logger.info(
                    f"Successfully enqueued task {task.name} to queue '{queue_name}' with ID: {task.id}"
                )
                return task.id
            except Exception as e:
                # Log the full exception to see the real error
                logger.error(f"Failed to enqueue task {task.id}: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500, detail=f"Failed to enqueue task: {str(e)}"
                )


# API Endpoints


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "3.0.0",
        "queue_system": "ClearML",
        "multi_user": True,
    }


# @app.post("/queues")
# async def list_queues(request: ClearMLCredentials):
#     """List available ClearML queues using provided credentials."""
#     try:
#         # Set environment temporarily to list queues
#         original_env = {}
#         env_vars = {
#             'CLEARML_API_ACCESS_KEY': request.access_key,
#             'CLEARML_API_SECRET_KEY': request.secret_key
#         }

#         for key in env_vars:
#             original_env[key] = os.environ.get(key)
#             os.environ[key] = env_vars[key]

#         try:
#             from clearml.backend_api.session.client import APIClient
#             client = APIClient()
#             resp = client.queues.get_all()
#             queues = []
#             if hasattr(resp, 'queues'):
#                 queues = [{"name": q.name, "id": q.id} for q in resp.queues]

#             return {"queues": queues, "total": len(queues)}
#         finally:
#             # Restore environment
#             for key, value in original_env.items():
#                 if value is None:
#                     os.environ.pop(key, None)
#                 else:
#                     os.environ[key] = value

#     except Exception as e:
#         logger.error(f"Error listing queues: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to list queues: {str(e)}")


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
            "revision": hf_req.revision or "",
            "s3_config": hf_req.s3_config.dict(),
            "clearml_config": hf_req.clearml.dict(),
        }
        user_credentials = parameters["clearml_config"]
        logger.info(parameters["s3_config"])
        # Create ClearML task with user-specific credentials

        task_id = ClearMLTaskManager.create_and_enqueue_task(
            task_name=f"HuggingFace_{hf_req.dataset_name.replace('/', '-')}",
            script_name="huggingface_processor.py",
            parameters=parameters,
            user_credentials=user_credentials,
            queue_name=hf_req.queue_name,
        )

        return {
            "status": "success",
            "message": "HuggingFace dataset ingestion task created and enqueued",
            "task_id": task_id,
            "dataset_name": hf_req.dataset_name,
            "user_name": hf_req.user_name,
            "queue_name": hf_req.queue_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating HuggingFace ingestion task: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create ingestion task: {str(e)}"
        )


@app.post("/kaggle")
async def ingest_kaggle(request: KaggleDownloadRequest):
    """Create and enqueue a ClearML task for Kaggle dataset ingestion."""
    try:
        # Prepare task parameters
        parameters = {
            "task_type": "kaggle",
            "dataset_id": request.dataset_id,
            "user_name": request.user_name,
            "kaggle_username": request.kaggle.user_name,
            "kaggle_key": request.kaggle.key,
            "private": request.private,
            "dataset_tag": request.dataset_tag or "",
            "s3_config": request.s3_config.dict(),
            "clearml_config": request.clearml.dict(),
        }
        user_credentials = parameters["clearml_config"]

        task_id = ClearMLTaskManager.create_and_enqueue_task(
            task_name=f"Kaggle_{request.dataset_id}",
            script_name="kaggle_processor.py",
            parameters=parameters,
            user_credentials=user_credentials,
            queue_name=request.queue_name,
        )
        return {
            "status": "success",
            "message": "Kaggle dataset ingestion task created and enqueued",
            "task_id": task_id,
            "dataset_id": request.dataset_id,
            "user_name": request.user_name,
            "queue_name": request.queue_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating Kaggle ingestion task: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create ingestion task: {str(e)}"
        )


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
            "private": request.private,
            "dataset_tag": request.dataset_tag or "",
            "s3_config": request.s3_config.dict(),
            "clearml_config": request.clearml.dict(),
        }
        user_credentials = parameters["clearml_config"]

        task_id = ClearMLTaskManager(
            task_name=f"OpenML_{request.dataset_name or request.dataset_id}",
            script_name="openml_processor.py",
            parameters=parameters,
            user_credentials=user_credentials,
            queue_name=request.queue_name,
        )

        return {
            "status": "success",
            "message": f"OpenML dataset ingestion task created and enqueued to queue '{request.queue_name}'",
            "task_id": task_id,
            "dataset_id": request.dataset_id,
            "user_name": request.user_name,
            "queue_name": request.queue_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating OpenML ingestion task: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create ingestion task: {str(e)}"
        )


@app.post("/s3")
async def ingest_s3(request: Request, s3_request: S3DownloadRequest):
    """Create and enqueue a ClearML task for S3 dataset ingestion."""
    try:
        raw_body = await request.body()
        logger.info(f"Raw request body: {raw_body.decode()}")

        # --- REFACTORED PARAMETER PREPARATION ---

        # Use a default empty S3Credentials object if none is provided to avoid errors
        source_creds = s3_request.s3_source or S3Credentials()
        target_creds = s3_request.s3_target or S3Credentials()

        # Prepare task parameters by accessing the nested objects
        parameters = {
            "task_type": "s3",
            "access_key": source_creds.access_key,
            "secret_key": source_creds.secret_key,
            "endpoint_url": source_creds.endpoint_url,
            "bucket_name": source_creds.bucket_name,
            # Core dataset parameters
            "s3_file_path": s3_request.s3_file_path,
            "dataset_name": s3_request.dataset_name,
            "user_name": s3_request.user_name,
            "private": s3_request.private,
            "dataset_tag": s3_request.dataset_tag,
            "s3_config_target": {
                "endpoint_url_target": target_creds.endpoint_url,
                "access_key_target": target_creds.access_key,
                "secret_key_target": target_creds.secret_key,
                "bucket_name_target": target_creds.bucket_name,
            },
            # ClearML config - Pydantic's .dict() is perfect for this
            "clearml_config": s3_request.clearml.dict(),
        }

        # This part remains the same as it was already well-structured
        user_credentials = parameters["clearml_config"]

        task_id = ClearMLTaskManager.create_and_enqueue_task(
            task_name=f"S3_{s3_request.dataset_name.replace('/', '-')}",
            script_name="s3_processor.py",
            parameters=parameters,
            user_credentials=user_credentials,
            queue_name=s3_request.queue_name,
        )

        return {
            "status": "success",
            "message": f"S3 dataset ingestion task created and enqueued to queue '{s3_request.queue_name}'",
            "task_id": task_id,
            "dataset_name": s3_request.dataset_name,
            "user_name": s3_request.user_name,
            "queue_name": s3_request.queue_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating S3 ingestion task: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to create ingestion task: {str(e)}"
        )


@app.post("/task-status")
async def get_task_status(request: TaskStatusRequest):
    """Get the status of a ClearML task using provided credentials."""
    try:
        credentials = {
            "access_key": request.clearml.access_key,
            "secret_key": request.clearml.secret_key,
        }
        with clearml_client_session(credentials=credentials):
            from clearml import Task

            task = Task.get_task(task_id=request.task_id)
            return {
                "task_id": request.task_id,
                "name": task.name,
                "status": task.get_status(),
                "project": task.get_project_name(),
                "created": task.data.created,
                "last_update": task.data.last_update,
            }
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Task not found or error retrieving status: {str(e)}",
        )


# Updated model for listing tasks


@app.post("/tasks")
async def list_tasks(request: ListTasksRequest):
    """List recent ClearML tasks using provided credentials."""
    try:
        credentials = {
            "access_key": request.clearml.access_key,
            "secret_key": request.clearml.secret_key,
        }
        with clearml_client_session(credentials=credentials):
            from clearml.backend_api.session.client import APIClient

            client = APIClient()

            # Get project ID
            projects = client.projects.get_all(
                name=f"DatasetIngestion_{request.user_name}"
            )
            logger.info(projects)
            if not projects:
                return {
                    "tasks": [],
                    "message": "No project found with name: DatasetIngestion",
                }

            project_id = projects[0].id
            logger.info(project_id)
            # Get tasks for the project
            tasks_response = client.tasks.get_all(
                project=[project_id], page_size=request.limit, order_by=["-created"]
            )
            logger.info(tasks_response)
            tasks = []
            for task in tasks_response:
                tasks.append(
                    {
                        "id": task.id,
                        "name": task.name,
                        "status": task.status,
                        "created": task.created,
                        "last_update": task.last_update,
                        "type": task.type,
                    }
                )

            return {"tasks": tasks, "total": len(tasks)}
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")


@app.post("/clearml-datasets")
async def list_clearml_datasets(request: ListClearMLDatasetsRequest):
    """List ClearML datasets using provided credentials."""
    try:
        credentials = {
            "access_key": request.clearml.access_key,
            "secret_key": request.clearml.secret_key,
        }
        with clearml_client_session(credentials=credentials):
            from clearml import Dataset

            datasets = Dataset.list_datasets(
                dataset_project=f"DatasetIngestion_{request.user_name}",
                partial_name="",
                only_completed=True,
            )

            dataset_list = []
            for dataset_dict in datasets:
                dataset_list.append(
                    {
                        "id": dataset_dict.get("id"),
                        "name": dataset_dict.get("name"),
                        "project": dataset_dict.get("project"),
                        "created": dataset_dict.get("created"),
                        "status": dataset_dict.get("status"),
                        "size": dataset_dict.get("size", 0),
                    }
                )

            return {"datasets": dataset_list, "total": len(dataset_list)}

    except Exception as e:
        logger.error(f"Error listing ClearML datasets: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list ClearML datasets: {str(e)}"
        )


# Helper function to generate presigned URLs
def generate_presigned_urls(
    s3_client, bucket_name: str, s3_key: str, expiration: int = 3600
):
    """Generate presigned URLs for all files under the S3 path"""
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_key)

        presigned_urls = []

        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    obj_key = obj["Key"]
                    obj_size = obj["Size"]

                    # Skip directories
                    if obj_key.endswith("/"):
                        continue

                    try:
                        presigned_url = s3_client.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": bucket_name, "Key": obj_key},
                            ExpiresIn=expiration,
                        )

                        # Extract filename from key
                        filename = obj_key.split("/")[-1]
                        relative_path = obj_key[len(s3_key) :].lstrip("/")
                        if not relative_path:
                            relative_path = filename

                        presigned_urls.append(
                            {
                                "filename": filename,
                                "relative_path": relative_path,
                                "s3_key": obj_key,
                                "size": obj_size,
                                "download_url": presigned_url,
                            }
                        )
                    except ClientError as e:
                        logger.error(
                            f"Error generating presigned URL for {obj_key}: {e}"
                        )
                        continue

        return presigned_urls
    except Exception as e:
        logger.error(f"Error generating presigned URLs: {e}")
        raise


# Helper function to create streaming zip
def create_streaming_zip(s3_client, bucket_name: str, s3_key: str):
    """Create a streaming zip response of S3 files"""

    def generate_zip():
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            try:
                paginator = s3_client.get_paginator("list_objects_v2")
                page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_key)

                for page in page_iterator:
                    if "Contents" in page:
                        for obj in page["Contents"]:
                            obj_key = obj["Key"]

                            # Skip directories
                            if obj_key.endswith("/"):
                                continue

                            try:
                                # Get object from S3
                                response = s3_client.get_object(
                                    Bucket=bucket_name, Key=obj_key
                                )
                                file_data = response["Body"].read()

                                # Add to zip with relative path
                                relative_path = obj_key[len(s3_key) :].lstrip("/")
                                if not relative_path:
                                    relative_path = obj_key.split("/")[-1]

                                zip_file.writestr(relative_path, file_data)
                                logger.info(
                                    f"Added {obj_key} to zip as {relative_path}"
                                )

                            except Exception as e:
                                logger.error(f"Error adding {obj_key} to zip: {e}")
                                continue

            except Exception as e:
                logger.error(f"Error creating zip: {e}")
                raise

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    return generate_zip()


# Helper function for streaming individual files
async def stream_files_info(
    s3_client, bucket_name: str, s3_key: str, expiration: int = 3600
):
    """Get streaming URLs and file information"""
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_key)

        files_info = []

        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    obj_key = obj["Key"]
                    obj_size = obj["Size"]

                    # Skip directories
                    if obj_key.endswith("/"):
                        continue

                    # Generate streaming URL
                    streaming_url = (
                        f"/stream-file/{quote(bucket_name)}/{quote(obj_key)}"
                    )

                    filename = obj_key.split("/")[-1]
                    relative_path = obj_key[len(s3_key) :].lstrip("/")
                    if not relative_path:
                        relative_path = filename

                    files_info.append(
                        {
                            "filename": filename,
                            "relative_path": relative_path,
                            "s3_key": obj_key,
                            "size": obj_size,
                            "streaming_url": streaming_url,
                        }
                    )

        return files_info
    except Exception as e:
        logger.error(f"Error getting files info for streaming: {e}")
        raise


@app.post("/download-dataset")
async def download_dataset(request: DownloadDatasetRequest):
    """Download a dataset with multiple download methods using provided credentials."""
    try:
        logger.info(
            f"Starting download for dataset: {request.dataset_name}, user: {request.user_name}, method: {request.download_method} , clearml_access: {request.clearml.access_key} , clearml_secret: {request.clearml.secret_key} "
        )

        credentials = {
            "access_key": request.clearml.access_key,
            "secret_key": request.clearml.secret_key,
        }
        with clearml_client_session(credentials=credentials):
            from clearml import Dataset

            # Find the ClearML dataset by name
            datasets = Dataset.list_datasets(
                partial_name=request.dataset_name, only_completed=True
            )
            print(f"list of datasets: {datasets}")
            if not datasets:
                raise HTTPException(
                    status_code=404,
                    detail=f"No completed datasets found with name containing '{request.dataset_name}'",
                )

            # Find exact match by name and user
            target_dataset = None
            target_dataset_obj = None

            for dataset_dict in datasets:
                try:
                    dataset_obj = Dataset.get(dataset_id=dataset_dict["id"])
                    dataset_metadata = dataset_obj.get_metadata()
                    # logging.info(f"username of dataset: {dataset_metadata.get("user_name")} ")

                    metadata_name_match = (
                        dataset_metadata.get("dataset_name") == request.dataset_name
                    )
                    dict_name_match = dataset_dict.get("name") == request.dataset_name
                    user_match = dataset_metadata.get("user_name") == request.user_name
                    has_s3_path = dataset_metadata.get("s3_path") is not None

                    if (
                        (metadata_name_match or dict_name_match)
                        and user_match
                        and has_s3_path
                    ):
                        target_dataset = dataset_dict
                        target_dataset_obj = dataset_obj
                        break

                except Exception as e:
                    logger.warning(
                        f"Could not get metadata for dataset {dataset_dict['id']}: {e}"
                    )
                    continue

            if not target_dataset or not target_dataset_obj:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset '{request.dataset_name}' not found for user '{request.user_name}'",
                )

            # Extract S3 information
            dataset_metadata = target_dataset_obj.get_metadata()
            s3_path = dataset_metadata.get("s3_path")

            if not s3_path:
                raise HTTPException(
                    status_code=400,
                    detail="Dataset does not have S3 path information in metadata",
                )

            # Parse S3 path
            if not s3_path.startswith("s3://"):
                raise HTTPException(
                    status_code=400, detail=f"Invalid S3 path format: {s3_path}"
                )

            s3_path_parts = s3_path[5:].split("/", 1)
            if len(s3_path_parts) < 2:
                raise HTTPException(
                    status_code=400, detail=f"Invalid S3 path format: {s3_path}"
                )

            s3_bucket = s3_path_parts[0]
            s3_key = s3_path_parts[1]

            # Initialize S3 client
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=request.s3_config.access_key,
                aws_secret_access_key=request.s3_config.secret_key,
                endpoint_url=request.s3_config.endpoint_url,
            )

            # Handle different download methods
            if request.download_method == "presigned_urls":
                # Generate presigned URLs for all files
                presigned_urls = generate_presigned_urls(
                    s3_client, s3_bucket, s3_key, request.expiration
                )

                return {
                    "status": "success",
                    "download_method": "presigned_urls",
                    "message": "Presigned URLs generated successfully",
                    "dataset_info": {
                        "dataset_id": target_dataset["id"],
                        "dataset_name": request.dataset_name,
                        "user_name": request.user_name,
                        "clearml_project": target_dataset["project"],
                        "created": target_dataset["created"],
                    },
                    "s3_info": {
                        "bucket": s3_bucket,
                        "key": s3_key,
                        "full_path": s3_path,
                    },
                    "download_info": {
                        "total_files": len(presigned_urls),
                        "expiration_seconds": request.expiration,
                        "files": presigned_urls,
                    },
                    "instructions": {
                        "description": "Use the download_url for each file to download directly to your local machine",
                        "note": f"URLs will expire in {request.expiration} seconds ({request.expiration // 3600} hours)",
                    },
                }

            elif request.download_method == "stream_zip":
                # Create and stream a zip file
                zip_data = create_streaming_zip(s3_client, s3_bucket, s3_key)

                zip_filename = f"{request.dataset_name}.zip"

                return StreamingResponse(
                    io.BytesIO(zip_data),
                    media_type="application/zip",
                    headers={
                        "Content-Disposition": f"attachment; filename={zip_filename}",
                        "Content-Length": str(len(zip_data)),
                    },
                )

            elif request.download_method == "direct_stream":
                # Provide streaming endpoints for individual files
                files_info = await stream_files_info(
                    s3_client, s3_bucket, s3_key, request.expiration
                )

                return {
                    "status": "success",
                    "download_method": "direct_stream",
                    "message": "Streaming URLs generated successfully",
                    "dataset_info": {
                        "dataset_id": target_dataset["id"],
                        "dataset_name": request.dataset_name,
                        "user_name": request.user_name,
                        "clearml_project": target_dataset["project"],
                        "created": target_dataset["created"],
                    },
                    "s3_info": {
                        "bucket": s3_bucket,
                        "key": s3_key,
                        "full_path": s3_path,
                    },
                    "download_info": {
                        "total_files": len(files_info),
                        "files": files_info,
                    },
                    "instructions": {
                        "description": "Use the streaming_url for each file to stream download directly",
                        "base_url": "Your API base URL",
                        "note": "These URLs stream files directly through this API",
                    },
                }

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid download_method: {request.download_method}. Valid options: presigned_urls, stream_zip, direct_stream",
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in download_dataset: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal error during dataset download: {str(e)}"
        )


# Updated model for listing datasets with direct credentials
# class ListDatasetsRequest(BaseModel):
#     clearml_access_key: str
#     clearml_secret_key: str
#     user_name: Optional[str] = None
#     private_only: Optional[bool] = False

# @app.post("/list-datasets")
# async def list_datasets(request: ListDatasetsRequest):
#     """List available datasets using provided credentials."""
#     try:
#         # Set environment for ClearML operations
#         original_env = {}
#         env_vars = {
#             'CLEARML_API_ACCESS_KEY': request.clearml_access_key,
#             'CLEARML_API_SECRET_KEY': request.clearml_secret_key
#         }

#         for key in env_vars:
#             original_env[key] = os.environ.get(key)
#             os.environ[key] = env_vars[key]

#         try:
#             from clearml import Dataset

#             # Get all datasets from the project
#             datasets = Dataset.list_datasets(
#                 dataset_project="DatasetIngestion",
#                 partial_name="",
#                 only_completed=True
#             )
#             print(datasets)

#             filtered_datasets = []

#             for dataset_dict in datasets:
#                 try:
#                     dataset_obj = Dataset.get(dataset_id=dataset_dict["id"])
#                     dataset_metadata = dataset_obj.get_metadata()

#                     # Apply filters
#                     if request.user_name and dataset_metadata.get("user_name") != request.user_name:
#                         continue
#                     if request.private_only and not dataset_metadata.get("private", False):
#                         continue

#                     filtered_datasets.append({
#                         "id": dataset_dict.get("id"),
#                         "name": dataset_dict.get("name"),
#                         "project": dataset_dict.get("project"),
#                         "created": dataset_dict.get("created"),
#                         "status": dataset_dict.get("status"),
#                         "size": dataset_dict.get("size", 0),
#                         "metadata": {
#                             "dataset_name": dataset_metadata.get("dataset_name"),
#                             "user_name": dataset_metadata.get("user_name"),
#                             "private": dataset_metadata.get("private", False),
#                             "dataset_tag": dataset_metadata.get("dataset_tag", ""),
#                             "s3_path": dataset_metadata.get("s3_path")
#                         }
#                     })

#                 except Exception as e:
#                     logger.warning(f"Could not get metadata for dataset {dataset_dict['id']}: {e}")
#                     continue

#             return {"datasets": filtered_datasets, "total": len(filtered_datasets)}

#         finally:
#             # Restore environment
#             for key, value in original_env.items():
#                 if value is None:
#                     os.environ.pop(key, None)
#                 else:
#                     os.environ[key] = value

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error listing datasets: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to list datasets: {e}")


# Add this endpoint to get download methods information
@app.get("/download-methods")
async def get_download_methods():
    """Get information about available download methods."""
    return {
        "methods": {
            "presigned_urls": {
                "description": "Generate presigned URLs for direct download from S3",
                "pros": [
                    "Direct S3 download",
                    "No server bandwidth usage",
                    "Fastest for large files",
                ],
                "cons": ["URLs expire", "Requires S3 access"],
                "best_for": "Large datasets, multiple files",
            },
            "stream_zip": {
                "description": "Stream all files as a single ZIP archive through the API",
                "pros": ["Single download", "Works without direct S3 access"],
                "cons": [
                    "Uses server bandwidth",
                    "Memory intensive for large datasets",
                ],
                "best_for": "Small to medium datasets, when S3 access is restricted",
            },
            "direct_stream": {
                "description": "Stream individual files through the API",
                "pros": ["Granular file access", "Works without direct S3 access"],
                "cons": ["Multiple requests needed", "Uses server bandwidth"],
                "best_for": "Selective file downloads, when S3 access is restricted",
            },
        },
        "recommendation": "Use 'presigned_urls' for best performance with large datasets",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8279)
