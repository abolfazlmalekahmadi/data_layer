import os
import uuid
import logging
from clearml import Task, Dataset
from data.data_ingestion.handlers.s3_handler import S3Handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get the current task
    task = Task.current_task()
    if not task:
        logger.error("No ClearML task found")
        return
    
    # Get parameters from the task
    params = task.get_parameters_as_dict().get('General', {})

    
    try:
        # Extract parameters
        access_key = params.get("access_key")
        secret_key = params.get("secret_key")
        endpoint_url = params.get("endpoint_url")
        bucket_name = params.get("bucket_name")
        s3_file_path = params.get("s3_file_path")
        dataset_name = params.get("dataset_name")
        user_name = params.get("user_name")
        private = params.get("private", False)
        dataset_tag = params.get("dataset_tag")
        s3_config_target = params.get("s3_config_target", {})
        s3_config_target = {
            key.replace("_target", ""): value
            for key, value in s3_config_target.items()
                }
        print("10"*10)

        print(s3_config_target)
        logger.info(f"Processing S3 dataset: {dataset_name}")
        
        # Create S3 handler
        s3_handler = S3Handler(
            access_key=access_key,
            secret_key=secret_key,
            endpoint_url=endpoint_url,
            bucket_name=bucket_name,
        )
        
        # Process the dataset
        result = s3_handler.process_s3_dataset(
            s3_file_path=s3_file_path,
            dataset_name=dataset_name,
            user_name=user_name,
            private=private,
            dataset_tag=dataset_tag if dataset_tag else None,
            s3_config_target=s3_config_target
        )
        
        # Create ClearML dataset if processing was successful
        if result and result.get("status") == "success":
            unique_id = uuid.uuid4().hex[:8]
            clearml_dataset = Dataset.create(
                dataset_project="DatasetIngestion",
                dataset_name=f"S3_{dataset_name}_{unique_id}",
                parent_datasets=[],
                use_current_task=True
            )
            
            # Add metadata as artifact
            task.upload_artifact("processing_result", result)
            
            # If there's a local file path, add it to the dataset
            if "file_path" in result:
                clearml_dataset.add_files(result["file_path"])
                clearml_dataset.upload()
            
            clearml_dataset.finalize()
            logger.info(f"Created ClearML dataset: S3_{dataset_name}_{unique_id}")
        
        # Mark task as completed
        task.mark_completed()
        
    except Exception as e:
        logger.error(f"Error processing S3 dataset: {e}")
        task.mark_failed(status_message=str(e))
        raise

if __name__ == "__main__":
    main()