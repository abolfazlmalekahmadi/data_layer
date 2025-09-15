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
        
        clearml_credentials = params.get("clearml_credentials")
        if clearml_credentials:
            # Set environment variables for this specific execution
            # os.environ['CLEARML_WEB_HOST'] = clearml_credentials['web_host']
            # os.environ['CLEARML_API_HOST'] = clearml_credentials['api_host']
            # os.environ['CLEARML_FILES_HOST'] = clearml_credentials['files_host']
            # print(clearml_credentials['access_key'])
            os.environ['CLEARML_API_ACCESS_KEY'] = clearml_credentials['access_key']
            os.environ['CLEARML_API_SECRET_KEY'] = clearml_credentials['secret_key']
            logger.info(f"Set ClearML credentials for user execution {os.environ['CLEARML_API_ACCESS_KEY']} {os.environ['CLEARML_API_SECRET_KEY']}")
        
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
            s3_config_target=s3_config_target,
            clearml_config=clearml_credentials

        )
        
        # Mark task as completed
        task.mark_completed()
        
    except Exception as e:
        logger.error(f"Error processing S3 dataset: {e}")
        task.mark_failed(status_message=str(e))
        raise

if __name__ == "__main__":
    main()