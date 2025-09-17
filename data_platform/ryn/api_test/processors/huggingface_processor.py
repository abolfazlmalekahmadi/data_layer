import os
import logging
from clearml import Task

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
    logger.info(f"params: {params}")
    try:
        # Extract ClearML credentials from task parameters
        clearml_credentials = params.get("clearml_credentials")
        logger.info(f"clearml parameters: {clearml_credentials}")
        if clearml_credentials:
            # Set environment variables for this specific execution
            # os.environ['CLEARML_WEB_HOST'] = clearml_credentials['web_host']
            # os.environ['CLEARML_API_HOST'] = clearml_credentials['api_host']
            # os.environ['CLEARML_FILES_HOST'] = clearml_credentials['files_host']
            # print(clearml_credentials['access_key'])
            os.environ['CLEARML_API_ACCESS_KEY'] = clearml_credentials['access_key']
            os.environ['CLEARML_API_SECRET_KEY'] = clearml_credentials['secret_key']
            logger.info("Set ClearML credentials for user execution")
        
        dataset_name = params.get("dataset_name")
        if not dataset_name:
            raise ValueError("Required parameter 'dataset_name' is missing from the task configuration.")
        
        # Extract parameters
        dataset_config = params.get("dataset_config", "")
        user_name = params.get("user_name")
        private = params.get("private", False)
        revision = params.get("revision", "")
        s3_config = params.get("s3_config", {})
        
        logger.info(f"Processing HuggingFace dataset: {dataset_name} for user: {user_name} s3 config: {s3_config}")
        
        # Import handler after setting environment
        from data.data_ingestion.handlers.huggingface_handler import HuggingFaceHandler
        
        # Create handler and process the dataset
        handler = HuggingFaceHandler()
        result = handler.process(
            dataset_name=dataset_name,
            dataset_config=dataset_config if dataset_config else None,
            user_name=user_name,
            private=private,
            revision=revision if revision else None,
            s3_config=s3_config,
            clearml_config=clearml_credentials
            
        )
        
        logger.info(f"Successfully processed dataset: {result}")
        
        # Mark task as completed
        task.mark_completed()
        
    except Exception as e:
        logger.error(f"Error processing HuggingFace dataset: {e}")
        task.mark_failed(status_message=str(e))
        raise

if __name__ == "__main__":
    main()