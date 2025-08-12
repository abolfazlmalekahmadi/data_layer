import os
import uuid
import logging
from clearml import Task, Dataset
from data.data_ingestion.handlers.kaggle_handler import KaggleHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get the current task
    task = Task.current_task()
    if not task:
        logger.error("No ClearML task found")
        return
    
    # Get parameters from the task
    params = task.get_parameters()
    
    try:
        # Extract parameters
        dataset_id = params.get("dataset_id")
        user_name = params.get("user_name")
        kaggle_username = params.get("kaggle_username")
        kaggle_key = params.get("kaggle_key")
        private = params.get("private", False)
        
        logger.info(f"Processing Kaggle dataset: {dataset_id}")
        
        # Create handler and process the dataset
        handler = KaggleHandler()
        result = handler.process(
            dataset_id=dataset_id,
            user_name=user_name,
            kaggle_username=kaggle_username,
            kaggle_key=kaggle_key,
            private=private
        )
        
        # Create ClearML dataset if processing was successful
        if result and result.get("status") == "success":
            unique_id = uuid.uuid4().hex[:8]
            clearml_dataset = Dataset.create(
                dataset_project="DatasetIngestion",
                dataset_name=f"Kaggle_{dataset_id.replace('/', '_')}_{unique_id}",
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
            logger.info(f"Created ClearML dataset: Kaggle_{dataset_id.replace('/', '_')}_{unique_id}")
        
        # Mark task as completed
        task.mark_completed()
        
    except Exception as e:
        logger.error(f"Error processing Kaggle dataset: {e}")
        task.mark_failed(status_message=str(e))
        raise

if __name__ == "__main__":
    main()