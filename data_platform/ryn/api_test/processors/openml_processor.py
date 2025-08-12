import os
import uuid
import logging
from clearml import Task, Dataset
from data.data_ingestion.handlers.openml_handler import OpenMLHandler

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
        dataset_name = params.get("dataset_name")
        user_name = params.get("user_name")
        private = params.get("private", False)
        
        logger.info(f"Processing OpenML dataset: {dataset_name or dataset_id}")
        
        # Create handler and process the dataset
        handler = OpenMLHandler()
        result = handler.process(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            user_name=user_name,
            private=private
        )
        
        # Create ClearML dataset if processing was successful
        if result and result.get("status") == "success":
            unique_id = uuid.uuid4().hex[:8]
            clearml_dataset = Dataset.create(
                dataset_project="DatasetIngestion",
                dataset_name=f"OpenML_{dataset_name or dataset_id}_{unique_id}",
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
            logger.info(f"Created ClearML dataset: OpenML_{dataset_name or dataset_id}_{unique_id}")
        
        # Mark task as completed
        task.mark_completed()
        
    except Exception as e:
        logger.error(f"Error processing OpenML dataset: {e}")
        task.mark_failed(status_message=str(e))
        raise

if __name__ == "__main__":
    main()
