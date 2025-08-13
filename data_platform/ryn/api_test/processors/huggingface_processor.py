import os
import uuid
import logging
from clearml import Task, Dataset
from data.data_ingestion.handlers.huggingface_handler import HuggingFaceHandler

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
    raise ValueError(f"Ali:{params}")
    try:
        dataset_name = params.get("dataset_name")
        if not dataset_name:
            raise ValueError("Required parameter 'dataset_name' is missing from the task configuration.")
        # Extract parameters
        dataset_config = params.get("dataset_config", "")
        user_name = params.get("user_name")
        private = params.get("private", False)
        dataset_tag = params.get("dataset_tag", "")
        s3_config = params.get("s3_config", {})
        
        logger.info(f"Processing HuggingFace dataset: {dataset_name}")
        
        # Create handler and process the dataset
        handler = HuggingFaceHandler()
        result = handler.process(
            dataset_name=dataset_name,
            dataset_config=dataset_config if dataset_config else None,
            user_name=user_name,
            private=private,
            dataset_tag=dataset_tag if dataset_tag else None,
            s3_config=s3_config
        )
        
        # Create ClearML dataset if processing was successful
        if result and result.get("status") == "success":
            unique_id = uuid.uuid4().hex[:8]
            clearml_dataset = Dataset.create(
                dataset_project="DatasetIngestion",
                dataset_name=f"HF_{dataset_name}_{unique_id}",
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
            logger.info(f"Created ClearML dataset: HF_{dataset_name}_{unique_id}")
        
        # Mark task as completed
        task.mark_completed()
        
    except Exception as e:
        logger.error(f"Error processing HuggingFace dataset: {e}")
        task.mark_failed(status_message=str(e))
        raise

if __name__ == "__main__":
    main()