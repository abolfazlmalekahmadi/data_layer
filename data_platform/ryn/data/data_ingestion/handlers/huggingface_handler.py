from datasets import load_dataset, get_dataset_infos
from huggingface_hub import HfApi
from pathlib import Path
from datetime import datetime
from fastapi import HTTPException
import tempfile
import logging
from data.data_ingestion.models.metadata import DatasetMetadata
from data.data_ingestion.handlers.utils import summarize_dataset
from data.data_ingestion.handlers.conditions import DatasetConditionChecker
from data.data_ingestion.storage_handler import DatasetStorageHandler

logger = logging.getLogger(__name__)

class HuggingFaceHandler:
    def process(self, dataset_name: str , dataset_config : str, user_name: str, private: bool , dataset_tag :str) -> dict  :
        """
        Download a Hugging Face dataset, persist it, and return metadata.
        """
        try:
            storage_handler = DatasetStorageHandler()  # This will call __init__ and create directories
            try:
                DatasetConditionChecker().check_huggingface_size(dataset_name, dataset_config)
            except ValueError as size_error:
                logger.error(f"Dataset size check failed for '{dataset_name}': {size_error}")
                raise HTTPException(status_code=400, detail=f"Dataset size check failed: {size_error}")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                logger.info(f"Temporary directory created: {temp_path}")

                ds = load_dataset(dataset_name, dataset_config, cache_dir=str(temp_path / "cache"))
                api = HfApi()
                info = api.repo_info(dataset_name, repo_type="dataset")
                local_dataset_dir = temp_path / "huggingface_dataset"
                ds.save_to_disk(str(local_dataset_dir))

                ds_id = storage_handler.generate_dataset_id()
                metadata = DatasetMetadata(
                    dataset_id=ds_id,
                    dataset_name= dataset_name.replace('/', '-'),
                    dataset_config=dataset_config,
                    last_commit=info.sha,
                    last_modified=info.last_modified.isoformat(),
                    user_name=user_name,
                    private=private,
                    dataset_tag = dataset_tag,
                    source="huggingface",
                    created_at=datetime.now().isoformat(),
                    file_path="",
                    summary=f"Hugging Face dataset with splits: {list(ds.keys())}"
                )

                stored_path = storage_handler.store_dataset(local_dataset_dir, metadata)

                return {
                    "status": "success",
                    "message": f"Hugging Face dataset '{dataset_name}' stored.",
                    "dataset_id": ds_id,
                    "stored_path": stored_path
                }
        except Exception as e:
            logger.error(f"Error processing Hugging Face dataset '{dataset_name}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error with Hugging Face dataset: {e}")

def process_huggingface_dataset(dataset_name, dataset_config , user_name, private , dataset_tag):
    return HuggingFaceHandler().process(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        user_name=user_name,
        private=private,
        dataset_tag = dataset_tag
    )