# data/data_ingestion/handlers/huggingface_handler.py
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
    def process(self, dataset_name: str , dataset_config : str, user_name: str, private: bool , dataset_tag :str, s3_config: dict = None , clearml_config: dict =None) -> dict  :
        """
        Download a Hugging Face dataset, persist it to PVC, and optionally upload to S3.
        """
        storage_handler = None
        try:
            mount_dataset_name = dataset_name.replace('/', '-')
            storage_handler = DatasetStorageHandler(mount_dataset_name)
            
            try:
                DatasetConditionChecker().check_huggingface_size(dataset_name, dataset_config)
            except ValueError as size_error:
                logger.error(f"Dataset size check failed for '{dataset_name}': {size_error}")
                raise HTTPException(status_code=400, detail=f"Dataset size check failed: {size_error}")

            # Download directly into PVC-backed temp dir so it persists
            temp_dir = storage_handler.temp_dir
            logger.info(f"Using PVC-backed temp directory: {temp_dir}")

            ds = load_dataset(dataset_name, dataset_config, cache_dir=str(temp_dir / "cache"))
            api = HfApi()
            info = api.repo_info(dataset_name, repo_type="dataset")
            local_dataset_dir = temp_dir / "huggingface_dataset"
            # save to disk under the PVC temp dir
            ds.save_to_disk(str(local_dataset_dir))

            ds_id = storage_handler.generate_dataset_id()
            metadata = DatasetMetadata(
                dataset_id=ds_id,
                dataset_name= dataset_name.replace('/', '-'),
                dataset_config=dataset_config,
                last_commit=getattr(info, "sha", None),
                last_modified=getattr(info, "last_modified", None).isoformat() if getattr(info, "last_modified", None) else None,
                user_name=user_name,
                private=private,
                dataset_tag = dataset_tag,
                source="huggingface",
                created_at=datetime.now().isoformat(),
                file_path="",
                summary=f"Hugging Face dataset with splits: {list(ds.keys())}"
            )

            # move dataset to PVC permanent place and upload to s3 if s3_config provided
            stored_path = storage_handler.store_dataset(local_dataset_dir, metadata, s3_config=s3_config , clearml_config=clearml_config)
            # register witth clearml
            # storage_handler.register_with_clearml(metadata, stored_path)
            return {
                "status": "success",
                "message": f"Hugging Face dataset '{dataset_name}' stored.",
                "dataset_id": ds_id,
                "stored_path": stored_path,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing Hugging Face dataset '{dataset_name}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error with Hugging Face dataset: {e}")
        finally:
            try:
                # ds may not be present on error
                if 'ds' in locals():
                    try:
                        ds.cleanup_cache_files()
                    except Exception:
                        pass
                    del ds
                import gc; gc.collect()
            finally:
                if storage_handler:
                    # cleanup temp dir (store_dataset also calls cleanup but re-ensure)
                    storage_handler.cleanup_temp()

def process_huggingface_dataset(dataset_name, dataset_config , user_name, private , dataset_tag, s3_config=None):
    return HuggingFaceHandler().process(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        user_name=user_name,
        private=private,
        dataset_tag = dataset_tag,
        s3_config = s3_config
    )
