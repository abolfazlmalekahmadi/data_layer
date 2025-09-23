# data/data_ingestion/handlers/huggingface_handler.py
import gc
import logging
from datetime import datetime

from fastapi import HTTPException
from huggingface_hub import HfApi, snapshot_download

from data.data_ingestion.handlers.conditions import DatasetConditionChecker
from data.data_ingestion.models.metadata import DatasetMetadata
from data.data_ingestion.storage_handler import DatasetStorageHandler

from data.data_validation.validation import DirectoryValidator

logger = logging.getLogger(__name__)


class HuggingFaceHandler:
    config = [{'name': 'Validate User CSVs', 'path_pattern': '**/*.csv', 'validator': 'tabular'},
            {'name': 'Validate Transaction Parquet Files', 'path_pattern': '**/*.parquet', 'validator': 'tabular'}, 
            {'name': 'Validate for Image files', 'path_pattern': '**/*.jpg', 'validator': 'image'}, 
            {'name': 'Validate for Image files(png)', 'path_pattern': '**/*.png', 'validator': 'image'}]
    def process(
        self,
        dataset_name: str,
        dataset_config: str,
        user_name: str,
        private: bool,
        s3_config: dict = None,
        clearml_config: dict = None,
        revision: str = "main",
    ) -> dict:
        """
        Download a Hugging Face dataset, persist it to PVC, and optionally upload to S3.
        """
        storage_handler = None
        try:
            mount_dataset_name = dataset_name.replace("/", "-")
            storage_handler = DatasetStorageHandler(mount_dataset_name)

            try:
                DatasetConditionChecker().check_huggingface_size(
                    dataset_name, revision=revision
                )
            except ValueError as size_error:
                logger.error(
                    f"Dataset size check failed for '{dataset_name}': {size_error}"
                )
                raise HTTPException(
                    status_code=400, detail=f"Dataset size check failed: {size_error}"
                )

            api = HfApi()
            # Fetch repo info to get metadata like the exact commit SHA for the given revision
            info = api.repo_info(dataset_name, repo_type="dataset", revision=revision)

            # --- Step 2: Download raw files using snapshot_download ---
            temp_dir = storage_handler.temp_dir
            logger.info(f"Using PVC-backed temp directory: {temp_dir}")

            # This directory will contain the downloaded files, mirroring the HF repo structure
            local_dataset_dir = temp_dir / "huggingface_dataset_files"
            local_dataset_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Downloading raw files for dataset '{dataset_name}' at revision '{revision}'..."
            )

            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                revision=revision,
                cache_dir=str(temp_dir / "cache"),
                local_dir=str(local_dataset_dir),
                # CRITICAL: Use False to ensure files are copied, not symlinked.
                # This is necessary so we can move the directory to permanent storage later.
                local_dir_use_symlinks=False,
            )

            ds_id = storage_handler.generate_dataset_id()
            metadata = DatasetMetadata(
                dataset_id=ds_id,
                dataset_name=dataset_name.replace("/", "-"),
                dataset_config=dataset_config,
                last_commit=getattr(info, "sha", None),
                last_modified=getattr(info, "last_modified", None).isoformat()
                if getattr(info, "last_modified", None)
                else None,
                user_name=user_name,
                private=private,
                revision=revision,
                source="huggingface",
                created_at=datetime.now().isoformat(),
                file_path="",
                summary=f"Raw file download of the Hugging Face dataset '{dataset_name}' at revision '{revision}'.",
            )

            # --- Step 3: Validate downloaded files ---
            logger.info(f"Validating downloaded files in: {local_dataset_dir}")
            validator = DirectoryValidator(self.config)
            report = validator.validate(str(local_dataset_dir))
            logger.info(f"Validation report: {report}")
            if report.has_failures():
                error_details = [
                    {
                        "file": result.file_path,
                        "errors": result.errors,
                        "warnings": result.warnings,
                    }
                    for result in report.results
                    if not result.is_valid
                ]
                logger.error(
                    f"Validation failed for some files in dataset '{dataset_name}'."
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Validation failed for some files.",
                        "errors": error_details,
                    },
                )

            # move dataset to PVC permanent place and upload to s3 if s3_config provided
            stored_path = storage_handler.store_dataset(
                local_dataset_dir,
                metadata,
                s3_config=s3_config,
                clearml_config=clearml_config,
            )
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
            logger.error(
                f"Error processing Hugging Face dataset '{dataset_name}': {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail=f"Error with Hugging Face dataset: {e}"
            )
        finally:
            # --- Step 5: Cleanup ---
            # No 'ds' object to clean up, just the temp directory
            gc.collect()
            if storage_handler:
                storage_handler.cleanup_temp()


def process_huggingface_dataset(
    dataset_name, dataset_config, user_name, private, dataset_tag, s3_config=None
):
    return HuggingFaceHandler().process(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        user_name=user_name,
        private=private,
        dataset_tag=dataset_tag,
        s3_config=s3_config,
    )
