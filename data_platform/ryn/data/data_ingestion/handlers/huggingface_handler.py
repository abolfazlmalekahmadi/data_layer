import gc
import logging
import os
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException
from huggingface_hub import HfApi, snapshot_download

from data.data_ingestion.handlers.conditions import DatasetConditionChecker
from data.data_ingestion.models.metadata import DatasetMetadata
from data.data_ingestion.storage_handler import DatasetStorageHandler
from data.data_validation.validation import DirectoryValidator

logger = logging.getLogger(__name__)


class HuggingFaceHandler:
    config = [
        {
            "name": "Validate User CSVs",
            "path_pattern": "**/*.csv",
            "validator": "tabular",
        },
        {
            "name": "Validate Transaction Parquet Files",
            "path_pattern": "**/*.parquet",
            "validator": "tabular",
        },
        {
            "name": "Validate for Image files",
            "path_pattern": "**/*.jpg",
            "validator": "image",
        },
        {
            "name": "Validate for Image files(png)",
            "path_pattern": "**/*.png",
            "validator": "image",
        },
        {
            "name": "Validate for JSON files",
            "path_pattern": "**/*.json",
            "validator": "json",  # Assuming a JSON validator exists
        },
        {
            "name": "Validate for Audio files",
            "path_pattern": "**/*.wav",
            "validator": "audio",  # Assuming an Audio validator exists
        },
        {"name": "Validate for Audio files(mp3)",
         "path_pattern": "**/*.mp3",
            "validator": "audio"  # Assuming an Audio validator exists
        },
        {"name": "Validate for Text files",
         "path_pattern": "**/*.txt",
            "validator": "text"  # Assuming a Text validator exists
        },
    ]

    def _validate_and_cleanup_dataset(
        self, local_dataset_dir: Path, dataset_name: str
    ) -> tuple[str, str, list]:
        """
        Validates a local dataset, removes invalid files, and returns validation status.

        Args:
            local_dataset_dir: The path to the downloaded dataset directory.
            dataset_name: The original Hugging Face dataset name for logging.

        Returns:
            A tuple containing the final status, a descriptive message, and a list of error details.

        Raises:
            HTTPException: If all files in the dataset fail validation.
        """
        logger.info(f"Validating downloaded files in: {local_dataset_dir}")
        validator = DirectoryValidator(self.config)
        report = validator.validate(Path(local_dataset_dir))
        logger.info(f"Validation report: {report}")

        error_details = []
        final_status = "success"
        final_message = f"Hugging Face dataset '{dataset_name}' stored successfully."

        if report.has_failures():
            final_status = "completed_with_errors"
            final_message = (
                "Dataset stored, but some files failed validation and were excluded."
            )
            logger.warning(
                f"Validation failed for some files in dataset '{dataset_name}'. "
                "These files will be excluded from the final stored dataset."
            )

            invalid_file_paths = set()
            for result in report.results:
                if not result.is_valid:
                    error_details.append(
                        {
                            "file": result.file_path,
                            "errors": result.errors,
                            "warnings": result.warnings,
                        }
                    )
                    invalid_file_paths.add(result.file_path)

            logger.info(f"Removing {len(invalid_file_paths)} invalid file(s)...")
            logger.info(error_details)
            for file_path in invalid_file_paths:
                try:
                    os.remove(file_path)
                    logger.debug(f"Removed invalid file: {file_path}")
                except OSError as e:
                    logger.error(f"Error removing invalid file {file_path}: {e}")

        # Check if any valid files remain after cleanup
        files = (
            [
                f.name
                for f in local_dataset_dir.rglob("*")
                if f.is_file()
                and f.suffix in (".csv", ".parquet", ".jpg", ".png", ".json")
            ]
            if local_dataset_dir.is_dir()
            else []
        )
        logger.info(f"{files}, {local_dataset_dir.is_dir()}")
        if not files:
            final_status = "failed"
            final_message = (
                "All files in the dataset failed validation. No files were stored."
            )
            logger.error(
                f"All files in the Hugging Face dataset '{dataset_name}' failed validation."
            )
            raise HTTPException(
                status_code=400,
                detail="All files in the dataset failed validation. No files were stored.",
            )

        return final_status, final_message, error_details

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
        Download a Hugging Face dataset, validate files, persist only the valid files
        to PVC/S3, and report any validation failures.
        """
        storage_handler = None
        try:
            mount_dataset_name = dataset_name.replace("/", "-")
            storage_handler = DatasetStorageHandler(mount_dataset_name)

            # Step 1: Size Check
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

            # Step 2: Download
            api = HfApi()
            info = api.repo_info(dataset_name, repo_type="dataset", revision=revision)

            temp_dir = storage_handler.temp_dir
            logger.info(f"Using PVC-backed temp directory: {temp_dir}")

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

            # Step 3: Validate and Cleanup
            final_status, final_message, error_details = (
                self._validate_and_cleanup_dataset(local_dataset_dir, dataset_name)
            )

            # Step 4: Store valid files
            stored_path = storage_handler.store_dataset(
                local_dataset_dir,
                metadata,
                s3_config=s3_config,
                clearml_config=clearml_config,
            )

            # Step 5: Construct final response
            response = {
                "status": final_status,
                "message": final_message,
                "dataset_id": ds_id,
                "stored_path": stored_path,
            }
            if error_details:
                response["validation_errors"] = error_details

            return response

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
