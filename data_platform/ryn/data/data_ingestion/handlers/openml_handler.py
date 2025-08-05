import openml
import tempfile
import pandas as pd
from pathlib import Path
from datetime import datetime
from fastapi import HTTPException
import logging

from data.data_ingestion.models.metadata import DatasetMetadata
from data.data_ingestion.storage_handler import DatasetStorageHandler, TEMP_DIR
from data.data_ingestion.handlers.conditions import DatasetConditionChecker
from data.data_ingestion.handlers.utils import summarize_dataset

logger = logging.getLogger(__name__)
def process_openml_dataset(dataset_id: int, dataset_name: str, user_name: str, private: bool , dataset_tag : str) -> dict:
    try:
        storage_handler = DatasetStorageHandler()

        # Try to get dataset by name if dataset_name is provided
        dataset = None
        name_based = False

        if dataset_name:
            try:
                matching_datasets = openml.datasets.list_datasets(output_format='dataframe')
                matched_row = matching_datasets[matching_datasets['name'] == dataset_name]

                if not matched_row.empty:
                    dataset_id_from_name = int(matched_row.iloc[0]['did'])
                    DatasetConditionChecker().check_openml_size(dataset_id_from_name)
                    dataset = openml.datasets.get_dataset(dataset_id_from_name, download_data=False, download_qualities=True)
                    name_based = True
                else:
                    logger.warning(f"No dataset found on OpenML with name '{dataset_name}'. Falling back to ID.")
            except Exception as name_err:
                logger.warning(f"Failed to load dataset by name '{dataset_name}': {name_err}. Falling back to ID.")

        # If not found by name, fallback to dataset ID
        if dataset is None:
            DatasetConditionChecker().check_openml_size(dataset_id)
            dataset = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=True)

        # Get data
        X, y, _, _ = dataset.get_data(dataset_format="dataframe")
        df = pd.concat([X, y], axis=1) if y is not None else X

        # Save to temp CSV
        with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Created temporary directory for OpenML download: {temp_path}")

            local_path = temp_path / f"{dataset.name}.csv"
            df.to_csv(local_path, index=False)

            # Generate metadata
            ds_id = storage_handler.generate_dataset_id()
            final_dataset_name = dataset_name if name_based else dataset.name
            metadata = DatasetMetadata(
                dataset_id=ds_id,
                dataset_name=final_dataset_name,
                dataset_config=str(dataset.version),
                last_commit=str(dataset.version_label),
                last_modified=dataset.upload_date,
                user_name=user_name,
                private=private,
                source="openml",
                created_at=datetime.now().isoformat(),
                file_path="",
                summary=summarize_dataset(local_path),
                dataset_tag = dataset_tag
            )

            stored_path = storage_handler.store_dataset(local_path, metadata)

            return {
                "status": "success",
                "message": f"OpenML dataset '{final_dataset_name}' stored.",
                "dataset_id": ds_id,
                "stored_path": stored_path,
            }

    except Exception as e:
        logger.error(f"Error processing OpenML dataset '{dataset_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error with OpenML dataset: {e}")
