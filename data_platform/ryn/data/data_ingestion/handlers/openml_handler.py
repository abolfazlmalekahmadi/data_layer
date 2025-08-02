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

def process_openml_dataset(dataset_id: int, dataset_name: str, user_name: str, private: bool) -> dict:
    try:
        DatasetConditionChecker().check_openml_size(dataset_id)

        with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Created temporary directory for OpenML download: {temp_path}")

            dataset = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=True)
            X, y, _, _ = dataset.get_data(dataset_format="dataframe")
            df = pd.concat([X, y], axis=1) if y is not None else X

            local_path = temp_path / f"{dataset.name}.csv"
            df.to_csv(local_path, index=False)

            storage_handler = DatasetStorageHandler()

            ds_id = storage_handler.generate_dataset_id()
            metadata = DatasetMetadata(
                dataset_id=ds_id,
                dataset_name=dataset_name or dataset.name,
                dataset_config=str(dataset.version),
                last_commit=str(dataset.version_label),
                last_modified=dataset.upload_date,
                user_name=user_name,
                private=private,
                source="openml",
                created_at=datetime.now().isoformat(),
                file_path="",
                summary=summarize_dataset(local_path)
            )

            stored_path = storage_handler.store_dataset(local_path, metadata)

            return {
                "status": "success",
                "message": f"OpenML dataset '{dataset_id}' stored.",
                "dataset_id": ds_id,
                "stored_path": stored_path,
            }

    except Exception as e:
        logger.error(f"Error processing OpenML dataset '{dataset_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error with OpenML dataset: {e}")