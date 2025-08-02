import os
import shutil
import openml
from datasets import get_dataset_infos
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class DatasetConditionChecker:
    def has_enough_space(self, size_bytes: int, buffer_ratio: float = 1.2) -> bool:
        """Check if there's enough available disk space with a buffer."""
        total, used, free = shutil.disk_usage("/")
        return free >= size_bytes * buffer_ratio

    def check_huggingface_size(self, dataset_name: str, dataset_config: str = None):
        try:
            infos = get_dataset_infos(dataset_name)
            for config, info in infos.items():
                if dataset_config is None or config == dataset_config:
                    estimated_size = (
                        getattr(info, "size_in_bytes", 0)
                        or getattr(info, "dataset_size", 0)
                        or getattr(info, "download_size", 0)
                    )
                    if estimated_size > 0 and not self.has_enough_space(estimated_size):
                        raise HTTPException(status_code=507, detail="Not enough disk space for Hugging Face dataset.")
        except Exception as e:
            logger.error(f"Failed to estimate Hugging Face dataset size: {e}")

    def check_openml_size(self, dataset_id: int):
        try:
            dataset = openml.datasets.get_dataset(dataset_id, download_all_files=False)
            instances = dataset.qualities.get("NumberOfInstances", 0)
            features = dataset.qualities.get("NumberOfFeatures", 0)

            max_bytes_per_value = 100  # conservative estimate
            estimated_size = instances * features * max_bytes_per_value

            if estimated_size > 0 and not self.has_enough_space(estimated_size):
                raise HTTPException(status_code=507, detail="Not enough disk space for OpenML dataset.")
        except Exception as e:
            logger.error(f"Failed to estimate OpenML dataset size: {e}")