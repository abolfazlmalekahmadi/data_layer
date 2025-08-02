from .handlers.huggingface_handler import process_huggingface_dataset
from .handlers.kaggle_handler import process_kaggle_dataset
from .handlers.openml_handler import process_openml_dataset
from .handlers.s3_handler import S3Handler
from .models.requests import (
    HuggingFaceDownloadRequest,
    KaggleDownloadRequest,
    OpenMLDownloadRequest,
    S3DownloadRequest,
)

def download_dataset(source: str, **kwargs):
    """
    Download and store a dataset from one of the supported sources.

    Args:
        source (str): One of 'huggingface', 'kaggle', 'openml', or 's3'.
        **kwargs: Arguments required by each source's download function.

    Returns:
        dict: Information about the stored dataset.
    """
    if source == "huggingface":
        return process_huggingface_dataset(**kwargs)
    elif source == "kaggle":
        return process_kaggle_dataset(**kwargs)
    elif source == "openml":
        return process_openml_dataset(**kwargs)
    elif source == "s3":
        # Instantiate S3Handler with the required arguments
        s3_handler = S3Handler(
            access_key=kwargs["access_key"],
            secret_key=kwargs["secret_key"],
            endpoint_url=kwargs["endpoint_url"],
            bucket_name=kwargs["bucket_name"]
        )
        # Call process_s3_dataset with the necessary arguments
        return s3_handler.process_s3_dataset(
            kwargs["s3_file_path"], kwargs["dataset_name"], kwargs["user_name"], kwargs["private"]
        )
    else:
        raise ValueError(f"Unsupported source '{source}'. Supported sources are: huggingface, kaggle, openml, s3.")