from pydantic import BaseModel
from typing import Optional

class S3DownloadRequest(BaseModel):
    s3_file_path: str
    user_name: str
    dataset_name: Optional[str] = None
    private: bool = False
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    bucket_name: Optional[str] = None

class KaggleDownloadRequest(BaseModel):
    dataset_id: str
    user_name: str
    dataset_name: Optional[str] = None
    private: bool = False
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None

class HuggingFaceDownloadRequest(BaseModel):
    dataset_name: str
    dataset_config: Optional[str] = None
    dataset_name_override: Optional[str] = None
    user_name: str
    private: bool = False

class OpenMLDownloadRequest(BaseModel):
    dataset_id: int
    user_name: str
    dataset_name: Optional[str] = None
    private: bool = False

class ListDatasetsRequest(BaseModel):
    user_name: Optional[str] = None
    private_only: bool = False

class DownloadDatasetRequest(BaseModel):
    dataset_id: str
    user_name: str
