from pydantic import BaseModel
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

class S3DownloadRequest(BaseModel):
    s3_file_path: str
    user_name: str
    dataset_name: Optional[str] = None
    private: bool = False
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    bucket_name: Optional[str] = None
    dataset_tag: Optional[str] = Field(default=None, alias="dataset_tag")
    
class KaggleDownloadRequest(BaseModel):
    dataset_id: str
    user_name: str
    dataset_name: Optional[str] = None
    private: bool = False
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None
    dataset_tag : str

class HuggingFaceDownloadRequest(BaseModel):
    dataset_name: str
    dataset_config: str | None = None
    user_name: str
    private: bool = False
    dataset_tag: str | None = None
    endpoint_url: str
    access_key: str
    secret_key: str
    bucket_name: str

class OpenMLDownloadRequest(BaseModel):
    dataset_id: int
    user_name: str
    dataset_name: Optional[str] = None
    private: bool = False
    dataset_tag : str

class ListDatasetsRequest(BaseModel):
    user_name: Optional[str] = None
    private_only: bool = False

class DownloadDatasetRequest(BaseModel):
    dataset_id: str
    user_name: str
