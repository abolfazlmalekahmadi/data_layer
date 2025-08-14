import boto3
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_folder_to_s3(local_folder: str, bucket_name: str, s3_path: str, endpoint_url: str, access_key: str, secret_key: str):
    """Upload all files in a local folder to S3 while preserving folder structure."""
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        logger.info(f"Initialized S3 client for bucket: {bucket_name}")

        # Convert local_folder to Path object
        local_folder_path = Path(local_folder)
        if not local_folder_path.is_dir():
            raise FileNotFoundError(f"Local folder {local_folder} does not exist")

        # Walk through folder and upload files
        for file_path in local_folder_path.rglob('*'):
            if file_path.is_file():
                # Create S3 key from relative path
                relative_path = file_path.relative_to(local_folder_path)
                s3_key = f"{s3_path}/{relative_path}".replace("\\", "/")
                
                logger.info(f"Uploading {file_path} to s3://{bucket_name}/{s3_key}")
                
                # Upload file
                s3_client.upload_file(
                    str(file_path),
                    bucket_name,
                    s3_key,
                    ExtraArgs={'ContentType': 'application/octet-stream'}
                )
                logger.info(f"Successfully uploaded {file_path}")

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

if __name__ == "__main__":
    # S3 configuration
    s3_config = {
        "endpoint_url": "https://s3.cloud-ai.ir",
        "access_key": "OAF0MC26UA7DV9WS11X5",
        "secret_key": "6SY2dTxhcIVEsjbfpjRUBhe3k7mMJIjZpccwvw3d",
        "bucket_name": "mlops"
    }

    # Local folder and S3 path
    local_folder = "/home/user01/abolfazl/tools2/data_platform/ryn/data/PV_Datasets/87a6c869-58fe-485d-8e36-ac47cbc6dc28"
    s3_path = "Datasets/fka-awesome-chatgpt-prompts"

    try:
        upload_folder_to_s3(
            local_folder=local_folder,
            bucket_name=s3_config["bucket_name"],
            s3_path=s3_path,
            endpoint_url=s3_config["endpoint_url"],
            access_key=s3_config["access_key"],
            secret_key=s3_config["secret_key"]
        )
        print("Folder uploaded successfully")
    except Exception as e:
        print(f"Folder upload failed: {e}")