import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import os
from pathlib import Path
import hashlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_sha256(file_path: str) -> str:
    """Calculate SHA256 hash of a file."""
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate SHA256 for {file_path}: {e}")
        raise

def upload_folder_to_s3(local_folder: str, bucket_name: str, s3_path: str, endpoint_url: str, access_key: str, secret_key: str, disable_payload_signing: bool = False):
    """Upload all files in a local folder to S3."""
    try:
        # Initialize S3 client
        config = Config(
            retries={'max_attempts': 3, 'mode': 'standard'},
            signature_version='s3v4',
            s3={'payload_signing_enabled': not disable_payload_signing}
        )
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=config
        )
        logger.info(f"Initialized S3 client for endpoint: {endpoint_url}, bucket: {bucket_name}, payload_signing_enabled: {not disable_payload_signing}")

        # Verify bucket access
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} is accessible")
        except ClientError as e:
            logger.error(f"Failed to access bucket {bucket_name}: {e}")
            raise

        # Convert local_folder to Path object
        local_folder_path = Path(local_folder)
        if not local_folder_path.is_dir():
            raise FileNotFoundError(f"Local folder {local_folder} does not exist or is not a directory")

        # Force single-part upload
        boto3.setup_default_session()
        boto3.resource('s3').meta.client.meta.config.s3 = {'multipart_threshold': 1024 * 1024 * 1024}

        # Walk through the folder and upload each file
        for root, _, files in os.walk(local_folder_path):
            for file in files:
                local_file_path = Path(root) / file
                relative_path = local_file_path.relative_to(local_folder_path)
                s3_key = f"{s3_path}/{relative_path}".replace("\\", "/")

                # Calculate SHA256 hash
                file_sha256 = calculate_sha256(str(local_file_path))
                logger.info(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}")
                logger.info(f"File SHA256: {file_sha256}")

                # Set Content-Type based on file extension
                content_type = 'text/plain' if file.endswith('.txt') else 'application/octet-stream'
                extra_args = {'ContentType': content_type}
                if not file.endswith('.txt'):
                    extra_args['ContentEncoding'] = 'binary'

                # Add explicit SHA256 header for debugging
                with open(local_file_path, 'rb') as f:
                    file_content = f.read()
                    sha256_hash = hashlib.sha256(file_content).hexdigest()
                    extra_args['ContentSHA256'] = sha256_hash

                # Upload file to S3
                try:
                    s3_client.upload_file(
                        str(local_file_path),
                        bucket_name,
                        s3_key,
                        ExtraArgs=extra_args
                    )
                    logger.info(f"Successfully uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")

                    # Verify upload
                    temp_file = f"/tmp/{file}"
                    s3_client.download_file(bucket_name, s3_key, temp_file)
                    downloaded_sha256 = calculate_sha256(temp_file)
                    if downloaded_sha256 == file_sha256:
                        logger.info(f"Upload verified: SHA256 matches for {s3_key}")
                    else:
                        logger.error(f"Upload verification failed: SHA256 mismatch for {s3_key}")
                    os.remove(temp_file)
                except ClientError as e:
                    logger.error(f"Failed to upload {local_file_path}: {e}")
                    raise

    except ClientError as e:
        logger.error(f"Failed to upload to S3: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during S3 upload: {e}")
        raise

if __name__ == "__main__":
    # S3 configuration
    s3_config = {
        "endpoint_url": "https://s3.cloud-ai.ir",
        "access_key": "OAF0MC26UA7DV9WS11X5",
        "secret_key": "6SY2dTxhcIVEsjbfpjRUBhe3k7mMJIjZpccwvw3d",
        "bucket_name": "mlops"
    }

    # Local folder to upload
    local_folder = "/home/user01/abolfazl/tools2/data_platform/ryn/data/PV_Datasets/87a6c869-58fe-485d-8e36-ac47cbc6dc28"
    s3_path = "Datasets/fka-awesome-chatgpt-prompts"

    # Create a test text file
    test_file = Path(local_folder) / "test.txt"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    with open(test_file, 'w') as f:
        f.write("Test content for S3 upload")
    logger.info(f"Created test file: {test_file}")

    # Try with and without payload signing
    for disable_payload_signing in [False, True]:
        try:
            logger.info(f"Attempting upload with payload_signing_enabled: {not disable_payload_signing}")
            upload_folder_to_s3(
                local_folder=local_folder,
                bucket_name=s3_config["bucket_name"],
                s3_path=s3_path,
                endpoint_url=s3_config["endpoint_url"],
                access_key=s3_config["access_key"],
                secret_key=s3_config["secret_key"],
                disable_payload_signing=disable_payload_signing
            )
            print(f"Folder upload completed successfully with payload_signing_enabled: {not disable_payload_signing}")
            break
        except Exception as e:
            print(f"Folder upload failed with payload_signing_enabled={not disable_payload_signing}: {e}")
            if disable_payload_signing:
                print("All attempts failed")