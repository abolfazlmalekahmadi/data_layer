# Robust S3 Uploader SDK

A Python SDK for robust, resumable file and folder uploads to AWS S3 and other S3-compatible storage services (like Ceph, MinIO, etc.).

This SDK is designed to handle common challenges with data transfer, such as network interruptions, large file uploads, and directory synchronization. It provides a simple interface while managing the complexity of multipart uploads, checksum verification, and state management under the hood.

## ✨ Key Features

-   **Resumable Uploads**: Automatically saves progress for large files. If an upload is interrupted (e.g., network error, `Ctrl+C`), it can be resumed from the exact point it left off.
-   **Automatic Multipart Upload**: Intelligently switches between standard and multipart uploads based on file size, optimizing for performance and reliability with large files.
-   **Directory Synchronization**: Upload entire folders with flexible sync modes:
    -   `SYNC`: (Default) Uploads only new or modified files by comparing size and MD5 checksum.
    -   `SKIP`: Skips any file that already exists in the destination, regardless of content.
    -   `OVERWRITE`: Re-uploads all files, overwriting any existing ones.
-   **Checksum Verification**: The `SYNC` mode uses MD5 checksums stored in S3 metadata to ensure data integrity and prevent unnecessary re-uploads.
-   **Thread-Safe Progress Bar**: A clean, real-time progress bar shows the upload status, including transfer speed and an estimated time of arrival (ETA).
-   **Broad Compatibility**: Works with both AWS S3 (via `region_name`) and S3-compatible endpoints (via `endpoint_url`).
-   **Connection Verification**: Checks credentials and connectivity upon initialization to fail fast if the configuration is invalid.

##



## 🚀 Usage

### 1. Initialization

First, import and instantiate the `S3Uploader` class.

**For AWS S3:**
Provide your AWS credentials and the target region.

```python
from data.data_ingestion.handlers.local_handler import S3Uploader

aws_uploader = S3Uploader(
    aws_access_key_id="YOUR_AWS_ACCESS_KEY",
    aws_secret_access_key="YOUR_AWS_SECRET_KEY",
    region_name="us-east-1"
)
```

**For S3-Compatible Storage (e.g., Ceph, MinIO):**
Provide your credentials and the service's `endpoint_url`. The `region_name` is optional but may be required by some services.

```python
from data.data_ingestion.handlers.local_handler import S3Uploader

ceph_uploader = S3Uploader(
    aws_access_key_id="YOUR_CEPH_ACCESS_KEY",
    aws_secret_access_key="YOUR_CEPH_SECRET_KEY",
    endpoint_url="http://your-ceph-gateway:80"
)
```

### 2. Uploading a Folder

The most common use case is synchronizing a local folder with an S3 bucket.

```python
# The local directory you want to upload
local_data_folder = "./my_local_data"

# The name of the S3 bucket
bucket = "my-backup-bucket"

# An optional "folder" within the bucket to place the files
s3_prefix = "project-alpha/raw-data"

# Upload the folder using the 'SYNC' mode
aws_uploader.upload_folder(
    local_folder_path=local_data_folder,
    bucket_name=bucket,
    s3_folder_prefix=s3_prefix,
    sync_mode="SYNC"  # Can be "SYNC", "SKIP", or "OVERWRITE"
)
```

### 3. Uploading a Single File

You can also upload a single file. The SDK will automatically handle whether to use a standard or resumable multipart upload.

```python
aws_uploader.upload_file(
    local_path="./logs/large_log_file.bin",
    bucket_name="my-backup-bucket",
    s3_key="archive/large_log_file.bin"
)
```

### Full Example Script

Here is a complete example demonstrating a folder upload to a Ceph endpoint.

```python
# main_example.py

import os
from s3_sdk import S3Uploader
from botocore.exceptions import ClientError

def main():
    """
    Example of uploading a local folder to an S3-compatible endpoint.
    """
    # --- Configuration ---
    ENDPOINT = "http://172.15.20.153"
    ACCESS_KEY = "YOUR_ACCESS_KEY"
    SECRET_KEY = "YOUR_SECRET_KEY"
    BUCKET_NAME = "my-test-bucket"
    LOCAL_FOLDER = "test_data" # Create this folder and add some files

    # Create dummy folder and a large file for testing
    if not os.path.exists(LOCAL_FOLDER):
        os.makedirs(LOCAL_FOLDER)
        print(f"Created test folder: '{LOCAL_FOLDER}'")
        with open(os.path.join(LOCAL_FOLDER, 'large_file.bin'), 'wb') as f:
            f.write(os.urandom(150 * 1024 * 1024)) # 150 MB file
        with open(os.path.join(LOCAL_FOLDER, 'small_file.txt'), 'w') as f:
            f.write("This is a small text file.")

    try:
        # 1. Initialize the uploader
        uploader = S3Uploader(
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            endpoint_url=ENDPOINT,
            verbose=True
        )

        # 2. Ensure the bucket exists (optional, but good practice for local setups)
        try:
            uploader.s3_client.create_bucket(Bucket=BUCKET_NAME)
            print(f"✅ Bucket '{BUCKET_NAME}' created.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                print(f"✔️ Bucket '{BUCKET_NAME}' already exists.")
            else:
                raise

        # 3. Upload the folder
        uploader.upload_folder(
            local_folder_path=LOCAL_FOLDER,
            bucket_name=BUCKET_NAME,
            s3_folder_prefix="sdk-upload-test",
            sync_mode="SYNC"
        )

    except (ValueError, ConnectionError) as e:
        print(f"\n❌ An error occurred: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
```

## 🧠 Core Concepts Explained

### Sync Modes

-   **`OVERWRITE`**: Always uploads the file. If a file with the same name exists in the bucket, it will be replaced. Use this when you want to force a refresh of all data.
-   **`SKIP`**: If a file with the same name already exists in the bucket, the local file is skipped. This is the fastest mode if you only ever add new files and never modify existing ones.
-   **`SYNC`**: This is the smartest mode. It checks if a remote file exists.
    -   If it doesn't exist, the local file is uploaded.
    -   If it does exist, it compares the local file's size and MD5 hash against the remote file's metadata. The file is only uploaded if the size or hash differs. This is the most reliable way to keep a local directory synchronized with S3.

### Resumable Uploads

For any file larger than the configured `multipart_threshold` (default 100MB), the SDK initiates a multipart upload. During this process, it creates a temporary cache file next to the original file (e.g., `large_file.bin.s3_upload.json`).

This JSON file stores the `UploadId` and a list of parts that have been successfully uploaded. If the script is stopped and re-run, the SDK reads this file, validates it, and resumes uploading only the remaining parts.

The cache file is **automatically deleted** upon successful completion of the upload.

## 🛠️ Advanced Configuration

For fine-grained control over the upload process, you can pass a `TransferConfig` object from `boto3`. This allows you to change multipart thresholds, chunk sizes, and concurrency.

```python
from boto3.s3.transfer import TransferConfig

# Configure uploads to use smaller chunks and higher concurrency
config = TransferConfig(
    multipart_threshold=50 * 1024 * 1024,   # 50 MB
    max_concurrency=20,
    multipart_chunksize=8 * 1024 * 1024,    # 8 MB
    use_threads=True
)

uploader.upload_folder(
    local_folder_path=local_data_folder,
    bucket_name=bucket,
    transfer_config=config
)
```
