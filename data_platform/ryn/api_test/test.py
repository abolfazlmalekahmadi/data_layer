# import aiohttp
# import asyncio

# async def send_request(session, url, payload):
#     async with session.post(url, json=payload) as response:
#         return await response.json()

# async def main():
#     # url = "http://127.0.0.1:6991/huggingface"
#     # payload = {
#     #     "dataset_name": "glue",
#     #     "dataset_config": "axx",
#     #     "dataset_name_override": "fka/awesome-chatgpt-prompts",
#     #     "user_name": "Ali",
#     #     "private": False
#     # }
#     # url = "http://127.0.0.1:6992/openml"
#     # payload = {
#     #     "dataset_name": "daaaaa",
#     #     "dataset_id": 31,
#     #     "user_name": "Alis",
#     #     "private": False
#     # }
    
#     # url = "http://127.0.0.1:6992/s3"
#     # payload = {
#     #     'access_key' : "OAF0MC26UA7DV9WS11X5",
#     #     'secret_key' :"6SY2dTxhcIVEsjbfpjRUBhe3k7mMJIjZpccwvw3d",
#     #     'endpoint_url':"https://s3.cloud-ai.ir",
#     #     'bucket_name':"mlops",
#     #     's3_file_path':"classification_cat_dog.zip",
#     #     'dataset_name':"credit-g",
#     #     'user_name' :"ryn",
#     #     'private': False
#     # }
    
#     async with aiohttp.ClientSession() as session:
#         tasks = [send_request(session, url, payload) for _ in range(10)]
#         responses = await asyncio.gather(*tasks)
#         for i, resp in enumerate(responses, 1):
#             print(f"Response {i}: {resp}")

# asyncio.run(main())



import os
import logging
from clearml import Dataset

# Set ClearML credentials
os.environ['CLEARML_API_ACCESS_KEY'] = "8EEF2F95D9BF64B9D035B07800F2E02A"
os.environ['CLEARML_API_SECRET_KEY'] = "9A30C02DC1D7E430A8C1354EA56DF0FDC559710E2230CB51779A2A99AD08BA1D"
# logging.info(f'set clearml config .... access: {clearml_config["access_key"]} ,secret: {clearml_config["secret_key"]}')

# Create the dataset
dataset = Dataset.create(
    dataset_name="test",
    dataset_project="importing",
    description="Dataset ID: "
)

# Add your local file to the dataset
local_file_path = "/home/coder/persistent/agent-cpu.yaml"  # Replace with your actual file path

# Method 1: Add a single file
dataset.add_files(local_file_path)
print(f"dataset name is : {dataset.id}")

# Method 2: Add file with a specific name in the dataset
# dataset.add_files(local_file_path, dataset_path="custom_name.txt")

# Method 3: Add entire folder (if you have multiple files)
# local_folder_path = "path/to/your/folder"
# dataset.add_files(local_folder_path)

# Method 4: Add files with wildcard pattern
# dataset.add_files("path/to/files/*.csv")

# Upload and finalize the dataset
dataset.upload()
dataset.finalize()

# Get the dataset ID for future reference
dataset_id = dataset.id
logging.info(f"Dataset created and uploaded successfully. Dataset ID: {dataset_id}")

# Optional: Get dataset statistics
print(f"Dataset URL: {dataset.get_default_storage()}")
print(f"Number of files: {len(dataset.list_files())}")