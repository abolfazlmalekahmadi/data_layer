# test_sdk.py
from data.data_ingestion import download_dataset

# response = download_dataset(
#     source="huggingface",
#     dataset_name="fka/awesome-chatgpt-prompts",
#     dataset_config="default",
#     user_name="ryn1",
#     private = False,
#     dataset_tag = "ML"
# )

# response = download_dataset(

# source="openml",
# dataset_id=31,
# dataset_name="credit-g",
# user_name="ryn",
# private=False

# )

response = download_dataset(
    
source="s3",
access_key="OAF0MC26UA7DV9WS11X5",
secret_key="6SY2dTxhcIVEsjbfpjRUBhe3k7mMJIjZpccwvw3d",
endpoint_url="https://s3.cloud-ai.ir",
bucket_name="mlops",
s3_file_path="classification_cat_dog.zip",
dataset_name="creghjkjdit-g",
user_name="ryn",
private=False,
dataset_tag = "ML"
)
print(response)