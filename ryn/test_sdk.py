# test_sdk.py
from data.data_ingestion import download_dataset

# response = download_dataset(
#     source="huggingface",
#     dataset_name="glue",
#     dataset_config="sst2",
#     dataset_name_override = "onn",
#     user_name="mahdi",
#     private = False
# )

response = download_dataset(
    source="openml",
    dataset_id="31",
    dataset_name="sst2",
    user_name="mahdi",
    private = False
)


print(response)