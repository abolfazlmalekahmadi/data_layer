# from clearml.backend_api.session.client import APIClient

# client = APIClient()

# try:
#     # First check if it exists
#     resp = client.queues.get_all(name="dataset_ingestion")
#     if resp.queues:
#         print(f"Queue already exists with ID: {resp.queues[0].id}")
#     else:
#         print("Queue doesn't exist - you'll need to create it manually in the UI")
# except Exception as e:
#     print(f"Error checking queue: {e}")



from clearml import Dataset

# Replace with your dataset name
dataset_name = "classification_cat_dog_abolfazl"

# Get the dataset by name
dataset = Dataset.get(dataset_name=dataset_name)

# Fetch and print the dataset metadata
metadata = dataset.get_metadata()
print("Metadata for dataset:", metadata)
# print(metadata)

        # Step 1: Find the ClearML dataset by name and user
# datasets = Dataset.list_datasets(
#              # Your project name for datasets
#             partial_name=dataset_name,
#             only_completed=True
#         )

# print(datasets)