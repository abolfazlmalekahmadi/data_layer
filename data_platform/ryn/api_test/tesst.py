from clearml import Dataset
# Create a new dataset entry in ClearML
dataset = Dataset.create(
    dataset_name="s3_import_dataset",
    dataset_project="importing"
)

dataset.add_external_files(
    source_url="s3://akbr/main/",
    dataset_path="akbr/"
)

dataset.upload()
dataset.finalize()