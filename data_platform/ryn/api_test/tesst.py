from clearml import Dataset as ClearMLDataset
stored_path = "http://s3.cloud-ai.ir/"
metadata = {
    'dataset_name':"glue",
    'dataset_id':"cda84caa-e9a2-4777-ba9e-70a7bbb2ff8d"
}
dataset = ClearMLDataset.create(
    dataset_name=metadata["dataset_name"],
    dataset_project="importing",
    description=f"Dataset ID: {metadata["dataset_id"]}\nStored path: {stored_path}"
)
dataset.add_external_files(
    source_url="s3://mlops/glue",
    dataset_path="clearml-test"
)
dataset.upload()
dataset.finalize()

    # logger.error(f"Failed to register dataset with ClearML: {e}")