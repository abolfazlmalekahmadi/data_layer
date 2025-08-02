
from clearml import Task, Dataset as ClearMLDataset
from datasets import load_dataset
import pandas as pd
import random
import os
from datasets import DatasetDict

# Function to split a single-split dataset into train, validation, and test
def split_dataset_if_needed(dataset, split_name="train", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    if len(dataset.keys()) == 1 and split_name in dataset:
        task.get_logger().report_text(f"Single split detected ({split_name}). Splitting into train, validation, and test...")
        # Ensure ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1"
        
        # Split into train and temp (val+test)
        train_test_split = dataset[split_name].train_test_split(test_size=(val_ratio + test_ratio), seed=seed)
        train_ds = train_test_split["train"]
        temp_ds = train_test_split["test"]
        
        # Split temp into validation and test
        val_test_ratio = test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0
        val_test_split = temp_ds.train_test_split(test_size=val_test_ratio, seed=seed)
        val_ds = val_test_split["train"]
        test_ds = val_test_split["test"]
        
        # Create new DatasetDict
        return DatasetDict({
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds
        })
    return dataset

# Initialize ClearML Task
task = Task.init(project_name="HuggingFace Pipeline", task_name="Scalable Dataset Processing")

# Dataset configuration (modify these for different datasets)
dataset_name = "ylecun/mnist"  # Example: "pourmand1376/fa-wikipedia"
dataset_config = None # Set to None if no config (e.g., for fa-wikipedia)
sample_size = 5  # Number of random samples to log per split
seed = 42  # For reproducibility

# Step 1: Download the dataset
task.get_logger().report_text(f"Starting download for {dataset_name} (config: {dataset_config})...")
ds = load_dataset(dataset_name, dataset_config)
task.get_logger().report_text("Dataset downloaded successfully.")

# Step 2: Handle splits (split if only one split exists)
task.get_logger().report_text(f"Available splits: {list(ds.keys())}")
ds = split_dataset_if_needed(ds, split_name="train", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=seed)
task.get_logger().report_text(f"Processed splits: {list(ds.keys())}")

# Step 3: Process each split
for split_name in ds.keys():
    task.get_logger().report_text(f"Processing split: {split_name}")
    
    # Convert split to pandas DataFrame
    df = ds[split_name].to_pandas()
    
    # Save split as CSV
    dataset_path = f"{dataset_name.replace('/', '_')}_{split_name}.csv"
    df.to_csv(dataset_path, index=False)
    
    # Upload as ClearML artifact
    artifact_name = f"{dataset_name.replace('/', '_')}_{split_name}"
    task.upload_artifact(name=artifact_name, artifact_object=dataset_path)
    task.get_logger().report_text(f"Saved and uploaded artifact: {dataset_path}")
    
    # Select and log random samples
    task.get_logger().report_text(f"Selecting {sample_size} random samples for {split_name}...")
    random_samples = df.sample(n=min(sample_size, len(df)), random_state=seed).to_dict(orient="records")
    # Log samples dynamically based on available features
    # What is this for?
    # for i, sample in enumerate(random_samples):
    #     task.get_logger().report_text(f"Sample {i+1} ({split_name}):")
    #     for feature, value in sample.items():
    #         # Handle string truncation for long text fields
    #         display_value = str(value)[:200] + "..." if isinstance(value, str) and len(str(value)) > 200 else value
    #         task.get_logger().report_text(f"{feature}: {display_value}")
    #     task.get_logger().report_table(
    #         title=f"Sample {i+1} ({split_name})",
    #         series=f"Random Samples ({split_name})",
    #         iteration=0,
    #         table_plot=pd.DataFrame([sample])
    #     )

# Step 4: Save all splits to ClearML Dataset for versioning
task.get_logger().report_text("Creating ClearML Dataset...")
clearml_dataset = ClearMLDataset.create(
    dataset_name=dataset_name.replace('/', '_'),
    dataset_project="HuggingFace Pipeline",
    dataset_tags=[dataset_name.split('/')[-1], "huggingface"]
)
for split_name in ds.keys():
    dataset_path = f"{dataset_name.replace('/', '_')}_{split_name}.csv"
    clearml_dataset.add_files(dataset_path)
    task.get_logger().report_text(f"Added {dataset_path} to ClearML Dataset")
clearml_dataset.upload()
task.get_logger().report_text("Dataset uploaded successfully")
clearml_dataset.finalize()
task.get_logger().report_text(f"Dataset registered in ClearML Dataset: {clearml_dataset.id}")

# Close the task
task.close()