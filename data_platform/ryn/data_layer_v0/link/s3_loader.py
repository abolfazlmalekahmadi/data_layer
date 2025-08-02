import os
import pandas as pd
from clearml import Dataset

# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
# df = pd.read_csv(url)

# # save locally
# os.makedirs("datasets/airline_passengers", exist_ok=True)
# # df.to_csv("airline-passengers.csv", index=False)
# csv_path = "datasets/airline_passengers/airline-passengers.csv"
# df.to_csv(csv_path, index=False)

# dataset = Dataset.create(dataset_name="airline_passengers_dataset",
#                         dataset_project="TimeSeries SDK",
#                         dataset_tags=["airline", "time_series"])
# dataset.add_files(csv_path)
# dataset.upload()
# dataset.finalize()
# print("✅ Airline Passengers dataset uploaded to ClearML")

# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"
# df = pd.read_csv(url)

# os.makedirs("datasets/sunspots", exist_ok=True)
# csv_path = "datasets/sunspots/monthly-sunspots.csv"
# df.to_csv(csv_path, index=False)

# dataset = Dataset.create(
#     dataset_name="monthly_sunspots_dataset",
#     dataset_project="TimeSeries SDK",
#     dataset_tags=["sunspots", "time_series"]
# )
# dataset.add_files(csv_path)
# dataset.upload()
# dataset.finalize()

# print("✅ Sunspots dataset uploaded to ClearML")

def upload_dataset_to_clearml(url ,dataset_name, dataset_project, dataset_tags, csv_path):
    """
    Upload a dataset to ClearML.
    
    Args:
        url (str): URL of the dataset to download.
        dataset_name (str): Name of the dataset.
        dataset_project (str): Project name in ClearML.
        dataset_tags (list): List of tags for the dataset.
        csv_path (str): Path to the CSV file to upload.
    """
    # Download the dataset
    df = pd.read_csv(url)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ {dataset_name} downloaded from {url}")

    # Create a new dataset
    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_tags=dataset_tags
    )
    
    # Add files to the dataset
    dataset.add_files(csv_path)
    
    # Upload the files
    dataset.upload()
    
    # Finalize the dataset
    dataset.finalize()
    
    print(f"✅ {dataset_name} uploaded to ClearML")
# Example usage
upload_dataset_to_clearml(
    url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv",
    dataset_name="monthly_sunspots_dataset",
    dataset_project="TimeSeries SDK",
    dataset_tags=["sunspots", "time_series"],
    csv_path="datasets/sunspots/monthly-sunspots.csv"
)

upload_dataset_to_clearml(
    url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
    dataset_name="airline_passengers_dataset",
    dataset_project="TimeSeries SDK",
    dataset_tags=["airline", "time_series"],
    csv_path="datasets/airline_passengers/airline-passengers.csv"
)
