from clearml.backend_api.session.client import APIClient

client = APIClient()

try:
    # First check if it exists
    resp = client.queues.get_all(name="dataset_ingestion")
    if resp.queues:
        print(f"Queue already exists with ID: {resp.queues[0].id}")
    else:
        print("Queue doesn't exist - you'll need to create it manually in the UI")
except Exception as e:
    print(f"Error checking queue: {e}")