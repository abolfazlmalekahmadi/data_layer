import requests
import json

# Define your ClearML API server URL and API key
API_URL = 'http://localhost:8008'
ACCESS_KEY = 'U02YLU0ROH0XOEMEI22W0EM7KUN5JN'
SECRET_KEY = 'm1vTcahGNjF2hT9kZXeYbLKYaqN6z0IC3k5HR2f9lwbvqoUt10L6-oTePeYnQc1DBus'

# Task creation endpoint
task_creation_endpoint = f"{API_URL}/tasks.create"

# Define the headers for authentication
headers = {
    # 'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
}

# Define the request payload
task_payload = {
    "name": "demo",
    "project": "2ddc59c355834e2d8d226f530f0b1a28",  # Project ID
    "type": "training",
    "script": {
        "repository": "https://github.com/TheHypnotic/automl_example",
        "entry_point": "timeseries_clearml_sdk.py"
    },

}
# Send the POST request to create the task
# response = requests.post(task_creation_endpoint, headers=headers, data=json.dumps(task_payload))
response = requests.post(
    task_creation_endpoint,
    headers=headers,
    data=json.dumps(task_payload),
    auth=requests.auth.HTTPBasicAuth(ACCESS_KEY, SECRET_KEY)
)

# Check the response
if response.status_code == 200:
    print("Task created successfully!")
    print(response.json())
else:
    print(f"Failed to create task. Status code: {response.status_code}")
    print(response.text)
