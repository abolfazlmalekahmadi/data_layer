from clearml.backend_api.session.client import APIClient
from clearml.backend_api.session import Session

import inspect

# Replace with your actual keys
credentials = {
    "access_key": "6C8CD3D76920C6Dasdad1F81FBBBCD188734C",
    "secret_key": "F3F98D9B5361768BF497DFFB8F63F6E9115C5314FDED76FF18E3B82997DFB6D5"
}

# Initialize the client with credentials
session = Session(key="6C8CD3D76920C6Dasdad1F81FBBBCD188734C",secret="F3F98D9B5361768BF497DFFB8F63F6E9115C5314FDED76FF18E3B82997DFB6D5")

client= APIClient(session=session)
methods = inspect.getmembers(session, predicate=inspect.isfunction)
print(methods)