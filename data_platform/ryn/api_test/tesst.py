from clearml.backend_api.session import Session
from clearml.backend_api.session.client import APIClient



session = Session(
    api_key="260F9D73553606E792EEF02EE1913D15",
    secret_key="B67919A8F659354DFF0ED5E28B33DF99C4F0693C38BCF5E523148C157B0E9301",
)

client = APIClient(session=session)

# original_client = APIClient.get_client()
client.set_credentials(
            api_key=credentials.get("access_key"),
            secret_key=credentials.get("secret_key"),
            # You might need to specify the api_server if not using the default
            # api_server="https://api.clear.ml"
        )