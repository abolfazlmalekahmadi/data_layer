from clearml.backend_api.session import Session
from clearml.backend_api.session.client import APIClient



session = Session(
    user_name="testdario75",
    api_key="DFBDA091008F1E348E7ECA6E5576DCFB",
    secret_key="578DC5356DC1209538D0C7ABF79929E7492A640CD0C5FE0149102923E2B18C6A",
)

client = APIClient(session=session)

# original_client = APIClient.get_client()
# client.set_credentials(
#             api_key=credentials.get("access_key"),
#             secret_key=credentials.get("secret_key"),
#             # You might need to specify the api_server if not using the default
#             # api_server="https://api.clear.ml"
#         )