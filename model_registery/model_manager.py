import os
import psycopg2
import boto3
from dotenv import load_dotenv

# PostgreSQL Configuration
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Load environment variables from .env file
load_dotenv()
if DB_NAME is None:
    DB_NAME = os.getenv("DB_NAME")
if DB_USER is None:
    DB_USER = os.getenv("DB_USER")
if DB_PASSWORD is None:
    DB_PASSWORD = os.getenv("DB_PASSWORD")
if DB_HOST is None:
    DB_HOST = os.getenv("DB_HOST")
if DB_PORT is None:
    DB_PORT = os.getenv("DB_PORT")

# Ceph Configuration
CEPH_ENDPOINT = os.getenv("CEPH_ENDPOINT")
CEPH_ACCESS_KEY = os.getenv("CEPH_ACCESS_KEY")
CEPH_SECRET_KEY = os.getenv("CEPH_SECRET_KEY")
BUCKET_NAME = "mlops"

print("######## DB info", DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT)

class ModelRegistry:
    def __init__(self):
        """Initialize S3 client for Ceph connection."""
        self.s3 = boto3.client(
            "s3",
            endpoint_url=CEPH_ENDPOINT,
            aws_access_key_id=CEPH_ACCESS_KEY,
            aws_secret_access_key=CEPH_SECRET_KEY
        )

    def get_connection(self):
        """Establish and return a connection to PostgreSQL."""
        try:
            return psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
        except psycopg2.Error as e:
            print(f"Database connection failed: {e}")
            return None

    def validate_user(self, username, user_token):
        """Check if a username and token are valid."""
        conn = self.get_connection()
        if not conn:
            return False
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE username = %s AND token = %s;", (username, user_token))
        exists = cur.fetchone()
        cur.close()
        conn.close()
        return bool(exists)

    def upload_to_ceph(self, local_path, remote_path):
        """Upload a file or folder to Ceph S3."""
        print(f"Uploading '{local_path}' to '{BUCKET_NAME}/{remote_path}'...")
        if os.path.isdir(local_path):
            base_dir = os.path.basename(local_path)
            for root, _, files in os.walk(local_path):
                for file in files:
                    local_file = os.path.join(root, file)
                    s3_path = os.path.join(remote_path, base_dir, os.path.relpath(local_file, local_path)).replace("\\", "/")
                    self.upload_file(local_file, s3_path)
        else:
            self.upload_file(local_path, remote_path)

    def upload_file(self, local_file_path, remote_path):
        """Upload a single file to Ceph S3."""
        if remote_path.endswith('/'):
            remote_path = os.path.join(remote_path, os.path.basename(local_file_path))
        try:
            self.s3.upload_file(local_file_path, BUCKET_NAME, remote_path)
            print(f"Uploaded '{local_file_path}' to '{BUCKET_NAME}/{remote_path}'")
        except Exception as e:
            print(f"Error uploading file: {e}")

    def download_file(self, remote_path, local_path):
        """Download a single file from Ceph S3."""
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            self.s3.download_file(BUCKET_NAME, remote_path, local_path)
            print(f"Downloaded '{remote_path}' to '{local_path}'")
        except Exception as e:
            print(f"Error downloading file '{remote_path}': {e}")

    def download_folder(self, remote_folder, local_folder):
        """Download all files inside a folder from Ceph while keeping folder structure."""
        if not remote_folder.endswith("/"):
            remote_folder += "/"

        response = self.s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=remote_folder)
        if "Contents" not in response:
            print(f"Folder '{remote_folder}' does not exist.")
            return
        print(local_folder)
        print(f"Downloading folder '{remote_folder}' to '{local_folder}'...")
        os.makedirs(local_folder, exist_ok=True)
        
        for obj in response["Contents"]:
            file_key = obj["Key"]
            relative_path = file_key[len(remote_folder):]  
            local_file_path = os.path.join(local_folder, relative_path)

            self.download_file(file_key, local_file_path)

    def register_model(self, username, user_token, model_name, model_version, local_path, train_code_path, extra_json_path):
        """Register a model, upload its files to Ceph, and store metadata in PostgreSQL."""
        if not self.validate_user(username, user_token):
            print("Invalid username or token.")
            return

        conn = self.get_connection()
        if not conn:
            return
        cur = conn.cursor()
        filename = os.path.basename(local_path)
        print("Uploading model files to Ceph...")
        if username == 'mlops':
            model_remote_path = f"models/{model_name}/{model_version}/"
            train_remote_path = f"scripts/{model_name}/{model_version}/train.py"
            extra_remote_path = f"metadata/{model_name}/{model_version}/extra.json"
        else:
            model_remote_path = f"users/{username}/models/{model_name}/{model_version}/"
            train_remote_path = f"users/{username}/scripts/{model_name}/{model_version}/train.py"
            extra_remote_path = f"users/{username}/metadata/{model_name}/{model_version}/extra.json"
        
        self.upload_to_ceph(local_path, model_remote_path)
        self.upload_file(train_code_path, train_remote_path)
        self.upload_file(extra_json_path, extra_remote_path)

        cur.execute("INSERT INTO models (username, model_name, model_version, file_name) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;", (username, model_name, model_version, filename))
        print(f"Model '{model_name}' version '{model_version}' registered successfully.")

        conn.commit()
        cur.close()
        conn.close()

    def download_model(self, username, user_token, model_name, model_version, local_path):
        """Retrieve a model from Ceph and save it locally."""
        if not self.validate_user(username, user_token):
            print("Invalid username or token.")
            return
        conn = self.get_connection()
        if not conn:
            return
        cur = conn.cursor()
        cur.execute("SELECT file_name FROM models WHERE username=%s and model_name=%s and model_version=%s;", (username, model_name, model_version))
        filename = cur.fetchall()[0][0]
        print("Downloading model files from Ceph...")
        
        if username == 'mlops':
            model_remote_path = os.path.join(f"models/{model_name}/{model_version}/", filename)
            train_remote_path = f"scripts/{model_name}/{model_version}/train.py"
            extra_remote_path = f"metadata/{model_name}/{model_version}/extra.json"
        else:
            model_remote_path = os.path.join(f"users/{username}/models/{model_name}/{model_version}/", filename)
            train_remote_path = f"users/{username}/scripts/{model_name}/{model_version}/train.py"
            extra_remote_path = f"users/{username}/metadata/{model_name}/{model_version}/extra.json"

        model_local_path = os.path.join(local_path, "model", filename)
        if "." in filename:
            self.download_file(model_remote_path, model_local_path)
        else:
            self.download_folder(model_remote_path, model_local_path)

        train_local_path = os.path.join(local_path, "train.py")
        self.download_file(train_remote_path, train_local_path)

        extra_local_path = os.path.join(local_path, "extra.json")
        self.download_file(extra_remote_path, extra_local_path)

        print(f"Model '{model_name}' version '{model_version}' downloaded successfully.")
        return os.path.join(local_path, "model", filename)

# Example Execution
if __name__ == "__main__":
    registry = ModelRegistry()
    # registry.download_folder("models/llm/qwen-small/Qwen2.5-0.5B-Instruct", "/home/user01/model_data/Qwen2.5-0.5B-Instruct")

    # # Register a model
    registry.register_model(
        username="mlops",
        user_token="c525a364-b08c-4ad7-a445-ac1e2410f447",
        model_name="llm-small",
        model_version="1.0",
        local_path="/home/user01/model_data/Qwen2.5-0.5B-Instruct",
        train_code_path="./test.py",
        extra_json_path="./test.json"
    )
    # path = registry.download_model(username='mlops', 
    #                         user_token='c525a364-b08c-4ad7-a445-ac1e2410f447', 
    #                         model_name='llm-small', 
    #                         model_version='1.0', 
    #                         local_path='model_dir/')
    # print(path)
    # registry.upload_to_ceph("model_dir/extra.json", 'test/')
    # # Download the model
    # registry.download_model(
    #     username="mlops",
    #     user_token="2381da4f-f173-48bf-8124-f99a38b136eb",
    #     model_name="teeth_segmentation20",
    #     model_version="1.0",
    #     local_path="/home/mlops/mohammad/localPathTest"
    # )
