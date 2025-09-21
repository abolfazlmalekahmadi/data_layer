import json
import subprocess
import threading
import time
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import uuid
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s/%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define test credentials
CREDENTIALS = [
    {
        "user_name": "testdario75",
        "clearml": {
            "access_key": "6C8CD3D76920C6D1F81FBBBCD188734C",
            "secret_key": "F3F98D9B5361768BF497DFFB8F63F6E9115C5314FDED76FF18E3B82997DFB6D5"
        },
        "s3_source": {
            "access_key": "8HZE0U3P5VOSCPEOUN4G",
            "secret_key": "ihLqlXsauVYmiV83uu5kDdzAZzjlLlXYx05OIOwg",
            "endpoint_url": "http://172.15.20.153",
            "bucket_name": "testdario7-bucket"
        },
        "s3_target": {
            "access_key": "8HZE0U3P5VOSCPEOUN4G",
            "secret_key": "ihLqlXsauVYmiV83uu5kDdzAZzjlLlXYx05OIOwg",
            "endpoint_url": "http://172.15.20.153",
            "bucket_name": "testdario7-bucket"
        },
        "queue_name": "testdario7_queue"
    },
    {
        "user_name": "datadevnewuserv2",
        "clearml": {
            "access_key": "DFBDA091008F1E348E7ECA6E5576DCFB",
            "secret_key": "578DC5356DC1209538D0C7ABF79929E7492A640CD0C5FE0149102923E2B18C6A"
        },
        "s3_source": {
            "access_key": "9IZE1I6T4EYKW2HUHO4T",
            "secret_key": "8R7nIne82Ci2oEkUnPdxRcEPum1m7iMLd8ZCZzxs",
            "endpoint_url": "http://172.15.20.153",
            "bucket_name": "datadevnewuserv2-bucket-bucket"
        },
        "s3_target": {
            "access_key": "9IZE1I6T4EYKW2HUHO4T",
            "secret_key": "8R7nIne82Ci2oEkUnPdxRcEPum1m7iMLd8ZCZzxs",
            "endpoint_url": "http://172.15.20.153",
            "bucket_name": "datadevnewuserv2-bucket-bucket"
        },
        "queue_name": "datadevnewuserv2_queue"
    },
    {
        "user_name": "dummy_user",
        "clearml": {
            "access_key": "DUMMY_ACCESS_KEY",
            "secret_key": "DUMMY_SECRET_KEY"
        },
        "s3_source": {
            "access_key": "9IZE1I6T4EYKW2HUHO4T",
            "secret_key": "8R7nIne82Ci2oEkUnPdxRcEPum1m7iMLd8ZCZzxs",
            "endpoint_url": "http://172.15.20.153",
            "bucket_name": "datadevnewuserv2-bucket-bucket"
        },
        "s3_target": {
            "access_key": "9IZE1I6T4EYKW2HUHO4T",
            "secret_key": "8R7nIne82Ci2oEkUnPdxRcEPum1m7iMLd8ZCZzxs",
            "endpoint_url": "http://172.15.20.153",
            "bucket_name": "datadevnewuserv2-bucket-bucket"
        },
        "queue_name": "default"
    }
]

# Metrics tracking
metrics = {
    "requests": 0,
    "failures": 0,
    "response_times": [],
    "start_time": None,
    "lock": threading.Lock(),
    "user_metrics": {
        "testdario75": {"requests": 0, "failures": 0},
        "datadevnewuserv2": {"requests": 0, "failures": 0},
        "dummy_user": {"requests": 0, "failures": 0}
    }
}

def send_request(credentials, api_url="http://localhost:8279"):
    user_name = credentials["user_name"]
    dataset_name = f"airline-passengers-{uuid.uuid4().hex[:8]}"  # Unique dataset name
    payload = {
        "s3_file_path": "ceph-upload-test/airline-passengers.csv",
        "dataset_name": dataset_name,
        "user_name": user_name,
        "private": True,
        "dataset_tag": "ML",
        "queue_name": credentials.get("queue_name", "default"),
        "clearml": credentials["clearml"],
        "s3_source": {
            "access_key": "8HZE0U3P5VOSCPEOUN4G",
            "secret_key": "ihLqlXsauVYmiV83uu5kDdzAZzjlLlXYx05OIOwg",
            "endpoint_url": "http://172.15.20.153",
            "bucket_name": "testdario7-bucket"
        },
        "s3_target": {
            "access_key": "8HZE0U3P5VOSCPEOUN4G",
            "secret_key": "ihLqlXsauVYmiV83uu5kDdzAZzjlLlXYx05OIOwg",
            "endpoint_url": "http://172.15.20.153",
            "bucket_name": "testdario7-bucket"
        }
    }

    # Convert payload to JSON string for curl
    payload_json = json.dumps(payload)
    
    # Measure start time
    start_time = time.time()
    
    try:
        # Run curl command
        result = subprocess.run(
            [
                "curl", "-s", "-w", "%{http_code}",
                "-X", "POST",
                f"{api_url}/s3",
                "-H", "Content-Type: application/json",
                "-d", payload_json
            ],
            capture_output=True,
            text=True,
            timeout=60  # Increased timeout to handle slow responses
        )
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Get response body and status code
        status_code = int(result.stdout[-3:].strip())
        response_body = result.stdout[:-3].strip()
        
        with metrics["lock"]:
            metrics["requests"] += 1
            metrics["user_metrics"][user_name]["requests"] += 1
            metrics["response_times"].append(response_time)
        
        try:
            response_data = json.loads(response_body)
            if status_code == 200:
                if response_data.get("user_name") != user_name:
                    with metrics["lock"]:
                        metrics["failures"] += 1
                        metrics["user_metrics"][user_name]["failures"] += 1
                    logger.error(f"Race condition detected for {user_name}: got user {response_data.get('user_name')}")
                elif response_data.get("dataset_name") != dataset_name:
                    with metrics["lock"]:
                        metrics["failures"] += 1
                        metrics["user_metrics"][user_name]["failures"] += 1
                    logger.error(f"Dataset name mismatch for {user_name}: got {response_data.get('dataset_name')}")
                else:
                    logger.info(f"Success for {user_name}: Task {response_data.get('task_id')} created for dataset {dataset_name}")
            elif status_code == 500 and user_name == "dummy_user":
                with metrics["lock"]:
                    metrics["user_metrics"][user_name]["failures"] += 1
                logger.info(f"Expected failure for dummy_user: {response_data.get('detail', 'No detail')}")
            else:
                with metrics["lock"]:
                    metrics["failures"] += 1
                    metrics["user_metrics"][user_name]["failures"] += 1
                logger.error(f"Failed for {user_name}: {status_code}, {response_data.get('detail', 'No detail')}")
        except json.JSONDecodeError:
            with metrics["lock"]:
                metrics["failures"] += 1
                metrics["user_metrics"][user_name]["failures"] += 1
            logger.error(f"Invalid JSON for {user_name}: {response_body}")
            
    except subprocess.TimeoutExpired:
        with metrics["lock"]:
            metrics["failures"] += 1
            metrics["user_metrics"][user_name]["failures"] += 1
        logger.error(f"Request timeout for {user_name}")
    except Exception as e:
        with metrics["lock"]:
            metrics["failures"] += 1
            metrics["user_metrics"][user_name]["failures"] += 1
        logger.error(f"Error for {user_name}: {str(e)}")

def worker_thread(credentials, api_url, run_time, thread_id):
    logger.info(f"Thread {thread_id} started for user {credentials['user_name']} with credentials: {credentials['clearml']['access_key']}")
    end_time = time.time() + run_time
    while time.time() < end_time:
        send_request(credentials, api_url)
        # Random wait time between 1-5 seconds
        time.sleep(random.uniform(1, 5))

def print_summary():
    end_time = time.time()
    duration = end_time - metrics["start_time"]
    reqs_per_sec = metrics["requests"] / duration if duration > 0 else 0
    fails_per_sec = metrics["failures"] / duration if duration > 0 else 0
    response_times = metrics["response_times"]
    
    avg_time = int(statistics.mean(response_times)) if response_times else 0
    min_time = int(min(response_times)) if response_times else 0
    max_time = int(max(response_times)) if response_times else 0
    median_time = int(statistics.median(response_times)) if response_times else 0
    
    print("\nType Name # reqs # fails | Avg Min Max Med | req/s failures/s")
    print("-" * 80)
    print(f"POST /s3 {metrics['requests']} {metrics['failures']}({metrics['failures']/metrics['requests']*100:.2f}%) | {avg_time} {min_time} {max_time} {median_time} | {reqs_per_sec:.2f} {fails_per_sec:.2f}")
    print("-" * 80)
    print("Per-User Metrics:")
    for user, user_data in metrics["user_metrics"].items():
        print(f"{user}: {user_data['requests']} requests, {user_data['failures']} failures ({user_data['failures']/user_data['requests']*100:.2f}%)")
    print("-" * 80)
    print(f"Total {metrics['requests']} {metrics['failures']}({metrics['failures']/metrics['requests']*100:.2f}%) | {reqs_per_sec:.2f}")

def main():
    api_url = "http://localhost:8279"  # Replace with forwarded URL if needed
    num_users = 20  # Simulate 20 concurrent users
    run_time = 120  # Run for 2 minutes (120 seconds)
    
    logger.info("Starting stress test for /s3 endpoint")
    metrics["start_time"] = time.time()
    
    # Create thread pool
    with ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = []
        for i in range(num_users):
            # Assign credentials round-robin
            credentials = CREDENTIALS[i % len(CREDENTIALS)]
            futures.append(executor.submit(worker_thread, credentials, api_url, run_time, i))
        
        # Wait for all threads to complete
        for future in futures:
            future.result()
    
    logger.info("Stress test completed")
    print_summary()

if __name__ == "__main__":
    main()