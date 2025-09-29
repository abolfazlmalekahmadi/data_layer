import os
import yaml
import pandas as pd
from pathlib import Path
import cv2
import numpy as np 
import shutil
import scipy.io.wavfile as wavfile

# Import the main orchestrator class
from data.data_validation.validation import DirectoryValidator

def setup_test_directory(root_dir: Path):
    """Creates a sample directory with data files and a config for testing."""
    print(f"--- Setting up test directory at '{root_dir}' ---")

    # Define paths
    users_path = root_dir / "landing/users"
    transactions_path = root_dir / "processed/transactions/day=1"
    images_path = root_dir / "assets/product_images"
    large_files_path = root_dir / "landing/large_files"
    audio_path = root_dir / "assets/audio_files"
    audio_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)

    large_files_path.mkdir(parents=True, exist_ok=True)
    
    # Create directories
    users_path.mkdir(parents=True, exist_ok=True)
    transactions_path.mkdir(parents=True, exist_ok=True)

    HUGE_FILE_ROWS = 2_500_000
    # print(f"Generating a large CSV with {HUGE_FILE_ROWS:,} rows. This may take a moment...")
    # huge_data = {
    #     'user_id': range(HUGE_FILE_ROWS),
    #     'email': [f'user_{i}@massive-data.com' for i in range(HUGE_FILE_ROWS)],
    #     "username": [f'user_{i}' for i in range(HUGE_FILE_ROWS)],
    #     'status': ['active'] * HUGE_FILE_ROWS
    # }
    # for i in range(15):
    #     huge_data[f"username{i}"]= huge_data["username"]
    # huge_data["nuu"] = [None  for i in range(HUGE_FILE_ROWS)]
    # huge_df = pd.DataFrame(huge_data)
    # random_index = np.random.choice(huge_df.index)
    # huge_df.at[random_index, "email"] = "https://hello.com"
    # huge_df.at[random_index, "status"] = "http://hello.com"

    # print(huge_df.at[random_index,"email"],random_index)
    # huge_file_path = large_files_path / "huge_user_file.csv"
    # huge_df.to_csv(huge_file_path, index=False)
    # print(f"Created: {huge_file_path}")
    zips_path = root_dir / "landing/zips" # NEW: Directory for zip files
    zips_path.mkdir(parents=True, exist_ok=True)

    # ... (all previous file creation is the same) ...

    # --- NEW: Create a CSV and then zip it up ---
    zipped_csv_data = {
        "order_id": ["A101", "B202"],
        "product_id": [123, 456],
        "quantity": [2, 1]
    }
    temp_csv_path = root_dir / "temp_orders.csv"
    pd.DataFrame(zipped_csv_data).to_csv(temp_csv_path, index=False)


    zipped_csv_data2 = {
        "order_id": ["A10{i}" for i in range(HUGE_FILE_ROWS)],
        "product_id": [123 for i in range(HUGE_FILE_ROWS)],
        "quantity": [2 for i in range(HUGE_FILE_ROWS)],
        "username": [f'https://user_{i}.com' for i in range(HUGE_FILE_ROWS)]
    }
    temp_csv_path = root_dir / "temp_orders2.csv"
    pd.DataFrame(zipped_csv_data2).to_csv(temp_csv_path, index=False)
    
    # Create the zip archive
    zip_archive_path = zips_path / "orders_archive" # Note: .zip is added automatically
    shutil.make_archive(str(zip_archive_path), 'zip', root_dir, "temp_orders.csv")
    shutil.make_archive(str(zip_archive_path), 'zip', root_dir, "temp_orders2.csv")
    os.remove(temp_csv_path) # Clean up the temporary source CSV
    print(f"Created archive: {zip_archive_path}.zip")

    # 1. Create a GOOD CSV file that should pass validation
    # good_data = {
    #     "user_id": [1, 2, 3],
    #     "email": ["one@test.com", "two@test.com", "three@test.com"],
    #     "age": [25, 30, 35]
    # }
    # pd.DataFrame(good_data).to_csv(users_path / "users_good.csv", index=False)
    # print(f"Created: {users_path / 'users_good.csv'}")

    # 2. Create a BAD CSV file that should fail validation
    #    - 'user_id' contains a null value (violates not_null)
    #    - 'email' column is missing (violates required_columns)
    # link_data = {
    #     "user_id": [9, 10],
    #     "email": ["nine@test.com", "ten@test.com"],
    #     "comment": ["All good", "Please see details at http://malicious-site.com"]
    # }
    # pd.DataFrame(link_data).to_csv(users_path / "users_with_links.csv", index=False)
    # print(f"Created: {users_path / 'users_with_links.csv'}")
    # bad_data = {
    #     "user_id": [4, None, 6],
    #     "age": [40, 45, 50]
    # }
    # pd.DataFrame(bad_data).to_csv(users_path / "users_bad.csv", index=False)
    # print(f"Created: {users_path / 'users_bad.csv'}")

    # 3. Create a valid Parquet file to test a different file type and pattern
    # transaction_data = {
    #     "transaction_id": ["txn_101", "txn_102"],
    #     "user_id": [1, 3],
    #     "amount": [99.99, 12.50]
    # }
    # pd.DataFrame(transaction_data).to_parquet(transactions_path / "transactions.parquet")
    # print(f"Created: {transactions_path / 'transactions.parquet'}")

    # valid_image_path = images_path / "product_good.jpg"
    # good_image_data = np.zeros((512, 512, 3), dtype=np.uint8) # Start with black
    # good_image_data[0, 0] = [255, 255, 255] # Add a single white pixel to make it valid
    # cv2.imwrite(str(valid_image_path), good_image_data)
    # print(f"Created: {valid_image_path}")

    # # # 2. A completely BLACK image (should fail)
    # black_image_path = images_path / "product_black.jpg"
    # black_image_data = np.zeros((512, 512, 3), dtype=np.uint8)
    # cv2.imwrite(str(black_image_path), black_image_data)
    # print(f"Created: {black_image_path}")
    
    # # 3. A completely WHITE image (should fail)
    # white_image_path = images_path / "product_white.jpg"
    # white_image_data = np.ones((512,512 , 3), dtype=np.uint8) * 255
    # cv2.imwrite(str(white_image_path), white_image_data)
    # print(f"Created: {white_image_path}")

    # # 4. An INVALID image file (a text file with a .jpg extension)
    # invalid_image_path = images_path / "product_corrupt.jpg"
    # with open(invalid_image_path, "w") as f:
    #     f.write("This is not an image!")
    # print(f"Created: {invalid_image_path}")
    # 4. Create the YAML configuration file
    # config_content = """
    # - name: "Validate User CSVs"
    #   path_pattern: "landing/users/*.csv"
    #   validator: "tabular"
    #   rules:
    #     dataset:
    #       required_columns:
    #         - "user_id"
    #         - "email"
    #     columns:
    #       user_id:
    #         not_null: true

    # - name: "Validate Transaction Parquet Files"
    #   path_pattern: "processed/transactions/**/*.parquet"
    #   validator: "tabular"
    #   rules:
    #     dataset:
    #       required_columns:
    #         - "transaction_id"
    #         - "user_id"
    # """
    
    # 1. Good audio file (2-second 440Hz sine wave)
    sample_rate = 16000
    good_audio_path = audio_path / "good_audio.wav"
    duration_sec = 2
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2.0 * np.pi * 440.0 * t)
    wavfile.write(str(good_audio_path), sample_rate, data.astype(np.int16))
    print(f"Created: {good_audio_path}")

    # 2. Silent audio file
    silent_audio_path = audio_path / "silent_audio.wav"
    silent_data = np.zeros(sample_rate, dtype=np.int16) # 1 second of silence
    wavfile.write(str(silent_audio_path), sample_rate, silent_data)
    print(f"Created: {silent_audio_path}")
    
    # 3. Too short audio file (0.05 seconds)
    short_audio_path = audio_path / "short_audio.wav"
    short_duration_sec = 0.05
    t_short = np.linspace(0., short_duration_sec, int(sample_rate * short_duration_sec), endpoint=False)
    short_data = amplitude * np.sin(2. * np.pi * 440. * t_short)
    wavfile.write(str(short_audio_path), sample_rate, short_data.astype(np.int16))
    print(f"Created: {short_audio_path}")
    
    # 4. Corrupt audio file
    corrupt_audio_path = audio_path / "corrupt_audio.wav"
    with open(corrupt_audio_path, "w") as f:
        f.write("This is not valid audio data.")
    print(f"Created: {corrupt_audio_path}")
    config_path = root_dir / "validation_config.yaml"
    # with open(config_path, "w") as f:
    #     f.write(config_content)
    # print(f"Created: {config_path}")
    logs_path = root_dir / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)

    # 1. Good log file (should pass all checks)
    good_log_path = logs_path / "app_2023-10-27.log"
    with open(good_log_path, "w", encoding="utf-8") as f:
        f.write("INFO: Application started successfully.\n")
        f.write("DEBUG: Configuration loaded from environment.\n")
        f.write("INFO: Processing request #12345.\n")
    print(f"Created (should pass): {good_log_path}")

    # 2. Empty log file (should fail 'disallow_empty_file')
    empty_log_path = logs_path / "empty.log"
    open(empty_log_path, 'a').close()
    print(f"Created (should fail - empty): {empty_log_path}")

    # 3. Log file with a forbidden pattern (should fail 'forbidden_patterns')
    secret_log_path = logs_path / "app_error.log"
    with open(secret_log_path, "w", encoding="utf-8") as f:
        f.write("ERROR: Failed to connect to database.\n")
        f.write("FATAL: Credentials compromised. User provided password='MyWeakPassword123'\n")
        f.write("FATAL: Credentials compromised. User provided password='MyWeakPassword123'\n")

    print(f"Created (should fail - forbidden pattern): {secret_log_path}")

    # 4. Log file with a very long line (should fail 'max_line_length')
    long_line_log_path = logs_path / "long_line.log"
    with open(long_line_log_path, "w", encoding="utf-8") as f:
        f.write("This is a normal line.\n")
        f.write("This is a very long line designed to fail the max_line_length check " + ("." * 10000) + "\n")
    print(f"Created (should fail - long line): {long_line_log_path}")

    # 5. File with bad encoding (should fail 'file_encoding')
    bad_encoding_path = logs_path / "legacy_output.log"
    # Write with UTF-16, but validator will try to read as UTF-8
    with open(bad_encoding_path, "w", encoding="utf-16") as f:
        f.write("This file contains special characters: ö, é, å\n")
    print(f"Created (should fail - bad encoding): {bad_encoding_path}")

    # 6. File with a non-printable control character (should fail 'control_character')
    control_char_path = logs_path / "control_chars.log"
    with open(control_char_path, "w", encoding="utf-8") as f:
        f.write("Line with a bell character: \x07\n") # \x07 is the ASCII BELL character
    print(f"Created (should fail - control character): {control_char_path}")

    # 7. Log file with links for replacement test
    link_log_path = logs_path / "user_comments.log"
    with open(link_log_path, "w", encoding="utf-8") as f:
        f.write("User 101 reported an issue. See ticket at https://jira.company.com/browse/PROJ-123\n")
        f.write("No issues for user 102.\n")
        f.write("User 103 provided feedback via http://feedback.com and also mentioned www.google.com\n")
    print(f"Created (should warn and replace links): {link_log_path}")
    print("--- Test setup complete ---")
    return config_path


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    
    # Define the test data directory to be created inside the script's directory.
    test_dir = script_dir / "landing"
    # 1. Set up the test environment
    config_path = setup_test_directory(test_dir)

    # 2. Load the configuration from the YAML file
    print("\n--- Loading configuration ---")
    config = [
        {
            "name": "Validate User CSVs",
            "path_pattern": "**/*.csv",
            "validator": "tabular",
        },
        {
            "name": "Validate Transaction Parquet Files",
            "path_pattern": "**/*.parquet",
            "validator": "tabular",
        },
        {
            "name": "Validate for Image files",
            "path_pattern": "**/*.jpg",
            "validator": "image",
        },
        {
            "name": "Validate for Image files(png)",
            "path_pattern": "**/*.png",
            "validator": "image",
        },
        {
            "name": "Validate for JSON files",
            "path_pattern": "**/*.json",
            "validator": "json",  # Assuming a JSON validator exists
        },
        {
            "name": "Validate for Audio files",
            "path_pattern": "**/*.wav",
            "validator": "audio",  # Assuming an Audio validator exists
        },
        {"name": "Validate for Audio files(mp3)",
         "path_pattern": "**/*.mp3",
            "validator": "audio"  # Assuming an Audio validator exists
        },
        {"name": "Validate for Text files",
         "path_pattern": "**/*.txt",
            "validator": "text"  # Assuming a Text validator exists
        },
        {
            "name": "Validate Log Files",
            "path_pattern": "**/*.log",
            "validator": "text",
        }
        ]
       

    print(config)
    print("Configuration loaded successfully.")

    # 3. Initialize the DirectoryValidator with the config
    validator = DirectoryValidator(config=config)

    # 4. Run the validation on the root directory
    print("\n--- Running Validation ---")
    report = validator.validate(Path(test_dir))

    # 5. Display the results in a user-friendly format
    print("\n--- Validation Report ---")
    print(report)

    print("-" * 25)

    for result in report.results:
        status = "✅ PASSED" if result.is_valid else "❌ FAILED"
        duration_info = f"({result.duration_seconds:.4f}s)" if result.duration_seconds is not None else ""
        print(f"File: {result.file_path} {duration_info}")
        print(f"Status: {status}")
        
        if not result.is_valid:
            print("Errors:")
            for error in result.errors:
                print(error)
        for warning in result.warnings:
            print(f"   warning: {warning}")

        print("-" * 25)
    
    

if __name__ == "__main__":
    main()