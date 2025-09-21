import os
import yaml
import pandas as pd
from pathlib import Path

# Import the main orchestrator class
from validation import DirectoryValidator

def setup_test_directory(root_dir: Path):
    """Creates a sample directory with data files and a config for testing."""
    print(f"--- Setting up test directory at '{root_dir}' ---")

    # Define paths
    users_path = root_dir / "landing/users"
    transactions_path = root_dir / "processed/transactions/day=1"
    
    # Create directories
    users_path.mkdir(parents=True, exist_ok=True)
    transactions_path.mkdir(parents=True, exist_ok=True)

    # 1. Create a GOOD CSV file that should pass validation
    good_data = {
        "user_id": [1, 2, 3],
        "email": ["one@test.com", "two@test.com", "three@test.com"],
        "age": [25, 30, 35]
    }
    pd.DataFrame(good_data).to_csv(users_path / "users_good.csv", index=False)
    print(f"Created: {users_path / 'users_good.csv'}")

    # 2. Create a BAD CSV file that should fail validation
    #    - 'user_id' contains a null value (violates not_null)
    #    - 'email' column is missing (violates required_columns)
    bad_data = {
        "user_id": [4, None, 6],
        "age": [40, 45, 50]
    }
    pd.DataFrame(bad_data).to_csv(users_path / "users_bad.csv", index=False)
    print(f"Created: {users_path / 'users_bad.csv'}")

    # 3. Create a valid Parquet file to test a different file type and pattern
    transaction_data = {
        "transaction_id": ["txn_101", "txn_102"],
        "user_id": [1, 3],
        "amount": [99.99, 12.50]
    }
    pd.DataFrame(transaction_data).to_parquet(transactions_path / "transactions.parquet")
    print(f"Created: {transactions_path / 'transactions.parquet'}")

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
    config_path = root_dir / "validation_config.yaml"
    # with open(config_path, "w") as f:
    #     f.write(config_content)
    # print(f"Created: {config_path}")
    print("--- Test setup complete ---")
    return config_path


def main():
    """Main execution function."""
    test_dir = Path("/home/coder/persistent/datalayer/tools/data_platform/ryn/data/data_validation/landing")
    
    # 1. Set up the test environment
    config_path = setup_test_directory(test_dir)

    # 2. Load the configuration from the YAML file
    print("\n--- Loading configuration ---")
    with open("/home/coder/persistent/datalayer/tools/data_platform/ryn/data/data_validation/validation_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully.")

    # 3. Initialize the DirectoryValidator with the config
    validator = DirectoryValidator(config=config)

    # 4. Run the validation on the root directory
    print("\n--- Running Validation ---")
    report = validator.validate(str(test_dir))

    # 5. Display the results in a user-friendly format
    print("\n--- Validation Report ---")
    print(f"Directory: {report.directory_path}")
    print(f"Overall Status: {report.overall_status}")
    print("-" * 25)

    for result in report.results:
        status = "✅ PASSED" if result.is_valid else "❌ FAILED"
        print(f"File: {result.file_path}")
        print(f"Status: {status}")
        if not result.is_valid:
            print("Errors:")
            for error in result.errors:
                print(f"  - Check: {error.get('check')}")
                if 'column' in error:
                    print(f"    Column: {error.get('column')}")
                print(f"    Details: {error.get('error')}")
        print("-" * 25)


if __name__ == "__main__":
    main()