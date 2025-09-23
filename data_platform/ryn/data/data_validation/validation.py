
from typing import Dict, Any, List

import pandas as pd
import os
from validators import TabularValidator,ImageValidator,JsonValidator,BaseValidator
from dataclasses import dataclass, field
from typing import List, Dict, Any

import glob
from pathlib import Path
from typing import Dict, Any, List
import zipfile
import tempfile

from structures import ValidationReport


class DirectoryValidator:
    """
    Orchestrates the validation of an entire directory containing mixed file types.
    """
    def __init__(self, config: List[Dict[str, Any]], unzip_first: bool = True):
        """
        Initializes the orchestrator with a validation configuration.

        Args:
            config (List[Dict[str, Any]]): A list of validation tasks, typically
                                          loaded from a YAML or JSON file.
        """
        self.config = config
        self.validator_registry: Dict[str, BaseValidator] = {}
        self.unzip_first = unzip_first
        # This crucial step maps string names from the config (e.g., "tabular")
        # to the actual validator classes.
        self._register_validators()

    def _register_validators(self):
        """
        Initializes the validator registry. This makes the system extensible.
        To add a new validator, you just add a line here.
        """
        self.validator_registry = {
            "tabular": TabularValidator,
            "image": ImageValidator, # Example for the future
            # "json": JsonValidator,   # Example for the future
        }
        print(f"Registered validators: {list(self.validator_registry.keys())}")

    def validate(self, directory_path: str) -> ValidationReport:
        """
        Top-level validation method. Manages the unzipping pre-processing step.
        """
        files_to_cleanup = []
        if self.unzip_first:
            # Step 1: Unzip files and get a list of the files that were created.
            files_to_cleanup = self._unzip_all(source_dir=directory_path)
        
        # Step 2: Run the validation process on the directory, which now contains the extracted files.
        report = self._perform_validation_on_dir(directory_path)
        
        # Step 3: After validation is complete, clean up the files that were extracted.
        if files_to_cleanup:
            self._cleanup_files(files_to_cleanup)

        return report
    def _unzip_all(self, source_dir: str) -> List[str]:
        """
        Finds and extracts all .zip files into the same directory.
        
        Returns:
            List[str]: A list of the absolute paths of all files that were extracted.
        """
        zip_files = glob.glob(str(Path(source_dir) / "**/*.zip"), recursive=True)
        print(f"Zip files found for extraction: {len(zip_files)}")
        if not zip_files:
            print("No .zip files found to extract.")
            return []

        extracted_file_paths = []
        for zip_path in zip_files:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    print(f"  - Extracting '{zip_path}'...")
                    # Get list of members to track them for cleanup
                    for member in zip_ref.infolist():
                        if not member.is_dir():
                            # Construct the full path where the file will be
                            full_path = Path(source_dir) / member.filename
                            extracted_file_paths.append(str(full_path))
                    
                    # Extract all members
                    zip_ref.extractall(source_dir)
            except zipfile.BadZipFile:
                print(f"  - WARNING: Skipping corrupt or invalid zip file: '{zip_path}'")
            except Exception as e:
                print(f"  - WARNING: Failed to extract '{zip_path}': {e}")
        
        return extracted_file_paths
    def _cleanup_files(self, file_paths: List[str]):
        """
        Deletes a list of files, typically the ones extracted from an archive.
        """
        print("\n--- Cleaning up extracted files ---")
        if not file_paths:
            print("No files to clean up.")
            return

        for file_path in file_paths:
            try:
                os.remove(file_path)
                print(f"  - Deleted: {file_path}")
            except FileNotFoundError:
                print(f"  - WARNING: Could not delete '{file_path}', as it was not found.")
            except Exception as e:
                print(f"  - ERROR: Failed to delete '{file_path}': {e}")
        print("--- Cleanup complete ---")


    def _perform_validation_on_dir(self, directory_to_validate: str) -> ValidationReport:
        """
        Contains the core validation loop that runs on a specified directory.
        (This is the refactored logic from the original `validate` method).
        """

        report = ValidationReport(directory_path=directory_to_validate)
        print(f"Starting validation in directory: {directory_to_validate}")

        for task in self.config:
            path_pattern = task.get("path_pattern")
            validator_name = task.get("validator")
            rules = task.get("rules", {})
            task_name = task.get("name", "Unnamed Task")

            if not path_pattern or not validator_name:
                print(f"Skipping invalid task configuration: {task}")
                continue

            validator_class = self.validator_registry.get(validator_name)
            if not validator_class:
                print(f"Unknown validator '{validator_name}' in task '{task_name}'. Skipping...")
                continue

            # print(f"\n--- Running Task: {task_name} ---")
            # print(f"Using validator: {validator_name}")
            # print(f"Path pattern: {path_pattern}")

            # Resolve the full glob pattern
            full_pattern = str(Path(directory_to_validate) / path_pattern)
            matched_files = glob.glob(full_pattern, recursive=True)

            if not matched_files:
                print(f"No files matched the pattern '{path_pattern}' in '{directory_to_validate}'.")
                continue

            # print(f"Found {len(matched_files)} files to validate.")

            # Initialize the validator instance with the specified rules
            validator_instance = validator_class(rules=rules)

            for file_path in matched_files:
                # print(f"Validating file: {file_path}")
                try:
                    result = validator_instance.validate(file_path)
                    report.add_result(result)
                except Exception as e:
                    print(f"  - ERROR: Validation failed for '{file_path}': {e}")


        
        return report

    def __call__(self, directory_path: str) -> ValidationReport:
        """Makes the instance callable as a shortcut for the `validate` method."""
        return self.validate(directory_path)