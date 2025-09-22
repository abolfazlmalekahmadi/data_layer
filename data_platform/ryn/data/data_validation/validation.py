
from typing import Dict, Any, List

import pandas as pd

from validators import TabularValidator,ImageValidator,JsonValidator,BaseValidator
from dataclasses import dataclass, field
from typing import List, Dict, Any

import glob
from pathlib import Path
from typing import Dict, Any, List

from structures import ValidationReport


class DirectoryValidator:
    """
    Orchestrates the validation of an entire directory containing mixed file types.
    """
    def __init__(self, config: List[Dict[str, Any]]):
        """
        Initializes the orchestrator with a validation configuration.

        Args:
            config (List[Dict[str, Any]]): A list of validation tasks, typically
                                          loaded from a YAML or JSON file.
        """
        self.config = config
        self.validator_registry: Dict[str, BaseValidator] = {}
        
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
        Executes the full validation plan for the given directory.

        1. Iterates through each task in the configuration.
        2. Finds all files matching the task's `path_pattern`.
        3. Instantiates the appropriate validator with the task's rules.
        4. Runs validation for each file and collects the results.
        """
        print(f"Starting validation for directory: '{directory_path}'...")
        report = ValidationReport(directory_path=directory_path)
        root_path = Path(directory_path)

        for task in self.config:
            path_pattern = task.get("path_pattern")
            validator_name = task.get("validator")
            rules = task.get("rules", {})  # Safely get rules, default to empty

            if not path_pattern or not validator_name:
                print(f"WARNING: Skipping invalid task in config: {task}")
                continue

            validator_class = self.validator_registry.get(validator_name)
            if not validator_class:
                print(f"WARNING: Validator '{validator_name}' not found in registry. Skipping task.")
                continue
            
            # Construct the full search pattern and find matching files
            full_search_path = str(root_path / path_pattern)
            matched_files = glob.glob(full_search_path, recursive=True)

            if not matched_files:
                print(f"INFO: No files found for pattern: '{full_search_path}'")
                continue

            print(f"Found {len(matched_files)} file(s) for task '{task.get('name', 'Unnamed Task')}'")
            for file_path in matched_files:
                # For each file, create a new validator instance with the correct rules
                validator_instance = validator_class(rules=rules)
                result = validator_instance.validate(file_path)
                report.add_result(result)
        
        print(f"Validation complete. Report status: {report.overall_status}")
        return report

    def __call__(self, directory_path: str) -> ValidationReport:
        """Makes the instance callable as a shortcut for the `validate` method."""
        return self.validate(directory_path)