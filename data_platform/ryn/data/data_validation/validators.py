
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from structures import ValidationResult
import pandas as pd
from pathlib import Path


class BaseValidator(ABC):
    """
    Abstract Base Class (Interface) for all specialized validators.

    This class defines the "contract" that any validator must follow.
    It guarantees that every validator will have an `__init__` method that
    accepts rules and a `validate` method that takes a file path and
    returns a ValidationResult.
    """

    def __init__(self, rules: Dict[str, Any] | None = None):
        """
        Initializes the validator with a specific set of rules.

        Args:
            rules (Dict[str, Any] | None): A dictionary defining the validation rules.
                                          Can be None if the validator is rule-free.
        """
        self.rules = rules or {}

    @abstractmethod
    def validate(self, file_path: str) -> ValidationResult:
        """
        The core validation method that all subclasses must implement.

        This method should perform the actual validation logic for a given file
        and return a standardized result object.

        Args:
            file_path (str): The path to the file to be validated.

        Returns:
            ValidationResult: An object containing the outcome of the validation.
        """
        pass



class TabularValidator(BaseValidator):
    """
    Validates structured data files (e.g., CSV, Parquet).
    
    It checks for things like required columns, data types, nulls, etc.
    based on the rules provided during initialization.
    """
    def validate(self, file_path: str) -> ValidationResult:
        """
        Executes the validation logic for a tabular data file.

        1. Attempts to load the data into a pandas DataFrame.
        2. Runs dataset-level checks (e.g., required columns).
        3. Runs column-level checks (e.g., not null).
        4. Collects all errors and returns a ValidationResult.
        """
        errors = []
        
        # 1. Load data
        try:
            df = self._load_data(file_path)
        except Exception as e:
            # If loading fails, we can't continue. Return a failure result immediately.
            errors.append({"check": "file_read", "error": f"Failed to read file: {e}"})
            return ValidationResult(
                file_path=file_path,
                is_valid=False,
                validator_used="tabular",
                errors=errors
            )
            
        # 2. Run checks and collect errors
        errors.extend(self._check_required_columns(df))
        errors.extend(self._check_not_null(df))
        # ... more checks will be added here later

        # 3. Create and return the final result
        is_valid = len(errors) == 0
        return ValidationResult(
            file_path=file_path,
            is_valid=is_valid,
            validator_used="tabular",
            errors=errors
        )

    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Loads data from a file path into a pandas DataFrame based on extension."""
        file_extension = Path(file_path).suffix.lower()
        if file_extension == '.csv':
            return pd.read_csv(file_path)
        elif file_extension == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: '{file_extension}'")

    # --- Private Helper Methods for Specific Checks ---

    def _check_required_columns(self, df: pd.DataFrame) -> List[Dict]:
        """Checks if all required columns are present in the DataFrame."""
        errors = []
        required_cols = self.rules.get("dataset", {}).get("required_columns", [])
        if not required_cols:
            return errors  # No columns were required

        actual_cols = set(df.columns)
        missing_cols = set(required_cols) - actual_cols
        
        if missing_cols:
            errors.append({
                "check": "required_columns",
                "error": f"Missing required columns: {sorted(list(missing_cols))}"
            })
        return errors

    def _check_not_null(self, df: pd.DataFrame) -> List[Dict]:
        """Checks for null values in columns that are specified as 'not_null'."""
        errors = []
        column_rules = self.rules.get("columns", {})
        if not column_rules:
            return errors

        for col_name, rules in column_rules.items():
            if rules.get("not_null") and col_name in df.columns:
                null_count = df[col_name].isnull().sum()
                if null_count > 0:
                    errors.append({
                        "check": "not_null",
                        "column": col_name,
                        "error": f"Found {null_count} null values."
                    })
        return errors


class ImageValidator(BaseValidator):
    """
    Validates image files.
    """
    def validate(self, file_path: str) -> ValidationResult:
        """
        1. Opens the image file (e.g., using Pillow or OpenCV).
        2. Applies rules for dimensions, file size, format, aspect ratio, etc.
        3. Returns the result.
        """
        pass
    
    # Internal helpers like _check_dimensions, _check_format, etc.

class JsonValidator(BaseValidator):
    """
    Validates JSON files, typically against a JSON Schema.
    """
    def validate(self, file_path: str) -> ValidationResult:
        """
        1. Loads the JSON file.
        2. Loads the reference JSON Schema (often specified in the rules).
        3. Validates the data against the schema.
        4. Returns the result.
        """
        pass