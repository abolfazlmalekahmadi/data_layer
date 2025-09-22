
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from structures import ValidationResult
import pandas as pd
from pathlib import Path
import cv2
import re
import numpy as np 


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
        warnings = []
        
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
                errors=errors,
                warnings=warnings
            )
            
        # 2. Run checks and collect errors
        errors.extend(self._check_for_links(df))
        errors.extend(self._check_required_columns(df))
        errors.extend(self._check_not_null(df))
        # ... more checks will be added here later


        warnings.extend(self._check_fully_null_entities(df))

        # 3. Create and return the final result
        is_valid = len(errors) == 0
        return ValidationResult(
            file_path=file_path,
            is_valid=is_valid,
            validator_used="tabular",
            errors=errors,
            warnings=warnings
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
    def _check_fully_null_entities(self, df: pd.DataFrame) -> List[Dict]:
        """
        Checks for and creates warnings for fully null columns or rows.
        This does not cause the validation to fail.
        """
        warnings = []
        
        # 1. Check for fully null columns
        null_columns = df.columns[df.isnull().all()]
        for col_name in null_columns:
            warnings.append({
                "check": "fully_null_column",
                "column": col_name,
                "warning": f"Column '{col_name}' is entirely null."
            })

        # 2. Check for fully null rows
        null_row_count = df.isnull().all(axis=1).sum()
        if null_row_count > 0:
            warnings.append({
                "check": "fully_null_row",
                "warning": f"Found {null_row_count} row(s) that are entirely null."
            })
            
        return warnings
    def _check_for_links(self, df: pd.DataFrame) -> List[Dict]:
        """
        Scans all string-like columns for URLs. This is a safety check.
        Any detected link will be reported as an error.
        """
        errors = []
        # This regex looks for common URL prefixes.
        link_pattern = re.compile(r'(https?://|ftp://|www\.)', re.IGNORECASE)
        
        # Select only columns that could contain strings ('object' dtype)
        string_columns = df.select_dtypes(include=['object']).columns

        for col_name in string_columns:
            # Ensure we're working with strings and handle nulls gracefully
            # na=False treats NaN/None values as not containing a link.
            matches = df[col_name].astype(str).str.contains(link_pattern, na=False, regex=True)
            
            if matches.any():
                # Find the first offending value to include as an example in the error
                first_offender = df[col_name][matches].iloc[0]
                errors.append({
                    "check": "contains_links",
                    "column": col_name,
                    "error": f"Disallowed link found in column.",
                    "example": str(first_offender)[:100] # Show up to 100 chars
                })
        return errors

class ImageValidator(BaseValidator):
    """
    Validates image files by attempting to load them.
    """
    def validate(self, file_path: str) -> ValidationResult:
        """
        Applies a simple 'is loadable' check using OpenCV.
        
        Returns a valid result if cv2.imread() successfully loads the image,
        and an invalid result if it returns None or an exception occurs.
        """
        errors = []
        is_valid = False # Assume failure until proven otherwise

        try:
            # Attempt to read the image file
            image = cv2.imread(file_path)
            
            if image is not None:
                # If the image object is not None, the file is a valid, loadable image
                is_valid = True
            else:
                # cv2.imread returns None for corrupt, unsupported, or non-existent files
                errors.append({
                    "check": "image_loadable",
                    "error": "OpenCV could not read the image. It may be corrupt, have an invalid path, or be an unsupported format."
                })

        except Exception as e:
            # Catch any other unexpected OS-level or library errors
            errors.append({
                "check": "file_read_exception",
                "error": f"An unexpected error occurred while trying to read the image: {e}"
            })

        errors.extend(self._check_solid_color(image))

        is_valid = len(errors) == 0
        return ValidationResult(
            file_path=file_path,
            is_valid=is_valid,
            validator_used="image",
            errors=errors
            # warnings list is empty by default, which is fine
        )
    def _check_solid_color(self, image: np.ndarray) -> list:
        """
        Checks if the image is completely one color (e.g., all black or all white).
        This check only runs if 'disallow_solid_color' is true in the rules.
        """
        errors = []
        # Check if the rule is enabled in the config
        # if not self.rules.get("disallow_solid_color"):
        #     return errors # Rule not enabled, so we skip the check

        # Efficiently check if all pixel values are the same
        min_val = np.min(image)
        max_val = np.max(image)

        # print(min_val,max_val)
        if min_val == max_val:
            # All pixels have the same value, so it's a solid color
            if min_val == 0:
                errors.append({
                    "check": "solid_color",
                    "error": "Image is completely black."
                })
            elif min_val == 255:
                errors.append({
                    "check": "solid_color",
                    "error": "Image is completely white."
                })
            else:
                 errors.append({
                    "check": "solid_color",
                    "error": f"Image is a solid color with pixel value {min_val}."
                })
        return errors
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