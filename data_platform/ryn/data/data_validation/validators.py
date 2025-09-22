
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from structures import ValidationResult
import pandas as pd
from pathlib import Path
import cv2
import re
import numpy as np 
from collections import defaultdict
import pyarrow.parquet as pq
from utils import time_validator

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


# Add polars to your imports
import polars as pl
from typing import Dict, Any, List

# ... (other imports and classes are the same) ...

class TabularValidator(BaseValidator):

    def __init__(self, rules: Dict[str, Any] | None = None):
        """Initializes with rules."""
        super().__init__(rules)
        # We no longer need to manage chunksize here

    @time_validator
    def validate(self, file_path: str) -> ValidationResult:
        """
        Executes validation logic using Polars for high performance.
        """
        errors = []
        warnings = []

        try:
            # Step 1: Lazily scan the file (no data is loaded yet)
            lazy_df = self._scan_data(file_path)
            
            # Step 2: Run checks that only need the schema (column names)
            errors.extend(self._check_required_columns(lazy_df.columns))

            # Step 3: Define all other checks as Polars expressions
            (
                not_null_exprs,
                link_check_exprs,
                fully_null_col_exprs,
            ) = self._build_validation_expressions(lazy_df.columns)

            # Step 4: Execute all checks in a single pass over the data
            results = lazy_df.select(
                *not_null_exprs,
                *link_check_exprs,
                *fully_null_col_exprs,
                # Expression to count fully null rows
                pl.all_horizontal(pl.all().is_null()).alias("fully_null_row_count").sum()
            ).collect() # <-- This is where the work happens

            # Step 5: Process the results of the query
            errors.extend(self._process_polars_results(results))
            warnings.extend(self._process_polars_warnings(results))

        except Exception as e:
            errors.append({"check": "file_read_process", "error": f"Failed to read or process file: {e}"})
            return ValidationResult(file_path, False, "tabular", errors=errors)

        is_valid = len(errors) == 0
        return ValidationResult(file_path, is_valid, "tabular", errors, warnings)

    def _scan_data(self, file_path: str) -> pl.LazyFrame:
        """Lazily scans a file into a Polars LazyFrame."""
        file_extension = Path(file_path).suffix.lower()
        if file_extension == '.csv':
            return pl.scan_csv(file_path)
        elif file_extension == '.parquet':
            return pl.scan_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type for Polars: '{file_extension}'")
            
    def _check_required_columns(self, actual_cols: List[str]) -> List[Dict]:
        """Checks for required columns using the schema."""
        errors = []
        required_cols = self.rules.get("dataset", {}).get("required_columns", [])
        if not required_cols:
            return errors

        missing_cols = set(required_cols) - set(actual_cols)
        if missing_cols:
            errors.append({
                "check": "required_columns",
                "error": f"Missing required columns: {sorted(list(missing_cols))}",
            })
        return errors

    def _build_validation_expressions(self, columns: List[str]) -> tuple:
        """Creates lists of Polars expressions for all our checks."""
        not_null_exprs = []
        link_check_exprs = []
        fully_null_col_exprs = []
        
        column_rules = self.rules.get("columns", {})
        link_pattern = r'(?:https?://|ftp://|www\.)'

        for col_name in columns:
            # Expression for not-null checks
            if column_rules.get(col_name, {}).get("not_null"):
                expr = pl.col(col_name).is_null().sum().alias(f"not_null_{col_name}")
                not_null_exprs.append(expr)
            
            # Expression for link checks (on string columns)
            # Note: Polars needs explicit casting for some operations
            expr_link = pl.col(col_name).cast(pl.Utf8).str.contains(link_pattern).any().alias(f"link_{col_name}")
            link_check_exprs.append(expr_link)

            # Expression to see if a column is entirely null
            expr_null_col = pl.col(col_name).is_null().all().alias(f"fully_null_{col_name}")
            fully_null_col_exprs.append(expr_null_col)
            
        return not_null_exprs, link_check_exprs, fully_null_col_exprs

    def _process_polars_results(self, results_df: pl.DataFrame) -> List[Dict]:
        """Processes the collected Polars results to generate error messages."""
        errors = []
        row = results_df.row(0, named=True) # Get the single result row as a dict
        for col_name, value in row.items():
            if col_name.startswith("not_null_") and value > 0:
                original_col = col_name.replace("not_null_", "")
                errors.append({"check": "not_null", "column": original_col, "error": f"Found {value} null values."})
            elif col_name.startswith("link_") and value is True:
                original_col = col_name.replace("link_", "")
                errors.append({"check": "contains_links", "column": original_col, "error": "Disallowed link found in column."})
        return errors

    def _process_polars_warnings(self, results_df: pl.DataFrame) -> List[Dict]:
        """Processes the collected Polars results to generate warning messages."""
        warnings = []
        row = results_df.row(0, named=True)
        for col_name, value in row.items():
            if col_name.startswith("fully_null_") and value is True:
                original_col = col_name.replace("fully_null_", "")
                warnings.append({"check": "fully_null_column", "column": original_col, "warning": f"Column '{original_col}' is entirely null."})

        if row.get("fully_null_row_count", 0) > 0:
            warnings.append({"check": "fully_null_row", "warning": f"Found {row['fully_null_row_count']} row(s) that are entirely null."})
        return warnings

class ImageValidator(BaseValidator):
    """
    Validates image files by attempting to load them.
    """
    @time_validator
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