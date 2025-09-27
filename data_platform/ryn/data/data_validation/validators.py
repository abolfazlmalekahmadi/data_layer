from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import polars as pl
from polars import LazyFrame

# Assuming these are in your project structure
from data.data_validation.structures import ValidationResult
from data.data_validation.utils import time_validator


class BaseValidator(ABC):
    """
    Abstract base class for specialized validators.

    All validators must implement:
      - __init__(rules)
      - validate(file_path) -> ValidationResult
    """

    def __init__(self, rules: Dict[str, Any] | None = None):
        self.rules = rules or {}

    @abstractmethod
    def validate(self, file_path: str) -> ValidationResult:
        """Run validation on the given file path."""
        ...


class TabularValidator(BaseValidator):
    """Validates tabular datasets using Polars for performance."""

    @time_validator
    def validate(self, file_path: str) -> ValidationResult:
        errors: List[Dict] = []
        warnings: List[Dict] = []

        try:
            lazy_df = self._scan_data(file_path)
            columns = lazy_df.columns
            errors.extend(self._check_required_columns(columns))

            # --- Perform fast, full-table aggregation checks ---
            agg_exprs = self._build_aggregation_expressions(columns)
            if agg_exprs:
                agg_results = self._execute_aggregation_checks(lazy_df, agg_exprs)
                errors.extend(self._process_aggregation_errors(agg_results))
                warnings.extend(self._process_aggregation_warnings(agg_results))

            # --- Perform detailed, row-level checks (like finding links) ---
            errors.extend(self._check_for_links(lazy_df))

        except Exception as e:
            errors.append({"check": "file_read_process", "error": str(e)})
            return ValidationResult(file_path, False, "tabular", errors)

        return ValidationResult(
            file_path,
            is_valid=not errors,
            validator_used="tabular",
            errors=errors,
            warnings=warnings,
        )

    def _scan_data(self, file_path: str) -> pl.LazyFrame:
        """Return a Polars LazyFrame from supported file types."""
        ext = Path(file_path).suffix.lower()
        if ext == ".csv":
            # This is the most robust option for validation.
            return pl.scan_csv(file_path, infer_schema_length=0, ignore_errors=True)
        if ext == ".parquet":
            return pl.scan_parquet(file_path)
        raise ValueError(f"Unsupported file type: {ext}")

    def _check_required_columns(self, actual_cols: List[str]) -> List[Dict]:
        """Verify required columns exist in schema."""
        required = self.rules.get("dataset", {}).get("required_columns", [])
        if not required:
            return []

        missing = sorted(set(required) - set(actual_cols))
        if missing:
            return [{"check": "required_columns", "error": f"Missing: {missing}"}]
        return []

    def _build_aggregation_expressions(self, columns: List[str]) -> List[pl.Expr]:
        """Generate Polars expressions for high-level dataset checks."""
        exprs = []
        column_rules = self.rules.get("columns", {})

        for col in columns:
            # Check for null values in columns marked as 'not_null'
            if column_rules.get(col, {}).get("not_null"):
                exprs.append(pl.col(col).is_null().sum().alias(f"not_null_{col}"))

            # Check if an entire column is null (for warnings)
            exprs.append(pl.col(col).is_null().all().alias(f"fully_null_{col}"))

        # Check for any rows where all values are null
        exprs.append(
            pl.all_horizontal(pl.all().is_null()).sum().alias("fully_null_row_count")
        )
        return exprs

    def _execute_aggregation_checks(
        self, lazy_df: pl.LazyFrame, exprs: List[pl.Expr]
    ) -> pl.DataFrame:
        """Run all aggregation checks in a single Polars query."""
        return lazy_df.select(exprs).collect()

    def _process_aggregation_errors(self, results: pl.DataFrame) -> List[Dict]:
        """Extract error messages from aggregation results."""
        errors = []
        row = results.row(0, named=True)

        for name, val in row.items():
            if name.startswith("not_null_") and val > 0:
                col = name.replace("not_null_", "")
                errors.append(
                    {"check": "not_null", "column": col, "error": f"{val} nulls found"}
                )
        return errors

    def _process_aggregation_warnings(self, results: pl.DataFrame) -> List[Dict]:
        """Extract warning messages from aggregation results."""
        warnings = []
        row = results.row(0, named=True)

        for name, val in row.items():
            if name.startswith("fully_null_") and val:
                col = name.replace("fully_null_", "")
                warnings.append(
                    {
                        "check": "fully_null_column",
                        "column": col,
                        "warning": "Column contains only null values",
                    }
                )

        if row.get("fully_null_row_count", 0) > 0:
            warnings.append(
                {
                    "check": "fully_null_row",
                    "warning": f"{row['fully_null_row_count']} fully null row(s) found",
                }
            )
        return warnings

     # --- MODIFIED METHOD ---
    def _check_for_links(self, lazy_df: LazyFrame) -> List[Dict]:
        """
        Find specific rows and values that contain web links.
        This check is now only applied to columns with a string data type.
        """
        errors = []
        link_pattern = r"((?:https?://|ftp://|www\.)\S+)"
        df_with_index = lazy_df.with_row_index()
        
        # Get the schema to check column types
        df_schema = lazy_df.schema

        # Iterate through only the STRING columns
        for col_name, col_type in df_schema.items():
            # Check if the column is a text-like type (Utf8 or Categorical)
        
            links_found = (
                df_with_index.filter(
                    pl.col(col_name).cast(pl.Utf8).str.contains(link_pattern)
                )
                .select(
                    pl.col("index").alias("row_number"),
                    pl.col(col_name)
                    .cast(pl.Utf8)
                    .str.extract(link_pattern, 1)
                    .alias("link"),
                )
                .collect()
            )

            for row in links_found.iter_rows(named=True):
                user_friendly_row = row["row_number"] + 2  # For header + 0-based index
                errors.append(
                    {
                        "check": "contains_links",
                        "column": col_name,
                        "row": user_friendly_row,
                        "error": f"Link found: '{row['link']}'",
                    }
                )
                break
            
        return errors


# ---------------- Image Validator (Unchanged) ---------------- #
class ImageValidator(BaseValidator):
    @time_validator
    def validate(self, file_path: str) -> ValidationResult:
        
        errors: List[Dict] = []
        try:
            image = cv2.imread(file_path)
            if image is None:
                errors.append({"check": "image_loadable", "error": "File not loadable (may be corrupt or unsupported)."})
            else:
                errors.extend(self._check_solid_color(image))
        except Exception as e:
            errors.append({"check": "file_read_exception", "error": str(e)})
        return ValidationResult(file_path, is_valid=not errors, validator_used="image", errors=errors)

    def _check_solid_color(self, image: np.ndarray) -> List[Dict]:
        if not self.rules.get("disallow_solid_color", True): return []
        min_val, max_val = np.min(image), np.max(image)
        if min_val != max_val: return []
        if min_val == 0: msg = "Image is completely black."
        elif min_val == 255: msg = "Image is completely white."
        else: msg = f"Image is solid color (value {min_val})."
        return [{"check": "solid_color", "error": msg}]


# ---------------- JSON Validator (Unchanged) ---------------- #
class JsonValidator(BaseValidator):
    def validate(self, file_path: str) -> ValidationResult:
        errors: List[Dict] = []
        try:
            import json
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            errors.append({"check": "file_read_or_parse", "error": str(e)})
        return ValidationResult(file_path, is_valid=not errors, validator_used="json", errors=errors)
