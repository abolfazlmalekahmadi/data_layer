from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import polars as pl

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
            errors.extend(self._check_required_columns(lazy_df.columns))

            exprs = self._build_validation_expressions(lazy_df.columns)
            results = self._execute_checks(lazy_df, exprs)

            errors.extend(self._process_errors(results))
            warnings.extend(self._process_warnings(results))

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
            return pl.scan_csv(file_path,infer_schema_length=0)
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

    def _build_validation_expressions(
        self, columns: List[str]
    ) -> Tuple[List[pl.Expr], List[pl.Expr], List[pl.Expr]]:
        """Generate Polars expressions for null, link, and fully-null checks."""
        not_null_exprs, link_exprs, null_col_exprs = [], [], []
        column_rules = self.rules.get("columns", {})
        link_pattern = r"(?:https?://|ftp://|www\.)"

        for col in columns:
            if column_rules.get(col, {}).get("not_null"):
                not_null_exprs.append(pl.col(col).is_null().sum().alias(f"not_null_{col}"))

            link_exprs.append(
                pl.col(col)
                .cast(pl.Utf8)
                .str.contains(link_pattern)
                .any()
                .alias(f"link_{col}")
            )

            null_col_exprs.append(pl.col(col).is_null().all().alias(f"fully_null_{col}"))

        return not_null_exprs, link_exprs, null_col_exprs

    def _execute_checks(
        self, lazy_df: pl.LazyFrame, exprs: Tuple[List[pl.Expr], List[pl.Expr], List[pl.Expr]]
    ) -> pl.DataFrame:
        """Run all checks in a single Polars query."""
        not_null_exprs, link_exprs, null_col_exprs = exprs
        return lazy_df.select(
            *not_null_exprs,
            *link_exprs,
            *null_col_exprs,
            pl.all_horizontal(pl.all().is_null()).alias("fully_null_row_count").sum(),
        ).collect()

    def _process_errors(self, results: pl.DataFrame) -> List[Dict]:
        """Extract error messages from Polars results."""
        errors = []
        row = results.row(0, named=True)

        for name, val in row.items():
            if name.startswith("not_null_") and val > 0:
                col = name.replace("not_null_", "")
                errors.append({"check": "not_null", "column": col, "error": f"{val} nulls"})
            elif name.startswith("link_") and val:
                col = name.replace("link_", "")
                errors.append({"check": "contains_links", "column": col, "error": "Link found"})

        return errors

    def _process_warnings(self, results: pl.DataFrame) -> List[Dict]:
        """Extract warning messages from Polars results."""
        warnings = []
        row = results.row(0, named=True)

        for name, val in row.items():
            if name.startswith("fully_null_") and val:
                col = name.replace("fully_null_", "")
                warnings.append({"check": "fully_null_column", "column": col, "warning": "All nulls"})

        if row.get("fully_null_row_count", 0) > 0:
            warnings.append({
                "check": "fully_null_row",
                "warning": f"{row['fully_null_row_count']} fully null row(s)",
            })

        return warnings


# ---------------- Image Validator ---------------- #

class ImageValidator(BaseValidator):
    """Validates image files by attempting to load them and checking basic quality."""

    @time_validator
    def validate(self, file_path: str) -> ValidationResult:
        errors: List[Dict] = []

        try:
            image = cv2.imread(file_path)
            if image is None:
                errors.append({
                    "check": "image_loadable",
                    "error": "File not loadable (may be corrupt or unsupported).",
                })
            else:
                errors.extend(self._check_solid_color(image))

        except Exception as e:
            errors.append({"check": "file_read_exception", "error": str(e)})

        return ValidationResult(
            file_path,
            is_valid=not errors,
            validator_used="image",
            errors=errors,
        )

    def _check_solid_color(self, image: np.ndarray) -> List[Dict]:
        """Check if the image is a single solid color, if rule is enabled."""
        if not self.rules.get("disallow_solid_color", True):
            return []

        min_val, max_val = np.min(image), np.max(image)
        if min_val != max_val:
            return []

        if min_val == 0:
            msg = "Image is completely black."
        elif min_val == 255:
            msg = "Image is completely white."
        else:
            msg = f"Image is solid color (value {min_val})."

        return [{"check": "solid_color", "error": msg}]


# ---------------- JSON Validator ---------------- #

class JsonValidator(BaseValidator):
    """Validates JSON files, typically against a schema."""

    def validate(self, file_path: str) -> ValidationResult:
        errors: List[Dict] = []

        try:
            import json
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # TODO: Add JSON schema validation if `self.rules` contains a schema

        except Exception as e:
            errors.append({"check": "file_read_or_parse", "error": str(e)})

        return ValidationResult(
            file_path,
            is_valid=not errors,
            validator_used="json",
            errors=errors,
        )
