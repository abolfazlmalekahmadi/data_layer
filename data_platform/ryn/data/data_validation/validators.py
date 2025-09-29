from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


import re
import cv2
import numpy as np
import polars as pl
from polars import LazyFrame

from data.data_validation.structures import ValidationResult
from data.data_validation.utils import time_validator

import librosa 


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
    def validate(self, file_path: Path) -> ValidationResult:
        """Run validation on the given file path."""
        ...


class TabularValidator(BaseValidator):
    """Validates tabular datasets using Polars for performance."""

    @time_validator
    def validate(self, file_path: Path) -> ValidationResult:
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

    def _scan_data(self, file_path: Path) -> pl.LazyFrame:
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

        # Iterate through columns
        for col_name, col_type in df_schema.items():
            

            if col_type not in [pl.Struct, pl.List, pl.Object]:
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
                    user_friendly_row = (
                        row["row_number"] + 2
                    )  # For header + 0-based index
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


class ImageValidator(BaseValidator):
    @time_validator
    def validate(self, file_path: Path) -> ValidationResult:
        errors: List[Dict] = []
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                errors.append(
                    {
                        "check": "image_loadable",
                        "error": "File not loadable (may be corrupt or unsupported).",
                    }
                )
            else:
                errors.extend(self._check_solid_color(image))
        except Exception as e:
            errors.append({"check": "file_read_exception", "error": str(e)})
        return ValidationResult(
            file_path, is_valid=not errors, validator_used="image", errors=errors
        )

    def _check_solid_color(self, image: np.ndarray) -> List[Dict]:
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



class JsonValidator(BaseValidator):
    def validate(self, file_path: Path) -> ValidationResult:
        errors: List[Dict] = []
        try:
            import json

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            errors.append({"check": "file_read_or_parse", "error": str(e)})
        return ValidationResult(
            file_path, is_valid=not errors, validator_used="json", errors=errors
        )


class AudioValidator(BaseValidator):
    """
    Validates audio files using librosa.

    Checks for:
    - Loadability: If the file can be read as audio.
    - Silence: If the audio is completely or nearly silent.
    - Minimum Duration: If the audio is shorter than a specified threshold.
    """

    @time_validator
    def validate(self, file_path: Path) -> ValidationResult:
        errors: List[Dict] = []
        try:
            # sr=None preserves the native sample rate
            audio_data, sample_rate = librosa.load(str(file_path), sr=None)
            
            # Perform checks on the loaded audio data
            errors.extend(self._check_silence(audio_data))
            errors.extend(self._check_duration(audio_data, sample_rate))

        except Exception as e:
            # This will catch errors from librosa if the file is corrupt/unsupported
            errors.append({"check": "audio_loadable", "error": str(e)})

        return ValidationResult(
            file_path, is_valid=not errors, validator_used="audio", errors=errors
        )

    def _check_silence(self, audio_data: np.ndarray) -> List[Dict]:
        """Check if the audio is effectively silent."""
        if not self.rules.get("disallow_silence", True):
            return []
        
        # A small threshold to account for floating point inaccuracies or faint noise
        silence_threshold = self.rules.get("silence_threshold", 1e-6)

        if np.max(np.abs(audio_data)) < silence_threshold:
            return [{"check": "silent_audio", "error": "Audio is effectively silent."}]
        return []

    def _check_duration(self, audio_data: np.ndarray, sample_rate: int | float) -> List[Dict]:
        """Check if the audio meets a minimum duration."""
        min_duration_sec = 0.5
        if not min_duration_sec:
            return []

        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        if duration < min_duration_sec:
            return [
                {
                    "check": "minimum_duration",
                    "error": f"Duration is {duration:.2f}s, less than required {min_duration_sec}s.",
                }
            ]
        return []
    


class TextValidator(BaseValidator):
    """
    Validates text-based files with built-in, sensible defaults.
    This version is designed to run without an external configuration.

    It checks for common issues like secrets, overly long lines, and bad encoding.
    """

    def __init__(self, rules: Dict[str, Any] | None = None):
        """Initializes the validator. The 'rules' are ignored in this implementation."""
        # We explicitly do not use the rules dictionary to rely on built-in logic.
        super().__init__({})
        
        # --- Define Hardcoded, "No-Config" Rules ---
        self.MAX_FILE_SIZE_MB = 2000  # Prevent validating huge binary files
        self.MAX_LINE_LENGTH = 10000 # Flag unnaturally long lines
        self.ENCODING = "utf-8"
        
        self.FORBIDDEN_PATTERNS = [
            re.compile(r"(?i)-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----"),
            re.compile(r"(?i)api_key\s*[:=]\s*['\"]?([a-zA-Z0-9_.-]{16,})['\"]?"),
            re.compile(r"(?i)secret_key\s*[:=]\s*['\"]?([a-zA-Z0-9_.-]{16,})['\"]?"),
            re.compile(r"(?i)password\s*[:=]\s*['\"]?(.{8,})['\"]?"),
            re.compile(r"sk_live_[0-9a-zA-Z]{24}"),  # Stripe Live API Key
            re.compile(r"AKIA[0-9A-Z]{16}"),       # AWS Access Key ID
            re.compile(r"xoxp-[0-9a-zA-Z-]{20,}"), # Slack Token
            re.compile(r"\b[a-fA-F0-9]{40,}\b")    # Generic long hex string (e.g., SHA1/256)
        ]

    @time_validator
    def validate(self, file_path: Path) -> ValidationResult:
        errors: List[Dict] = []
        warnings: List[Dict] = []

        # 1. File-level checks (size, emptiness)
        try:
            errors.extend(self._check_file_size(file_path))
            if errors:
                return ValidationResult(
                    file_path, is_valid=False, validator_used="text", errors=errors
                )
        except Exception as e:
            errors.append({"check": "file_metadata_read", "error": str(e)})
            return ValidationResult(
                file_path, is_valid=False, validator_used="text", errors=errors
            )

        # 2. Line-by-line checks for memory efficiency
        failed_checks = set()
        try:
            with open(file_path, "r", encoding=self.ENCODING) as f:
                for line_num, line in enumerate(f, 1):
                    # Check for long lines
                    if "line_length" not in failed_checks:
                        err = self._check_line_length(line, line_num)
                        if err:
                            errors.append(err)
                            failed_checks.add("line_length")
                    
                    # Check for forbidden patterns
                    if "forbidden_content" not in failed_checks:
                        err = self._check_forbidden_patterns(line, line_num)
                        if err:
                            errors.extend(err)
                            failed_checks.add("forbidden_content")
                    if "contains_links" not in failed_checks:
                        war = self.check_line_link(line,line_num)
                        if war:
                            warnings.append(war)
                            failed_checks.add("contains_links")
        except UnicodeDecodeError:
            errors.append({
                "check": "file_encoding",
                "error": f"File is not valid {self.ENCODING}. Check file encoding.",
            })
        except Exception as e:
            errors.append({"check": "file_read_process", "error": str(e)})

        return ValidationResult(
            file_path, is_valid=not errors, validator_used="text", errors=errors, warnings=warnings
        )

    def _check_file_size(self, file_path: Path) -> List[Dict]:
        file_size_bytes = file_path.stat().st_size
        if file_size_bytes == 0:
            return [{"check": "empty_file", "error": "File is empty."}]
        
        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            return [{
                "check": "file_size",
                "error": f"File size {file_size_mb:.2f}MB exceeds maximum {self.MAX_FILE_SIZE_MB}MB.",
            }]
        return []
    
    def check_line_link(self, line: str,line_num: int) -> Dict | None:
        link_pattern = r"((?:https?://|ftp://|www\.)\S+)"

        for match in re.finditer(link_pattern, line):
            return { 
            "check": "contains_links",
            "row": line_num,
            "error": f"Line contains link matching: '{match.group(0)}'",
            }
        return None
    
    def _check_line_length(self, line: str, line_num: int) -> Dict | None:
        if len(line) > self.MAX_LINE_LENGTH:
            return {
                "check": "line_length",
                "row": line_num,
                "error": f"Line is too long ({len(line)} chars, max is {self.MAX_LINE_LENGTH}).",
            }
        return None

    def _check_forbidden_patterns(self, line: str, line_num: int) -> List[Dict] | None:
        errors: List[Dict] = []

        for pattern in self.FORBIDDEN_PATTERNS:
            for match in re.finditer(pattern, line):
                errors.append({
                    "check": "forbidden_content",
                    "row": line_num,
                    "error": f"Line contains sensitive pattern matching: '{match.group(0)}'",
                })
                break

        return errors if errors else None