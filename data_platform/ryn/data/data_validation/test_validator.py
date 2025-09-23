import pytest
import yaml
from pathlib import Path

from data.data_validation.validation import DirectoryValidator
from data.data_validation.structures import ValidationReport, ValidationResult


@pytest.fixture(scope="session")
def landing_dir() -> Path:
    """Path to the root landing directory (contains files to validate)."""
    return Path(__file__).parent / "landing"


@pytest.fixture(scope="session")
def config(landing_dir: Path):
    """Load validation config from YAML."""
    config_path = landing_dir / "validation_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def report(landing_dir: Path, config) -> ValidationReport:
    """Run the full validator once and reuse results across tests."""
    validator = DirectoryValidator(config=config)
    return validator.validate(str(landing_dir))


# ----------------------------
# Helper: get result by suffix
# ----------------------------
def get_result(report: ValidationReport, suffix: str) -> ValidationResult:
    """Find a ValidationResult by filename suffix (e.g., 'product_good.jpg')."""
    matches = [r for r in report.results if r.file_path.endswith(suffix)]
    assert matches, f"No validation result found for {suffix}"
    return matches[0]


# ----------------------------
# Tests for image validation
# ----------------------------
@pytest.mark.parametrize(
    "filename,is_valid,expected_error",
    [
        ("product_good.jpg", True, None),
        ("product_black.jpg", False, "solid_color"),
        ("product_white.jpg", False, "solid_color"),
        ("product_corrupt.jpg", False, "image_loadable"),
    ],
)
def test_images(report, filename, is_valid, expected_error):
    result = get_result(report, filename)

    # If it fails, show actual errors for debugging
    assert result.is_valid == is_valid, (
        f"{filename}: expected valid={is_valid}, "
        f"but got valid={result.is_valid}, errors={result.errors}"
    )

    if not is_valid:
        actual_checks = [e["check"] for e in result.errors]
        assert expected_error in actual_checks, (
            f"{filename}: expected error '{expected_error}', "
            f"but got errors={actual_checks}"
        )


# ----------------------------
# Tests for parquet validation
# ----------------------------
def test_transactions_parquet(report):
    result = get_result(report, "transactions.parquet")
    assert result.is_valid, f"transactions.parquet errors={result.errors}"


# ----------------------------
# Tests for zipped CSVs
# ----------------------------
def test_unzipped_csvs_exist(report):
    """Check that zipped CSVs inside orders_archive.zip were validated."""
    csv_results = [r for r in report.results if r.file_path.endswith(".csv")]

    print("\n--- CSV Validation Results ---")
    for r in csv_results:
        print(f"CSV Validation - {r.file_path}: valid={r.is_valid}, errors={r.errors}")
    assert csv_results, "Expected unzipped CSV files to be validated"
    assert any(not r.is_valid for r in csv_results), (
        "Expected at least one CSV with errors, "
        f"but got {[r.errors for r in csv_results]}"
    )


# ----------------------------
# Global report consistency
# ----------------------------
def test_report_contains_expected_file_types(report):
    """The report should contain results for images, parquet, and csv files."""
    file_types = {Path(r.file_path).suffix for r in report.results}
    assert {".jpg", ".csv", ".parquet"}.issubset(file_types), (
        f"Unexpected file types in report: {file_types}"
    )
