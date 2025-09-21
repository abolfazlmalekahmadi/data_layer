from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass, field

# --- Data Structures for Results ---

from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ValidationResult:
    """Holds the outcome of validating a single file."""
    file_path: str
    is_valid: bool
    validator_used: str
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def __repr__(self):
        status = "PASSED" if self.is_valid else "FAILED"
        return f"ValidationResult(file='{self.file_path}', status='{status}', errors={len(self.errors)})"

@dataclass
class ValidationReport:
    """Aggregates results from all files validated in a directory run."""
    directory_path: str
    results: List[ValidationResult] = field(default_factory=list)
    
    @property
    def overall_status(self) -> str:
        if not self.results:
            return "NO_FILES_VALIDATED"
        if all(r.is_valid for r in self.results):
            return "SUCCESS"
        if any(r.is_valid for r in self.results):
            return "PARTIAL_FAILURE"
        return "COMPLETE_FAILURE"
        
    def add_result(self, result: ValidationResult):
        """Adds a single file's validation result to the report."""
        self.results.append(result)

    def has_failures(self) -> bool:
        """Returns True if any validation failed."""
        return any(not r.is_valid for r in self.results)

    def __repr__(self):
        return (f"ValidationReport(directory='{self.directory_path}', "
                f"status='{self.overall_status}', "
                f"files_validated={len(self.results)})")


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

# --- Specialized Validator Implementations ---

class TabularValidator(BaseValidator):
    """
    Validates structured data files (CSV, Parquet, etc.).
    This is an evolution of the previously discussed schema.
    """
    def validate(self, file_path: str) -> ValidationResult:
        """
        1. Reads the tabular file (CSV, Parquet) into a pandas DataFrame.
        2. Applies column-level and dataset-level rules.
        3. Returns the result.
        """
        pass
    
    # Internal helper methods for specific checks (_check_not_null, _check_unique, etc.)
    # would be implemented here, as in the previous schema.

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

# --- The Main Orchestrator Class ---

class DirectoryValidator:
    """
    Orchestrates the validation of an entire directory containing mixed file types.

    It uses a configuration file to map file patterns to the appropriate
    specialized validator and its rules.
    """

    def __init__(self, config: List[Dict[str, Any]]):
        """
        Initializes the orchestrator with a validation configuration.

        Args:
            config (List[Dict[str, Any]]): A list of validation tasks.
                Example config (e.g., from a YAML file):
                [
                    {
                        "name": "Validate User CSVs",
                        "path_pattern": "data/users/*.csv",
                        "validator": "tabular",
                        "rules": { "dataset": {"min_rows": 10}, "columns": { ... } }
                    },
                    {
                        "name": "Validate Product Images",
                        "path_pattern": "images/products/*.jpg",
                        "validator": "image",
                        "rules": { "min_height": 200, "max_filesize_kb": 512 }
                    },
                    {
                        "name": "Validate Order JSONs",
                        "path_pattern": "orders/*.json",
                        "validator": "json",
                        "rules": { "schema_path": "schemas/order_schema.json" }
                    }
                ]
        """
        # A mapping from a string name (like "tabular") to the actual validator class.
        self.validator_registry: Dict[str, BaseValidator] = {}
        # The configuration defining which validators to run on which files.
        self.config = config
        
        # This method would populate self.validator_registry
        self._register_validators() 

    def _register_validators(self):
        """
        Initializes the validator registry with known validator types.
        This makes the system easily extensible.
        """
        pass

    def __call__(self, directory_path: str) -> ValidationReport:
        """
        Makes the instance callable. A convenience wrapper for `validate`.
        """
        pass

    def validate(self, directory_path: str) -> ValidationReport:
        """
        Executes the full validation plan for the given directory.

        1. Scans the directory for files matching the patterns in the config.
        2. For each matching file, instantiates the specified validator with its rules.
        3. Calls the `validate` method on the specialized validator.
        4. Collects all ValidationResult objects into a final ValidationReport.

        Args:
            directory_path (str): The root directory to start the validation from.

        Returns:
            ValidationReport: A comprehensive report of the entire run.
        """
        pass