from dataclasses import dataclass, field
from typing import Dict, Any, List



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