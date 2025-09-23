import os
import glob
import zipfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Type

from validators import TabularValidator, ImageValidator, JsonValidator, BaseValidator
from structures import ValidationReport, ValidationResult


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class ArchiveHandler:
    """
    Handles extraction and cleanup of archive files in a directory.
    """

    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)

    def unzip_all(self) -> List[str]:
        """
        Extracts all .zip files under the directory.

        Returns:
            List[str]: Absolute paths of all extracted files.
        """
        zip_files = glob.glob(str(self.source_dir / "**/*.zip"), recursive=True)
        logger.info("Found %d zip file(s) in %s", len(zip_files), self.source_dir)

        if not zip_files:
            return []

        extracted_files: List[str] = []
        for zip_path in zip_files:
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    logger.info("Extracting %s", zip_path)
                    for member in zip_ref.infolist():
                        if not member.is_dir():
                            extracted_files.append(str(self.source_dir / member.filename))
                    zip_ref.extractall(self.source_dir)
            except zipfile.BadZipFile:
                logger.warning("Skipping corrupt zip file: %s", zip_path)
            except Exception as e:
                logger.error("Failed to extract %s: %s", zip_path, e)

        return extracted_files

    def cleanup(self, file_paths: List[str]):
        """
        Deletes a list of files, typically extracted from an archive.
        """
        if not file_paths:
            return

        logger.info("Cleaning up %d extracted file(s)", len(file_paths))
        for file_path in file_paths:
            try:
                os.remove(file_path)
                logger.debug("Deleted %s", file_path)
            except FileNotFoundError:
                logger.warning("File not found for cleanup: %s", file_path)
            except Exception as e:
                logger.error("Failed to delete %s: %s", file_path, e)


class DirectoryValidator:
    """
    Orchestrates the validation of an entire directory containing mixed file types.
    """

    def __init__(self, config: List[Dict[str, Any]], unzip_first: bool = True):
        self.config = config
        self.unzip_first = unzip_first
        self._validator_registry: Dict[str, Type[BaseValidator]] = {}
        self._register_default_validators()

    def _register_default_validators(self):
        """Registers built-in validators."""
        self._validator_registry = {
            "tabular": TabularValidator,
            "image": ImageValidator,
            # "json": JsonValidator,  # enable when implemented
        }
        logger.info("Registered validators: %s", list(self._validator_registry.keys()))

    def register_validator(self, name: str, validator_cls: Type[BaseValidator]):
        """Allows external code to register new validators dynamically."""
        if not issubclass(validator_cls, BaseValidator):
            raise TypeError("Validator must subclass BaseValidator")
        self._validator_registry[name] = validator_cls
        logger.info("Registered custom validator: %s", name)

    def validate(self, directory_path: str) -> ValidationReport:
        """
        Top-level validation method. Manages unzipping, validation, and cleanup.
        """
        files_to_cleanup: List[str] = []
        if self.unzip_first:
            archive_handler = ArchiveHandler(directory_path)
            files_to_cleanup = archive_handler.unzip_all()

        report = self._validate_directory(directory_path)

        if files_to_cleanup:
            ArchiveHandler(directory_path).cleanup(files_to_cleanup)

        return report

    def _validate_directory(self, directory_path: str) -> ValidationReport:
        """
        Core validation loop over configured tasks.
        """
        report = ValidationReport(directory_path=directory_path)
        logger.info("Starting validation in directory: %s", directory_path)

        for task in self.config:
            path_pattern = task.get("path_pattern")
            validator_name = task.get("validator")
            rules = task.get("rules", {})
            task_name = task.get("name", "Unnamed Task")

            if not path_pattern or not validator_name:
                logger.warning("Skipping invalid task config: %s", task)
                continue

            validator_cls = self._validator_registry.get(validator_name)
            if not validator_cls:
                logger.warning("Unknown validator '%s' in task '%s'. Skipping.", validator_name, task_name)
                continue

            # Resolve file matches
            full_pattern = str(Path(directory_path) / path_pattern)
            matched_files = glob.glob(full_pattern, recursive=True)
            if not matched_files:
                logger.info("No files matched pattern '%s' in '%s'", path_pattern, directory_path)
                continue

            validator = validator_cls(rules=rules)
            for file_path in matched_files:
                try:
                    result: ValidationResult = validator.validate(file_path)
                    report.add_result(result)
                except Exception as e:
                    logger.error("Validation failed for %s: %s", file_path, e)

        return report

    def __call__(self, directory_path: str) -> ValidationReport:
        return self.validate(directory_path)
