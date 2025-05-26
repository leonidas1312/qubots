"""
Upload Validator for Qubots Models

Provides comprehensive validation for the model upload process to the Rastion platform,
including pre-upload validation, upload monitoring, and post-upload verification.
"""

import os
import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from ..rastion_client import get_global_client, QubotPackager
import qubots.rastion as rastion


@dataclass
class UploadResult:
    """Result of an upload validation test."""
    success: bool
    repository_name: str = ""
    repository_url: str = ""
    upload_time: float = 0.0
    validation_details: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    error_details: Dict[str, Any] = field(default_factory=dict)


class UploadError(Exception):
    """Custom exception for upload validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class UploadValidator:
    """
    Comprehensive validator for model upload process.
    
    This class provides methods to validate models before upload,
    monitor the upload process, and verify successful upload.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the upload validator.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] UPLOAD {level}: {message}")
    
    def validate_pre_upload(self, model_path: str) -> Dict[str, Any]:
        """
        Validate model before upload.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Validation results dictionary
            
        Raises:
            UploadError: If validation fails
        """
        self.log(f"Pre-upload validation for: {model_path}")
        
        path = Path(model_path)
        if not path.exists():
            raise UploadError(f"Model path does not exist: {model_path}")
        
        validation_results = {
            "model_path": str(path),
            "files_found": [],
            "config_validation": {},
            "code_validation": {},
            "requirements_validation": {}
        }
        
        # Check file structure
        files_found = [f.name for f in path.iterdir() if f.is_file()]
        validation_results["files_found"] = files_found
        
        required_files = ["qubot.py", "config.json"]
        missing_files = [f for f in required_files if f not in files_found]
        
        if missing_files:
            raise UploadError(f"Missing required files: {missing_files}")
        
        # Validate config.json
        config_validation = self._validate_config_file(path / "config.json")
        validation_results["config_validation"] = config_validation
        
        # Validate qubot.py
        code_validation = self._validate_code_file(path / "qubot.py", config_validation["config"])
        validation_results["code_validation"] = code_validation
        
        # Validate requirements if present
        requirements_file = path / "requirements.txt"
        if requirements_file.exists():
            req_validation = self._validate_requirements_file(requirements_file)
            validation_results["requirements_validation"] = req_validation
        
        self.log("Pre-upload validation passed")
        return validation_results
    
    def _validate_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Validate the config.json file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise UploadError(f"Invalid JSON in config.json: {e}")
        
        # Check required fields
        required_fields = ["type", "entry_point", "class_name", "framework"]
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise UploadError(f"Missing required config fields: {missing_fields}")
        
        # Validate field values
        valid_types = ["problem", "optimizer"]
        if config["type"] not in valid_types:
            raise UploadError(f"Invalid type '{config['type']}'. Must be one of: {valid_types}")
        
        if config["framework"] != "qubots":
            raise UploadError(f"Invalid framework '{config['framework']}'. Must be 'qubots'")
        
        return {
            "valid": True,
            "config": config,
            "required_fields_present": True,
            "type": config["type"],
            "class_name": config["class_name"],
            "entry_point": config["entry_point"]
        }
    
    def _validate_code_file(self, code_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the qubot.py file."""
        try:
            with open(code_path, 'r') as f:
                code_content = f.read()
        except Exception as e:
            raise UploadError(f"Cannot read qubot.py: {e}")
        
        class_name = config["class_name"]
        
        # Check if class is defined
        if f"class {class_name}" not in code_content:
            raise UploadError(f"Class '{class_name}' not found in qubot.py")
        
        # Check for required imports based on type
        model_type = config["type"]
        if model_type == "problem":
            required_imports = ["BaseProblem", "CombinatorialProblem", "ContinuousProblem", "DiscreteProblem"]
        else:  # optimizer
            required_imports = ["BaseOptimizer", "PopulationBasedOptimizer", "LocalSearchOptimizer"]
        
        has_required_import = any(imp in code_content for imp in required_imports)
        if not has_required_import:
            raise UploadError(f"Missing required qubots imports for {model_type}")
        
        return {
            "valid": True,
            "class_found": True,
            "class_name": class_name,
            "has_required_imports": has_required_import,
            "file_size": len(code_content)
        }
    
    def _validate_requirements_file(self, req_path: Path) -> Dict[str, Any]:
        """Validate the requirements.txt file."""
        try:
            with open(req_path, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except Exception as e:
            raise UploadError(f"Cannot read requirements.txt: {e}")
        
        return {
            "valid": True,
            "requirements_count": len(requirements),
            "requirements": requirements
        }
    
    def validate_upload_process(self, model_path: str, repository_name: str, 
                              description: str, overwrite: bool = True) -> UploadResult:
        """
        Validate the complete upload process.
        
        Args:
            model_path: Path to the model directory
            repository_name: Name for the repository
            description: Description for the model
            overwrite: Whether to overwrite existing repository
            
        Returns:
            UploadResult with validation details
        """
        start_time = time.time()
        
        try:
            # Step 1: Pre-upload validation
            self.log("Step 1: Pre-upload validation")
            pre_validation = self.validate_pre_upload(model_path)
            
            # Step 2: Authentication check
            self.log("Step 2: Authentication check")
            client = get_global_client()
            if not (client and hasattr(client, 'token') and client.token):
                raise UploadError("Not authenticated with Rastion platform")
            
            # Step 3: Package validation
            self.log("Step 3: Package validation")
            package_validation = self._validate_package_creation(model_path)
            
            # Step 4: Upload execution
            self.log("Step 4: Upload execution")
            upload_url = rastion.upload_qubots_model(
                path=model_path,
                repository_name=repository_name,
                description=description,
                overwrite=overwrite
            )
            
            # Step 5: Post-upload verification
            self.log("Step 5: Post-upload verification")
            post_validation = self._validate_post_upload(repository_name)
            
            upload_time = time.time() - start_time
            
            return UploadResult(
                success=True,
                repository_name=repository_name,
                repository_url=upload_url,
                upload_time=upload_time,
                validation_details={
                    "pre_validation": pre_validation,
                    "package_validation": package_validation,
                    "post_validation": post_validation
                }
            )
            
        except Exception as e:
            upload_time = time.time() - start_time
            
            return UploadResult(
                success=False,
                repository_name=repository_name,
                upload_time=upload_time,
                error_message=str(e),
                error_details={
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            )
    
    def _validate_package_creation(self, model_path: str) -> Dict[str, Any]:
        """Validate that the model can be packaged correctly."""
        try:
            # Load the model to test packaging
            config_path = Path(model_path) / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Test package creation
            package = QubotPackager.package_model_from_path(
                model_path,
                config["class_name"],
                f"Test package for {config['class_name']}"
            )
            
            expected_files = {"qubot.py", "config.json", "requirements.txt", "README.md"}
            package_files = set(package.keys())
            
            return {
                "valid": True,
                "package_files": list(package_files),
                "expected_files_present": expected_files.issubset(package_files),
                "package_size": sum(len(content) for content in package.values())
            }
            
        except Exception as e:
            raise UploadError(f"Package creation failed: {e}")
    
    def _validate_post_upload(self, repository_name: str) -> Dict[str, Any]:
        """Validate that the upload was successful."""
        try:
            # Wait a moment for platform processing
            time.sleep(2)
            
            # Try to load the uploaded model
            model = rastion.load_qubots_model(repository_name)
            
            return {
                "valid": True,
                "model_loadable": True,
                "model_type": type(model).__name__,
                "repository_name": repository_name
            }
            
        except Exception as e:
            # Upload might have succeeded but loading failed
            return {
                "valid": False,
                "model_loadable": False,
                "load_error": str(e),
                "repository_name": repository_name
            }
