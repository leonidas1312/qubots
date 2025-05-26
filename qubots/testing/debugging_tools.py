"""
Debugging Tools for Qubots Platform Integration

Provides comprehensive debugging utilities for troubleshooting issues during
the upload/load/execute workflow, including error analysis, performance profiling,
and detailed reporting.
"""

import os
import sys
import time
import json
import traceback
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

from ..rastion_client import get_global_client
import qubots.rastion as rastion


@dataclass
class DebugReport:
    """Comprehensive debug report for troubleshooting."""
    timestamp: float = field(default_factory=time.time)
    system_info: Dict[str, Any] = field(default_factory=dict)
    environment_info: Dict[str, Any] = field(default_factory=dict)
    authentication_status: Dict[str, Any] = field(default_factory=dict)
    model_analysis: Dict[str, Any] = field(default_factory=dict)
    error_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class ErrorAnalyzer:
    """Analyzes errors and provides troubleshooting recommendations."""
    
    def __init__(self):
        self.error_patterns = {
            "authentication": [
                "not authenticated", "authentication failed", "invalid token",
                "unauthorized", "403", "401"
            ],
            "network": [
                "connection", "timeout", "network", "dns", "unreachable",
                "connection refused", "connection reset"
            ],
            "model_structure": [
                "missing required files", "config.json", "qubot.py",
                "class not found", "invalid json"
            ],
            "import_error": [
                "import error", "module not found", "no module named",
                "importerror", "modulenotfounderror"
            ],
            "execution_error": [
                "execution failed", "optimization failed", "runtime error",
                "attribute error", "type error"
            ]
        }
    
    def analyze_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze an error and provide troubleshooting recommendations.
        
        Args:
            error: The exception that occurred
            context: Additional context about when the error occurred
            
        Returns:
            Analysis results with recommendations
        """
        error_msg = str(error).lower()
        error_type = type(error).__name__
        
        analysis = {
            "error_type": error_type,
            "error_message": str(error),
            "error_category": "unknown",
            "severity": "medium",
            "recommendations": [],
            "context": context or {}
        }
        
        # Categorize error
        for category, patterns in self.error_patterns.items():
            if any(pattern in error_msg for pattern in patterns):
                analysis["error_category"] = category
                break
        
        # Generate recommendations based on category
        analysis["recommendations"] = self._get_recommendations(analysis["error_category"], error_msg)
        
        # Determine severity
        if analysis["error_category"] in ["authentication", "network"]:
            analysis["severity"] = "high"
        elif analysis["error_category"] in ["model_structure", "import_error"]:
            analysis["severity"] = "medium"
        else:
            analysis["severity"] = "low"
        
        return analysis
    
    def _get_recommendations(self, category: str, error_msg: str) -> List[str]:
        """Get recommendations based on error category."""
        recommendations = []
        
        if category == "authentication":
            recommendations.extend([
                "Check if you are authenticated: rastion.is_authenticated()",
                "Re-authenticate with: rastion.authenticate('your_token')",
                "Verify your Gitea token is valid and has the correct permissions",
                "Check if the token has expired"
            ])
        
        elif category == "network":
            recommendations.extend([
                "Check your internet connection",
                "Verify the Rastion platform URL is accessible",
                "Check if you're behind a firewall or proxy",
                "Try again after a few moments (temporary network issue)"
            ])
        
        elif category == "model_structure":
            recommendations.extend([
                "Ensure your model directory contains qubot.py and config.json",
                "Validate your config.json syntax with a JSON validator",
                "Check that the class name in config.json matches the class in qubot.py",
                "Verify all required fields are present in config.json"
            ])
        
        elif category == "import_error":
            recommendations.extend([
                "Check that qubots is properly installed: pip install qubots",
                "Verify your model imports the correct qubots base classes",
                "Check for typos in import statements",
                "Ensure all dependencies are installed"
            ])
        
        elif category == "execution_error":
            recommendations.extend([
                "Check that your model's methods are implemented correctly",
                "Verify parameter types and values are correct",
                "Test your model locally before uploading",
                "Check for compatibility between problem and optimizer"
            ])
        
        else:
            recommendations.extend([
                "Check the full error traceback for more details",
                "Try running the operation again",
                "Verify your model follows the qubots framework conventions"
            ])
        
        return recommendations


class PerformanceProfiler:
    """Profiles performance of model operations."""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = []
        self.memory_usage = []
    
    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = time.time()
        self.checkpoints = []
        self.memory_usage = []
        self._record_checkpoint("profiling_start")
    
    def checkpoint(self, name: str):
        """Record a performance checkpoint."""
        if self.start_time is None:
            self.start_profiling()
        
        self._record_checkpoint(name)
    
    def _record_checkpoint(self, name: str):
        """Record a checkpoint with timing and memory info."""
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
        except:
            memory_mb = 0
        
        checkpoint = {
            "name": name,
            "timestamp": current_time,
            "elapsed_seconds": elapsed,
            "memory_mb": memory_mb
        }
        
        self.checkpoints.append(checkpoint)
        self.memory_usage.append(memory_mb)
    
    def get_profile_report(self) -> Dict[str, Any]:
        """Get a performance profile report."""
        if not self.checkpoints:
            return {"error": "No profiling data available"}
        
        total_time = self.checkpoints[-1]["elapsed_seconds"]
        max_memory = max(self.memory_usage) if self.memory_usage else 0
        min_memory = min(self.memory_usage) if self.memory_usage else 0
        
        return {
            "total_time": total_time,
            "checkpoints": self.checkpoints,
            "memory_stats": {
                "max_memory_mb": max_memory,
                "min_memory_mb": min_memory,
                "memory_delta_mb": max_memory - min_memory
            },
            "performance_summary": {
                "total_checkpoints": len(self.checkpoints),
                "avg_checkpoint_interval": total_time / len(self.checkpoints) if len(self.checkpoints) > 1 else 0
            }
        }


class DebuggingTools:
    """
    Comprehensive debugging toolkit for qubots platform integration.
    
    Provides utilities for system analysis, error diagnosis, and troubleshooting.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize debugging tools.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.error_analyzer = ErrorAnalyzer()
        self.profiler = PerformanceProfiler()
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] DEBUG {level}: {message}")
    
    def generate_debug_report(self, model_path: Optional[str] = None,
                            error: Optional[Exception] = None) -> DebugReport:
        """
        Generate a comprehensive debug report.
        
        Args:
            model_path: Path to model being debugged (optional)
            error: Error that occurred (optional)
            
        Returns:
            Comprehensive debug report
        """
        self.log("Generating debug report...")
        
        report = DebugReport()
        
        # System information
        report.system_info = self._collect_system_info()
        
        # Environment information
        report.environment_info = self._collect_environment_info()
        
        # Authentication status
        report.authentication_status = self._check_authentication_status()
        
        # Model analysis (if path provided)
        if model_path:
            report.model_analysis = self._analyze_model(model_path)
        
        # Error analysis (if error provided)
        if error:
            report.error_analysis = self.error_analyzer.analyze_error(error)
            report.recommendations.extend(report.error_analysis["recommendations"])
        
        # General recommendations
        report.recommendations.extend(self._get_general_recommendations(report))
        
        return report
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        try:
            return {
                "platform": sys.platform,
                "python_version": sys.version,
                "python_executable": sys.executable,
                "working_directory": os.getcwd(),
                "cpu_count": os.cpu_count(),
                "memory_available": psutil.virtual_memory().available / 1024 / 1024 if psutil else "unknown"
            }
        except Exception as e:
            return {"error": f"Failed to collect system info: {e}"}
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect environment information."""
        try:
            env_info = {
                "qubots_available": False,
                "qubots_version": "unknown",
                "python_path": sys.path[:3],  # First 3 entries
                "installed_packages": []
            }
            
            # Check qubots availability
            try:
                import qubots
                env_info["qubots_available"] = True
                env_info["qubots_version"] = getattr(qubots, '__version__', 'unknown')
            except ImportError:
                pass
            
            # Check key dependencies
            key_packages = ["numpy", "requests", "psutil"]
            for package in key_packages:
                try:
                    __import__(package)
                    env_info["installed_packages"].append(package)
                except ImportError:
                    pass
            
            return env_info
        except Exception as e:
            return {"error": f"Failed to collect environment info: {e}"}
    
    def _check_authentication_status(self) -> Dict[str, Any]:
        """Check authentication status with Rastion platform."""
        try:
            auth_info = {
                "authenticated": False,
                "client_available": False,
                "token_present": False,
                "connection_test": "not_tested"
            }
            
            # Check if client is available
            try:
                client = get_global_client()
                auth_info["client_available"] = True
                
                if client and hasattr(client, 'token') and client.token:
                    auth_info["token_present"] = True
                    auth_info["authenticated"] = True
                
            except Exception as e:
                auth_info["client_error"] = str(e)
            
            # Test connection if authenticated
            if auth_info["authenticated"]:
                try:
                    # Try a simple operation to test connection
                    rastion.discover_models()
                    auth_info["connection_test"] = "success"
                except Exception as e:
                    auth_info["connection_test"] = "failed"
                    auth_info["connection_error"] = str(e)
            
            return auth_info
        except Exception as e:
            return {"error": f"Failed to check authentication: {e}"}
    
    def _analyze_model(self, model_path: str) -> Dict[str, Any]:
        """Analyze a model for potential issues."""
        try:
            path = Path(model_path)
            analysis = {
                "path_exists": path.exists(),
                "files_found": [],
                "config_analysis": {},
                "code_analysis": {}
            }
            
            if not path.exists():
                analysis["error"] = f"Model path does not exist: {model_path}"
                return analysis
            
            # List files
            analysis["files_found"] = [f.name for f in path.iterdir() if f.is_file()]
            
            # Analyze config.json
            config_path = path / "config.json"
            if config_path.exists():
                analysis["config_analysis"] = self._analyze_config_file(config_path)
            else:
                analysis["config_analysis"] = {"error": "config.json not found"}
            
            # Analyze qubot.py
            code_path = path / "qubot.py"
            if code_path.exists():
                analysis["code_analysis"] = self._analyze_code_file(code_path)
            else:
                analysis["code_analysis"] = {"error": "qubot.py not found"}
            
            return analysis
        except Exception as e:
            return {"error": f"Failed to analyze model: {e}"}
    
    def _analyze_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Analyze config.json file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            analysis = {
                "valid_json": True,
                "required_fields": {},
                "optional_fields": {},
                "issues": []
            }
            
            # Check required fields
            required_fields = ["type", "entry_point", "class_name", "framework"]
            for field in required_fields:
                analysis["required_fields"][field] = field in config
                if field not in config:
                    analysis["issues"].append(f"Missing required field: {field}")
            
            # Check optional but recommended fields
            optional_fields = ["metadata", "parameters", "default_params"]
            for field in optional_fields:
                analysis["optional_fields"][field] = field in config
            
            return analysis
        except json.JSONDecodeError as e:
            return {"valid_json": False, "json_error": str(e)}
        except Exception as e:
            return {"error": f"Failed to analyze config: {e}"}
    
    def _analyze_code_file(self, code_path: Path) -> Dict[str, Any]:
        """Analyze qubot.py file."""
        try:
            with open(code_path, 'r') as f:
                code_content = f.read()
            
            analysis = {
                "file_size": len(code_content),
                "line_count": len(code_content.splitlines()),
                "has_imports": False,
                "has_class_definition": False,
                "potential_issues": []
            }
            
            # Check for imports
            if "import" in code_content:
                analysis["has_imports"] = True
            
            # Check for class definition
            if "class " in code_content:
                analysis["has_class_definition"] = True
            
            # Check for potential issues
            if "from qubots" not in code_content and "import qubots" not in code_content:
                analysis["potential_issues"].append("No qubots imports found")
            
            if not analysis["has_class_definition"]:
                analysis["potential_issues"].append("No class definition found")
            
            return analysis
        except Exception as e:
            return {"error": f"Failed to analyze code: {e}"}
    
    def _get_general_recommendations(self, report: DebugReport) -> List[str]:
        """Get general recommendations based on the debug report."""
        recommendations = []
        
        # Authentication recommendations
        if not report.authentication_status.get("authenticated", False):
            recommendations.append("Authenticate with Rastion platform using rastion.authenticate('your_token')")
        
        # Environment recommendations
        if not report.environment_info.get("qubots_available", False):
            recommendations.append("Install qubots library: pip install qubots")
        
        # Model recommendations
        if report.model_analysis and "error" in report.model_analysis:
            recommendations.append("Check that the model path exists and contains required files")
        
        return recommendations
    
    def print_debug_report(self, report: DebugReport):
        """Print a formatted debug report."""
        print("\n" + "=" * 80)
        print("🔍 QUBOTS DEBUG REPORT")
        print("=" * 80)
        print(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}")
        
        # System Info
        print(f"\n🖥️ System Information:")
        for key, value in report.system_info.items():
            print(f"   {key}: {value}")
        
        # Environment Info
        print(f"\n🌍 Environment Information:")
        for key, value in report.environment_info.items():
            print(f"   {key}: {value}")
        
        # Authentication Status
        print(f"\n🔐 Authentication Status:")
        for key, value in report.authentication_status.items():
            print(f"   {key}: {value}")
        
        # Model Analysis
        if report.model_analysis:
            print(f"\n📁 Model Analysis:")
            for key, value in report.model_analysis.items():
                print(f"   {key}: {value}")
        
        # Error Analysis
        if report.error_analysis:
            print(f"\n❌ Error Analysis:")
            for key, value in report.error_analysis.items():
                print(f"   {key}: {value}")
        
        # Recommendations
        if report.recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "=" * 80)
