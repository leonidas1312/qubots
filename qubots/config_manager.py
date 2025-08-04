"""
Configuration manager for qubots framework.
Handles local vs remote git hosting configuration.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GitConfig:
    """Git hosting configuration."""
    base_url: str
    api_base: str
    clone_url_template: str
    ssh_port: Optional[int] = None
    
    def get_clone_url(self, owner: str, repo: str) -> str:
        """Get clone URL for a repository."""
        return self.clone_url_template.format(owner=owner, repo=repo)


class QubotsConfig:
    """
    Configuration manager for qubots framework.
    Supports both local Gitea and remote hosting.
    """
    
    DEFAULT_CONFIGS = {
        "local": GitConfig(
            base_url="http://localhost:3000",
            api_base="http://localhost:3000/api/v1",
            clone_url_template="http://localhost:3000/{owner}/{repo}.git",
            ssh_port=2223
        ),
        "rastion": GitConfig(
            base_url="https://hub.rastion.com",
            api_base="https://hub.rastion.com/api/v1", 
            clone_url_template="https://hub.rastion.com/{owner}/{repo}.git"
        )
    }
    
    def __init__(self, config_path: str = "~/.qubots/config.json"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path).expanduser()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Return default configuration
        return {
            "active_profile": "local",
            "profiles": {
                "local": {
                    "git_config": "local",
                    "token": None,
                    "username": None
                },
                "rastion": {
                    "git_config": "rastion", 
                    "token": None,
                    "username": None
                }
            },
            "cache_dir": "~/.cache/qubots_hub"
        }
    
    def _save_config(self):
        """Save configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def get_active_profile(self) -> str:
        """Get the active profile name."""
        return self._config.get("active_profile", "local")
    
    def set_active_profile(self, profile: str):
        """Set the active profile."""
        if profile not in self._config["profiles"]:
            raise ValueError(f"Profile '{profile}' does not exist")
        self._config["active_profile"] = profile
        self._save_config()
    
    def get_git_config(self, profile: Optional[str] = None) -> GitConfig:
        """Get git configuration for a profile."""
        profile = profile or self.get_active_profile()
        profile_config = self._config["profiles"][profile]
        git_config_name = profile_config["git_config"]
        
        if git_config_name in self.DEFAULT_CONFIGS:
            return self.DEFAULT_CONFIGS[git_config_name]
        else:
            raise ValueError(f"Unknown git config: {git_config_name}")
    
    def get_auth_token(self, profile: Optional[str] = None) -> Optional[str]:
        """Get authentication token for a profile."""
        profile = profile or self.get_active_profile()
        return self._config["profiles"][profile].get("token")
    
    def set_auth_token(self, token: str, profile: Optional[str] = None):
        """Set authentication token for a profile."""
        profile = profile or self.get_active_profile()
        self._config["profiles"][profile]["token"] = token
        self._save_config()
    
    def get_username(self, profile: Optional[str] = None) -> Optional[str]:
        """Get username for a profile."""
        profile = profile or self.get_active_profile()
        return self._config["profiles"][profile].get("username")
    
    def set_username(self, username: str, profile: Optional[str] = None):
        """Set username for a profile."""
        profile = profile or self.get_active_profile()
        self._config["profiles"][profile]["username"] = username
        self._save_config()
    
    def get_cache_dir(self) -> str:
        """Get cache directory path."""
        return os.path.expanduser(self._config.get("cache_dir", "~/.cache/qubots_hub"))
    
    def is_authenticated(self, profile: Optional[str] = None) -> bool:
        """Check if profile is authenticated."""
        profile = profile or self.get_active_profile()
        profile_config = self._config["profiles"][profile]
        return bool(profile_config.get("token") and profile_config.get("username"))
    
    def create_profile(self, name: str, git_config: str, token: Optional[str] = None, 
                      username: Optional[str] = None):
        """Create a new profile."""
        if git_config not in self.DEFAULT_CONFIGS:
            raise ValueError(f"Unknown git config: {git_config}")
        
        self._config["profiles"][name] = {
            "git_config": git_config,
            "token": token,
            "username": username
        }
        self._save_config()
    
    def list_profiles(self) -> Dict[str, Dict[str, Any]]:
        """List all profiles."""
        return self._config["profiles"].copy()
    
    def get_profile_info(self, profile: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a profile."""
        profile = profile or self.get_active_profile()
        profile_config = self._config["profiles"][profile].copy()
        git_config = self.get_git_config(profile)
        
        return {
            "name": profile,
            "git_config": profile_config["git_config"],
            "base_url": git_config.base_url,
            "api_base": git_config.api_base,
            "authenticated": self.is_authenticated(profile),
            "username": profile_config.get("username"),
            "has_token": bool(profile_config.get("token"))
        }


# Global configuration instance
_global_config: Optional[QubotsConfig] = None


def get_config() -> QubotsConfig:
    """Get or create the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = QubotsConfig()
    return _global_config


def set_profile(profile: str):
    """Set the active profile globally."""
    get_config().set_active_profile(profile)


def get_active_git_config() -> GitConfig:
    """Get the git configuration for the active profile."""
    return get_config().get_git_config()


def is_local_mode() -> bool:
    """Check if running in local mode."""
    return get_config().get_active_profile() == "local"
