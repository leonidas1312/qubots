"""
Rregistry and discovery system for qubots.
Provides centralized management, search, and recommendation capabilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import hashlib
from datetime import datetime
from .base_problem import BaseProblem, ProblemMetadata, ProblemType
from .base_optimizer import BaseOptimizer, OptimizerMetadata, OptimizerType, OptimizerFamily

class RegistryType(Enum):
    """Types of registry entries."""
    PROBLEM = "problem"
    OPTIMIZER = "optimizer"
    BENCHMARK = "benchmark"
    DATASET = "dataset"

@dataclass
class RegistryEntry:
    """Entry in the qubots registry."""
    id: str
    name: str
    description: str
    registry_type: RegistryType
    author: str
    version: str
    tags: Set[str] = field(default_factory=set)
    
    # Repository information
    repository_url: str = ""
    repository_path: str = ""
    commit_hash: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Usage statistics
    download_count: int = 0
    rating: float = 0.0
    rating_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    python_requirements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "registry_type": self.registry_type.value,
            "author": self.author,
            "version": self.version,
            "tags": list(self.tags),
            "repository_url": self.repository_url,
            "repository_path": self.repository_path,
            "commit_hash": self.commit_hash,
            "metadata": self.metadata,
            "download_count": self.download_count,
            "rating": self.rating,
            "rating_count": self.rating_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "dependencies": self.dependencies,
            "python_requirements": self.python_requirements
        }

class QubotRegistry:
    """
    Enhanced registry for qubots with search, recommendation, and management capabilities.
    """
    
    def __init__(self, registry_path: str = "~/.qubots/registry"):
        """
        Initialize the registry.
        
        Args:
            registry_path: Path to store registry data
        """
        self.registry_path = os.path.expanduser(registry_path)
        os.makedirs(self.registry_path, exist_ok=True)
        
        self.entries = {}
        self.load_registry()
    
    def register_problem(self, problem: BaseProblem, repository_info: Dict[str, str] = None) -> str:
        """
        Register a problem in the registry.
        
        Args:
            problem: Problem instance to register
            repository_info: Repository information
            
        Returns:
            Registry ID for the problem
        """
        metadata = problem.metadata
        repo_info = repository_info or {}
        
        entry_id = self._generate_id(metadata.name, metadata.author, metadata.version)
        
        entry = RegistryEntry(
            id=entry_id,
            name=metadata.name,
            description=metadata.description,
            registry_type=RegistryType.PROBLEM,
            author=metadata.author,
            version=metadata.version,
            tags=metadata.tags.copy(),
            repository_url=repo_info.get("url", ""),
            repository_path=repo_info.get("path", ""),
            commit_hash=repo_info.get("commit", ""),
            metadata=metadata.to_dict()
        )
        
        self.entries[entry_id] = entry
        self.save_registry()
        return entry_id
    
    def register_optimizer(self, optimizer: BaseOptimizer, repository_info: Dict[str, str] = None) -> str:
        """
        Register an optimizer in the registry.
        
        Args:
            optimizer: Optimizer instance to register
            repository_info: Repository information
            
        Returns:
            Registry ID for the optimizer
        """
        metadata = optimizer.metadata
        repo_info = repository_info or {}
        
        entry_id = self._generate_id(metadata.name, metadata.author, metadata.version)
        
        entry = RegistryEntry(
            id=entry_id,
            name=metadata.name,
            description=metadata.description,
            registry_type=RegistryType.OPTIMIZER,
            author=metadata.author,
            version=metadata.version,
            repository_url=repo_info.get("url", ""),
            repository_path=repo_info.get("path", ""),
            commit_hash=repo_info.get("commit", ""),
            metadata=metadata.to_dict()
        )
        
        self.entries[entry_id] = entry
        self.save_registry()
        return entry_id
    
    def search(self, query: str = "", registry_type: Optional[RegistryType] = None,
               tags: Optional[List[str]] = None, author: Optional[str] = None) -> List[RegistryEntry]:
        """
        Search the registry.
        
        Args:
            query: Text query to search in name and description
            registry_type: Filter by registry type
            tags: Filter by tags
            author: Filter by author
            
        Returns:
            List of matching registry entries
        """
        results = []
        
        for entry in self.entries.values():
            # Filter by type
            if registry_type and entry.registry_type != registry_type:
                continue
            
            # Filter by author
            if author and entry.author.lower() != author.lower():
                continue
            
            # Filter by tags
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            # Text search
            if query:
                query_lower = query.lower()
                if (query_lower not in entry.name.lower() and 
                    query_lower not in entry.description.lower() and
                    not any(query_lower in tag.lower() for tag in entry.tags)):
                    continue
            
            results.append(entry)
        
        # Sort by relevance (download count and rating)
        results.sort(key=lambda x: (x.download_count, x.rating), reverse=True)
        return results
    
    def get_recommendations(self, entry_id: str, limit: int = 5) -> List[RegistryEntry]:
        """
        Get recommendations based on an entry.
        
        Args:
            entry_id: ID of the entry to base recommendations on
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended entries
        """
        if entry_id not in self.entries:
            return []
        
        base_entry = self.entries[entry_id]
        recommendations = []
        
        for entry in self.entries.values():
            if entry.id == entry_id:
                continue
            
            # Calculate similarity score
            score = self._calculate_similarity(base_entry, entry)
            recommendations.append((score, entry))
        
        # Sort by similarity score and return top recommendations
        recommendations.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in recommendations[:limit]]
    
    def get_compatible_optimizers(self, problem_id: str) -> List[RegistryEntry]:
        """
        Get optimizers compatible with a specific problem.
        
        Args:
            problem_id: ID of the problem
            
        Returns:
            List of compatible optimizer entries
        """
        if problem_id not in self.entries:
            return []
        
        problem_entry = self.entries[problem_id]
        if problem_entry.registry_type != RegistryType.PROBLEM:
            return []
        
        problem_metadata = problem_entry.metadata
        problem_type = problem_metadata.get("problem_type", "")
        
        compatible_optimizers = []
        
        for entry in self.entries.values():
            if entry.registry_type != RegistryType.OPTIMIZER:
                continue
            
            optimizer_metadata = entry.metadata
            
            # Check compatibility based on problem type
            if problem_type == "continuous" and optimizer_metadata.get("supports_continuous", True):
                compatible_optimizers.append(entry)
            elif problem_type == "discrete" and optimizer_metadata.get("supports_discrete", True):
                compatible_optimizers.append(entry)
            elif problem_type == "combinatorial" and optimizer_metadata.get("supports_discrete", True):
                compatible_optimizers.append(entry)
            elif problem_type == "mixed_integer" and optimizer_metadata.get("supports_mixed_integer", False):
                compatible_optimizers.append(entry)
        
        # Sort by rating and download count
        compatible_optimizers.sort(key=lambda x: (x.rating, x.download_count), reverse=True)
        return compatible_optimizers
    
    def get_entry(self, entry_id: str) -> Optional[RegistryEntry]:
        """Get entry by ID."""
        return self.entries.get(entry_id)
    
    def update_entry(self, entry_id: str, **updates):
        """Update an entry."""
        if entry_id in self.entries:
            entry = self.entries[entry_id]
            for key, value in updates.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)
            entry.updated_at = datetime.now()
            self.save_registry()
    
    def rate_entry(self, entry_id: str, rating: float):
        """Add a rating to an entry."""
        if entry_id in self.entries and 1.0 <= rating <= 5.0:
            entry = self.entries[entry_id]
            # Update running average
            total_rating = entry.rating * entry.rating_count + rating
            entry.rating_count += 1
            entry.rating = total_rating / entry.rating_count
            self.save_registry()
    
    def increment_download_count(self, entry_id: str):
        """Increment download count for an entry."""
        if entry_id in self.entries:
            self.entries[entry_id].download_count += 1
            self.save_registry()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_entries = len(self.entries)
        problems = sum(1 for e in self.entries.values() if e.registry_type == RegistryType.PROBLEM)
        optimizers = sum(1 for e in self.entries.values() if e.registry_type == RegistryType.OPTIMIZER)
        
        authors = set(e.author for e in self.entries.values())
        total_downloads = sum(e.download_count for e in self.entries.values())
        
        return {
            "total_entries": total_entries,
            "problems": problems,
            "optimizers": optimizers,
            "unique_authors": len(authors),
            "total_downloads": total_downloads,
            "average_rating": sum(e.rating for e in self.entries.values()) / total_entries if total_entries > 0 else 0
        }
    
    def export_registry(self, filename: str):
        """Export registry to JSON file."""
        export_data = {
            "registry_version": "1.0",
            "export_timestamp": datetime.now().isoformat(),
            "entries": {entry_id: entry.to_dict() for entry_id, entry in self.entries.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def load_registry(self):
        """Load registry from disk."""
        registry_file = os.path.join(self.registry_path, "registry.json")
        if os.path.exists(registry_file):
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                for entry_id, entry_data in data.get("entries", {}).items():
                    entry = RegistryEntry(
                        id=entry_data["id"],
                        name=entry_data["name"],
                        description=entry_data["description"],
                        registry_type=RegistryType(entry_data["registry_type"]),
                        author=entry_data["author"],
                        version=entry_data["version"],
                        tags=set(entry_data.get("tags", [])),
                        repository_url=entry_data.get("repository_url", ""),
                        repository_path=entry_data.get("repository_path", ""),
                        commit_hash=entry_data.get("commit_hash", ""),
                        metadata=entry_data.get("metadata", {}),
                        download_count=entry_data.get("download_count", 0),
                        rating=entry_data.get("rating", 0.0),
                        rating_count=entry_data.get("rating_count", 0),
                        created_at=datetime.fromisoformat(entry_data.get("created_at", datetime.now().isoformat())),
                        updated_at=datetime.fromisoformat(entry_data.get("updated_at", datetime.now().isoformat())),
                        dependencies=entry_data.get("dependencies", []),
                        python_requirements=entry_data.get("python_requirements", [])
                    )
                    self.entries[entry_id] = entry
            except Exception as e:
                print(f"Error loading registry: {e}")
    
    def save_registry(self):
        """Save registry to disk."""
        registry_file = os.path.join(self.registry_path, "registry.json")
        export_data = {
            "registry_version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "entries": {entry_id: entry.to_dict() for entry_id, entry in self.entries.items()}
        }
        
        with open(registry_file, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _generate_id(self, name: str, author: str, version: str) -> str:
        """Generate unique ID for an entry."""
        content = f"{name}_{author}_{version}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _calculate_similarity(self, entry1: RegistryEntry, entry2: RegistryEntry) -> float:
        """Calculate similarity score between two entries."""
        score = 0.0
        
        # Same type bonus
        if entry1.registry_type == entry2.registry_type:
            score += 0.3
        
        # Tag similarity
        common_tags = entry1.tags.intersection(entry2.tags)
        total_tags = entry1.tags.union(entry2.tags)
        if total_tags:
            score += 0.4 * (len(common_tags) / len(total_tags))
        
        # Same author bonus
        if entry1.author == entry2.author:
            score += 0.2
        
        # Rating similarity
        rating_diff = abs(entry1.rating - entry2.rating)
        score += 0.1 * (1 - rating_diff / 4.0)  # Normalize by max rating difference
        
        return score

# Global registry instance
_global_registry = None

def get_global_registry() -> QubotRegistry:
    """Get the global registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = QubotRegistry()
    return _global_registry
