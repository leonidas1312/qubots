# Changelog

All notable changes to the Qubots project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-XX - PyPI Production Release

### Added
- **Repository Reorganization for PyPI**: Complete restructuring for production deployment
  - Organized examples by domain (routing, scheduling, logistics, finance, energy, fantasy_football)
  - Created comprehensive documentation structure with tutorials and guides
  - Added complete test suite with unit, integration, and benchmark tests
  - Separated basic examples from domain-specific examples

- **Enhanced Documentation**:
  - Complete rewrite of main README.md with installation options and quick start
  - Step-by-step tutorials for getting started and creating custom optimizers
  - Domain-specific tutorials for all supported optimization areas
  - Comprehensive Rastion platform integration guide
  - Fantasy football 3-file structure tutorial and examples

- **Fantasy Football Optimization**:
  - Complete 3-file structure implementation (local_testing.py, optimizer_only.py, rastion_integration.py)
  - Production-ready fantasy football genetic algorithm optimizer
  - Rastion platform integration for fantasy sports optimization
  - Comprehensive tutorial with educational content

- **Testing Infrastructure**:
  - Unit tests for all core qubots functionality
  - Integration tests for Rastion platform
  - Benchmark tests for performance validation
  - Example validation tests to ensure all examples work
  - Continuous integration configuration

- **PyPI Configuration**:
  - Optional dependencies for domain-specific features (routing, continuous, finance, energy, fantasy_football)
  - Development dependencies for testing and code quality
  - Proper package metadata and classifiers
  - Tool configuration for pytest, black, isort, mypy

- **Production-ready release** of Qubots optimization framework
- Comprehensive integration support for OR-Tools, CasADi, Feloopy, and GAMSPY
- Enhanced package configuration with proper dependencies and metadata
- Domain-specific optimization classes for routing, scheduling, logistics, finance, and energy
- Advanced benchmarking and evaluation system
- Registry and discovery system for collaborative optimization
- Type hints support with py.typed marker

### Changed
- Updated package version from 0.1.6 to 1.0.0
- Improved README.md with production-ready examples and integration guides
- Enhanced pyproject.toml with comprehensive metadata and optional dependencies
- Cleaned up codebase by removing development artifacts

### Removed
- Development artifacts (__pycache__, .egg-info directories)
- Temporary development files (ENHANCED_FEATURES.md, remote.py)
- Fantasy football example directory (moved to separate examples)
- Root-level requirements.txt (dependencies now in pyproject.toml)

### Fixed
- Variable name consistency in README examples
- Import structure and module organization
- Package build configuration for PyPI distribution

### Security
- Improved dependency version constraints
- Removed development-only code from production package

## [0.1.6] - Previous Development Version

### Added
- Initial implementation of qubots framework
- Basic problem and optimizer interfaces
- Auto-loading functionality from GitHub repositories
- Registry system for qubot discovery
- Specialized problem and optimizer classes
- Benchmarking capabilities

---

For more details about each release, visit the [GitHub releases page](https://github.com/Rastion/qubots/releases).
