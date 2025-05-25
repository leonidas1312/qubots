# Rastion Platform Integration Guide

This comprehensive guide covers everything you need to know about integrating qubots with the Rastion platform for collaborative optimization development, sharing, and deployment.

## 🎯 Overview

The Rastion platform provides a cloud-based repository system for optimization models, similar to Hugging Face for machine learning models. It enables:

- **Model Sharing**: Upload and share optimization models with the community
- **Model Discovery**: Search and discover models created by others
- **Collaborative Development**: Work together on optimization projects
- **Version Management**: Track changes and manage model versions
- **Playground Integration**: Interactive testing and experimentation

## 🚀 Getting Started

### 1. Account Setup

1. **Visit the Platform**: Go to [https://rastion.com](https://rastion.com)
2. **Create Account**: Sign up for a free account
3. **Generate Token**: Go to Profile Settings > Applications > Generate Token
4. **Save Token**: Store your token securely (never commit to version control)

### 2. Authentication

```python
import qubots.rastion as rastion

# Authenticate with your token
rastion.authenticate("your_gitea_token_here")

# Verify authentication
try:
    user_info = rastion.get_user_info()
    print(f"✅ Authenticated as: {user_info['username']}")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
```

### 3. Environment Setup

For security, use environment variables:

```bash
# Set environment variable
export RASTION_TOKEN="your_token_here"

# Or create .env file
echo "RASTION_TOKEN=your_token_here" > .env
```

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Authenticate using environment variable
token = os.getenv('RASTION_TOKEN')
if token:
    rastion.authenticate(token)
else:
    print("⚠️ RASTION_TOKEN not found in environment")
```

## 📤 Uploading Models

### Basic Upload

```python
from qubots import BaseOptimizer, OptimizerMetadata
import qubots.rastion as rastion

class MyOptimizer(BaseOptimizer):
    def __init__(self):
        metadata = OptimizerMetadata(
            name="My Custom Optimizer",
            description="A novel optimization algorithm",
            author="Your Name",
            version="1.0.0"
        )
        super().__init__(metadata)
    
    def _optimize_implementation(self, problem, initial_solution=None):
        # Your optimization logic
        pass

# Upload the optimizer
optimizer = MyOptimizer()
url = rastion.upload_model(
    model=optimizer,
    name="my_custom_optimizer",
    description="A novel optimization algorithm for routing problems",
    requirements=["numpy", "scipy", "qubots"]
)

print(f"✅ Model uploaded: {url}")
```

### Advanced Upload Options

```python
# Upload with detailed configuration
url = rastion.upload_model(
    model=optimizer,
    name="advanced_genetic_algorithm",
    description="Advanced genetic algorithm with adaptive parameters",
    requirements=["numpy>=1.20.0", "scipy>=1.7.0", "qubots>=1.0.0"],
    tags=["genetic", "metaheuristic", "adaptive"],
    category="evolutionary",
    private=False,  # Public model
    license="MIT",
    documentation="docs/my_algorithm.md",
    examples=["examples/usage_example.py"],
    benchmark_results="benchmarks/performance_report.json"
)
```

### Upload Validation

```python
# Validate model before upload
try:
    validation_result = rastion.validate_model(optimizer)
    if validation_result.is_valid:
        print("✅ Model validation passed")
        url = rastion.upload_model(optimizer, name="validated_optimizer")
    else:
        print(f"❌ Validation failed: {validation_result.errors}")
except Exception as e:
    print(f"❌ Validation error: {e}")
```

## 📥 Loading Models

### Basic Loading

```python
# Load any public model
problem = rastion.load_qubots_model("traveling_salesman_problem")
optimizer = rastion.load_qubots_model("genetic_algorithm_tsp")

# Load with specific user
model = rastion.load_qubots_model("custom_optimizer", username="researcher123")

# Load specific version
model = rastion.load_qubots_model("my_model", revision="v1.2.0")
```

### Advanced Loading Options

```python
# Load with custom parameters
optimizer = rastion.load_qubots_model(
    "genetic_algorithm",
    override_params={
        "population_size": 200,
        "max_generations": 500,
        "mutation_rate": 0.05
    }
)

# Load with caching
model = rastion.load_qubots_model(
    "large_model",
    cache=True,  # Cache locally for faster subsequent loads
    cache_dir="./model_cache"
)

# Load with dependency management
model = rastion.load_qubots_model(
    "complex_model",
    install_requirements=True,  # Automatically install dependencies
    virtual_env="rastion_env"   # Use specific virtual environment
)
```

### Error Handling

```python
try:
    model = rastion.load_qubots_model("nonexistent_model")
except rastion.ModelNotFoundError as e:
    print(f"Model not found: {e}")
except rastion.AuthenticationError as e:
    print(f"Authentication required: {e}")
except rastion.DependencyError as e:
    print(f"Missing dependencies: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 🔍 Model Discovery

### Search Models

```python
# Search by keyword
genetic_models = rastion.search_models("genetic algorithm")
print(f"Found {len(genetic_models)} genetic algorithm models")

# Search by category
routing_models = rastion.search_models(category="routing")

# Search by tags
metaheuristic_models = rastion.search_models(tags=["metaheuristic", "optimization"])

# Advanced search
results = rastion.search_models(
    query="portfolio optimization",
    category="finance",
    tags=["risk", "return"],
    author="expert_user",
    min_rating=4.0,
    sort_by="popularity",
    limit=20
)
```

### Browse Models

```python
# List all available models
all_models = rastion.discover_models()

# List models by category
finance_models = rastion.discover_models(category="finance")

# List trending models
trending = rastion.get_trending_models(period="week")

# List your models
my_models = rastion.list_my_models()

# List models you've starred
starred = rastion.list_starred_models()
```

### Model Information

```python
# Get detailed model information
model_info = rastion.get_model_info("username/model_name")
print(f"Name: {model_info['name']}")
print(f"Description: {model_info['description']}")
print(f"Author: {model_info['author']}")
print(f"Downloads: {model_info['download_count']}")
print(f"Rating: {model_info['rating']}")
print(f"Last Updated: {model_info['updated_at']}")

# Get model statistics
stats = rastion.get_model_stats("username/model_name")
print(f"Total downloads: {stats['downloads']}")
print(f"Weekly downloads: {stats['weekly_downloads']}")
print(f"Stars: {stats['stars']}")
print(f"Forks: {stats['forks']}")
```

## 🎮 Playground Integration

### Execute in Playground

```python
# Execute optimization in Rastion playground
result = rastion.execute_playground_optimization(
    problem_name="traveling_salesman_problem",
    optimizer_name="genetic_algorithm_tsp",
    parameters={
        "population_size": 100,
        "max_generations": 200
    }
)

print(f"Playground result: {result.best_value}")
print(f"Execution time: {result.runtime_seconds}")
```

### Share Playground Sessions

```python
# Create shareable playground session
session = rastion.create_playground_session(
    problem="my_problem",
    optimizer="my_optimizer",
    parameters={"param1": "value1"},
    description="Demonstration of my optimization approach"
)

print(f"Shareable URL: {session.url}")
```

### Playground Collaboration

```python
# Invite collaborators to playground session
rastion.invite_to_session(
    session_id="session_123",
    collaborators=["user1", "user2"],
    permissions=["view", "edit", "run"]
)

# Fork existing session
new_session = rastion.fork_session(
    original_session_id="session_123",
    new_name="My Modified Approach"
)
```

## 🔧 Advanced Features

### Model Collections

```python
# Create model collection
collection = rastion.create_collection(
    name="Routing Algorithms",
    description="Collection of vehicle routing optimizers",
    models=["vrp_genetic", "vrp_simulated_annealing", "vrp_tabu_search"]
)

# Add models to collection
rastion.add_to_collection(collection.id, ["new_vrp_model"])

# Share collection
rastion.share_collection(collection.id, public=True)
```

### Model Versioning

```python
# Create new version
new_version = rastion.create_version(
    model_name="my_optimizer",
    version="v2.0.0",
    changes="Added multi-objective support",
    breaking_changes=True
)

# Compare versions
comparison = rastion.compare_versions(
    model_name="my_optimizer",
    version1="v1.0.0",
    version2="v2.0.0"
)

# Rollback to previous version
rastion.rollback_version(
    model_name="my_optimizer",
    target_version="v1.5.0"
)
```

### Collaboration Features

```python
# Add collaborators to model
rastion.add_collaborator(
    model_name="my_optimizer",
    username="colleague",
    permissions=["read", "write"]
)

# Create pull request for model changes
pr = rastion.create_pull_request(
    model_name="community_optimizer",
    title="Improve convergence speed",
    description="Optimized selection mechanism",
    changes=["optimizer.py", "README.md"]
)

# Review and merge pull request
rastion.review_pull_request(pr.id, approved=True, comments="Looks good!")
rastion.merge_pull_request(pr.id)
```

## 📊 Analytics and Monitoring

### Usage Analytics

```python
# Get model usage analytics
analytics = rastion.get_model_analytics("my_optimizer")
print(f"Total runs: {analytics['total_runs']}")
print(f"Success rate: {analytics['success_rate']}")
print(f"Average runtime: {analytics['avg_runtime']}")

# Get user analytics
user_analytics = rastion.get_user_analytics()
print(f"Models created: {user_analytics['models_created']}")
print(f"Total downloads: {user_analytics['total_downloads']}")
print(f"Community rating: {user_analytics['rating']}")
```

### Performance Monitoring

```python
# Monitor model performance
performance = rastion.monitor_model_performance(
    model_name="my_optimizer",
    metrics=["runtime", "success_rate", "solution_quality"],
    time_period="30d"
)

# Set up alerts
rastion.create_alert(
    model_name="my_optimizer",
    condition="success_rate < 0.95",
    notification_method="email"
)
```

## 🔒 Security and Privacy

### Private Models

```python
# Upload private model
url = rastion.upload_model(
    model=optimizer,
    name="proprietary_algorithm",
    description="Internal optimization algorithm",
    private=True,  # Only you can access
    organization="my_company"  # Share with organization
)

# Manage access permissions
rastion.set_model_permissions(
    model_name="proprietary_algorithm",
    permissions={
        "user1": ["read"],
        "user2": ["read", "write"],
        "team_lead": ["read", "write", "admin"]
    }
)
```

### API Keys and Tokens

```python
# Generate API key for programmatic access
api_key = rastion.generate_api_key(
    name="production_deployment",
    permissions=["model_read", "model_execute"],
    expires_in="90d"
)

# Revoke API key
rastion.revoke_api_key(api_key.id)

# List active API keys
keys = rastion.list_api_keys()
```

## 🛠️ Best Practices

### 1. Model Documentation

```python
# Include comprehensive documentation
url = rastion.upload_model(
    model=optimizer,
    name="well_documented_optimizer",
    description="Clear, concise description",
    documentation={
        "readme": "README.md",
        "api_docs": "docs/api.md",
        "examples": ["examples/basic_usage.py", "examples/advanced_usage.py"],
        "changelog": "CHANGELOG.md"
    }
)
```

### 2. Testing and Validation

```python
# Include test suite
url = rastion.upload_model(
    model=optimizer,
    name="tested_optimizer",
    test_suite={
        "unit_tests": "tests/test_optimizer.py",
        "integration_tests": "tests/test_integration.py",
        "benchmark_tests": "tests/test_benchmarks.py"
    },
    ci_config=".github/workflows/test.yml"
)
```

### 3. Dependency Management

```python
# Specify exact dependencies
requirements = [
    "numpy==1.21.0",
    "scipy==1.7.0",
    "qubots>=1.0.0,<2.0.0"
]

url = rastion.upload_model(
    model=optimizer,
    requirements=requirements,
    python_version=">=3.8,<3.12"
)
```

### 4. Performance Optimization

```python
# Optimize for platform execution
class PlatformOptimizedOptimizer(BaseOptimizer):
    def __init__(self, **params):
        # Use platform-specific optimizations
        if rastion.is_platform_execution():
            params['use_gpu'] = rastion.gpu_available()
            params['max_memory'] = rastion.get_memory_limit()
        
        super().__init__(metadata, **params)
```

## 🚨 Troubleshooting

### Common Issues

1. **Authentication Errors**
```python
# Check token validity
if not rastion.is_token_valid():
    print("Token expired or invalid")
    rastion.refresh_token()
```

2. **Upload Failures**
```python
# Check model size and format
model_size = rastion.get_model_size(optimizer)
if model_size > rastion.MAX_MODEL_SIZE:
    print("Model too large, consider compression")

# Validate model format
if not rastion.is_valid_model_format(optimizer):
    print("Invalid model format")
```

3. **Dependency Issues**
```python
# Check dependency compatibility
conflicts = rastion.check_dependency_conflicts(requirements)
if conflicts:
    print(f"Dependency conflicts: {conflicts}")
```

### Getting Help

- **Documentation**: [https://rastion.com/docs](https://rastion.com/docs)
- **Community Forum**: [https://rastion.com/community](https://rastion.com/community)
- **Support**: [support@rastion.com](mailto:support@rastion.com)
- **GitHub Issues**: [https://github.com/Rastion/qubots/issues](https://github.com/Rastion/qubots/issues)

## 🔗 Integration Examples

Check out these complete integration examples:

- **[Fantasy Football Integration](../tutorials/fantasy_football.md)**: Complete 3-file structure
- **[Routing Optimization](../tutorials/routing_optimization.md)**: Vehicle routing with Rastion
- **[Financial Optimization](../tutorials/finance_optimization.md)**: Portfolio optimization workflow

---

The Rastion platform integration makes qubots a powerful collaborative optimization framework. Start with simple uploads and gradually explore advanced features as your needs grow! 🚀
