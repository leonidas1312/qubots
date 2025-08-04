# Qubots Local Development Setup

This guide helps you set up a fully local development environment for qubots, eliminating the need for cloud services during development.

## 🎯 What You Get

- **Local Gitea Instance**: Self-hosted Git repositories at `http://localhost:3000`
- **No Cloud Dependencies**: Develop entirely offline
- **Same API**: Identical interface to the cloud version
- **Easy Migration**: Switch between local and cloud profiles seamlessly

## 📋 Prerequisites

- **Docker & Docker Compose**: For running Gitea
- **Python 3.8+**: For qubots framework
- **Git**: For repository operations

### Install Prerequisites

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install docker.io docker-compose git python3 python3-pip
sudo usermod -aG docker $USER  # Add user to docker group
# Log out and back in for group changes to take effect
```

**macOS:**
```bash
brew install docker docker-compose git python3
# Start Docker Desktop application
```

**Windows:**
- Install Docker Desktop from https://docker.com
- Install Git from https://git-scm.com
- Install Python from https://python.org

## 🚀 Quick Setup

### 1. Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/leonidas1312/qubots.git
cd qubots

# Run the setup script
python setup_local.py
```

This will:
- Start local Gitea instance
- Configure qubots for local mode
- Create example repositories
- Show next steps

### 2. Manual Setup

If you prefer manual setup:

```bash
# Start Gitea
docker-compose up -d gitea

# Install qubots in development mode
pip install -e .

# Set local profile
python -c "from qubots import set_profile; set_profile('local')"
```

## ⚙️ Configuration

### Complete Gitea Setup

1. Open http://localhost:3000 in your browser
2. Complete the initial setup form:
   - **Database Type**: SQLite3 (default)
   - **Application Name**: Qubots Local
   - **Repository Root Path**: `/data/git/repositories`
   - **Git LFS Root Path**: `/data/git/lfs`
   - **Domain**: `localhost`
   - **SSH Port**: `2222`
   - **HTTP Port**: `3000`
   - **Application URL**: `http://localhost:3000/`

3. Create an administrator account
4. Complete the setup

### Authenticate qubots

```bash
# Using the CLI tool
python -m qubots.cli auth --username YOUR_USERNAME

# Or programmatically
python -c "
from qubots import get_local_gitea_client
client = get_local_gitea_client()
client.authenticate('username', 'password')
"
```

## 🧪 Testing the Setup

### Test with Example Repository

```python
from qubots import AutoProblem

# Load the example problem (created by setup)
problem = AutoProblem.from_repo("examples/local_test_problem")

# Test it
solution = problem.random_solution()
score = problem.evaluate_solution(solution)
print(f"Solution: {solution}")
print(f"Score: {score}")
```

### Create Your First Repository

```python
from qubots import get_local_gitea_client

client = get_local_gitea_client()

# Create a new repository
client.create_repository(
    name="my-first-problem",
    description="My first optimization problem",
    private=False
)
```

## 🛠️ CLI Commands

The qubots CLI provides convenient commands for managing your local setup:

```bash
# Check system status
python -m qubots.cli status

# List repositories
python -m qubots.cli repo list

# Create a repository
python -m qubots.cli repo create --name my-problem --description "My problem"

# Manage profiles
python -m qubots.cli profile list
python -m qubots.cli profile set local
```

## 🔄 Profile Management

Qubots supports multiple profiles for different environments:

```python
from qubots import get_config, set_profile

# Switch to local development
set_profile("local")

# Switch back to cloud (if configured)
set_profile("rastion")

# Check current profile
config = get_config()
print(f"Active profile: {config.get_active_profile()}")
```

## 📁 Repository Structure

Each qubots repository needs these files:

```
my-problem/
├── qubot.py          # Main implementation
├── config.json       # Configuration and metadata
├── requirements.txt  # Python dependencies
└── README.md        # Documentation (optional)
```

### Example config.json

```json
{
  "type": "problem",
  "entry_point": "qubot",
  "class_name": "MyProblem",
  "default_params": {
    "size": 100
  },
  "metadata": {
    "name": "My Optimization Problem",
    "description": "A custom optimization problem",
    "domain": "logistics",
    "tags": ["optimization", "custom"],
    "difficulty": "intermediate",
    "problem_type": "discrete"
  }
}
```

## 🐳 Docker Services

The `docker-compose.yml` includes:

- **gitea**: Git repository hosting (port 3000)
- **qubots-web**: Web interface (port 3001, optional)
- **qubots-api**: REST API (port 8000, optional)

Start specific services:
```bash
# Just Gitea (minimal setup)
docker-compose up -d gitea

# With web interface
docker-compose --profile web-interface up -d

# Full stack
docker-compose --profile web-interface --profile api up -d
```

## 🔧 Troubleshooting

### Gitea Not Starting
```bash
# Check Docker status
docker ps

# View Gitea logs
docker-compose logs gitea

# Restart Gitea
docker-compose restart gitea
```

### Authentication Issues
```bash
# Check if Gitea is accessible
curl http://localhost:3000/api/healthz

# Reset authentication
python -c "
from qubots import get_config
config = get_config()
config.set_auth_token(None)
"
```

### Repository Clone Issues
```bash
# Check git configuration
git config --list

# Test manual clone
git clone http://localhost:3000/username/repo.git
```

## 🚀 Next Steps

1. **Create Your First Problem**: Follow the qubots documentation to create optimization problems
2. **Build Optimizers**: Implement solvers for your problems
3. **Web Interface**: Set up the visual workflow designer (coming soon)
4. **Share**: Export your work or migrate to cloud when ready

## 📚 Additional Resources

- [Qubots Documentation](https://docs.rastion.com)
- [Gitea Documentation](https://docs.gitea.io)
- [Docker Compose Reference](https://docs.docker.com/compose/)

## 🆘 Getting Help

- **Issues**: Create issues on the GitHub repository
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the official docs at docs.rastion.com
