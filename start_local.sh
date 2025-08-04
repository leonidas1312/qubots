#!/bin/bash

# Qubots Local Development Startup Script
# This script starts the local development environment

set -e

echo "🚀 Starting Qubots Local Development Environment"
echo "================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found. Please run this script from the qubots root directory."
    exit 1
fi

# Start Gitea
echo "🐳 Starting Gitea container..."
docker-compose up -d gitea

# Wait for Gitea to be ready
echo "⏳ Waiting for Gitea to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:3000/api/healthz > /dev/null 2>&1; then
        echo "✅ Gitea is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Gitea is taking longer than expected to start"
        echo "   Check the logs with: docker-compose logs gitea"
        break
    fi
    sleep 1
done

# Set qubots to local mode
echo "⚙️  Configuring qubots for local mode..."
python -c "
try:
    from qubots import set_profile
    set_profile('local')
    print('✅ Qubots configured for local mode')
except Exception as e:
    print(f'⚠️  Could not configure qubots: {e}')
    print('   You may need to install qubots first: pip install -e .')
"

echo ""
echo "🎉 Local development environment is ready!"
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo "1. Open http://localhost:3000 to complete Gitea setup"
echo "2. Create a user account in Gitea"
echo "3. Authenticate qubots: python -m qubots.cli auth"
echo "4. Test the setup: python test_local_setup.py"
echo ""
echo "🛠️  Useful Commands:"
echo "• qubots status           - Check system status"
echo "• qubots repo list        - List repositories"
echo "• qubots profile list     - List configuration profiles"
echo ""
echo "🌐 Gitea Web Interface: http://localhost:3000"
echo "📚 Documentation: https://docs.rastion.com"
echo ""
echo "To stop the environment: docker-compose down"
