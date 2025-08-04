#!/bin/bash

# Qubots Web Interface Startup Script
# This script starts the full web interface with visual workflow designer

set -e

echo "🚀 Starting Qubots Web Interface"
echo "================================="

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

# Start all services for web interface
echo "🐳 Starting all services..."
docker-compose --profile web-interface --profile api up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."

# Wait for Gitea
echo "  Checking Gitea..."
for i in {1..30}; do
    if curl -s http://localhost:3000/api/healthz > /dev/null 2>&1; then
        echo "  ✅ Gitea is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  ⚠️  Gitea is taking longer than expected to start"
        break
    fi
    sleep 1
done

# Wait for API
echo "  Checking API..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✅ API is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  ⚠️  API is taking longer than expected to start"
        break
    fi
    sleep 1
done

# Wait for Web Interface
echo "  Checking Web Interface..."
for i in {1..60}; do
    if curl -s http://localhost:3001 > /dev/null 2>&1; then
        echo "  ✅ Web Interface is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "  ⚠️  Web Interface is taking longer than expected to start"
        break
    fi
    sleep 2
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
echo "🎉 Qubots Web Interface is ready!"
echo "================================="
echo ""
echo "🌐 Access Points:"
echo "• Web Interface:  http://localhost:3001"
echo "• Gitea:          http://localhost:3000"
echo "• API:            http://localhost:8000"
echo ""
echo "📋 Next Steps:"
echo "1. Open http://localhost:3001 to access the visual workflow designer"
echo "2. Complete Gitea setup at http://localhost:3000 (if not done already)"
echo "3. Create a user account in Gitea"
echo "4. Authenticate: python -m qubots.cli auth"
echo ""
echo "🛠️  Features Available:"
echo "• Drag-and-drop workflow designer"
echo "• Component library browser"
echo "• Parameter configuration panels"
echo "• Real-time code generation"
echo "• JSON workflow export"
echo ""
echo "To stop all services: docker-compose --profile web-interface --profile api down"
