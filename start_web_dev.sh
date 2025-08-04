#!/bin/bash

# Qubots Web Interface Development Setup
# This script starts the web interface in development mode without Docker

set -e

echo "🚀 Starting Qubots Web Interface (Development Mode)"
echo "=================================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ and try again."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm and try again."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "web_interface/package.json" ]; then
    echo "❌ web_interface/package.json not found. Please run this script from the qubots root directory."
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "web_interface/node_modules" ]; then
    echo "📦 Installing web interface dependencies..."
    cd web_interface
    npm install
    cd ..
    echo "✅ Dependencies installed!"
fi

# Start the API backend in the background
echo "🔧 Starting API backend..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Python is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Install API dependencies
if [ ! -f "api/.venv/bin/activate" ]; then
    echo "📦 Setting up API virtual environment..."
    cd api
    $PYTHON_CMD -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cd ..
    echo "✅ API environment ready!"
fi

# Start API in background
echo "🚀 Starting API server..."
cd api
source .venv/bin/activate
$PYTHON_CMD main.py &
API_PID=$!
cd ..

# Wait for API to be ready
echo "⏳ Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  API is taking longer than expected to start"
        break
    fi
    sleep 1
done

# Start the web interface
echo "🌐 Starting web interface..."
cd web_interface

# Set environment variables
export REACT_APP_GITEA_URL=http://localhost:3000
export REACT_APP_API_URL=http://localhost:8000

# Start React development server
npm start &
WEB_PID=$!

# Wait for web interface to be ready
echo "⏳ Waiting for web interface to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ Web interface is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "⚠️  Web interface is taking longer than expected to start"
        break
    fi
    sleep 2
done

cd ..

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
    fi
    if [ ! -z "$WEB_PID" ]; then
        kill $WEB_PID 2>/dev/null || true
    fi
    echo "✅ Services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

echo ""
echo "🎉 Qubots Web Interface is ready!"
echo "================================="
echo ""
echo "🌐 Access Points:"
echo "• Web Interface:  http://localhost:3000"
echo "• API:            http://localhost:8000"
echo "• Gitea:          http://localhost:3000 (if running separately)"
echo ""
echo "📋 Development Features:"
echo "• Hot reload for React components"
echo "• API auto-restart on changes"
echo "• Real-time error reporting"
echo "• Development tools enabled"
echo ""
echo "🛠️  Available Features:"
echo "• Drag-and-drop workflow designer"
echo "• Component library browser"
echo "• Parameter configuration panels"
echo "• Real-time code generation"
echo "• JSON workflow export"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Keep the script running
wait
