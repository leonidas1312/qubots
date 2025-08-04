# Qubots Web Interface

A visual workflow designer for optimization problems - "n8n meets huggingface for the optimization community".

## 🎯 Features

### Visual Workflow Designer
- **Drag-and-drop interface** for building optimization workflows
- **Node-based editor** with problems, optimizers, and data sources
- **Real-time connections** between workflow components
- **Parameter configuration** with dynamic forms
- **Visual feedback** and validation

### Component Library
- **Browse available components** from local Gitea or cloud
- **Search and filter** by type, domain, difficulty
- **Component details** with ratings, downloads, and documentation
- **One-click installation** and integration

### Code Generation
- **Real-time Python code generation** from visual workflows
- **JSON export** for workflow definitions (MCP compatible)
- **Syntax highlighting** and code preview
- **Download workflows** as executable Python scripts

### Dashboard & Management
- **System status** and connection monitoring
- **Recent workflows** and component usage
- **Settings management** for local/cloud profiles
- **Authentication** with local Gitea

## 🚀 Quick Start

### Option 1: Full Docker Setup (Recommended)

```bash
# Start all services (Gitea + API + Web Interface)
./start_web_interface.sh

# Access the web interface
open http://localhost:3001
```

### Option 2: Development Mode

```bash
# Start in development mode (requires Node.js)
./start_web_dev.sh

# Access the web interface
open http://localhost:3000
```

### Option 3: Manual Setup

```bash
# 1. Start Gitea (if not already running)
docker-compose up -d gitea

# 2. Start API backend
cd api
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py

# 3. Start web interface (in another terminal)
cd web_interface
npm install
npm start
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   API Backend   │    │   Local Gitea   │
│   (React App)   │◄──►│   (FastAPI)     │◄──►│  (Repository)   │
│   Port 3001     │    │   Port 8000     │    │   Port 3000     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Flow    │    │   Qubots Core   │    │   Git Repos     │
│   (Workflow)    │    │   (Auto Load)   │    │   (Components)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎨 User Interface

### Dashboard
- **System overview** with statistics and status
- **Recent workflows** and components
- **Quick actions** for creating new workflows
- **Connection status** indicators

### Workflow Designer
- **Component sidebar** with drag-and-drop components
- **Canvas area** with React Flow editor
- **Parameter panel** for configuring selected nodes
- **Toolbar** with save, load, run, and export actions
- **Code preview** modal with syntax highlighting

### Component Library
- **Grid view** of available components
- **Search and filtering** capabilities
- **Component details** with metadata and ratings
- **Installation and management** features

### Settings
- **Profile management** (local vs cloud)
- **Connection configuration** for Gitea and API
- **User preferences** and authentication
- **System status** and diagnostics

## 🔧 Development

### Prerequisites
- **Node.js 16+** for the React frontend
- **Python 3.8+** for the API backend
- **Docker** for containerized deployment

### Project Structure
```
web_interface/
├── public/                 # Static assets
├── src/
│   ├── components/         # React components
│   │   ├── workflow/       # Workflow designer components
│   │   ├── Dashboard.js    # Main dashboard
│   │   ├── ComponentLibrary.js
│   │   ├── Settings.js
│   │   └── Layout.js       # Main layout
│   ├── App.js             # Main app component
│   ├── index.js           # Entry point
│   └── index.css          # Tailwind styles
├── package.json           # Dependencies
├── tailwind.config.js     # Tailwind configuration
└── Dockerfile            # Production build

api/
├── main.py               # FastAPI application
├── requirements.txt      # Python dependencies
└── Dockerfile           # API container
```

### Key Technologies
- **React 18** - Frontend framework
- **React Flow** - Node-based workflow editor
- **Tailwind CSS** - Utility-first styling
- **Lucide React** - Icon library
- **React Router** - Client-side routing
- **React Hook Form** - Form handling
- **React Hot Toast** - Notifications
- **FastAPI** - Python API framework
- **Pydantic** - Data validation

### Development Commands

```bash
# Frontend development
cd web_interface
npm install          # Install dependencies
npm start           # Start development server
npm run build       # Build for production
npm test            # Run tests

# Backend development
cd api
pip install -r requirements.txt
python main.py      # Start API server
uvicorn main:app --reload  # Start with auto-reload
```

## 🎯 Usage Guide

### Creating a Workflow

1. **Open the Workflow Designer**
   - Navigate to `/designer` or click "New Workflow"

2. **Add Components**
   - Drag components from the sidebar to the canvas
   - Choose from problems, optimizers, and data sources

3. **Connect Components**
   - Click and drag between node handles to create connections
   - Ensure proper flow from problems to optimizers

4. **Configure Parameters**
   - Click on nodes to open the parameter panel
   - Adjust settings using the dynamic forms
   - Save changes to update the workflow

5. **Generate Code**
   - Click "Export" to view generated Python code
   - Download as `.py` file or copy to clipboard
   - Export as JSON for workflow sharing

### Component Management

1. **Browse Library**
   - Visit the Component Library page
   - Use search and filters to find components
   - View details, ratings, and documentation

2. **Install Components**
   - Click "Install" on any component
   - Components become available in the workflow designer
   - View installation status and manage updates

### System Configuration

1. **Profile Setup**
   - Go to Settings to configure profiles
   - Switch between local and cloud modes
   - Set up authentication credentials

2. **Connection Testing**
   - Use the connection test features
   - Monitor system status on the dashboard
   - Troubleshoot connectivity issues

## 🔌 API Endpoints

The web interface communicates with the API backend:

```
GET  /api/components              # List available components
GET  /api/components/{id}         # Get component details
POST /api/components/{id}/install # Install component
POST /api/workflows/execute       # Execute workflow
POST /api/workflows/validate      # Validate workflow
POST /api/workflows/generate-code # Generate Python code
GET  /api/system/status          # Get system status
```

## 🎨 Customization

### Themes and Styling
- Built with Tailwind CSS for easy customization
- Color scheme defined in `tailwind.config.js`
- Component styles in `src/index.css`

### Adding New Node Types
1. Create component in `src/components/workflow/`
2. Register in `nodeTypes` object
3. Add to component sidebar data
4. Update code generation logic

### Extending the API
1. Add new endpoints in `api/main.py`
2. Update frontend service calls
3. Add corresponding UI components

## 🚀 Deployment

### Production Docker Build
```bash
# Build and start all services
docker-compose --profile web-interface --profile api up -d

# Or build individually
docker build -t qubots-web ./web_interface
docker build -t qubots-api ./api
```

### Environment Variables
```bash
# Web Interface
REACT_APP_GITEA_URL=http://localhost:3000
REACT_APP_API_URL=http://localhost:8000

# API Backend
GITEA_URL=http://gitea:3000
DATABASE_URL=sqlite:///data/qubots.db
```

## 🐛 Troubleshooting

### Common Issues

**Web interface won't start:**
- Check Node.js version (16+ required)
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`
- Check port 3000/3001 availability

**API connection failed:**
- Verify API is running on port 8000
- Check CORS settings in API
- Ensure qubots package is installed

**Components not loading:**
- Check Gitea connection
- Verify authentication status
- Review API logs for errors

**Workflow execution fails:**
- Validate workflow structure
- Check component parameters
- Review generated code for errors

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true

# Start with verbose output
npm start --verbose
python main.py --debug
```

## 📚 Additional Resources

- [Qubots Documentation](https://docs.rastion.com)
- [React Flow Documentation](https://reactflow.dev)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Tailwind CSS Documentation](https://tailwindcss.com)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
