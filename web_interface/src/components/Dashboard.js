import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Workflow, 
  Library, 
  Play, 
  Clock, 
  TrendingUp,
  GitBranch,
  Database,
  Brain,
  Plus,
  ArrowRight
} from 'lucide-react';

const Dashboard = () => {
  const [stats, setStats] = useState({
    workflows: 0,
    components: 0,
    executions: 0,
    avgRuntime: 0
  });

  const [recentWorkflows, setRecentWorkflows] = useState([]);
  const [recentComponents, setRecentComponents] = useState([]);

  useEffect(() => {
    // Mock data - replace with API calls
    setStats({
      workflows: 5,
      components: 12,
      executions: 23,
      avgRuntime: 2.4
    });

    setRecentWorkflows([
      {
        id: 1,
        name: 'Portfolio Optimization',
        description: 'Risk-minimizing portfolio allocation',
        lastModified: '2 hours ago',
        status: 'completed'
      },
      {
        id: 2,
        name: 'Supply Chain Network',
        description: 'Multi-facility logistics optimization',
        lastModified: '1 day ago',
        status: 'running'
      },
      {
        id: 3,
        name: 'Vehicle Routing',
        description: 'TSP with time windows',
        lastModified: '3 days ago',
        status: 'draft'
      }
    ]);

    setRecentComponents([
      {
        id: 1,
        name: 'Genetic Algorithm',
        type: 'optimizer',
        domain: 'metaheuristic',
        lastUsed: '1 hour ago'
      },
      {
        id: 2,
        name: 'Portfolio Problem',
        type: 'problem',
        domain: 'finance',
        lastUsed: '2 hours ago'
      },
      {
        id: 3,
        name: 'OR-Tools Solver',
        type: 'optimizer',
        domain: 'exact',
        lastUsed: '1 day ago'
      }
    ]);
  }, []);

  const StatCard = ({ icon: Icon, title, value, subtitle, color = 'primary' }) => (
    <div className="card">
      <div className="flex items-center">
        <div className={`p-3 rounded-lg bg-${color}-100`}>
          <Icon className={`h-6 w-6 text-${color}-600`} />
        </div>
        <div className="ml-4">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}
        </div>
      </div>
    </div>
  );

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'running': return 'text-blue-600 bg-blue-100';
      case 'draft': return 'text-gray-600 bg-gray-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case 'problem': return Database;
      case 'optimizer': return Brain;
      default: return Library;
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-2">
          Welcome to Qubots - your optimization workflow platform
        </p>
      </div>

      {/* Quick Actions */}
      <div className="mb-8">
        <div className="flex flex-wrap gap-4">
          <Link
            to="/designer"
            className="btn btn-primary flex items-center space-x-2"
          >
            <Plus className="h-4 w-4" />
            <span>New Workflow</span>
          </Link>
          <Link
            to="/library"
            className="btn btn-outline flex items-center space-x-2"
          >
            <Library className="h-4 w-4" />
            <span>Browse Components</span>
          </Link>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          icon={Workflow}
          title="Workflows"
          value={stats.workflows}
          subtitle="Total created"
          color="primary"
        />
        <StatCard
          icon={Library}
          title="Components"
          value={stats.components}
          subtitle="Available"
          color="green"
        />
        <StatCard
          icon={Play}
          title="Executions"
          value={stats.executions}
          subtitle="This month"
          color="blue"
        />
        <StatCard
          icon={Clock}
          title="Avg Runtime"
          value={`${stats.avgRuntime}s`}
          subtitle="Per execution"
          color="purple"
        />
      </div>

      {/* Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Recent Workflows */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Recent Workflows</h2>
            <Link
              to="/designer"
              className="text-primary-600 hover:text-primary-700 text-sm font-medium flex items-center"
            >
              View all
              <ArrowRight className="h-4 w-4 ml-1" />
            </Link>
          </div>
          
          <div className="space-y-4">
            {recentWorkflows.map((workflow) => (
              <div
                key={workflow.id}
                className="flex items-center justify-between p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer"
              >
                <div className="flex-1">
                  <h3 className="font-medium text-gray-900">{workflow.name}</h3>
                  <p className="text-sm text-gray-600">{workflow.description}</p>
                  <p className="text-xs text-gray-500 mt-1">{workflow.lastModified}</p>
                </div>
                <div className="ml-4">
                  <span className={`status-indicator ${getStatusColor(workflow.status)}`}>
                    {workflow.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
          
          {recentWorkflows.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <Workflow className="h-8 w-8 mx-auto mb-2 text-gray-400" />
              <p>No workflows yet</p>
              <Link to="/designer" className="text-primary-600 hover:text-primary-700 text-sm">
                Create your first workflow
              </Link>
            </div>
          )}
        </div>

        {/* Recent Components */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Recent Components</h2>
            <Link
              to="/library"
              className="text-primary-600 hover:text-primary-700 text-sm font-medium flex items-center"
            >
              Browse all
              <ArrowRight className="h-4 w-4 ml-1" />
            </Link>
          </div>
          
          <div className="space-y-4">
            {recentComponents.map((component) => {
              const Icon = getTypeIcon(component.type);
              return (
                <div
                  key={component.id}
                  className="flex items-center p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer"
                >
                  <div className="p-2 rounded-lg bg-gray-100 mr-3">
                    <Icon className="h-4 w-4 text-gray-600" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-medium text-gray-900">{component.name}</h3>
                    <p className="text-sm text-gray-600 capitalize">
                      {component.type} • {component.domain}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Used {component.lastUsed}</p>
                  </div>
                </div>
              );
            })}
          </div>
          
          {recentComponents.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <Library className="h-8 w-8 mx-auto mb-2 text-gray-400" />
              <p>No components used yet</p>
              <Link to="/library" className="text-primary-600 hover:text-primary-700 text-sm">
                Explore the library
              </Link>
            </div>
          )}
        </div>
      </div>

      {/* System Status */}
      <div className="mt-8">
        <div className="card">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <GitBranch className="h-5 w-5 text-gray-600" />
              <div>
                <h3 className="font-medium text-gray-900">System Status</h3>
                <p className="text-sm text-gray-600">Local development environment</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm text-gray-600">Gitea Connected</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm text-gray-600">Local Mode</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
