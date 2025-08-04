import React, { useState, useEffect } from 'react';
import { 
  Save, 
  RefreshCw, 
  GitBranch, 
  Server, 
  Key, 
  Database,
  Settings as SettingsIcon,
  CheckCircle,
  AlertCircle,
  Info
} from 'lucide-react';
import { toast } from 'react-hot-toast';

const Settings = () => {
  const [settings, setSettings] = useState({
    profile: 'local',
    giteaUrl: 'http://localhost:3000',
    apiUrl: 'http://localhost:8000',
    cacheDir: '~/.cache/qubots_hub',
    autoSave: true,
    darkMode: false,
    notifications: true
  });

  const [connectionStatus, setConnectionStatus] = useState({
    gitea: 'checking',
    api: 'checking'
  });

  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Load settings from localStorage or API
    const savedSettings = localStorage.getItem('qubots-settings');
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings));
    }

    // Check connection status
    checkConnections();
  }, []);

  const checkConnections = async () => {
    setConnectionStatus({ gitea: 'checking', api: 'checking' });

    // Check Gitea connection
    try {
      const giteaResponse = await fetch(`${settings.giteaUrl}/api/healthz`);
      setConnectionStatus(prev => ({
        ...prev,
        gitea: giteaResponse.ok ? 'connected' : 'error'
      }));
    } catch (error) {
      setConnectionStatus(prev => ({ ...prev, gitea: 'error' }));
    }

    // Check API connection
    try {
      const apiResponse = await fetch(`${settings.apiUrl}/health`);
      setConnectionStatus(prev => ({
        ...prev,
        api: apiResponse.ok ? 'connected' : 'error'
      }));
    } catch (error) {
      setConnectionStatus(prev => ({ ...prev, api: 'error' }));
    }
  };

  const handleSettingChange = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleSave = async () => {
    setLoading(true);
    try {
      // Save to localStorage
      localStorage.setItem('qubots-settings', JSON.stringify(settings));
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      toast.success('Settings saved successfully');
      
      // Recheck connections if URLs changed
      await checkConnections();
    } catch (error) {
      toast.error('Failed to save settings');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    if (window.confirm('Reset all settings to defaults?')) {
      const defaultSettings = {
        profile: 'local',
        giteaUrl: 'http://localhost:3000',
        apiUrl: 'http://localhost:8000',
        cacheDir: '~/.cache/qubots_hub',
        autoSave: true,
        darkMode: false,
        notifications: true
      };
      setSettings(defaultSettings);
      toast.success('Settings reset to defaults');
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'connected':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'checking':
        return <RefreshCw className="h-4 w-4 text-gray-400 animate-spin" />;
      default:
        return <Info className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'connected': return 'Connected';
      case 'error': return 'Connection failed';
      case 'checking': return 'Checking...';
      default: return 'Unknown';
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-600 mt-2">
          Configure your Qubots environment and preferences
        </p>
      </div>

      <div className="space-y-8">
        {/* Connection Settings */}
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <Server className="h-5 w-5 text-gray-600" />
            <h2 className="text-lg font-semibold text-gray-900">Connection Settings</h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="parameter-label">Profile</label>
              <select
                value={settings.profile}
                onChange={(e) => handleSettingChange('profile', e.target.value)}
                className="input"
              >
                <option value="local">Local Development</option>
                <option value="rastion">Rastion Cloud</option>
              </select>
              <p className="parameter-description">
                Choose between local development or cloud-hosted environment
              </p>
            </div>

            <div>
              <label className="parameter-label">Gitea URL</label>
              <div className="flex space-x-2">
                <input
                  type="url"
                  value={settings.giteaUrl}
                  onChange={(e) => handleSettingChange('giteaUrl', e.target.value)}
                  className="input flex-1"
                  placeholder="http://localhost:3000"
                />
                <div className="flex items-center space-x-2 px-3 py-2 bg-gray-50 rounded-lg">
                  {getStatusIcon(connectionStatus.gitea)}
                  <span className="text-sm text-gray-600">
                    {getStatusText(connectionStatus.gitea)}
                  </span>
                </div>
              </div>
              <p className="parameter-description">
                URL for your Gitea instance (repository hosting)
              </p>
            </div>

            <div>
              <label className="parameter-label">API URL</label>
              <div className="flex space-x-2">
                <input
                  type="url"
                  value={settings.apiUrl}
                  onChange={(e) => handleSettingChange('apiUrl', e.target.value)}
                  className="input flex-1"
                  placeholder="http://localhost:8000"
                />
                <div className="flex items-center space-x-2 px-3 py-2 bg-gray-50 rounded-lg">
                  {getStatusIcon(connectionStatus.api)}
                  <span className="text-sm text-gray-600">
                    {getStatusText(connectionStatus.api)}
                  </span>
                </div>
              </div>
              <p className="parameter-description">
                URL for the Qubots API backend
              </p>
            </div>

            <div className="pt-2">
              <button
                onClick={checkConnections}
                className="btn btn-outline"
                disabled={connectionStatus.gitea === 'checking' || connectionStatus.api === 'checking'}
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Test Connections
              </button>
            </div>
          </div>
        </div>

        {/* Storage Settings */}
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <Database className="h-5 w-5 text-gray-600" />
            <h2 className="text-lg font-semibold text-gray-900">Storage Settings</h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="parameter-label">Cache Directory</label>
              <input
                type="text"
                value={settings.cacheDir}
                onChange={(e) => handleSettingChange('cacheDir', e.target.value)}
                className="input"
                placeholder="~/.cache/qubots_hub"
              />
              <p className="parameter-description">
                Directory where downloaded repositories and cache files are stored
              </p>
            </div>
          </div>
        </div>

        {/* User Preferences */}
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <SettingsIcon className="h-5 w-5 text-gray-600" />
            <h2 className="text-lg font-semibold text-gray-900">User Preferences</h2>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <label className="parameter-label">Auto-save workflows</label>
                <p className="parameter-description">
                  Automatically save workflow changes
                </p>
              </div>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={settings.autoSave}
                  onChange={(e) => handleSettingChange('autoSave', e.target.checked)}
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
              </label>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="parameter-label">Dark mode</label>
                <p className="parameter-description">
                  Use dark theme for the interface
                </p>
              </div>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={settings.darkMode}
                  onChange={(e) => handleSettingChange('darkMode', e.target.checked)}
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
              </label>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="parameter-label">Notifications</label>
                <p className="parameter-description">
                  Show system notifications and alerts
                </p>
              </div>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={settings.notifications}
                  onChange={(e) => handleSettingChange('notifications', e.target.checked)}
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
              </label>
            </div>
          </div>
        </div>

        {/* Authentication */}
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <Key className="h-5 w-5 text-gray-600" />
            <h2 className="text-lg font-semibold text-gray-900">Authentication</h2>
          </div>

          <div className="space-y-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <GitBranch className="h-4 w-4 text-gray-600" />
                <span className="text-sm font-medium text-gray-700">Current Status</span>
              </div>
              <div className="text-sm text-gray-600">
                <div>Profile: <span className="font-medium">{settings.profile}</span></div>
                <div>Authentication: <span className="font-medium text-red-600">Not authenticated</span></div>
              </div>
            </div>

            <div className="text-sm text-gray-600">
              <p>To authenticate with your local Gitea instance, use the CLI:</p>
              <code className="block mt-2 p-2 bg-gray-100 rounded text-xs font-mono">
                python -m qubots.cli auth --username YOUR_USERNAME
              </code>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex justify-between">
          <button
            onClick={handleReset}
            className="btn btn-outline text-red-600 hover:bg-red-50"
          >
            Reset to Defaults
          </button>
          
          <button
            onClick={handleSave}
            disabled={loading}
            className="btn btn-primary"
          >
            {loading ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="h-4 w-4 mr-2" />
                Save Settings
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Settings;
