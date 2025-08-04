import React, { useState, useEffect } from 'react';
import { X, Save, RotateCcw, Info } from 'lucide-react';
import { toast } from 'react-hot-toast';

const ParameterPanel = ({ node, onUpdateParameters, onClose }) => {
  const [parameters, setParameters] = useState({});
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    if (node?.data?.parameters) {
      setParameters(node.data.parameters);
    }
  }, [node]);

  const handleParameterChange = (paramName, value) => {
    const newParameters = { ...parameters, [paramName]: value };
    setParameters(newParameters);
    setHasChanges(true);
  };

  const handleSave = () => {
    onUpdateParameters(parameters);
    setHasChanges(false);
    toast.success('Parameters updated');
  };

  const handleReset = () => {
    if (node?.data?.parameters) {
      setParameters(node.data.parameters);
      setHasChanges(false);
    }
  };

  const renderParameterInput = (paramName, paramConfig) => {
    const currentValue = parameters[paramName] ?? paramConfig.default;

    switch (paramConfig.type) {
      case 'number':
        return (
          <input
            type="number"
            value={currentValue}
            min={paramConfig.min}
            max={paramConfig.max}
            step={paramConfig.step || 'any'}
            onChange={(e) => handleParameterChange(paramName, parseFloat(e.target.value))}
            className="input"
          />
        );

      case 'string':
        return (
          <input
            type="text"
            value={currentValue}
            onChange={(e) => handleParameterChange(paramName, e.target.value)}
            className="input"
          />
        );

      case 'boolean':
        return (
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={currentValue}
              onChange={(e) => handleParameterChange(paramName, e.target.checked)}
              className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
            />
            <span className="text-sm text-gray-700">
              {currentValue ? 'Enabled' : 'Disabled'}
            </span>
          </label>
        );

      case 'select':
        return (
          <select
            value={currentValue}
            onChange={(e) => handleParameterChange(paramName, e.target.value)}
            className="input"
          >
            {paramConfig.options?.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        );

      case 'array':
        return (
          <textarea
            value={Array.isArray(currentValue) ? currentValue.join(', ') : currentValue}
            onChange={(e) => {
              const value = e.target.value.split(',').map(v => v.trim()).filter(v => v);
              handleParameterChange(paramName, value);
            }}
            placeholder="Enter comma-separated values"
            rows={3}
            className="input"
          />
        );

      default:
        return (
          <input
            type="text"
            value={currentValue}
            onChange={(e) => handleParameterChange(paramName, e.target.value)}
            className="input"
          />
        );
    }
  };

  if (!node) return null;

  const nodeConfig = node.data.config || {};
  const nodeParameters = nodeConfig.parameters || {};

  return (
    <div className="w-80 bg-white border-l border-gray-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              {node.data.label}
            </h3>
            <p className="text-sm text-gray-500 capitalize">
              {node.data.type} Configuration
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <div className="p-4 space-y-6">
          {/* Node Info */}
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="flex items-center space-x-2 mb-2">
              <Info className="h-4 w-4 text-gray-500" />
              <span className="text-sm font-medium text-gray-700">Node Information</span>
            </div>
            <div className="space-y-1 text-xs text-gray-600">
              <div><span className="font-medium">ID:</span> {node.id}</div>
              <div><span className="font-medium">Type:</span> {node.data.type}</div>
              {node.data.metadata?.domain && (
                <div><span className="font-medium">Domain:</span> {node.data.metadata.domain}</div>
              )}
              {node.data.metadata?.difficulty && (
                <div><span className="font-medium">Difficulty:</span> {node.data.metadata.difficulty}</div>
              )}
            </div>
          </div>

          {/* Parameters */}
          {Object.keys(nodeParameters).length > 0 ? (
            <div className="parameter-form">
              <h4 className="text-sm font-medium text-gray-900 mb-3">Parameters</h4>
              <div className="space-y-4">
                {Object.entries(nodeParameters).map(([paramName, paramConfig]) => (
                  <div key={paramName} className="parameter-group">
                    <label className="parameter-label">
                      {paramName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      {paramConfig.required && (
                        <span className="text-red-500 ml-1">*</span>
                      )}
                    </label>
                    
                    {renderParameterInput(paramName, paramConfig)}
                    
                    {paramConfig.description && (
                      <p className="parameter-description">
                        {paramConfig.description}
                      </p>
                    )}
                    
                    {paramConfig.type === 'number' && (paramConfig.min !== undefined || paramConfig.max !== undefined) && (
                      <p className="parameter-description">
                        Range: {paramConfig.min ?? '∞'} to {paramConfig.max ?? '∞'}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              <Settings className="h-8 w-8 mx-auto mb-2 text-gray-400" />
              <p className="text-sm">No configurable parameters</p>
            </div>
          )}

          {/* Current Values Preview */}
          {Object.keys(parameters).length > 0 && (
            <div className="bg-gray-50 rounded-lg p-3">
              <h5 className="text-sm font-medium text-gray-700 mb-2">Current Values</h5>
              <div className="space-y-1 text-xs font-mono text-gray-600">
                {Object.entries(parameters).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span>{key}:</span>
                    <span className="text-gray-900">
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      {Object.keys(nodeParameters).length > 0 && (
        <div className="p-4 border-t border-gray-200 bg-gray-50">
          <div className="flex space-x-2">
            <button
              onClick={handleSave}
              disabled={!hasChanges}
              className={`btn flex-1 ${hasChanges ? 'btn-primary' : 'btn-outline'}`}
            >
              <Save className="h-4 w-4 mr-2" />
              Save Changes
            </button>
            <button
              onClick={handleReset}
              disabled={!hasChanges}
              className="btn btn-outline"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>
          {hasChanges && (
            <p className="text-xs text-gray-500 mt-2 text-center">
              You have unsaved changes
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default ParameterPanel;
