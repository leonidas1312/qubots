import React from 'react';
import { Handle, Position } from 'reactflow';
import { Database, Brain, Package, Settings } from 'lucide-react';

const CustomNode = ({ data, selected }) => {
  const getNodeIcon = (type) => {
    switch (type) {
      case 'problem': return Database;
      case 'optimizer': return Brain;
      default: return Package;
    }
  };

  const getNodeColor = (type) => {
    switch (type) {
      case 'problem': return 'border-blue-500 bg-blue-50';
      case 'optimizer': return 'border-green-500 bg-green-50';
      default: return 'border-gray-500 bg-gray-50';
    }
  };

  const getNodeAccentColor = (type) => {
    switch (type) {
      case 'problem': return 'bg-blue-500';
      case 'optimizer': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const Icon = getNodeIcon(data.type);
  const hasParameters = data.parameters && Object.keys(data.parameters).length > 0;

  return (
    <div className={`node-card ${selected ? 'selected' : ''} ${getNodeColor(data.type)}`}>
      {/* Input Handle */}
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: '#6b7280' }}
      />

      {/* Node Header */}
      <div className="flex items-center space-x-2 mb-2">
        <div className={`p-1.5 rounded ${getNodeAccentColor(data.type)}`}>
          <Icon className="h-4 w-4 text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-medium text-gray-900 truncate">
            {data.label}
          </h3>
          <p className="text-xs text-gray-500 capitalize">
            {data.type}
          </p>
        </div>
        {hasParameters && (
          <Settings className="h-3 w-3 text-gray-400" />
        )}
      </div>

      {/* Node Content */}
      {data.metadata && (
        <div className="space-y-1">
          {data.metadata.domain && (
            <div className="text-xs text-gray-600">
              <span className="font-medium">Domain:</span> {data.metadata.domain}
            </div>
          )}
          {data.metadata.difficulty && (
            <div className="text-xs text-gray-600">
              <span className="font-medium">Level:</span> {data.metadata.difficulty}
            </div>
          )}
        </div>
      )}

      {/* Parameters Summary */}
      {hasParameters && (
        <div className="mt-2 pt-2 border-t border-gray-200">
          <div className="text-xs text-gray-500">
            {Object.keys(data.parameters).length} parameter(s) configured
          </div>
        </div>
      )}

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: '#6b7280' }}
      />
    </div>
  );
};

export default CustomNode;
