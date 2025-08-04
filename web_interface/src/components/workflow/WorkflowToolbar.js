import React, { useState } from 'react';
import { 
  Save, 
  FolderOpen, 
  Download, 
  Trash2, 
  Play, 
  Square,
  Code,
  Settings,
  FileText
} from 'lucide-react';
import { toast } from 'react-hot-toast';

const WorkflowToolbar = ({ 
  workflow, 
  setWorkflow, 
  onSave, 
  onLoad, 
  onExport, 
  onClear,
  selectedNode,
  onDeleteNode 
}) => {
  const [isRunning, setIsRunning] = useState(false);

  const handleRun = async () => {
    if (isRunning) return;
    
    setIsRunning(true);
    toast.loading('Running workflow...', { id: 'workflow-run' });
    
    try {
      // Simulate workflow execution
      await new Promise(resolve => setTimeout(resolve, 2000));
      toast.success('Workflow completed successfully!', { id: 'workflow-run' });
    } catch (error) {
      toast.error('Workflow execution failed', { id: 'workflow-run' });
    } finally {
      setIsRunning(false);
    }
  };

  const handleSave = () => {
    onSave();
    toast.success('Workflow saved');
  };

  const handleLoad = () => {
    onLoad();
    toast.success('Workflow loaded');
  };

  const handleExport = () => {
    onExport();
  };

  const handleClear = () => {
    if (window.confirm('Are you sure you want to clear the workflow? This action cannot be undone.')) {
      onClear();
      toast.success('Workflow cleared');
    }
  };

  const handleDeleteNode = () => {
    if (selectedNode && window.confirm(`Delete node "${selectedNode.data.label}"?`)) {
      onDeleteNode();
      toast.success('Node deleted');
    }
  };

  return (
    <div className="bg-white border-b border-gray-200 px-4 py-3">
      <div className="flex items-center justify-between">
        {/* Left side - Workflow info */}
        <div className="flex items-center space-x-4">
          <div>
            <input
              type="text"
              value={workflow.name}
              onChange={(e) => setWorkflow({ ...workflow, name: e.target.value })}
              className="text-lg font-semibold bg-transparent border-none focus:outline-none focus:ring-2 focus:ring-primary-500 rounded px-2 py-1"
              placeholder="Workflow Name"
            />
            <p className="text-sm text-gray-500">
              {workflow.description || 'No description'}
            </p>
          </div>
        </div>

        {/* Center - Main actions */}
        <div className="flex items-center space-x-2">
          <button
            onClick={handleRun}
            disabled={isRunning}
            className={`btn ${isRunning ? 'btn-secondary' : 'btn-success'} flex items-center space-x-2`}
          >
            {isRunning ? (
              <>
                <Square className="h-4 w-4" />
                <span>Running...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                <span>Run</span>
              </>
            )}
          </button>

          <div className="h-6 w-px bg-gray-300" />

          <button
            onClick={handleSave}
            className="btn btn-primary"
            title="Save Workflow"
          >
            <Save className="h-4 w-4" />
          </button>

          <button
            onClick={handleLoad}
            className="btn btn-outline"
            title="Load Workflow"
          >
            <FolderOpen className="h-4 w-4" />
          </button>

          <button
            onClick={handleExport}
            className="btn btn-outline"
            title="Export Code"
          >
            <Code className="h-4 w-4" />
          </button>
        </div>

        {/* Right side - Node actions and utilities */}
        <div className="flex items-center space-x-2">
          {selectedNode && (
            <>
              <div className="text-sm text-gray-600">
                Selected: <span className="font-medium">{selectedNode.data.label}</span>
              </div>
              <button
                onClick={handleDeleteNode}
                className="btn btn-error"
                title="Delete Selected Node"
              >
                <Trash2 className="h-4 w-4" />
              </button>
              <div className="h-6 w-px bg-gray-300" />
            </>
          )}

          <button
            onClick={handleClear}
            className="btn btn-outline text-red-600 hover:bg-red-50"
            title="Clear Workflow"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Workflow description input */}
      <div className="mt-3">
        <input
          type="text"
          value={workflow.description}
          onChange={(e) => setWorkflow({ ...workflow, description: e.target.value })}
          className="w-full text-sm bg-gray-50 border border-gray-200 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          placeholder="Add a description for this workflow..."
        />
      </div>
    </div>
  );
};

export default WorkflowToolbar;
