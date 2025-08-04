import React, { useState, useMemo } from 'react';
import { X, Copy, Download, Code, FileText } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { toast } from 'react-hot-toast';

const CodePreview = ({ workflow, onClose }) => {
  const [activeTab, setActiveTab] = useState('python');

  const tabs = [
    { id: 'python', name: 'Python Code', icon: Code },
    { id: 'json', name: 'JSON Export', icon: FileText },
    { id: 'mcp', name: 'MCP Export', icon: FileText },
    { id: 'npx', name: 'NPX Command', icon: Code },
  ];

  const generatedCode = useMemo(() => {
    return generatePythonCode(workflow);
  }, [workflow]);

  const generatedJSON = useMemo(() => {
    return JSON.stringify(workflow, null, 2);
  }, [workflow]);

  const generatedMCP = useMemo(() => {
    // Generate MCP-compatible export
    const mcpData = {
      name: `qubots-workflow-${workflow.name?.toLowerCase().replace(/\s+/g, '-') || 'unnamed'}`,
      version: "1.0.0",
      description: "Qubots optimization workflow",
      type: "module",
      main: "workflow.py",
      exports: {
        tools: workflow.nodes?.map(node => ({
          name: `qubots_${node.type}_${node.id}`,
          description: `${node.name || node.id} - ${node.type} component`,
          inputSchema: {
            type: "object",
            properties: node.parameters || {},
            required: []
          }
        })) || []
      },
      qubots: {
        workflow: workflow,
        compatibility: "1.0.0",
        mcp_version: "1.0.0"
      },
      mcp: {
        version: "1.0.0",
        compatibility: ["npx", "node", "python"],
        runtime: {
          python: {
            version: ">=3.8",
            dependencies: ["qubots", "numpy", "scipy"]
          }
        }
      }
    };
    return JSON.stringify(mcpData, null, 2);
  }, [workflow]);

  const generatedNPX = useMemo(() => {
    const workflowName = workflow.name?.toLowerCase().replace(/\s+/g, '-') || 'workflow';
    return `#!/bin/bash
# Qubots Workflow NPX Commands
# Generated for: ${workflow.name || 'Unnamed Workflow'}

echo "🚀 Qubots Workflow: ${workflow.name || 'Unnamed Workflow'}"
echo "================================="

# Create workflow file
cat > ${workflowName}.json << 'EOF'
${JSON.stringify(workflow, null, 2)}
EOF

echo "📁 Created workflow file: ${workflowName}.json"

# Validate workflow
echo "🔍 Validating workflow..."
npx -y @qubots/mcp-tools workflow-validate ${workflowName}.json

# Execute workflow
echo "▶️  Executing workflow..."
npx -y @qubots/mcp-tools workflow-execute ${workflowName}.json

# Generate code
echo "🔧 Generating Python code..."
npx -y @qubots/mcp-tools code-generate ${workflowName}.json -o ${workflowName}.py

echo "✅ Workflow execution complete!"
echo "📄 Files created:"
echo "   - ${workflowName}.json (workflow definition)"
echo "   - ${workflowName}.py (generated Python code)"
`;
  }, [workflow]);

  const copyToClipboard = (content) => {
    navigator.clipboard.writeText(content).then(() => {
      toast.success('Copied to clipboard!');
    }).catch(() => {
      toast.error('Failed to copy to clipboard');
    });
  };

  const downloadFile = (content, filename) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success(`Downloaded ${filename}`);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl h-3/4 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Generated Code</h2>
            <p className="text-sm text-gray-500">Export your workflow as code</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-200">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.name}</span>
              </button>
            );
          })}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {activeTab === 'python' && (
            <div className="h-full flex flex-col">
              <div className="flex items-center justify-between p-4 bg-gray-50 border-b border-gray-200">
                <div className="text-sm text-gray-600">
                  Generated Python code for your workflow
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => copyToClipboard(generatedCode)}
                    className="btn btn-outline btn-sm"
                  >
                    <Copy className="h-4 w-4 mr-1" />
                    Copy
                  </button>
                  <button
                    onClick={() => downloadFile(generatedCode, 'workflow.py')}
                    className="btn btn-primary btn-sm"
                  >
                    <Download className="h-4 w-4 mr-1" />
                    Download
                  </button>
                </div>
              </div>
              <div className="flex-1 overflow-auto">
                <SyntaxHighlighter
                  language="python"
                  style={tomorrow}
                  customStyle={{
                    margin: 0,
                    height: '100%',
                    background: '#1a1a1a',
                  }}
                  showLineNumbers
                >
                  {generatedCode}
                </SyntaxHighlighter>
              </div>
            </div>
          )}

          {activeTab === 'json' && (
            <div className="h-full flex flex-col">
              <div className="flex items-center justify-between p-4 bg-gray-50 border-b border-gray-200">
                <div className="text-sm text-gray-600">
                  JSON representation of your workflow
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => copyToClipboard(generatedJSON)}
                    className="btn btn-outline btn-sm"
                  >
                    <Copy className="h-4 w-4 mr-1" />
                    Copy
                  </button>
                  <button
                    onClick={() => downloadFile(generatedJSON, 'workflow.json')}
                    className="btn btn-primary btn-sm"
                  >
                    <Download className="h-4 w-4 mr-1" />
                    Download
                  </button>
                </div>
              </div>
              <div className="flex-1 overflow-auto">
                <SyntaxHighlighter
                  language="json"
                  style={tomorrow}
                  customStyle={{
                    margin: 0,
                    height: '100%',
                    background: '#1a1a1a',
                  }}
                  showLineNumbers
                >
                  {generatedJSON}
                </SyntaxHighlighter>
              </div>
            </div>
          )}

          {activeTab === 'mcp' && (
            <div className="h-full flex flex-col">
              <div className="flex items-center justify-between p-4 bg-gray-50 border-b border-gray-200">
                <div className="text-sm text-gray-600">
                  MCP (Model Context Protocol) compatible export
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => copyToClipboard(generatedMCP)}
                    className="btn btn-outline btn-sm"
                  >
                    <Copy className="h-4 w-4 mr-1" />
                    Copy
                  </button>
                  <button
                    onClick={() => downloadFile(generatedMCP, 'workflow-mcp.json')}
                    className="btn btn-primary btn-sm"
                  >
                    <Download className="h-4 w-4 mr-1" />
                    Download
                  </button>
                </div>
              </div>
              <div className="flex-1 overflow-auto">
                <SyntaxHighlighter
                  language="json"
                  style={tomorrow}
                  customStyle={{
                    margin: 0,
                    height: '100%',
                    background: '#1a1a1a',
                  }}
                  showLineNumbers
                >
                  {generatedMCP}
                </SyntaxHighlighter>
              </div>
            </div>
          )}

          {activeTab === 'npx' && (
            <div className="h-full flex flex-col">
              <div className="flex items-center justify-between p-4 bg-gray-50 border-b border-gray-200">
                <div className="text-sm text-gray-600">
                  NPX command script for easy execution
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => copyToClipboard(generatedNPX)}
                    className="btn btn-outline btn-sm"
                  >
                    <Copy className="h-4 w-4 mr-1" />
                    Copy
                  </button>
                  <button
                    onClick={() => downloadFile(generatedNPX, 'run-workflow.sh')}
                    className="btn btn-primary btn-sm"
                  >
                    <Download className="h-4 w-4 mr-1" />
                    Download
                  </button>
                </div>
              </div>
              <div className="flex-1 overflow-auto">
                <SyntaxHighlighter
                  language="bash"
                  style={tomorrow}
                  customStyle={{
                    margin: 0,
                    height: '100%',
                    background: '#1a1a1a',
                  }}
                  showLineNumbers
                >
                  {generatedNPX}
                </SyntaxHighlighter>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Helper function to generate Python code from workflow
function generatePythonCode(workflow) {
  const { nodes, edges } = workflow;
  
  if (!nodes || nodes.length === 0) {
    return `# Empty workflow
# Add some components to generate code!

from qubots import AutoProblem, AutoOptimizer

# Your workflow will appear here once you add components
`;
  }

  let code = `#!/usr/bin/env python3
"""
Generated Qubots Workflow
Auto-generated from visual workflow designer
"""

from qubots import AutoProblem, AutoOptimizer
import json

def main():
    """Main workflow execution function."""
    
`;

  // Find problems and optimizers
  const problems = nodes.filter(node => node.type === 'problem');
  const optimizers = nodes.filter(node => node.type === 'optimizer');

  // Generate problem loading code
  problems.forEach((problem, index) => {
    const varName = `problem_${index + 1}`;
    const repoId = problem.config?.repository || `examples/${problem.id}`;
    
    code += `    # Load ${problem.label}\n`;
    code += `    ${varName} = AutoProblem.from_repo("${repoId}"`;
    
    if (problem.parameters && Object.keys(problem.parameters).length > 0) {
      code += `,\n        override_params=${JSON.stringify(problem.parameters, null, 8).replace(/^/gm, '        ')}`;
    }
    
    code += `)\n`;
    code += `    print(f"Loaded problem: {${varName}.metadata.name}")\n\n`;
  });

  // Generate optimizer loading code
  optimizers.forEach((optimizer, index) => {
    const varName = `optimizer_${index + 1}`;
    const repoId = optimizer.config?.repository || `examples/${optimizer.id}`;
    
    code += `    # Load ${optimizer.label}\n`;
    code += `    ${varName} = AutoOptimizer.from_repo("${repoId}"`;
    
    if (optimizer.parameters && Object.keys(optimizer.parameters).length > 0) {
      code += `,\n        override_params=${JSON.stringify(optimizer.parameters, null, 8).replace(/^/gm, '        ')}`;
    }
    
    code += `)\n`;
    code += `    print(f"Loaded optimizer: {${varName}.metadata.name}")\n\n`;
  });

  // Generate optimization execution code
  if (problems.length > 0 && optimizers.length > 0) {
    code += `    # Run optimization\n`;
    code += `    result = optimizer_1.optimize(problem_1)\n`;
    code += `    \n`;
    code += `    # Display results\n`;
    code += `    print(f"Best solution: {result.best_solution}")\n`;
    code += `    print(f"Best value: {result.best_value}")\n`;
    code += `    print(f"Runtime: {result.runtime_seconds:.3f} seconds")\n`;
    code += `    print(f"Status: {result.termination_reason}")\n`;
  } else {
    code += `    # Add both problems and optimizers to run optimization\n`;
    code += `    print("Workflow loaded successfully!")\n`;
  }

  code += `

if __name__ == "__main__":
    main()
`;

  return code;
}

export default CodePreview;
