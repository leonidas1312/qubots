import React, { useState, useCallback, useRef } from 'react';
import {
  ReactFlow,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  MiniMap,
  Background,
  Panel,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import ComponentSidebar from './workflow/ComponentSidebar';
import ParameterPanel from './workflow/ParameterPanel';
import CodePreview from './workflow/CodePreview';
import WorkflowToolbar from './workflow/WorkflowToolbar';
import CustomNode from './workflow/CustomNode';

// Custom node types
const nodeTypes = {
  custom: CustomNode,
};

const initialNodes = [];
const initialEdges = [];

const WorkflowDesigner = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState(null);
  const [showCodePreview, setShowCodePreview] = useState(false);
  const [workflow, setWorkflow] = useState({
    name: 'Untitled Workflow',
    description: '',
    nodes: [],
    edges: [],
    parameters: {}
  });

  const reactFlowWrapper = useRef(null);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const type = event.dataTransfer.getData('application/reactflow');
      const componentData = JSON.parse(event.dataTransfer.getData('application/json'));

      if (typeof type === 'undefined' || !type) {
        return;
      }

      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const newNode = {
        id: `${type}-${Date.now()}`,
        type: 'custom',
        position,
        data: {
          label: componentData.name,
          type: componentData.type,
          config: componentData.config,
          metadata: componentData.metadata,
          parameters: componentData.parameters || {},
        },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes]
  );

  const onNodeClick = useCallback((event, node) => {
    setSelectedNode(node);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  const updateNodeParameters = useCallback((nodeId, parameters) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, parameters } }
          : node
      )
    );
  }, [setNodes]);

  const deleteSelectedNode = useCallback(() => {
    if (selectedNode) {
      setNodes((nds) => nds.filter((node) => node.id !== selectedNode.id));
      setEdges((eds) => eds.filter((edge) => 
        edge.source !== selectedNode.id && edge.target !== selectedNode.id
      ));
      setSelectedNode(null);
    }
  }, [selectedNode, setNodes, setEdges]);

  const generateWorkflowCode = useCallback(() => {
    const workflowData = {
      nodes: nodes.map(node => ({
        id: node.id,
        type: node.data.type,
        config: node.data.config,
        parameters: node.data.parameters,
        position: node.position
      })),
      edges: edges.map(edge => ({
        source: edge.source,
        target: edge.target,
        id: edge.id
      }))
    };

    return workflowData;
  }, [nodes, edges]);

  return (
    <div className="h-screen flex">
      {/* Component Sidebar */}
      <ComponentSidebar />

      {/* Main Workflow Canvas */}
      <div className="flex-1 flex flex-col">
        <WorkflowToolbar
          workflow={workflow}
          setWorkflow={setWorkflow}
          onSave={() => console.log('Save workflow')}
          onLoad={() => console.log('Load workflow')}
          onExport={() => setShowCodePreview(true)}
          onClear={() => {
            setNodes([]);
            setEdges([]);
            setSelectedNode(null);
          }}
          selectedNode={selectedNode}
          onDeleteNode={deleteSelectedNode}
        />

        <div className="flex-1 relative">
          <div ref={reactFlowWrapper} className="w-full h-full">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onInit={setReactFlowInstance}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onNodeClick={onNodeClick}
              onPaneClick={onPaneClick}
              nodeTypes={nodeTypes}
              fitView
              className="workflow-canvas"
            >
              <Controls />
              <MiniMap 
                nodeColor={(node) => {
                  switch (node.data.type) {
                    case 'problem': return '#3b82f6';
                    case 'optimizer': return '#22c55e';
                    case 'data': return '#8b5cf6';
                    default: return '#6b7280';
                  }
                }}
              />
              <Background variant="dots" gap={12} size={1} />
              
              <Panel position="top-left" className="bg-white p-2 rounded-lg shadow-sm border">
                <div className="text-sm text-gray-600">
                  Nodes: {nodes.length} | Edges: {edges.length}
                </div>
              </Panel>
            </ReactFlow>
          </div>
        </div>
      </div>

      {/* Parameter Panel */}
      {selectedNode && (
        <ParameterPanel
          node={selectedNode}
          onUpdateParameters={(parameters) => 
            updateNodeParameters(selectedNode.id, parameters)
          }
          onClose={() => setSelectedNode(null)}
        />
      )}

      {/* Code Preview Modal */}
      {showCodePreview && (
        <CodePreview
          workflow={generateWorkflowCode()}
          onClose={() => setShowCodePreview(false)}
        />
      )}
    </div>
  );
};

export default WorkflowDesigner;
