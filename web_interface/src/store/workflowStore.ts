import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'
import { Node, Edge, Connection } from '@xyflow/react'

export interface WorkflowNode extends Node {
  data: {
    label: string
    type: 'problem' | 'optimizer' | 'data'
    repository?: string
    parameters?: Record<string, any>
    metadata?: Record<string, any>
  }
}

export interface WorkflowEdge extends Edge {}

interface WorkflowState {
  // Workflow state
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
  selectedNode: WorkflowNode | null
  
  // UI state
  sidebarOpen: boolean
  parameterPanelOpen: boolean
  codePreviewOpen: boolean
  
  // Actions
  setNodes: (nodes: WorkflowNode[]) => void
  setEdges: (edges: WorkflowEdge[]) => void
  addNode: (node: WorkflowNode) => void
  updateNode: (id: string, data: Partial<WorkflowNode['data']>) => void
  removeNode: (id: string) => void
  addEdge: (connection: Connection) => void
  removeEdge: (id: string) => void
  setSelectedNode: (node: WorkflowNode | null) => void
  
  // UI actions
  toggleSidebar: () => void
  toggleParameterPanel: () => void
  toggleCodePreview: () => void
  
  // Workflow actions
  clearWorkflow: () => void
  loadWorkflow: (workflow: { nodes: WorkflowNode[], edges: WorkflowEdge[] }) => void
  exportWorkflow: () => { nodes: WorkflowNode[], edges: WorkflowEdge[] }
}

export const useWorkflowStore = create<WorkflowState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        nodes: [],
        edges: [],
        selectedNode: null,
        sidebarOpen: true,
        parameterPanelOpen: false,
        codePreviewOpen: false,
        
        // Node actions
        setNodes: (nodes) => set({ nodes }),
        setEdges: (edges) => set({ edges }),
        
        addNode: (node) => set((state) => ({
          nodes: [...state.nodes, node]
        })),
        
        updateNode: (id, data) => set((state) => ({
          nodes: state.nodes.map(node => 
            node.id === id 
              ? { ...node, data: { ...node.data, ...data } }
              : node
          )
        })),
        
        removeNode: (id) => set((state) => ({
          nodes: state.nodes.filter(node => node.id !== id),
          edges: state.edges.filter(edge => edge.source !== id && edge.target !== id),
          selectedNode: state.selectedNode?.id === id ? null : state.selectedNode
        })),
        
        addEdge: (connection) => set((state) => ({
          edges: [...state.edges, {
            id: `${connection.source}-${connection.target}`,
            source: connection.source!,
            target: connection.target!,
            type: 'smoothstep'
          }]
        })),
        
        removeEdge: (id) => set((state) => ({
          edges: state.edges.filter(edge => edge.id !== id)
        })),
        
        setSelectedNode: (node) => set({ selectedNode: node }),
        
        // UI actions
        toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
        toggleParameterPanel: () => set((state) => ({ parameterPanelOpen: !state.parameterPanelOpen })),
        toggleCodePreview: () => set((state) => ({ codePreviewOpen: !state.codePreviewOpen })),
        
        // Workflow actions
        clearWorkflow: () => set({
          nodes: [],
          edges: [],
          selectedNode: null
        }),
        
        loadWorkflow: (workflow) => set({
          nodes: workflow.nodes,
          edges: workflow.edges,
          selectedNode: null
        }),
        
        exportWorkflow: () => {
          const { nodes, edges } = get()
          return { nodes, edges }
        }
      }),
      {
        name: 'qubots-workflow-store',
        partialize: (state) => ({
          nodes: state.nodes,
          edges: state.edges,
          sidebarOpen: state.sidebarOpen,
          parameterPanelOpen: state.parameterPanelOpen,
          codePreviewOpen: state.codePreviewOpen
        })
      }
    ),
    {
      name: 'workflow-store'
    }
  )
)
