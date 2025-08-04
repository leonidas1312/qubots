import React, { useState, useEffect } from 'react';
import { Search, Package, Brain, Database, Filter } from 'lucide-react';
import { toast } from 'react-hot-toast';

const ComponentSidebar = () => {
  const [components, setComponents] = useState([]);
  const [filteredComponents, setFilteredComponents] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [loading, setLoading] = useState(true);

  const categories = [
    { id: 'all', name: 'All Components', icon: Package },
    { id: 'problem', name: 'Problems', icon: Database },
    { id: 'optimizer', name: 'Optimizers', icon: Brain },
  ];

  // Mock data - replace with API call
  const mockComponents = [
    {
      id: 'portfolio-problem',
      name: 'Portfolio Optimization',
      type: 'problem',
      description: 'Markowitz portfolio optimization problem',
      tags: ['finance', 'portfolio', 'risk'],
      config: {
        type: 'problem',
        entry_point: 'qubot',
        class_name: 'PortfolioOptimizationProblem'
      },
      metadata: {
        domain: 'finance',
        difficulty: 'intermediate'
      },
      parameters: {
        target_return: { type: 'number', default: 0.10, min: 0.0, max: 0.5 },
        risk_free_rate: { type: 'number', default: 0.02, min: 0.0, max: 0.1 }
      }
    },
    {
      id: 'tsp-problem',
      name: 'Traveling Salesman',
      type: 'problem',
      description: 'Classic TSP optimization problem',
      tags: ['routing', 'combinatorial', 'tsp'],
      config: {
        type: 'problem',
        entry_point: 'qubot',
        class_name: 'TSPProblem'
      },
      metadata: {
        domain: 'routing',
        difficulty: 'intermediate'
      },
      parameters: {
        num_cities: { type: 'number', default: 20, min: 3, max: 1000 }
      }
    },
    {
      id: 'genetic-optimizer',
      name: 'Genetic Algorithm',
      type: 'optimizer',
      description: 'Evolutionary optimization algorithm',
      tags: ['genetic', 'evolutionary', 'metaheuristic'],
      config: {
        type: 'optimizer',
        entry_point: 'qubot',
        class_name: 'GeneticOptimizer'
      },
      metadata: {
        family: 'evolutionary',
        complexity: 'medium'
      },
      parameters: {
        population_size: { type: 'number', default: 100, min: 10, max: 1000 },
        mutation_rate: { type: 'number', default: 0.01, min: 0.0, max: 1.0 },
        crossover_rate: { type: 'number', default: 0.8, min: 0.0, max: 1.0 }
      }
    },
    {
      id: 'ortools-optimizer',
      name: 'OR-Tools Solver',
      type: 'optimizer',
      description: 'Google OR-Tools optimization solver',
      tags: ['exact', 'linear', 'constraint'],
      config: {
        type: 'optimizer',
        entry_point: 'qubot',
        class_name: 'ORToolsOptimizer'
      },
      metadata: {
        family: 'exact',
        complexity: 'high'
      },
      parameters: {
        time_limit: { type: 'number', default: 300, min: 1, max: 3600 },
        num_workers: { type: 'number', default: 0, min: 0, max: 16 }
      }
    }
  ];

  useEffect(() => {
    // Simulate API call
    const loadComponents = async () => {
      try {
        setLoading(true);
        // In real app, this would be: const response = await api.getComponents();
        await new Promise(resolve => setTimeout(resolve, 500)); // Simulate delay
        setComponents(mockComponents);
        setFilteredComponents(mockComponents);
      } catch (error) {
        toast.error('Failed to load components');
        console.error('Error loading components:', error);
      } finally {
        setLoading(false);
      }
    };

    loadComponents();
  }, []);

  useEffect(() => {
    let filtered = components;

    // Filter by category
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(comp => comp.type === selectedCategory);
    }

    // Filter by search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(comp =>
        comp.name.toLowerCase().includes(term) ||
        comp.description.toLowerCase().includes(term) ||
        comp.tags.some(tag => tag.toLowerCase().includes(term))
      );
    }

    setFilteredComponents(filtered);
  }, [components, selectedCategory, searchTerm]);

  const onDragStart = (event, component) => {
    event.dataTransfer.setData('application/reactflow', component.type);
    event.dataTransfer.setData('application/json', JSON.stringify(component));
    event.dataTransfer.effectAllowed = 'move';
  };

  const getComponentIcon = (type) => {
    switch (type) {
      case 'problem': return Database;
      case 'optimizer': return Brain;
      default: return Package;
    }
  };

  const getComponentColor = (type) => {
    switch (type) {
      case 'problem': return 'border-blue-200 bg-blue-50';
      case 'optimizer': return 'border-green-200 bg-green-50';
      default: return 'border-gray-200 bg-gray-50';
    }
  };

  return (
    <div className="sidebar">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">Components</h2>
        
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search components..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input pl-10 text-sm"
          />
        </div>
      </div>

      {/* Categories */}
      <div className="p-4 border-b border-gray-200">
        <div className="space-y-1">
          {categories.map((category) => {
            const Icon = category.icon;
            return (
              <button
                key={category.id}
                onClick={() => setSelectedCategory(category.id)}
                className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                  selectedCategory === category.id
                    ? 'bg-primary-100 text-primary-900'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`}
              >
                <Icon className="mr-2 h-4 w-4" />
                {category.name}
                <span className="ml-auto text-xs text-gray-500">
                  {category.id === 'all' 
                    ? components.length 
                    : components.filter(c => c.type === category.id).length
                  }
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Components List */}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {loading ? (
          <div className="p-4 text-center text-gray-500">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto mb-2"></div>
            Loading components...
          </div>
        ) : filteredComponents.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            <Filter className="h-8 w-8 mx-auto mb-2 text-gray-400" />
            No components found
          </div>
        ) : (
          <div className="p-4 space-y-3">
            {filteredComponents.map((component) => {
              const Icon = getComponentIcon(component.type);
              return (
                <div
                  key={component.id}
                  draggable
                  onDragStart={(event) => onDragStart(event, component)}
                  className={`component-item ${getComponentColor(component.type)} cursor-grab active:cursor-grabbing`}
                >
                  <div className="flex items-start space-x-3">
                    <Icon className="h-5 w-5 text-gray-600 mt-0.5 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <h3 className="text-sm font-medium text-gray-900 truncate">
                        {component.name}
                      </h3>
                      <p className="text-xs text-gray-600 mt-1 line-clamp-2">
                        {component.description}
                      </p>
                      <div className="flex flex-wrap gap-1 mt-2">
                        {component.tags.slice(0, 3).map((tag) => (
                          <span
                            key={tag}
                            className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800"
                          >
                            {tag}
                          </span>
                        ))}
                        {component.tags.length > 3 && (
                          <span className="text-xs text-gray-500">
                            +{component.tags.length - 3}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default ComponentSidebar;
