import React, { useState, useEffect } from 'react';
import { Search, Filter, Package, Database, Brain, Star, Download, ExternalLink } from 'lucide-react';
import { toast } from 'react-hot-toast';

const ComponentLibrary = () => {
  const [components, setComponents] = useState([]);
  const [filteredComponents, setFilteredComponents] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedDomain, setSelectedDomain] = useState('all');
  const [loading, setLoading] = useState(true);

  const categories = [
    { id: 'all', name: 'All Components', count: 0 },
    { id: 'problem', name: 'Problems', count: 0 },
    { id: 'optimizer', name: 'Optimizers', count: 0 },
  ];

  const domains = [
    'all', 'finance', 'routing', 'scheduling', 'logistics', 
    'energy', 'manufacturing', 'healthcare', 'research'
  ];

  // Mock data - replace with API call
  const mockComponents = [
    {
      id: 'portfolio-problem',
      name: 'Portfolio Optimization Problem',
      type: 'problem',
      description: 'Markowitz mean-variance portfolio optimization problem that reads stock data from CSV files.',
      domain: 'finance',
      difficulty: 'intermediate',
      tags: ['portfolio', 'markowitz', 'finance', 'risk'],
      author: 'Qubots Team',
      downloads: 1250,
      rating: 4.8,
      lastUpdated: '2 days ago',
      repository: 'examples/portfolio_optimization_problem'
    },
    {
      id: 'tsp-problem',
      name: 'Traveling Salesman Problem',
      type: 'problem',
      description: 'Classic TSP optimization problem with support for various distance metrics.',
      domain: 'routing',
      difficulty: 'intermediate',
      tags: ['tsp', 'routing', 'combinatorial'],
      author: 'Qubots Team',
      downloads: 890,
      rating: 4.6,
      lastUpdated: '1 week ago',
      repository: 'examples/tsp'
    },
    {
      id: 'genetic-optimizer',
      name: 'Genetic Algorithm Optimizer',
      type: 'optimizer',
      description: 'Evolutionary optimization algorithm with customizable operators.',
      domain: 'metaheuristic',
      difficulty: 'intermediate',
      tags: ['genetic', 'evolutionary', 'metaheuristic'],
      author: 'Qubots Team',
      downloads: 2100,
      rating: 4.9,
      lastUpdated: '3 days ago',
      repository: 'examples/genetic_optimizer'
    },
    {
      id: 'ortools-optimizer',
      name: 'OR-Tools Solver',
      type: 'optimizer',
      description: 'Google OR-Tools optimization solver with constraint programming support.',
      domain: 'exact',
      difficulty: 'advanced',
      tags: ['exact', 'linear', 'constraint', 'google'],
      author: 'Qubots Team',
      downloads: 1680,
      rating: 4.7,
      lastUpdated: '5 days ago',
      repository: 'examples/ortools_optimizer'
    },
    {
      id: 'supply-chain-problem',
      name: 'Supply Chain Network Problem',
      type: 'problem',
      description: 'Multi-facility supply chain optimization with capacity constraints.',
      domain: 'logistics',
      difficulty: 'advanced',
      tags: ['supply-chain', 'logistics', 'network', 'capacity'],
      author: 'Qubots Team',
      downloads: 750,
      rating: 4.5,
      lastUpdated: '1 week ago',
      repository: 'examples/supply_chain_problem'
    },
    {
      id: 'molecular-problem',
      name: 'Molecular Conformation Problem',
      type: 'problem',
      description: 'Molecular structure optimization for drug discovery applications.',
      domain: 'research',
      difficulty: 'expert',
      tags: ['molecular', 'chemistry', 'drug-discovery', 'research'],
      author: 'Research Lab',
      downloads: 320,
      rating: 4.3,
      lastUpdated: '2 weeks ago',
      repository: 'examples/molecular_problem'
    }
  ];

  useEffect(() => {
    // Simulate API call
    const loadComponents = async () => {
      try {
        setLoading(true);
        await new Promise(resolve => setTimeout(resolve, 800));
        setComponents(mockComponents);
        setFilteredComponents(mockComponents);
      } catch (error) {
        toast.error('Failed to load components');
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

    // Filter by domain
    if (selectedDomain !== 'all') {
      filtered = filtered.filter(comp => comp.domain === selectedDomain);
    }

    // Filter by search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(comp =>
        comp.name.toLowerCase().includes(term) ||
        comp.description.toLowerCase().includes(term) ||
        comp.tags.some(tag => tag.toLowerCase().includes(term)) ||
        comp.author.toLowerCase().includes(term)
      );
    }

    setFilteredComponents(filtered);
  }, [components, selectedCategory, selectedDomain, searchTerm]);

  const getComponentIcon = (type) => {
    switch (type) {
      case 'problem': return Database;
      case 'optimizer': return Brain;
      default: return Package;
    }
  };

  const getComponentColor = (type) => {
    switch (type) {
      case 'problem': return 'text-blue-600 bg-blue-100';
      case 'optimizer': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-600 bg-green-100';
      case 'intermediate': return 'text-yellow-600 bg-yellow-100';
      case 'advanced': return 'text-orange-600 bg-orange-100';
      case 'expert': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const handleInstall = (component) => {
    toast.success(`Installing ${component.name}...`);
    // Simulate installation
    setTimeout(() => {
      toast.success(`${component.name} installed successfully!`);
    }, 2000);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Component Library</h1>
        <p className="text-gray-600 mt-2">
          Discover and install optimization components for your workflows
        </p>
      </div>

      {/* Filters */}
      <div className="mb-6 space-y-4">
        {/* Search */}
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search components..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input pl-10"
          />
        </div>

        {/* Filter buttons */}
        <div className="flex flex-wrap gap-4">
          <div className="flex items-center space-x-2">
            <Filter className="h-4 w-4 text-gray-500" />
            <span className="text-sm font-medium text-gray-700">Category:</span>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="input py-1 text-sm"
            >
              {categories.map((category) => (
                <option key={category.id} value={category.id}>
                  {category.name}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-700">Domain:</span>
            <select
              value={selectedDomain}
              onChange={(e) => setSelectedDomain(e.target.value)}
              className="input py-1 text-sm"
            >
              {domains.map((domain) => (
                <option key={domain} value={domain}>
                  {domain === 'all' ? 'All Domains' : domain.charAt(0).toUpperCase() + domain.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="mb-4 text-sm text-gray-600">
        {loading ? 'Loading...' : `${filteredComponents.length} components found`}
      </div>

      {/* Components Grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-1/2 mb-4"></div>
              <div className="h-3 bg-gray-200 rounded w-full mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-2/3"></div>
            </div>
          ))}
        </div>
      ) : filteredComponents.length === 0 ? (
        <div className="text-center py-12">
          <Package className="h-12 w-12 mx-auto mb-4 text-gray-400" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No components found</h3>
          <p className="text-gray-600">Try adjusting your search or filters</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredComponents.map((component) => {
            const Icon = getComponentIcon(component.type);
            return (
              <div key={component.id} className="card hover:shadow-lg transition-shadow">
                {/* Header */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <div className={`p-2 rounded-lg ${getComponentColor(component.type)}`}>
                      <Icon className="h-4 w-4" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 text-sm">{component.name}</h3>
                      <p className="text-xs text-gray-500 capitalize">{component.type}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-1 text-xs text-gray-500">
                    <Star className="h-3 w-3 fill-current text-yellow-400" />
                    <span>{component.rating}</span>
                  </div>
                </div>

                {/* Description */}
                <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                  {component.description}
                </p>

                {/* Tags */}
                <div className="flex flex-wrap gap-1 mb-3">
                  {component.tags.slice(0, 3).map((tag) => (
                    <span
                      key={tag}
                      className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800"
                    >
                      {tag}
                    </span>
                  ))}
                  {component.tags.length > 3 && (
                    <span className="text-xs text-gray-500">+{component.tags.length - 3}</span>
                  )}
                </div>

                {/* Metadata */}
                <div className="flex items-center justify-between text-xs text-gray-500 mb-4">
                  <div className="flex items-center space-x-3">
                    <span className={`status-indicator ${getDifficultyColor(component.difficulty)}`}>
                      {component.difficulty}
                    </span>
                    <span>{component.domain}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Download className="h-3 w-3" />
                    <span>{component.downloads}</span>
                  </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between pt-3 border-t border-gray-200">
                  <div className="text-xs text-gray-500">
                    by {component.author}
                  </div>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => window.open(`http://localhost:3000/${component.repository}`, '_blank')}
                      className="btn btn-outline btn-sm"
                      title="View Repository"
                    >
                      <ExternalLink className="h-3 w-3" />
                    </button>
                    <button
                      onClick={() => handleInstall(component)}
                      className="btn btn-primary btn-sm"
                    >
                      Install
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default ComponentLibrary;
