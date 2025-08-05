# Qubots Web Interface

Modern React-based visual workflow designer for the Qubots optimization platform.

## 🚀 Features

- **Visual Workflow Designer** - Drag-and-drop interface using React Flow
- **Component Library** - Browse and search optimization components
- **Real-time Parameter Configuration** - Dynamic forms with validation
- **Code Generation** - Export workflows as Python, JSON, or MCP format
- **Modern UI** - Built with React 18, TypeScript, and Tailwind CSS
- **State Management** - Zustand for efficient state handling
- **API Integration** - React Query for server state management

## 🛠️ Tech Stack

- **React 18.3** - Latest React with concurrent features
- **TypeScript 5.6** - Type-safe development
- **Vite 5.4** - Fast build tool and dev server
- **React Flow 12.3** - Modern flow-based editor
- **Tailwind CSS 3.4** - Utility-first CSS framework
- **Zustand 5.0** - Lightweight state management
- **React Query 5.62** - Server state management
- **React Hook Form 7.54** - Performant forms with validation
- **Framer Motion 11.15** - Smooth animations
- **Vitest 2.1** - Fast unit testing

## 🏃‍♂️ Quick Start

### Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Open http://localhost:3001
```

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

### Testing

```bash
# Run tests
npm test

# Run tests with UI
npm run test:ui

# Type checking
npm run type-check
```

### Linting

```bash
# Lint code
npm run lint

# Fix linting issues
npm run lint:fix
```

## 📁 Project Structure

```
src/
├── components/          # React components
│   ├── workflow/       # Workflow designer components
│   ├── Dashboard.tsx   # Main dashboard
│   └── Layout.tsx      # App layout
├── lib/                # Utilities and API
│   ├── api.ts         # API client with React Query
│   └── utils.ts       # Common utilities
├── store/              # State management
│   └── workflowStore.ts # Zustand workflow store
├── test/               # Test utilities
│   └── setup.ts       # Test setup
├── App.tsx            # Main app component
├── main.tsx           # App entry point
└── index.css          # Global styles
```

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file:

```env
VITE_API_URL=http://localhost:8000
VITE_GITEA_URL=http://localhost:3000
```

### Vite Configuration

The `vite.config.ts` includes:
- React plugin
- Path aliases (`@/` for `src/`)
- Proxy for API calls
- Optimized build with code splitting
- Test configuration

### TypeScript Configuration

Modern TypeScript setup with:
- ES2020 target
- Bundler module resolution
- Strict type checking
- Path mapping for imports

## 🎨 Styling

### Tailwind CSS

Custom configuration with:
- Extended color palette
- Custom fonts (Inter, JetBrains Mono)
- Custom animations
- Responsive design utilities

### Component Styling

```tsx
import { cn } from '@/lib/utils'

function MyComponent({ className, ...props }) {
  return (
    <div className={cn('base-styles', className)} {...props}>
      Content
    </div>
  )
}
```

## 📊 State Management

### Zustand Store

```tsx
import { useWorkflowStore } from '@/store/workflowStore'

function WorkflowComponent() {
  const { nodes, addNode, selectedNode } = useWorkflowStore()
  
  // Component logic
}
```

### React Query

```tsx
import { useComponents } from '@/lib/api'

function ComponentList() {
  const { data: components, isLoading, error } = useComponents()
  
  // Component logic
}
```

## 🧪 Testing

### Unit Tests

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import MyComponent from './MyComponent'

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />)
    expect(screen.getByText('Expected text')).toBeInTheDocument()
  })
})
```

### Test Utilities

The `src/test/setup.ts` provides:
- Jest DOM matchers
- Mock implementations
- Global test configuration

## 🐳 Docker

### Development

```bash
# Build development image
docker build -t qubots-web-dev .

# Run container
docker run -p 3001:3001 qubots-web-dev
```

### Production

The Dockerfile uses multi-stage build:
1. **Build stage** - Install deps and build app
2. **Production stage** - Serve with Nginx

## 🔄 API Integration

### API Client

The `src/lib/api.ts` provides:
- Axios configuration with interceptors
- TypeScript interfaces
- React Query hooks
- Error handling

### Usage

```tsx
import { useComponents, useCreateWorkflow } from '@/lib/api'

function MyComponent() {
  const { data: components } = useComponents()
  const createWorkflow = useCreateWorkflow()
  
  const handleCreate = () => {
    createWorkflow.mutate({
      name: 'New Workflow',
      description: 'Description',
      nodes: [],
      edges: []
    })
  }
}
```

## 🚀 Performance

### Optimizations

- **Code splitting** - Automatic route-based splitting
- **Tree shaking** - Remove unused code
- **Asset optimization** - Compress images and fonts
- **Caching** - Aggressive caching for static assets
- **Lazy loading** - Load components on demand

### Bundle Analysis

```bash
# Analyze bundle size
npm run build
npx vite-bundle-analyzer dist
```

## 🔧 Development

### Hot Reload

Vite provides instant hot reload for:
- React components
- CSS changes
- TypeScript updates

### Debugging

- React DevTools
- Redux DevTools (for Zustand)
- Network tab for API calls
- Console logging with proper source maps

## 📚 Resources

- [React Documentation](https://react.dev/)
- [Vite Guide](https://vitejs.dev/guide/)
- [React Flow Documentation](https://reactflow.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Zustand](https://github.com/pmndrs/zustand)
- [React Query](https://tanstack.com/query/latest)

## 🤝 Contributing

1. Follow the existing code style
2. Write tests for new features
3. Update documentation
4. Use conventional commits
5. Ensure all checks pass

## 📄 License

This project is part of the Qubots platform and follows the same license terms.
