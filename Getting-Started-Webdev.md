# Getting Started with Web Development for tldw_server Frontend

## Overview
This guide is designed for developers new to web development who want to build a frontend for the tldw_server project. The backend is a robust FastAPI server with comprehensive media processing, transcription, and AI capabilities.

## Technology Stack Recommendations

### Core Technologies (Start Here)
- **Next.js 14** with Pages Router - Simpler mental model for beginners than App Router
- **TypeScript** - Type safety from the start, matches your backend's discipline
- **shadcn/ui** - Copy-paste component library where you learn by seeing the code
- **TanStack Query** - Elegant API state management
- **Tailwind CSS** - Utility-first CSS without managing separate stylesheets

## Learning Path (4-6 Weeks)

### Week 1-2: Foundations
- **HTML/CSS Basics** (if needed)
  - MDN Web Docs HTML/CSS tutorials
  - Basic layouts, forms, responsive design
- **JavaScript ES6+ Essentials**
  - Variables, functions, arrays, objects
  - Promises, async/await
  - Array methods (map, filter, reduce)
  - Destructuring, spread operator
- **React Fundamentals**
  - Components and JSX
  - Props and state
  - Hooks (useState, useEffect, useContext)
  - Event handling

### Week 3-4: Next.js & TypeScript
- **Next.js Pages Router**
  - File-based routing
  - API routes
  - getServerSideProps/getStaticProps
  - Environment variables
- **TypeScript Basics**
  - Types and interfaces
  - Type inference
  - Generics basics
  - Working with API responses

### Week 5-6: Integration
- **Connecting to Your Backend**
  - API client setup
  - Error handling
  - Loading states
- **Authentication Flow**
  - JWT token management
  - Protected routes
  - User context
- **Data Fetching Patterns**
  - TanStack Query setup
  - Caching strategies
  - Optimistic updates

## Project Structure

```
tldw-frontend/
├── pages/                 # Routes (file = route)
│   ├── index.tsx         # Home page (/)
│   ├── login.tsx         # Login page (/login)
│   ├── media/
│   │   ├── index.tsx     # Media list (/media)
│   │   └── [id].tsx      # Media detail (/media/123)
│   ├── chat.tsx          # Chat interface
│   └── api/              # API proxy routes
│       └── auth/         # Auth endpoints
├── components/           # Reusable UI components
│   ├── layout/
│   │   ├── Header.tsx
│   │   ├── Footer.tsx
│   │   └── Layout.tsx
│   └── ui/              # Basic UI components
│       ├── Button.tsx
│       ├── Input.tsx
│       └── Card.tsx
├── lib/                  # Utilities and helpers
│   ├── api.ts           # API client configuration
│   ├── auth.ts          # Auth helpers
│   └── utils.ts         # General utilities
├── hooks/               # Custom React hooks
│   ├── useAuth.ts       # Authentication hook
│   └── useApi.ts        # API fetching hook
├── types/               # TypeScript type definitions
│   ├── api.ts           # API response types
│   └── index.ts         # General types
├── styles/              # Global styles
│   └── globals.css      # Tailwind imports
├── public/              # Static assets
├── .env.local           # Environment variables
└── package.json         # Dependencies

```

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [x] Set up Next.js project with TypeScript
- [ ] Configure environment variables for API URL
- [ ] Create basic layout components
- [ ] Implement API client wrapper
- [ ] Set up Tailwind CSS

### Phase 2: Authentication (Week 2)
- [ ] Create login/logout pages
- [ ] Implement JWT token management
- [ ] Build protected route wrapper
- [ ] Set up user context/state

### Phase 3: Core Features (Week 3-4)
- [ ] Media list view with pagination
- [ ] Search functionality
- [ ] File upload interface
- [ ] Basic chat interface (non-streaming)

### Phase 4: Enhanced Features (Week 5-6)
- [ ] Streaming chat responses (SSE)
- [ ] Media processing status updates
- [ ] Comprehensive error handling
- [ ] Responsive design refinement
- [ ] Loading states and skeletons

## Key Recommendations

### Do's
- ✅ Start with one feature end-to-end (e.g., login → view media list)
- ✅ Use the OpenAPI docs at `http://localhost:8000/docs` to understand endpoints
- ✅ Generate TypeScript types from your OpenAPI spec
- ✅ Test with real API calls early and often
- ✅ Use browser DevTools Network tab extensively
- ✅ Commit working code frequently
- ✅ Read error messages carefully - they often tell you exactly what's wrong
- ✅ Use console.log liberally while learning

### Don'ts
- ❌ Don't try to build everything at once
- ❌ Don't optimize prematurely
- ❌ Don't skip error handling
- ❌ Don't ignore TypeScript errors - they're helping you
- ❌ Don't build complex state management initially
- ❌ Don't be afraid to Google - everyone does it

## Essential Tools

### Development Environment
- **VS Code** with extensions:
  - ES7+ React/Redux/React-Native snippets
  - Tailwind CSS IntelliSense
  - Prettier - Code formatter
  - TypeScript React code snippets
- **Browser Extensions**:
  - React Developer Tools
  - Redux DevTools (if using Redux)
- **API Testing**:
  - Postman or Insomnia
  - Your backend's `/docs` endpoint

### Initial Dependencies
```json
{
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "axios": "^1.6.0",
    "@tanstack/react-query": "^5.0.0",
    "zustand": "^4.4.0",
    "react-hook-form": "^7.0.0",
    "tailwindcss": "^3.4.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.0.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@types/react": "^18.0.0",
    "@types/react-dom": "^18.0.0",
    "typescript": "^5.0.0",
    "autoprefixer": "^10.0.0",
    "postcss": "^8.0.0"
  }
}
```

## Learning Resources

### Free & Excellent
- **Next.js Official Tutorial**: [nextjs.org/learn](https://nextjs.org/learn)
- **React Documentation**: [react.dev](https://react.dev)
- **MDN Web Docs**: For JavaScript, HTML, CSS fundamentals
- **TypeScript Handbook**: [typescriptlang.org/docs](https://www.typescriptlang.org/docs)

### YouTube Channels
- **Traversy Media**: Practical, project-based tutorials
- **Web Dev Simplified**: Clear explanations of complex concepts
- **Fireship**: Quick, dense tutorials on modern web tech

### Interactive Learning
- **FreeCodeCamp**: Comprehensive curriculum
- **Scrimba**: Interactive coding tutorials
- **Frontend Mentor**: Real-world project challenges

## Your Backend Advantages

The tldw_server backend provides several advantages:

1. **Well-Documented API**: OpenAPI docs at `/docs` show all endpoints
2. **Standard OAuth2 Flow**: Industry-standard authentication
3. **Consistent Error Responses**: Predictable error handling
4. **TypeScript-Friendly**: Pydantic schemas translate well to TS types
5. **Comprehensive Features**: All major functionality exposed via API

## Initial Implementation Steps

### Step 1: Create Next.js App
```bash
npx create-next-app@latest tldw-frontend --typescript --no-app
cd tldw-frontend
```

### Step 2: Set Up API Client
Create `lib/api.ts`:
```typescript
import axios from 'axios';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;
```

### Step 3: Create Login Page
Connect to `/api/v1/auth/login` endpoint

### Step 4: Build Media List
Fetch from `/api/v1/media/list` endpoint

### Step 5: Add Search
Use `/api/v1/media/search` endpoint

### Step 6: Implement Chat
Start with basic chat, then add SSE streaming

### Step 7: File Upload
Connect to media ingestion endpoints

## Common Patterns

### API Calls with Error Handling
```typescript
const fetchMedia = async () => {
  try {
    setLoading(true);
    const response = await api.get('/api/v1/media/list');
    setData(response.data);
  } catch (error) {
    setError(error.message);
  } finally {
    setLoading(false);
  }
};
```

### Protected Routes
```typescript
const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
    }
  }, [user, loading]);

  if (loading) return <Spinner />;
  if (!user) return null;
  
  return children;
};
```

### Form Handling
```typescript
const LoginForm = () => {
  const { register, handleSubmit, formState: { errors } } = useForm();
  
  const onSubmit = async (data) => {
    try {
      const response = await api.post('/api/v1/auth/login', data);
      localStorage.setItem('token', response.data.access_token);
      router.push('/');
    } catch (error) {
      setError(error.response?.data?.detail || 'Login failed');
    }
  };
  
  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input {...register('username', { required: true })} />
      <input type="password" {...register('password', { required: true })} />
      <button type="submit">Login</button>
    </form>
  );
};
```

## Debugging Tips

### Common Issues & Solutions

1. **CORS Errors**
   - Check backend CORS configuration
   - Ensure API URL is correct in .env.local
   - Use API routes in Next.js as proxy if needed

2. **401 Unauthorized**
   - Check token is being sent in headers
   - Verify token isn't expired
   - Ensure login endpoint returns correct token format

3. **TypeScript Errors**
   - Don't use `any` as a quick fix - understand the type
   - Use the backend's OpenAPI spec to generate types
   - Start with simpler types, refine as you learn

4. **State Not Updating**
   - Remember React state updates are asynchronous
   - Use React DevTools to inspect component state
   - Check if you're mutating state directly (don't!)

5. **API Calls Failing**
   - Check Network tab in DevTools
   - Verify backend is running
   - Test endpoint with Postman first
   - Check request payload format

## Performance Tips (After Everything Works)

1. **Image Optimization**: Use Next.js Image component
2. **Code Splitting**: Dynamic imports for large components
3. **Caching**: Configure TanStack Query cache times
4. **Lazy Loading**: Implement for lists and heavy components
5. **Debouncing**: For search inputs and frequent API calls

## Security Considerations

1. **Never commit .env files** with real credentials
2. **Validate user input** before sending to API
3. **Store tokens securely** (httpOnly cookies preferred over localStorage)
4. **Sanitize HTML** if displaying user-generated content
5. **Use HTTPS** in production

## Next Steps After Basics

Once comfortable with the basics:
1. Add real-time features with WebSockets
2. Implement advanced search with filters
3. Add data visualization for media analytics
4. Build admin dashboard
5. Implement collaborative features
6. Add PWA capabilities

## Getting Help

- **Stack Overflow**: Tag questions with [next.js] [typescript] [react]
- **Reddit**: r/webdev, r/nextjs, r/reactjs
- **Discord**: Reactiflux, Next.js Discord
- **GitHub Issues**: For library-specific problems

## Remember

- 🎯 **Focus on one feature at a time**
- 🔄 **Iterate and improve gradually**
- 📚 **Read documentation before asking**
- 🐛 **Debugging is part of learning**
- 💪 **Every expert was once a beginner**
- 🎉 **Celebrate small wins**

Good luck building your frontend! The tldw_server backend is well-designed and will make your journey easier with its comprehensive API and documentation.