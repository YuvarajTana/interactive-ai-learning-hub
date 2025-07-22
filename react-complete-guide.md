# Complete React.js Guide: Beginner to Advanced

## Table of Contents
1. [Introduction to React](#introduction-to-react)
2. [Getting Started](#getting-started)
3. [JSX Fundamentals](#jsx-fundamentals)
4. [Components](#components)
5. [Props](#props)
6. [State Management](#state-management)
7. [Event Handling](#event-handling)
8. [Conditional Rendering](#conditional-rendering)
9. [Lists and Keys](#lists-and-keys)
10. [React Hooks](#react-hooks)
11. [Advanced Hook Patterns](#advanced-hook-patterns)
12. [Component Lifecycle](#component-lifecycle)
13. [Context API](#context-api)
14. [Performance Optimization](#performance-optimization)
15. [Error Boundaries](#error-boundaries)
16. [Routing with React Router](#routing-with-react-router)
17. [State Management Libraries](#state-management-libraries)
18. [Testing React Applications](#testing-react-applications)
19. [Advanced Patterns](#advanced-patterns)
20. [Production and Deployment](#production-and-deployment)

---

## Introduction to React

React is a JavaScript library for building user interfaces, particularly web applications. Created by Facebook (now Meta), React follows a component-based architecture that makes it easy to build complex UIs from small, reusable pieces of code.

### Key Features:
- **Component-Based**: Build encapsulated components that manage their own state
- **Virtual DOM**: Efficient updates through a virtual representation of the UI
- **One-Way Data Flow**: Predictable data flow makes debugging easier
- **JSX**: Write HTML-like syntax in JavaScript
- **Ecosystem**: Rich ecosystem of tools and libraries

---

## Getting Started

### Prerequisites
- Basic knowledge of HTML, CSS, and JavaScript
- Understanding of ES6+ features (arrow functions, destructuring, modules)
- Node.js installed on your machine

### Setting Up Your First React App

```bash
# Using Create React App
npx create-react-app my-app
cd my-app
npm start

# Using Vite (faster alternative)
npm create vite@latest my-react-app -- --template react
cd my-react-app
npm install
npm run dev
```

### Project Structure
```
my-app/
  src/
    App.js          # Main App component
    App.css         # App styles
    index.js        # Entry point
    index.css       # Global styles
  public/
    index.html      # HTML template
  package.json      # Dependencies and scripts
```

---

## JSX Fundamentals

JSX (JavaScript XML) allows you to write HTML-like syntax in JavaScript. It gets compiled to regular JavaScript function calls.

### Basic JSX Rules

```jsx
// JSX must have one parent element
const App = () => {
  return (
    <div>
      <h1>Hello, React!</h1>
      <p>Welcome to JSX</p>
    </div>
  );
};

// Use React.Fragment or <> </> for multiple elements without wrapper
const App = () => {
  return (
    <>
      <h1>Hello, React!</h1>
      <p>Welcome to JSX</p>
    </>
  );
};
```

### JavaScript Expressions in JSX

```jsx
const App = () => {
  const name = "John";
  const age = 25;
  
  return (
    <div>
      <h1>Hello, {name}!</h1>
      <p>You are {age} years old</p>
      <p>Next year you'll be {age + 1}</p>
      <p>Today is {new Date().toDateString()}</p>
    </div>
  );
};
```

### JSX Attributes

```jsx
const App = () => {
  const imageUrl = "https://example.com/image.jpg";
  const altText = "Beautiful landscape";
  
  return (
    <div>
      <img src={imageUrl} alt={altText} />
      <button className="primary-btn" onClick={() => alert('Clicked!')}>
        Click me
      </button>
    </div>
  );
};
```

---

## Components

Components are the building blocks of React applications. They are reusable pieces of UI that can accept inputs (props) and return JSX.

### Functional Components

```jsx
// Simple functional component
const Welcome = () => {
  return <h1>Welcome to React!</h1>;
};

// Component with parameters
const Greeting = ({ name, age }) => {
  return (
    <div>
      <h2>Hello, {name}!</h2>
      <p>Age: {age}</p>
    </div>
  );
};

// Using components
const App = () => {
  return (
    <div>
      <Welcome />
      <Greeting name="Alice" age={30} />
      <Greeting name="Bob" age={25} />
    </div>
  );
};
```

### Component Best Practices

```jsx
// 1. Use descriptive names
const UserProfile = ({ user }) => {
  return (
    <div className="user-profile">
      <h2>{user.name}</h2>
      <p>{user.email}</p>
    </div>
  );
};

// 2. Keep components small and focused
const Avatar = ({ src, alt, size = "medium" }) => {
  return (
    <img 
      className={`avatar avatar--${size}`}
      src={src} 
      alt={alt} 
    />
  );
};

// 3. Extract reusable logic
const Button = ({ children, variant = "primary", onClick, disabled }) => {
  return (
    <button 
      className={`btn btn--${variant}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
};
```

---

## Props

Props (properties) are how you pass data from parent components to child components. They are read-only and help make components reusable.

### Basic Props Usage

```jsx
// Child component receiving props
const UserCard = ({ name, email, avatar, isOnline }) => {
  return (
    <div className={`user-card ${isOnline ? 'online' : 'offline'}`}>
      <img src={avatar} alt={`${name}'s avatar`} />
      <h3>{name}</h3>
      <p>{email}</p>
      <span className="status">
        {isOnline ? 'Online' : 'Offline'}
      </span>
    </div>
  );
};

// Parent component passing props
const UserList = () => {
  const users = [
    { id: 1, name: "Alice", email: "alice@example.com", avatar: "avatar1.jpg", isOnline: true },
    { id: 2, name: "Bob", email: "bob@example.com", avatar: "avatar2.jpg", isOnline: false },
  ];
  
  return (
    <div className="user-list">
      {users.map(user => (
        <UserCard 
          key={user.id}
          name={user.name}
          email={user.email}
          avatar={user.avatar}
          isOnline={user.isOnline}
        />
      ))}
    </div>
  );
};
```

### Props Destructuring and Default Values

```jsx
// Destructuring props
const ProductCard = ({ title, price, image, onAddToCart, isInStock = true }) => {
  return (
    <div className="product-card">
      <img src={image} alt={title} />
      <h3>{title}</h3>
      <p className="price">${price}</p>
      <button 
        onClick={() => onAddToCart({ title, price })}
        disabled={!isInStock}
      >
        {isInStock ? 'Add to Cart' : 'Out of Stock'}
      </button>
    </div>
  );
};

// Props validation with PropTypes (optional)
import PropTypes from 'prop-types';

ProductCard.propTypes = {
  title: PropTypes.string.isRequired,
  price: PropTypes.number.isRequired,
  image: PropTypes.string.isRequired,
  onAddToCart: PropTypes.func.isRequired,
  isInStock: PropTypes.bool
};
```

---

## State Management

State represents data that can change over time. In functional components, we use the `useState` hook to manage state.

### useState Hook Basics

```jsx
import { useState } from 'react';

const Counter = () => {
  const [count, setCount] = useState(0);
  
  const increment = () => setCount(count + 1);
  const decrement = () => setCount(count - 1);
  const reset = () => setCount(0);
  
  return (
    <div>
      <h2>Count: {count}</h2>
      <button onClick={increment}>+</button>
      <button onClick={decrement}>-</button>
      <button onClick={reset}>Reset</button>
    </div>
  );
};
```

### Complex State Management

```jsx
const TodoApp = () => {
  const [todos, setTodos] = useState([]);
  const [inputValue, setInputValue] = useState('');
  
  const addTodo = () => {
    if (inputValue.trim()) {
      const newTodo = {
        id: Date.now(),
        text: inputValue,
        completed: false
      };
      setTodos([...todos, newTodo]);
      setInputValue('');
    }
  };
  
  const toggleTodo = (id) => {
    setTodos(todos.map(todo =>
      todo.id === id ? { ...todo, completed: !todo.completed } : todo
    ));
  };
  
  const deleteTodo = (id) => {
    setTodos(todos.filter(todo => todo.id !== id));
  };
  
  return (
    <div>
      <div>
        <input 
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Add a todo..."
        />
        <button onClick={addTodo}>Add</button>
      </div>
      
      <ul>
        {todos.map(todo => (
          <li key={todo.id}>
            <span 
              style={{
                textDecoration: todo.completed ? 'line-through' : 'none'
              }}
              onClick={() => toggleTodo(todo.id)}
            >
              {todo.text}
            </span>
            <button onClick={() => deleteTodo(todo.id)}>Delete</button>
          </li>
        ))}
      </ul>
    </div>
  );
};
```

### State Update Patterns

```jsx
const FormExample = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    age: 0
  });
  
  // Update object state
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value
    }));
  };
  
  // Update array state
  const [items, setItems] = useState([]);
  
  const addItem = (newItem) => {
    setItems(prevItems => [...prevItems, newItem]);
  };
  
  const removeItem = (index) => {
    setItems(prevItems => prevItems.filter((_, i) => i !== index));
  };
  
  const updateItem = (index, newValue) => {
    setItems(prevItems => 
      prevItems.map((item, i) => i === index ? newValue : item)
    );
  };
  
  return (
    <form>
      <input
        name="name"
        value={formData.name}
        onChange={handleInputChange}
        placeholder="Name"
      />
      <input
        name="email"
        value={formData.email}
        onChange={handleInputChange}
        placeholder="Email"
      />
      <input
        name="age"
        type="number"
        value={formData.age}
        onChange={handleInputChange}
        placeholder="Age"
      />
    </form>
  );
};
```

---

## Event Handling

React handles events using SyntheticEvents, which provide a consistent interface across different browsers.

### Basic Event Handling

```jsx
const EventExamples = () => {
  const [message, setMessage] = useState('');
  
  const handleClick = () => {
    alert('Button clicked!');
  };
  
  const handleMouseEnter = () => {
    console.log('Mouse entered');
  };
  
  const handleInputChange = (e) => {
    setMessage(e.target.value);
  };
  
  const handleSubmit = (e) => {
    e.preventDefault(); // Prevent form submission
    console.log('Form submitted with message:', message);
  };
  
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      console.log('Enter key pressed');
    }
  };
  
  return (
    <div>
      <button onClick={handleClick}>Click me</button>
      
      <div onMouseEnter={handleMouseEnter}>
        Hover over me
      </div>
      
      <form onSubmit={handleSubmit}>
        <input 
          type="text"
          value={message}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Type something..."
        />
        <button type="submit">Submit</button>
      </form>
    </div>
  );
};
```

### Event Object and Parameters

```jsx
const AdvancedEventHandling = () => {
  const handleButtonClick = (buttonName, e) => {
    e.preventDefault();
    console.log(`${buttonName} button clicked`);
    console.log('Event details:', e.type, e.target);
  };
  
  const handleListItemClick = (itemId) => (e) => {
    console.log(`Item ${itemId} clicked`);
  };
  
  return (
    <div>
      <button onClick={(e) => handleButtonClick('Primary', e)}>
        Primary Button
      </button>
      
      <ul>
        {[1, 2, 3].map(id => (
          <li key={id} onClick={handleListItemClick(id)}>
            Item {id}
          </li>
        ))}
      </ul>
    </div>
  );
};
```

---

## Conditional Rendering

React allows you to conditionally render elements based on certain conditions.

### If-Else Patterns

```jsx
const ConditionalExample = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Using ternary operator
  return (
    <div>
      {isLoggedIn ? (
        <div>
          <h2>Welcome back!</h2>
          <button onClick={() => setIsLoggedIn(false)}>Logout</button>
        </div>
      ) : (
        <div>
          <h2>Please log in</h2>
          <button onClick={() => setIsLoggedIn(true)}>Login</button>
        </div>
      )}
      
      {/* Logical AND operator */}
      {loading && <div className="spinner">Loading...</div>}
      
      {/* Multiple conditions */}
      {user && user.isAdmin && (
        <div className="admin-panel">
          <h3>Admin Panel</h3>
        </div>
      )}
      
      {/* Conditional CSS classes */}
      <div className={`notification ${isLoggedIn ? 'success' : 'warning'}`}>
        Status: {isLoggedIn ? 'Logged in' : 'Not logged in'}
      </div>
    </div>
  );
};
```

### Switch-Case Pattern

```jsx
const StatusDisplay = ({ status }) => {
  const renderStatus = () => {
    switch (status) {
      case 'loading':
        return <div className="status loading">Loading...</div>;
      case 'success':
        return <div className="status success">Operation successful!</div>;
      case 'error':
        return <div className="status error">An error occurred</div>;
      case 'idle':
      default:
        return <div className="status idle">Ready</div>;
    }
  };
  
  return (
    <div>
      <h3>Current Status:</h3>
      {renderStatus()}
    </div>
  );
};
```

---

## Lists and Keys

Rendering lists of data is a common pattern in React applications.

### Basic List Rendering

```jsx
const ListExample = () => {
  const fruits = ['Apple', 'Banana', 'Orange', 'Grape'];
  const users = [
    { id: 1, name: 'Alice', age: 25 },
    { id: 2, name: 'Bob', age: 30 },
    { id: 3, name: 'Charlie', age: 35 }
  ];
  
  return (
    <div>
      {/* Simple array */}
      <h3>Fruits:</h3>
      <ul>
        {fruits.map((fruit, index) => (
          <li key={index}>{fruit}</li>
        ))}
      </ul>
      
      {/* Array of objects */}
      <h3>Users:</h3>
      <div>
        {users.map(user => (
          <div key={user.id} className="user-card">
            <h4>{user.name}</h4>
            <p>Age: {user.age}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
```

### Dynamic Lists with State

```jsx
const DynamicList = () => {
  const [items, setItems] = useState([
    { id: 1, text: 'Learn React', completed: false },
    { id: 2, text: 'Build a project', completed: false },
  ]);
  const [newItem, setNewItem] = useState('');
  
  const addItem = () => {
    if (newItem.trim()) {
      const item = {
        id: Date.now(),
        text: newItem,
        completed: false
      };
      setItems([...items, item]);
      setNewItem('');
    }
  };
  
  const toggleItem = (id) => {
    setItems(items.map(item =>
      item.id === id ? { ...item, completed: !item.completed } : item
    ));
  };
  
  const deleteItem = (id) => {
    setItems(items.filter(item => item.id !== id));
  };
  
  return (
    <div>
      <div>
        <input 
          value={newItem}
          onChange={(e) => setNewItem(e.target.value)}
          placeholder="Add new item..."
        />
        <button onClick={addItem}>Add</button>
      </div>
      
      <ul>
        {items.map(item => (
          <li key={item.id} className={item.completed ? 'completed' : ''}>
            <span onClick={() => toggleItem(item.id)}>
              {item.text}
            </span>
            <button onClick={() => deleteItem(item.id)}>Delete</button>
          </li>
        ))}
      </ul>
    </div>
  );
};
```

---

## React Hooks

Hooks are functions that let you "hook into" React features from functional components.

### useEffect Hook

```jsx
import { useState, useEffect } from 'react';

const DataFetching = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Effect runs after component mounts
  useEffect(() => {
    const fetchUsers = async () => {
      try {
        setLoading(true);
        const response = await fetch('https://jsonplaceholder.typicode.com/users');
        const userData = await response.json();
        setUsers(userData);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    
    fetchUsers();
  }, []); // Empty dependency array means this effect runs once on mount
  
  // Effect with cleanup
  useEffect(() => {
    const timer = setInterval(() => {
      console.log('Timer tick');
    }, 1000);
    
    return () => clearInterval(timer); // Cleanup function
  }, []);
  
  // Effect that depends on state
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    document.title = `Count: ${count}`;
  }, [count]); // Runs when count changes
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  
  return (
    <div>
      <h2>Users</h2>
      <div>Count: {count}</div>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      
      <ul>
        {users.map(user => (
          <li key={user.id}>{user.name} - {user.email}</li>
        ))}
      </ul>
    </div>
  );
};
```

### Custom Hooks

```jsx
// Custom hook for API calls
const useApi = (url) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to fetch');
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [url]);
  
  return { data, loading, error };
};

// Custom hook for local storage
const useLocalStorage = (key, initialValue) => {
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      return initialValue;
    }
  });
  
  const setValue = (value) => {
    try {
      setStoredValue(value);
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error('Error saving to localStorage:', error);
    }
  };
  
  return [storedValue, setValue];
};

// Using custom hooks
const UserProfile = () => {
  const { data: user, loading, error } = useApi('/api/user/profile');
  const [preferences, setPreferences] = useLocalStorage('userPrefs', {});
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  
  return (
    <div>
      <h2>{user?.name}</h2>
      <p>{user?.email}</p>
    </div>
  );
};
```

---

## Advanced Hook Patterns

### useReducer Hook

```jsx
import { useReducer } from 'react';

const initialState = {
  count: 0,
  step: 1
};

const counterReducer = (state, action) => {
  switch (action.type) {
    case 'increment':
      return { ...state, count: state.count + state.step };
    case 'decrement':
      return { ...state, count: state.count - state.step };
    case 'reset':
      return { ...state, count: 0 };
    case 'setStep':
      return { ...state, step: action.payload };
    default:
      throw new Error(`Unknown action: ${action.type}`);
  }
};

const AdvancedCounter = () => {
  const [state, dispatch] = useReducer(counterReducer, initialState);
  
  return (
    <div>
      <h2>Count: {state.count}</h2>
      <p>Step: {state.step}</p>
      
      <div>
        <button onClick={() => dispatch({ type: 'increment' })}>
          +{state.step}
        </button>
        <button onClick={() => dispatch({ type: 'decrement' })}>
          -{state.step}
        </button>
        <button onClick={() => dispatch({ type: 'reset' })}>
          Reset
        </button>
      </div>
      
      <div>
        <input 
          type="number"
          value={state.step}
          onChange={(e) => dispatch({ 
            type: 'setStep', 
            payload: parseInt(e.target.value) || 1 
          })}
        />
      </div>
    </div>
  );
};
```

### useMemo and useCallback

```jsx
import { useState, useMemo, useCallback } from 'react';

const ExpensiveComponent = () => {
  const [count, setCount] = useState(0);
  const [todos, setTodos] = useState([]);
  const [filter, setFilter] = useState('all');
  
  // useMemo for expensive calculations
  const expensiveValue = useMemo(() => {
    console.log('Calculating expensive value...');
    return count * 1000;
  }, [count]); // Only recalculates when count changes
  
  // useMemo for filtered data
  const filteredTodos = useMemo(() => {
    console.log('Filtering todos...');
    return todos.filter(todo => {
      if (filter === 'completed') return todo.completed;
      if (filter === 'active') return !todo.completed;
      return true;
    });
  }, [todos, filter]);
  
  // useCallback for event handlers
  const addTodo = useCallback((text) => {
    setTodos(prev => [...prev, {
      id: Date.now(),
      text,
      completed: false
    }]);
  }, []);
  
  const toggleTodo = useCallback((id) => {
    setTodos(prev => prev.map(todo =>
      todo.id === id ? { ...todo, completed: !todo.completed } : todo
    ));
  }, []);
  
  return (
    <div>
      <h2>Count: {count}</h2>
      <p>Expensive Value: {expensiveValue}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      
      <TodoList 
        todos={filteredTodos}
        onAddTodo={addTodo}
        onToggleTodo={toggleTodo}
      />
      
      <FilterButtons 
        currentFilter={filter}
        onFilterChange={setFilter}
      />
    </div>
  );
};
```

---

## Context API

Context provides a way to pass data through the component tree without having to pass props down manually at every level.

### Creating and Using Context

```jsx
import { createContext, useContext, useState } from 'react';

// Create context
const ThemeContext = createContext();
const UserContext = createContext();

// Context provider component
const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState('light');
  
  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };
  
  const value = {
    theme,
    toggleTheme
  };
  
  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};

const UserProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const login = async (credentials) => {
    setLoading(true);
    try {
      // Simulate API call
      const userData = await fakeLogin(credentials);
      setUser(userData);
    } catch (error) {
      console.error('Login failed:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const logout = () => {
    setUser(null);
  };
  
  return (
    <UserContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </UserContext.Provider>
  );
};

// Custom hooks for consuming context
const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

const useUser = () => {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};

// Components using context
const Header = () => {
  const { theme, toggleTheme } = useTheme();
  const { user, logout } = useUser();
  
  return (
    <header className={`header header--${theme}`}>
      <h1>My App</h1>
      <div>
        <button onClick={toggleTheme}>
          Switch to {theme === 'light' ? 'dark' : 'light'} mode
        </button>
        {user ? (
          <div>
            <span>Welcome, {user.name}!</span>
            <button onClick={logout}>Logout</button>
          </div>
        ) : (
          <LoginButton />
        )}
      </div>
    </header>
  );
};

// App with providers
const App = () => {
  return (
    <ThemeProvider>
      <UserProvider>
        <div className="app">
          <Header />
          <Main />
        </div>
      </UserProvider>
    </ThemeProvider>
  );
};
```

---

## Performance Optimization

### React.memo

```jsx
import { memo, useState } from 'react';

// Memoized component
const ExpensiveChild = memo(({ data, onUpdate }) => {
  console.log('ExpensiveChild rendering...');
  
  return (
    <div>
      <h3>Expensive Component</h3>
      <p>Data: {JSON.stringify(data)}</p>
      <button onClick={onUpdate}>Update</button>
    </div>
  );
});

// Component with custom comparison
const OptimizedComponent = memo(({ user, theme }) => {
  return (
    <div className={`user-card user-card--${theme}`}>
      <h3>{user.name}</h3>
      <p>{user.email}</p>
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison logic
  return prevProps.user.id === nextProps.user.id &&
         prevProps.theme === nextProps.theme;
});
```

### Lazy Loading and Suspense

```jsx
import { lazy, Suspense } from 'react';

// Lazy load components
const Dashboard = lazy(() => import('./Dashboard'));
const Settings = lazy(() => import('./Settings'));
const Profile = lazy(() => import('./Profile'));

const App = () => {
  const [currentView, setCurrentView] = useState('dashboard');
  
  const renderView = () => {
    switch (currentView) {
      case 'dashboard':
        return <Dashboard />;
      case 'settings':
        return <Settings />;
      case 'profile':
        return <Profile />;
      default:
        return <Dashboard />;
    }
  };
  
  return (
    <div className="app">
      <nav>
        <button onClick={() => setCurrentView('dashboard')}>Dashboard</button>
        <button onClick={() => setCurrentView('settings')}>Settings</button>
        <button onClick={() => setCurrentView('profile')}>Profile</button>
      </nav>
      
      <main>
        <Suspense fallback={<div>Loading...</div>}>
          {renderView()}
        </Suspense>
      </main>
    </div>
  );
};
```

---

## Error Boundaries

Error boundaries are React components that catch JavaScript errors anywhere in their child component tree.

```jsx
import { Component } from 'react';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }
  
  static getDerivedStateFromError(error) {
    return { hasError: true };
  }
  
  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
    
    // Log error to error reporting service
    console.error('Error caught by boundary:', error, errorInfo);
  }
  
  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>Something went wrong.</h2>
          <details style={{ whiteSpace: 'pre-wrap' }}>
            {this.state.error && this.state.error.toString()}
            <br />
            {this.state.errorInfo.componentStack}
          </details>
        </div>
      );
    }
    
    return this.props.children;
  }
}

// Using Error Boundary
const App = () => {
  return (
    <ErrorBoundary>
      <Header />
      <ErrorBoundary>
        <Sidebar />
      </ErrorBoundary>
      <ErrorBoundary>
        <MainContent />
      </ErrorBoundary>
    </ErrorBoundary>
  );
};
```

---

## Advanced Patterns

### Higher-Order Components (HOCs)

```jsx
// HOC for adding loading state
const withLoading = (WrappedComponent) => {
  return function WithLoadingComponent({ isLoading, ...props }) {
    if (isLoading) {
      return <div className="spinner">Loading...</div>;
    }
    return <WrappedComponent {...props} />;
  };
};

// HOC for authentication
const withAuth = (WrappedComponent) => {
  return function WithAuthComponent(props) {
    const { user } = useUser();
    
    if (!user) {
      return <div>Please log in to access this page.</div>;
    }
    
    return <WrappedComponent {...props} user={user} />;
  };
};

// Using HOCs
const UserDashboard = ({ user }) => {
  return (
    <div>
      <h2>Welcome, {user.name}!</h2>
    </div>
  );
};

const EnhancedDashboard = withAuth(withLoading(UserDashboard));
```

### Render Props Pattern

```jsx
const DataFetcher = ({ url, children }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(url);
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [url]);
  
  return children({ data, loading, error });
};

// Using render props
const UserList = () => {
  return (
    <DataFetcher url="/api/users">
      {({ data: users, loading, error }) => {
        if (loading) return <div>Loading users...</div>;
        if (error) return <div>Error: {error}</div>;
        
        return (
          <ul>
            {users.map(user => (
              <li key={user.id}>{user.name}</li>
            ))}
          </ul>
        );
      }}
    </DataFetcher>
  );
};
```

### Compound Components

```jsx
const Modal = ({ children, isOpen, onClose }) => {
  if (!isOpen) return null;
  
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        {children}
      </div>
    </div>
  );
};

Modal.Header = ({ children }) => (
  <div className="modal-header">{children}</div>
);

Modal.Body = ({ children }) => (
  <div className="modal-body">{children}</div>
);

Modal.Footer = ({ children }) => (
  <div className="modal-footer">{children}</div>
);

// Using compound components
const App = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  return (
    <div>
      <button onClick={() => setIsModalOpen(true)}>Open Modal</button>
      
      <Modal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)}>
        <Modal.Header>
          <h2>Confirmation</h2>
        </Modal.Header>
        <Modal.Body>
          <p>Are you sure you want to delete this item?</p>
        </Modal.Body>
        <Modal.Footer>
          <button onClick={() => setIsModalOpen(false)}>Cancel</button>
          <button onClick={() => setIsModalOpen(false)}>Delete</button>
        </Modal.Footer>
      </Modal>
    </div>
  );
};
```

---

## Best Practices and Tips

### 1. Component Organization
- Keep components small and focused on a single responsibility
- Use descriptive names for components and props
- Organize files and folders logically
- Separate business logic from presentation logic

### 2. State Management
- Start with local state, lift up when needed
- Use useReducer for complex state logic
- Consider Context for app-wide state
- Avoid unnecessary re-renders with useMemo and useCallback

### 3. Performance
- Use React.memo for expensive components
- Implement lazy loading for large components
- Optimize bundle size with code splitting
- Profile your app with React DevTools

### 4. Error Handling
- Always implement error boundaries
- Handle loading and error states in UI
- Provide meaningful error messages
- Log errors for debugging

### 5. Accessibility
- Use semantic HTML elements
- Provide proper ARIA labels
- Ensure keyboard navigation works
- Test with screen readers

### 6. Testing
- Write unit tests for components and hooks
- Test user interactions, not implementation details
- Use React Testing Library for testing
- Implement integration and end-to-end tests

---

## Next Steps

Once you've mastered these React fundamentals, consider exploring:

- **Next.js** - React framework for production applications
- **TypeScript** - Add type safety to your React apps
- **State Management Libraries** - Redux, Zustand, or Recoil
- **Styling Solutions** - Styled Components, Emotion, or Tailwind CSS
- **Testing** - Jest, React Testing Library, Cypress
- **Mobile Development** - React Native
- **Desktop Applications** - Electron with React

Remember, the best way to learn React is by building projects. Start small, practice regularly, and gradually take on more complex challenges. The React ecosystem is vast and constantly evolving, so stay curious and keep learning!