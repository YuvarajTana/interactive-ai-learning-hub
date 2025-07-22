# Complete TypeScript Learning Guide: Basics to Advanced with React & Node.js

## Table of Contents
1. [TypeScript Basics](#typescript-basics)
2. [Intermediate TypeScript](#intermediate-typescript)
3. [Advanced TypeScript](#advanced-typescript)
4. [TypeScript with React](#typescript-with-react)
5. [TypeScript with Node.js](#typescript-with-nodejs)
6. [Full-Stack TypeScript (React + Node.js)](#full-stack-typescript)
7. [Best Practices & Tooling](#best-practices--tooling)

---

## TypeScript Basics

### 1. Setup and Installation
```bash
# Install TypeScript globally
npm install -g typescript

# Create a new project
mkdir my-ts-project && cd my-ts-project
npm init -y

# Install TypeScript locally
npm install --save-dev typescript @types/node

# Create tsconfig.json
npx tsc --init
```

### 2. Basic Types
```typescript
// Primitive types
let name: string = "John";
let age: number = 30;
let isActive: boolean = true;
let data: null = null;
let value: undefined = undefined;

// Arrays
let numbers: number[] = [1, 2, 3];
let strings: Array<string> = ["a", "b", "c"];

// Tuples
let person: [string, number] = ["John", 30];

// Enum
enum Color {
  Red,
  Green,
  Blue
}
let myColor: Color = Color.Red;

// Any and Unknown
let anything: any = 42;
let something: unknown = 42;

// Void and Never
function logMessage(): void {
  console.log("Hello");
}

function throwError(): never {
  throw new Error("Error occurred");
}
```

### 3. Functions
```typescript
// Function with typed parameters and return type
function add(a: number, b: number): number {
  return a + b;
}

// Optional parameters
function greet(name: string, greeting?: string): string {
  return `${greeting || "Hello"}, ${name}!`;
}

// Default parameters
function createUser(name: string, age: number = 18): object {
  return { name, age };
}

// Rest parameters
function sum(...numbers: number[]): number {
  return numbers.reduce((total, num) => total + num, 0);
}

// Function overloads
function combine(a: string, b: string): string;
function combine(a: number, b: number): number;
function combine(a: any, b: any): any {
  return a + b;
}
```

### 4. Objects and Interfaces
```typescript
// Object type annotation
let user: { name: string; age: number } = {
  name: "John",
  age: 30
};

// Interface
interface User {
  name: string;
  age: number;
  email?: string; // Optional property
  readonly id: number; // Readonly property
}

const user1: User = {
  id: 1,
  name: "John",
  age: 30
};

// Interface with methods
interface Calculator {
  add(a: number, b: number): number;
  subtract(a: number, b: number): number;
}

class BasicCalculator implements Calculator {
  add(a: number, b: number): number {
    return a + b;
  }
  
  subtract(a: number, b: number): number {
    return a - b;
  }
}
```

---

## Intermediate TypeScript

### 1. Union and Intersection Types
```typescript
// Union types
type StringOrNumber = string | number;
let value: StringOrNumber = "hello";
value = 42; // Also valid

// Intersection types
interface Person {
  name: string;
}

interface Employee {
  employeeId: number;
}

type Staff = Person & Employee;

const staff: Staff = {
  name: "John",
  employeeId: 123
};

// Discriminated unions
interface Circle {
  kind: "circle";
  radius: number;
}

interface Rectangle {
  kind: "rectangle";
  width: number;
  height: number;
}

type Shape = Circle | Rectangle;

function getArea(shape: Shape): number {
  switch (shape.kind) {
    case "circle":
      return Math.PI * shape.radius ** 2;
    case "rectangle":
      return shape.width * shape.height;
  }
}
```

### 2. Generics
```typescript
// Generic functions
function identity<T>(arg: T): T {
  return arg;
}

const stringIdentity = identity<string>("hello");
const numberIdentity = identity<number>(42);

// Generic interfaces
interface Repository<T> {
  create(item: T): T;
  findById(id: number): T | null;
  update(id: number, item: Partial<T>): T;
  delete(id: number): boolean;
}

class UserRepository implements Repository<User> {
  create(user: User): User {
    // Implementation
    return user;
  }
  
  findById(id: number): User | null {
    // Implementation
    return null;
  }
  
  update(id: number, user: Partial<User>): User {
    // Implementation
    return {} as User;
  }
  
  delete(id: number): boolean {
    // Implementation
    return true;
  }
}

// Generic constraints
interface Lengthwise {
  length: number;
}

function logLength<T extends Lengthwise>(arg: T): T {
  console.log(arg.length);
  return arg;
}
```

### 3. Type Assertions and Type Guards
```typescript
// Type assertions
let someValue: unknown = "hello world";
let strLength: number = (someValue as string).length;
// or
let strLength2: number = (<string>someValue).length;

// Type guards
function isString(value: any): value is string {
  return typeof value === "string";
}

function processValue(value: string | number) {
  if (isString(value)) {
    // TypeScript knows value is string here
    console.log(value.toUpperCase());
  } else {
    // TypeScript knows value is number here
    console.log(value.toFixed(2));
  }
}

// instanceof type guard
class Dog {
  bark() {
    console.log("Woof!");
  }
}

class Cat {
  meow() {
    console.log("Meow!");
  }
}

function makeSound(animal: Dog | Cat) {
  if (animal instanceof Dog) {
    animal.bark();
  } else {
    animal.meow();
  }
}
```

---

## Advanced TypeScript

### 1. Utility Types
```typescript
interface User {
  id: number;
  name: string;
  email: string;
  age: number;
}

// Partial - makes all properties optional
type PartialUser = Partial<User>;

// Required - makes all properties required
type RequiredUser = Required<PartialUser>;

// Pick - select specific properties
type UserBasicInfo = Pick<User, 'name' | 'email'>;

// Omit - exclude specific properties
type UserWithoutId = Omit<User, 'id'>;

// Record - create object type with specific keys and values
type Roles = 'admin' | 'user' | 'guest';
type Permissions = Record<Roles, string[]>;

// ReturnType - extract return type of function
function getUser() {
  return { id: 1, name: "John" };
}
type UserReturnType = ReturnType<typeof getUser>;
```

### 2. Mapped Types
```typescript
// Basic mapped type
type Readonly<T> = {
  readonly [P in keyof T]: T[P];
};

// Optional mapped type
type Optional<T> = {
  [P in keyof T]?: T[P];
};

// Custom mapped type with transformation
type Stringify<T> = {
  [P in keyof T]: string;
};

type StringifiedUser = Stringify<User>;
// Result: { id: string; name: string; email: string; age: string; }

// Conditional mapped types
type NonNullable<T> = {
  [P in keyof T]: T[P] extends null | undefined ? never : T[P];
};
```

### 3. Conditional Types
```typescript
// Basic conditional type
type IsString<T> = T extends string ? true : false;

type Test1 = IsString<string>; // true
type Test2 = IsString<number>; // false

// Conditional types with infer
type GetArrayElementType<T> = T extends (infer U)[] ? U : never;

type StringArrayElement = GetArrayElementType<string[]>; // string
type NumberArrayElement = GetArrayElementType<number[]>; // number

// Advanced conditional type
type FunctionReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

function getString(): string {
  return "hello";
}

type StringReturn = FunctionReturnType<typeof getString>; // string
```

### 4. Template Literal Types
```typescript
// Basic template literal type
type Greeting = `Hello, ${string}!`;

let greeting1: Greeting = "Hello, World!"; // Valid
let greeting2: Greeting = "Hello, TypeScript!"; // Valid

// Template literal types with unions
type Size = "small" | "medium" | "large";
type Color = "red" | "blue" | "green";
type Variant = `${Size}-${Color}`;

// Template literal patterns
type EventName<T extends string> = `on${Capitalize<T>}`;
type ClickEvent = EventName<"click">; // "onClick"

// Parsing template literals
type ParseRoute<T extends string> = T extends `/${infer Path}` ? Path : never;
type UserRoute = ParseRoute<"/users">; // "users"
```

---

## TypeScript with React

### 1. React Setup with TypeScript
```bash
# Create React app with TypeScript
npx create-react-app my-app --template typescript

# Or add TypeScript to existing React project
npm install --save-dev typescript @types/react @types/react-dom
```

### 2. Component Types
```typescript
import React, { FC, Component, ReactNode } from 'react';

// Functional component with props
interface Props {
  name: string;
  age?: number;
  children?: ReactNode;
}

// Using FC (FunctionComponent)
const UserCard: FC<Props> = ({ name, age, children }) => {
  return (
    <div>
      <h2>{name}</h2>
      {age && <p>Age: {age}</p>}
      {children}
    </div>
  );
};

// Without FC (preferred approach)
function UserProfile({ name, age, children }: Props) {
  return (
    <div>
      <h2>{name}</h2>
      {age && <p>Age: {age}</p>}
      {children}
    </div>
  );
}

// Class component
interface State {
  count: number;
}

class Counter extends Component<Props, State> {
  state: State = {
    count: 0
  };

  increment = () => {
    this.setState({ count: this.state.count + 1 });
  };

  render() {
    return (
      <div>
        <p>Count: {this.state.count}</p>
        <button onClick={this.increment}>Increment</button>
      </div>
    );
  }
}
```

### 3. Hooks with TypeScript
```typescript
import React, { useState, useEffect, useContext, useReducer, useRef } from 'react';

// useState
function UserForm() {
  const [name, setName] = useState<string>('');
  const [age, setAge] = useState<number | null>(null);
  const [users, setUsers] = useState<User[]>([]);

  // useEffect
  useEffect(() => {
    // Fetch users
    async function fetchUsers() {
      const response = await fetch('/api/users');
      const userData: User[] = await response.json();
      setUsers(userData);
    }
    
    fetchUsers();
  }, []);

  return (
    <form>
      <input 
        value={name} 
        onChange={(e) => setName(e.target.value)} 
        placeholder="Name"
      />
      <input 
        type="number"
        value={age || ''} 
        onChange={(e) => setAge(Number(e.target.value))} 
        placeholder="Age"
      />
    </form>
  );
}

// useRef
function FocusInput() {
  const inputRef = useRef<HTMLInputElement>(null);

  const focusInput = () => {
    inputRef.current?.focus();
  };

  return (
    <div>
      <input ref={inputRef} />
      <button onClick={focusInput}>Focus Input</button>
    </div>
  );
}

// useContext
interface ThemeContextType {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
}

const ThemeContext = React.createContext<ThemeContextType | undefined>(undefined);

function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
}

// useReducer
interface State {
  count: number;
  loading: boolean;
}

type Action = 
  | { type: 'INCREMENT' }
  | { type: 'DECREMENT' }
  | { type: 'SET_LOADING'; payload: boolean };

function counterReducer(state: State, action: Action): State {
  switch (action.type) {
    case 'INCREMENT':
      return { ...state, count: state.count + 1 };
    case 'DECREMENT':
      return { ...state, count: state.count - 1 };
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    default:
      return state;
  }
}

function CounterWithReducer() {
  const [state, dispatch] = useReducer(counterReducer, { count: 0, loading: false });

  return (
    <div>
      <p>Count: {state.count}</p>
      <button onClick={() => dispatch({ type: 'INCREMENT' })}>+</button>
      <button onClick={() => dispatch({ type: 'DECREMENT' })}>-</button>
    </div>
  );
}
```

### 4. Event Handling
```typescript
import React, { ChangeEvent, FormEvent, MouseEvent } from 'react';

function EventsExample() {
  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    console.log(e.target.value);
  };

  const handleSelectChange = (e: ChangeEvent<HTMLSelectElement>) => {
    console.log(e.target.value);
  };

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    console.log('Form submitted');
  };

  const handleButtonClick = (e: MouseEvent<HTMLButtonElement>) => {
    console.log('Button clicked');
  };

  return (
    <form onSubmit={handleSubmit}>
      <input onChange={handleInputChange} />
      <select onChange={handleSelectChange}>
        <option value="1">Option 1</option>
        <option value="2">Option 2</option>
      </select>
      <button type="button" onClick={handleButtonClick}>
        Click me
      </button>
      <button type="submit">Submit</button>
    </form>
  );
}
```

---

## TypeScript with Node.js

### 1. Node.js Setup with TypeScript
```bash
# Initialize project
mkdir my-node-app && cd my-node-app
npm init -y

# Install dependencies
npm install express cors helmet morgan
npm install --save-dev typescript @types/node @types/express @types/cors @types/helmet @types/morgan ts-node nodemon

# Create tsconfig.json
npx tsc --init
```

### 2. Express Server with TypeScript
```typescript
// src/app.ts
import express, { Application, Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';

const app: Application = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Types
interface User {
  id: number;
  name: string;
  email: string;
}

interface CreateUserRequest {
  name: string;
  email: string;
}

// In-memory data store (replace with database)
let users: User[] = [
  { id: 1, name: 'John Doe', email: 'john@example.com' },
  { id: 2, name: 'Jane Smith', email: 'jane@example.com' }
];

// Routes
app.get('/api/users', (req: Request, res: Response) => {
  res.json(users);
});

app.get('/api/users/:id', (req: Request, res: Response) => {
  const id = parseInt(req.params.id);
  const user = users.find(u => u.id === id);
  
  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }
  
  res.json(user);
});

app.post('/api/users', (req: Request<{}, User, CreateUserRequest>, res: Response<User>) => {
  const { name, email } = req.body;
  
  if (!name || !email) {
    return res.status(400).json({ error: 'Name and email are required' } as any);
  }
  
  const newUser: User = {
    id: users.length + 1,
    name,
    email
  };
  
  users.push(newUser);
  res.status(201).json(newUser);
});

app.put('/api/users/:id', (req: Request<{ id: string }, User, Partial<CreateUserRequest>>, res: Response<User>) => {
  const id = parseInt(req.params.id);
  const userIndex = users.findIndex(u => u.id === id);
  
  if (userIndex === -1) {
    return res.status(404).json({ error: 'User not found' } as any);
  }
  
  users[userIndex] = { ...users[userIndex], ...req.body };
  res.json(users[userIndex]);
});

app.delete('/api/users/:id', (req: Request, res: Response) => {
  const id = parseInt(req.params.id);
  const userIndex = users.findIndex(u => u.id === id);
  
  if (userIndex === -1) {
    return res.status(404).json({ error: 'User not found' });
  }
  
  users.splice(userIndex, 1);
  res.status(204).send();
});

// Error handling middleware
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// 404 handler
app.use((req: Request, res: Response) => {
  res.status(404).json({ error: 'Route not found' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

export default app;
```

### 3. Database Integration (with Prisma)
```typescript
// Install Prisma
// npm install prisma @prisma/client
// npx prisma init

// prisma/schema.prisma
/*
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id    Int     @id @default(autoincrement())
  email String  @unique
  name  String?
  posts Post[]
}

model Post {
  id        Int      @id @default(autoincrement())
  title     String
  content   String?
  published Boolean  @default(false)
  author    User     @relation(fields: [authorId], references: [id])
  authorId  Int
}
*/

// src/services/userService.ts
import { PrismaClient, User, Prisma } from '@prisma/client';

const prisma = new PrismaClient();

export class UserService {
  async getAllUsers(): Promise<User[]> {
    return await prisma.user.findMany({
      include: {
        posts: true
      }
    });
  }

  async getUserById(id: number): Promise<User | null> {
    return await prisma.user.findUnique({
      where: { id },
      include: {
        posts: true
      }
    });
  }

  async createUser(data: Prisma.UserCreateInput): Promise<User> {
    return await prisma.user.create({
      data
    });
  }

  async updateUser(id: number, data: Prisma.UserUpdateInput): Promise<User> {
    return await prisma.user.update({
      where: { id },
      data
    });
  }

  async deleteUser(id: number): Promise<User> {
    return await prisma.user.delete({
      where: { id }
    });
  }
}
```

### 4. Middleware and Error Handling
```typescript
// src/middleware/validation.ts
import { Request, Response, NextFunction } from 'express';
import { validationResult, ValidationChain } from 'express-validator';

export const validate = (validations: ValidationChain[]) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    await Promise.all(validations.map(validation => validation.run(req)));

    const errors = validationResult(req);
    if (errors.isEmpty()) {
      return next();
    }

    res.status(400).json({
      error: 'Validation failed',
      details: errors.array()
    });
  };
};

// src/middleware/auth.ts
import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';

interface AuthRequest extends Request {
  user?: { id: number; email: string };
}

export const authenticateToken = (req: AuthRequest, res: Response, next: NextFunction) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  jwt.verify(token, process.env.JWT_SECRET as string, (err, user) => {
    if (err) {
      return res.status(403).json({ error: 'Invalid token' });
    }
    
    req.user = user as { id: number; email: string };
    next();
  });
};
```

---

## Full-Stack TypeScript (React + Node.js)

### 1. Shared Types
```typescript
// shared/types.ts (used by both frontend and backend)
export interface User {
  id: number;
  name: string;
  email: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface CreateUserRequest {
  name: string;
  email: string;
}

export interface UpdateUserRequest {
  name?: string;
  email?: string;
}

export interface ApiResponse<T> {
  data: T;
  message?: string;
  error?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}
```

### 2. API Client (Frontend)
```typescript
// frontend/src/services/api.ts
import { User, CreateUserRequest, UpdateUserRequest, ApiResponse, PaginatedResponse } from '../../../shared/types';

class ApiClient {
  private baseURL: string;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // User endpoints
  async getUsers(page: number = 1, limit: number = 10): Promise<PaginatedResponse<User>> {
    return this.request<PaginatedResponse<User>>(`/api/users?page=${page}&limit=${limit}`);
  }

  async getUserById(id: number): Promise<ApiResponse<User>> {
    return this.request<ApiResponse<User>>(`/api/users/${id}`);
  }

  async createUser(userData: CreateUserRequest): Promise<ApiResponse<User>> {
    return this.request<ApiResponse<User>>('/api/users', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }

  async updateUser(id: number, userData: UpdateUserRequest): Promise<ApiResponse<User>> {
    return this.request<ApiResponse<User>>(`/api/users/${id}`, {
      method: 'PUT',
      body: JSON.stringify(userData),
    });
  }

  async deleteUser(id: number): Promise<void> {
    await this.request(`/api/users/${id}`, {
      method: 'DELETE',
    });
  }
}

export const apiClient = new ApiClient(process.env.REACT_APP_API_URL || 'http://localhost:3001');
```

### 3. React Hooks for API
```typescript
// frontend/src/hooks/useUsers.ts
import { useState, useEffect } from 'react';
import { User, CreateUserRequest, UpdateUserRequest } from '../../../shared/types';
import { apiClient } from '../services/api';

export function useUsers() {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchUsers = async () => {
    try {
      setLoading(true);
      const response = await apiClient.getUsers();
      setUsers(response.data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch users');
    } finally {
      setLoading(false);
    }
  };

  const createUser = async (userData: CreateUserRequest): Promise<User | null> => {
    try {
      const response = await apiClient.createUser(userData);
      const newUser = response.data;
      setUsers(prev => [...prev, newUser]);
      return newUser;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create user');
      return null;
    }
  };

  const updateUser = async (id: number, userData: UpdateUserRequest): Promise<User | null> => {
    try {
      const response = await apiClient.updateUser(id, userData);
      const updatedUser = response.data;
      setUsers(prev => prev.map(user => user.id === id ? updatedUser : user));
      return updatedUser;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update user');
      return null;
    }
  };

  const deleteUser = async (id: number): Promise<boolean> => {
    try {
      await apiClient.deleteUser(id);
      setUsers(prev => prev.filter(user => user.id !== id));
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete user');
      return false;
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  return {
    users,
    loading,
    error,
    createUser,
    updateUser,
    deleteUser,
    refetch: fetchUsers
  };
}
```

### 4. React Components
```typescript
// frontend/src/components/UserList.tsx
import React, { useState } from 'react';
import { User, CreateUserRequest } from '../../../shared/types';
import { useUsers } from '../hooks/useUsers';

interface UserListProps {
  onUserSelect?: (user: User) => void;
}

export function UserList({ onUserSelect }: UserListProps) {
  const { users, loading, error, createUser, deleteUser } = useUsers();
  const [showForm, setShowForm] = useState(false);

  if (loading) return <div>Loading users...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <div className="header">
        <h2>Users</h2>
        <button onClick={() => setShowForm(!showForm)}>
          {showForm ? 'Cancel' : 'Add User'}
        </button>
      </div>

      {showForm && (
        <UserForm
          onSubmit={async (userData) => {
            const newUser = await createUser(userData);
            if (newUser) {
              setShowForm(false);
            }
          }}
          onCancel={() => setShowForm(false)}
        />
      )}

      <div className="user-list">
        {users.map(user => (
          <UserCard
            key={user.id}
            user={user}
            onSelect={() => onUserSelect?.(user)}
            onDelete={() => deleteUser(user.id)}
          />
        ))}
      </div>
    </div>
  );
}

interface UserCardProps {
  user: User;
  onSelect: () => void;
  onDelete: () => void;
}

function UserCard({ user, onSelect, onDelete }: UserCardProps) {
  return (
    <div className="user-card">
      <h3>{user.name}</h3>
      <p>{user.email}</p>
      <div className="actions">
        <button onClick={onSelect}>View</button>
        <button onClick={onDelete} className="danger">Delete</button>
      </div>
    </div>
  );
}

interface UserFormProps {
  onSubmit: (userData: CreateUserRequest) => void;
  onCancel: () => void;
}

function UserForm({ onSubmit, onCancel }: UserFormProps) {
  const [formData, setFormData] = useState<CreateUserRequest>({
    name: '',
    email: ''
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (formData.name && formData.email) {
      onSubmit(formData);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="user-form">
      <input
        type="text"
        placeholder="Name"
        value={formData.name}
        onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
        required
      />
      <input
        type="email"
        placeholder="Email"
        value={formData.email}
        onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
        required
      />
      <div className="form-actions">
        <button type="submit">Create User</button>
        <button type="button" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}
```

---

## Best Practices & Tooling

### 1. TSConfig Configuration
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noFallthroughCasesInSwitch": true,
    "module": "ESNext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "declaration": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "removeComments": true,
    "noImplicitAny": true,
    "noImplicitReturns": true,
    "noImplicitThis": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "exactOptionalPropertyTypes": true
  },
  "include": [
    "src/**/*"
  ],
  "exclude": [
    "node_modules",
    "dist"
  ]
}
```

### 2. ESLint Configuration
```json
{
  "extends": [
    "@typescript-eslint/recommended",
    "@typescript-eslint/recommended-requiring-type-checking"
  ],
  "parser": "@typescript-eslint/parser",
  "parserOptions": {
    "project": "./tsconfig.json"
  },
  "plugins": ["@typescript-eslint"],
  "rules": {
    "@typescript-eslint/no-unused-vars": "error",
    "@typescript-eslint/explicit-function-return-type": "warn",
    "@typescript-eslint/no-explicit-any": "warn",
    "@typescript-eslint/no-non-null-assertion": "error"
  }
}
```

### 3. Package.json Scripts
```json
{
  "scripts": {
    "build": "tsc",
    "dev": "ts-node-dev --respawn --transpile-only src/index.ts",
    "start": "node dist/index.js",
    "lint": "eslint src/**/*.ts",
    "lint:fix": "eslint src/**/*.ts --fix",
    "test": "jest",
    "test:watch": "jest --watch",
    "type-check": "tsc --noEmit"
  }
}
```

### 4. Best Practices Summary

#### Type Safety
- Always enable `strict` mode in TypeScript
- Use `unknown` instead of `any` when possible
- Prefer type assertions with type guards
- Use discriminated unions for complex state management

#### Code Organization
- Separate types into dedicated files
- Use barrel exports (index.ts files) for cleaner imports
- Create shared types for frontend/backend communication
- Use consistent naming conventions

#### Performance
- Use React.memo for expensive components
- Implement proper error boundaries
- Use lazy loading for large components
- Optimize bundle size with proper tree shaking

#### Testing
- Write unit tests for utility functions
- Use React Testing Library for component tests
- Mock API calls in tests
- Test TypeScript types with type-only imports

#### Development Workflow
- Use pre-commit hooks for linting and type checking
- Set up automated testing in CI/CD
- Use semantic versioning
- Document complex type definitions

This guide provides a comprehensive path from TypeScript basics to advanced full-stack development. Practice each section thoroughly before moving to the next, and build real projects to solidify your understanding.