# Complete Playwright Browser Automation Guide: Basics to Advanced

## Table of Contents
1. [Introduction & Setup](#introduction--setup)
2. [Basic Browser Operations](#basic-browser-operations)
3. [Element Interaction](#element-interaction)
4. [Page Navigation & Waiting](#page-navigation--waiting)
5. [Form Handling](#form-handling)
6. [Screenshots & Videos](#screenshots--videos)
7. [Network Interception](#network-interception)
8. [Mobile & Device Emulation](#mobile--device-emulation)
9. [Authentication & Sessions](#authentication--sessions)
10. [Parallel Testing](#parallel-testing)
11. [Page Object Model](#page-object-model)
12. [Advanced Patterns](#advanced-patterns)
13. [Performance Testing](#performance-testing)
14. [Best Practices](#best-practices)

## Introduction & Setup

### What is Playwright?
Playwright is a Node.js library for automating Chromium, Firefox, and WebKit browsers. It provides a unified API for cross-browser automation with excellent reliability and speed.

### Installation
```bash
# Install Playwright
npm init playwright@latest

# Or add to existing project
npm install @playwright/test
npx playwright install
```

### Basic Project Structure
```
my-tests/
├── tests/
│   ├── example.spec.js
│   └── utils/
├── playwright.config.js
├── package.json
└── README.md
```

### Configuration (playwright.config.js)
```javascript
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
  ],
});
```

## Basic Browser Operations

### Simple Test Structure
```javascript
import { test, expect } from '@playwright/test';

test('basic test', async ({ page }) => {
  await page.goto('https://example.com');
  await expect(page).toHaveTitle(/Example/);
});
```

### Browser & Context Management
```javascript
import { chromium } from 'playwright';

// Manual browser management
const browser = await chromium.launch({ headless: false });
const context = await browser.newContext();
const page = await context.newPage();

await page.goto('https://example.com');

// Cleanup
await page.close();
await context.close();
await browser.close();
```

### Multiple Pages
```javascript
test('multiple pages', async ({ browser }) => {
  const context = await browser.newContext();
  
  const page1 = await context.newPage();
  const page2 = await context.newPage();
  
  await page1.goto('https://example.com');
  await page2.goto('https://github.com');
  
  // Switch between pages
  await page1.bringToFront();
  await page2.bringToFront();
});
```

## Element Interaction

### Locators
```javascript
// By text
await page.locator('text=Click me').click();

// By role
await page.getByRole('button', { name: 'Submit' }).click();

// By test ID
await page.getByTestId('submit-button').click();

// By CSS selector
await page.locator('.btn-primary').click();

// By XPath
await page.locator('xpath=//button[@class="submit"]').click();

// Chaining locators
await page.locator('.container').locator('button').first().click();
```

### Advanced Locator Techniques
```javascript
// Filter locators
await page.getByRole('listitem').filter({ hasText: 'Product 1' }).click();

// Locator with options
await page.locator('input', { hasText: 'Email' }).fill('test@example.com');

// nth element
await page.locator('li').nth(2).click();

// Get by label
await page.getByLabel('Email address').fill('user@example.com');

// Get by placeholder
await page.getByPlaceholder('Enter your name').fill('John Doe');
```

### Element State Checks
```javascript
// Wait for element
await page.locator('#submit').waitFor();

// Check visibility
const isVisible = await page.locator('#element').isVisible();

// Check if enabled
const isEnabled = await page.locator('#button').isEnabled();

// Check if checked (for checkboxes/radio)
const isChecked = await page.locator('#checkbox').isChecked();

// Get text content
const text = await page.locator('#title').textContent();

// Get attribute value
const href = await page.locator('a').getAttribute('href');
```

## Page Navigation & Waiting

### Navigation
```javascript
// Basic navigation
await page.goto('https://example.com');

// Navigation with options
await page.goto('https://example.com', {
  waitUntil: 'networkidle',
  timeout: 30000
});

// Go back/forward
await page.goBack();
await page.goForward();

// Reload
await page.reload();
```

### Waiting Strategies
```javascript
// Wait for element
await page.waitForSelector('#content');

// Wait for navigation
await page.waitForNavigation();

// Wait for load state
await page.waitForLoadState('networkidle');

// Wait for function
await page.waitForFunction(() => window.innerWidth < 100);

// Wait with timeout
await page.waitForTimeout(1000);

// Wait for response
await page.waitForResponse('**/api/data');

// Wait for request
await page.waitForRequest('**/api/submit');
```

### Auto-waiting with Locators
```javascript
// These automatically wait for element to be actionable
await page.getByRole('button').click();
await page.getByLabel('Email').fill('test@email.com');
await page.getByText('Submit').hover();
```

## Form Handling

### Input Fields
```javascript
// Text input
await page.getByLabel('Username').fill('john_doe');

// Clear and fill
await page.getByLabel('Email').clear();
await page.getByLabel('Email').fill('new@email.com');

// Append text
await page.getByLabel('Comments').fill('Initial text');
await page.getByLabel('Comments').fill('Additional text', { clear: false });
```

### Select Dropdowns
```javascript
// Select by value
await page.getByLabel('Country').selectOption('us');

// Select by text
await page.getByLabel('Country').selectOption({ label: 'United States' });

// Select multiple
await page.getByLabel('Languages').selectOption(['javascript', 'python']);
```

### Checkboxes and Radio Buttons
```javascript
// Check checkbox
await page.getByLabel('I agree').check();

// Uncheck
await page.getByLabel('Newsletter').uncheck();

// Radio button
await page.getByLabel('Male').check();
```

### File Uploads
```javascript
// Single file
await page.getByLabel('Upload file').setInputFiles('path/to/file.pdf');

// Multiple files
await page.getByLabel('Upload files').setInputFiles([
  'path/to/file1.pdf',
  'path/to/file2.jpg'
]);

// Upload from buffer
const buffer = await fs.readFile('path/to/file.pdf');
await page.getByLabel('Upload').setInputFiles({
  name: 'file.pdf',
  mimeType: 'application/pdf',
  buffer: buffer
});
```

### Complex Form Example
```javascript
test('complete form submission', async ({ page }) => {
  await page.goto('/registration');
  
  await page.getByLabel('First Name').fill('John');
  await page.getByLabel('Last Name').fill('Doe');
  await page.getByLabel('Email').fill('john@example.com');
  await page.getByLabel('Password').fill('SecurePass123');
  await page.getByLabel('Country').selectOption('US');
  await page.getByLabel('I agree to terms').check();
  await page.getByLabel('Profile photo').setInputFiles('photo.jpg');
  
  await page.getByRole('button', { name: 'Register' }).click();
  
  await expect(page.getByText('Registration successful')).toBeVisible();
});
```

## Screenshots & Videos

### Screenshots
```javascript
// Full page screenshot
await page.screenshot({ path: 'page.png' });

// Element screenshot
await page.locator('#chart').screenshot({ path: 'chart.png' });

// Screenshot with options
await page.screenshot({
  path: 'page.png',
  fullPage: true,
  clip: { x: 0, y: 0, width: 800, height: 600 }
});
```

### Videos
```javascript
// Enable video recording in config
const context = await browser.newContext({
  recordVideo: {
    dir: 'videos/',
    size: { width: 1280, height: 720 }
  }
});

// Or in test
test('with video', async ({ browser }) => {
  const context = await browser.newContext({
    recordVideo: { dir: 'videos/' }
  });
  const page = await context.newPage();
  
  // Your test actions
  await page.goto('https://example.com');
  
  await context.close();
});
```

### PDF Generation
```javascript
// Generate PDF
await page.pdf({
  path: 'page.pdf',
  format: 'A4',
  printBackground: true
});
```

## Network Interception

### Request Interception
```javascript
// Block images
await page.route('**/*.{png,jpg,jpeg}', route => route.abort());

// Modify requests
await page.route('**/api/**', route => {
  const headers = route.request().headers();
  headers['Authorization'] = 'Bearer token123';
  route.continue({ headers });
});

// Mock API responses
await page.route('**/api/users', route => {
  route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify([
      { id: 1, name: 'John Doe' },
      { id: 2, name: 'Jane Smith' }
    ])
  });
});
```

### Network Monitoring
```javascript
// Listen to requests
page.on('request', request => {
  console.log('Request:', request.url());
});

// Listen to responses
page.on('response', response => {
  console.log('Response:', response.url(), response.status());
});

// Wait for specific request
const request = await page.waitForRequest('**/api/data');
console.log('Request headers:', request.headers());

// Wait for response and check
const response = await page.waitForResponse('**/api/data');
const data = await response.json();
console.log('Response data:', data);
```

## Mobile & Device Emulation

### Device Emulation
```javascript
import { devices } from 'playwright';

// Use predefined device
test('mobile test', async ({ browser }) => {
  const context = await browser.newContext({
    ...devices['iPhone 12']
  });
  const page = await context.newPage();
  
  await page.goto('https://example.com');
});

// Custom device settings
const context = await browser.newContext({
  viewport: { width: 375, height: 667 },
  userAgent: 'Mozilla/5.0...',
  deviceScaleFactor: 2,
  isMobile: true,
  hasTouch: true
});
```

### Geolocation
```javascript
// Set geolocation
await context.setGeolocation({ latitude: 37.7749, longitude: -122.4194 });

// Grant geolocation permission
await context.grantPermissions(['geolocation']);
```

### Touch and Mobile Gestures
```javascript
// Tap
await page.locator('#button').tap();

// Double tap
await page.locator('#element').dblclick();

// Swipe
await page.locator('#slider').dragTo(page.locator('#target'));
```

## Authentication & Sessions

### Basic Authentication
```javascript
// HTTP Basic Auth
const context = await browser.newContext({
  httpCredentials: {
    username: 'user',
    password: 'pass'
  }
});
```

### Session Storage
```javascript
// Save authentication state
await page.goto('/login');
await page.getByLabel('Username').fill('user@example.com');
await page.getByLabel('Password').fill('password');
await page.getByRole('button', { name: 'Login' }).click();

// Save state
await page.context().storageState({ path: 'auth.json' });

// Use saved state
test('authenticated test', async ({ browser }) => {
  const context = await browser.newContext({
    storageState: 'auth.json'
  });
  const page = await context.newPage();
  
  await page.goto('/dashboard'); // Already logged in
});
```

### Cookie Management
```javascript
// Add cookies
await context.addCookies([
  {
    name: 'session',
    value: 'abc123',
    domain: 'example.com',
    path: '/'
  }
]);

// Get cookies
const cookies = await context.cookies();
console.log(cookies);
```

## Parallel Testing

### Test-level Parallelism
```javascript
// playwright.config.js
export default defineConfig({
  workers: 4, // Run 4 tests in parallel
  fullyParallel: true
});
```

### Browser-level Parallelism
```javascript
test.describe.parallel('parallel suite', () => {
  test('test 1', async ({ page }) => {
    // This test runs in parallel with test 2
  });
  
  test('test 2', async ({ page }) => {
    // This test runs in parallel with test 1
  });
});
```

### Worker Isolation
```javascript
// Each worker gets its own browser context
test('isolated test', async ({ page, context }) => {
  // This page is isolated from other tests
  await page.goto('/app');
});
```

## Page Object Model

### Basic Page Object
```javascript
// pages/LoginPage.js
export class LoginPage {
  constructor(page) {
    this.page = page;
    this.usernameInput = page.getByLabel('Username');
    this.passwordInput = page.getByLabel('Password');
    this.loginButton = page.getByRole('button', { name: 'Login' });
    this.errorMessage = page.getByText('Invalid credentials');
  }

  async goto() {
    await this.page.goto('/login');
  }

  async login(username, password) {
    await this.usernameInput.fill(username);
    await this.passwordInput.fill(password);
    await this.loginButton.click();
  }

  async expectErrorMessage() {
    await expect(this.errorMessage).toBeVisible();
  }
}
```

### Using Page Objects
```javascript
import { LoginPage } from '../pages/LoginPage';

test('login test', async ({ page }) => {
  const loginPage = new LoginPage(page);
  
  await loginPage.goto();
  await loginPage.login('user@example.com', 'password');
  
  await expect(page).toHaveURL('/dashboard');
});
```

### Advanced Page Object with Base Class
```javascript
// pages/BasePage.js
export class BasePage {
  constructor(page) {
    this.page = page;
  }

  async waitForPageLoad() {
    await this.page.waitForLoadState('networkidle');
  }

  async takeScreenshot(name) {
    await this.page.screenshot({ path: `screenshots/${name}.png` });
  }
}

// pages/ProductPage.js
export class ProductPage extends BasePage {
  constructor(page) {
    super(page);
    this.addToCartButton = page.getByRole('button', { name: 'Add to Cart' });
    this.productTitle = page.locator('.product-title');
    this.price = page.locator('.price');
  }

  async addToCart() {
    await this.addToCartButton.click();
    await this.waitForPageLoad();
  }

  async getProductInfo() {
    return {
      title: await this.productTitle.textContent(),
      price: await this.price.textContent()
    };
  }
}
```

## Advanced Patterns

### Custom Fixtures
```javascript
// fixtures.js
import { test as base } from '@playwright/test';
import { LoginPage } from '../pages/LoginPage';

export const test = base.extend({
  loginPage: async ({ page }, use) => {
    const loginPage = new LoginPage(page);
    await use(loginPage);
  },

  authenticatedPage: async ({ page, loginPage }, use) => {
    await loginPage.goto();
    await loginPage.login('user@example.com', 'password');
    await use(page);
  }
});

export { expect } from '@playwright/test';
```

### Data-Driven Testing
```javascript
const testData = [
  { username: 'user1@example.com', password: 'pass1' },
  { username: 'user2@example.com', password: 'pass2' },
  { username: 'user3@example.com', password: 'pass3' }
];

testData.forEach(({ username, password }) => {
  test(`login with ${username}`, async ({ page }) => {
    await page.goto('/login');
    await page.getByLabel('Username').fill(username);
    await page.getByLabel('Password').fill(password);
    await page.getByRole('button', { name: 'Login' }).click();
    
    await expect(page).toHaveURL('/dashboard');
  });
});
```

### Global Setup and Teardown
```javascript
// global-setup.js
async function globalSetup() {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  // Setup database, start servers, etc.
  await page.goto('/admin');
  await page.getByLabel('Username').fill('admin');
  await page.getByLabel('Password').fill('admin');
  await page.getByRole('button', { name: 'Login' }).click();
  
  // Save admin state
  await page.context().storageState({ path: 'admin-auth.json' });
  
  await browser.close();
}

export default globalSetup;
```

### Custom Matchers
```javascript
// playwright.config.js
import { expect } from '@playwright/test';

expect.extend({
  async toHaveSuccessMessage(page) {
    const successElement = page.locator('.success-message');
    const pass = await successElement.isVisible();
    
    return {
      message: () => `expected page to have success message`,
      pass
    };
  }
});

// Usage in tests
await expect(page).toHaveSuccessMessage();
```

## Performance Testing

### Timing Measurements
```javascript
test('performance test', async ({ page }) => {
  const startTime = Date.now();
  
  await page.goto('https://example.com');
  
  const loadTime = Date.now() - startTime;
  console.log(`Page load time: ${loadTime}ms`);
  
  expect(loadTime).toBeLessThan(3000);
});
```

### Network Performance
```javascript
test('network performance', async ({ page }) => {
  const responses = [];
  
  page.on('response', response => {
    responses.push({
      url: response.url(),
      status: response.status(),
      timing: response.timing()
    });
  });
  
  await page.goto('https://example.com');
  
  // Analyze responses
  const slowResponses = responses.filter(r => r.timing.responseEnd > 1000);
  expect(slowResponses).toHaveLength(0);
});
```

### Core Web Vitals
```javascript
test('core web vitals', async ({ page }) => {
  await page.goto('https://example.com');
  
  const vitals = await page.evaluate(() => {
    return new Promise(resolve => {
      new PerformanceObserver(list => {
        const entries = list.getEntries();
        resolve(entries.map(entry => ({
          name: entry.name,
          value: entry.value
        })));
      }).observe({ entryTypes: ['largest-contentful-paint', 'first-input'] });
      
      setTimeout(() => resolve([]), 5000);
    });
  });
  
  console.log('Web Vitals:', vitals);
});
```

## Best Practices

### 1. Use Auto-waiting
```javascript
// Good - uses auto-waiting
await page.getByRole('button', { name: 'Submit' }).click();

// Avoid - manual waits
await page.waitForSelector('button');
await page.click('button');
```

### 2. Prefer User-facing Locators
```javascript
// Good - user-facing locators
await page.getByRole('button', { name: 'Sign in' });
await page.getByLabel('Email address');
await page.getByText('Welcome back');

// Avoid - implementation details
await page.locator('#submit-btn');
await page.locator('.form-input[type="email"]');
```

### 3. Use Test IDs for Stability
```javascript
// Good for dynamic content
await page.getByTestId('product-card');

// HTML
<div data-testid="product-card">...</div>
```

### 4. Isolate Tests
```javascript
// Good - each test is independent
test('add item to cart', async ({ page }) => {
  await page.goto('/products');
  await page.getByTestId('product-1').getByRole('button', { name: 'Add to Cart' }).click();
  await expect(page.getByTestId('cart-count')).toHaveText('1');
});

test('remove item from cart', async ({ page }) => {
  // Setup cart state for this specific test
  await setupCartWithItems(page, 1);
  await page.getByTestId('remove-item').click();
  await expect(page.getByTestId('cart-count')).toHaveText('0');
});
```

### 5. Handle Dynamic Content
```javascript
// Wait for data to load
await expect(page.getByTestId('product-list')).toContainText('Product 1');

// Handle loading states
await expect(page.getByTestId('loading-spinner')).toBeHidden();
await expect(page.getByTestId('product-grid')).toBeVisible();
```

### 6. Error Handling
```javascript
test('handle errors gracefully', async ({ page }) => {
  try {
    await page.goto('/app');
    await page.getByRole('button', { name: 'Action' }).click();
  } catch (error) {
    await page.screenshot({ path: 'error-screenshot.png' });
    throw error;
  }
});
```

### 7. Configuration Best Practices
```javascript
// playwright.config.js
export default defineConfig({
  use: {
    // Base URL for shorter test code
    baseURL: 'http://localhost:3000',
    
    // Capture traces on failure
    trace: 'retain-on-failure',
    
    // Screenshots on failure
    screenshot: 'only-on-failure',
    
    // Videos on failure
    video: 'retain-on-failure'
  },
  
  // Retry failed tests
  retries: process.env.CI ? 2 : 0,
  
  // Reasonable timeout
  timeout: 30000,
  
  // Expect timeout
  expect: {
    timeout: 5000
  }
});
```

This comprehensive guide covers Playwright from basic setup to advanced automation patterns. Start with the basics and gradually incorporate more advanced techniques as you become comfortable with the fundamentals. Remember to always prioritize test reliability and maintainability over complex implementations.