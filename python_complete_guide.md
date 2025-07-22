# Complete Python Guide: Beginner to Advanced

## Table of Contents
1. [Getting Started](#getting-started)
2. [Python Basics](#python-basics)
3. [Data Types and Variables](#data-types-and-variables)
4. [Control Flow](#control-flow)
5. [Functions](#functions)
6. [Data Structures](#data-structures)
7. [Object-Oriented Programming](#object-oriented-programming)
8. [File Handling](#file-handling)
9. [Error Handling](#error-handling)
10. [Modules and Packages](#modules-and-packages)
11. [Advanced Concepts](#advanced-concepts)
12. [Best Practices](#best-practices)
13. [Real-World Applications](#real-world-applications)

---

## Getting Started

### Installation
- Download Python from [python.org](https://python.org)
- Use package managers: `brew install python` (macOS) or `apt install python3` (Linux)
- Verify installation: `python --version`

### Running Python
- Interactive mode: `python` or `python3`
- Script execution: `python script.py`
- IDLE: Built-in development environment

---

## Python Basics

### Your First Program
```python
print("Hello, World!")
```

### Comments
```python
# Single line comment
"""
Multi-line comment
or docstring
"""
```

### Python Philosophy (The Zen of Python)
```python
import this
```

---

## Data Types and Variables

### Basic Data Types
```python
# Integers
age = 25
year = 2024

# Floats
price = 19.99
temperature = -5.5

# Strings
name = "Alice"
message = 'Hello, World!'
multi_line = """This is a
multi-line string"""

# Booleans
is_active = True
is_complete = False

# None (null equivalent)
data = None
```

### String Operations
```python
# String formatting
name = "Bob"
age = 30

# f-strings (Python 3.6+)
message = f"My name is {name} and I'm {age} years old"

# .format() method
message = "My name is {} and I'm {} years old".format(name, age)

# % formatting (older style)
message = "My name is %s and I'm %d years old" % (name, age)

# String methods
text = "  Hello World  "
print(text.strip())        # Remove whitespace
print(text.lower())        # Lowercase
print(text.upper())        # Uppercase
print(text.replace("World", "Python"))  # Replace
print(text.split())        # Split into list
```

### Type Conversion
```python
# Converting types
num_str = "42"
num = int(num_str)         # String to integer
float_num = float(num_str) # String to float
back_to_str = str(num)     # Integer to string

# Checking types
print(type(num))           # <class 'int'>
print(isinstance(num, int)) # True
```

---

## Control Flow

### Conditional Statements
```python
age = 18

if age >= 18:
    print("You're an adult")
elif age >= 13:
    print("You're a teenager")
else:
    print("You're a child")

# Ternary operator
status = "adult" if age >= 18 else "minor"
```

### Loops

#### For Loops
```python
# Iterating over a range
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

for i in range(1, 6):
    print(i)  # 1, 2, 3, 4, 5

for i in range(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8

# Iterating over collections
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(fruit)

# Enumerate for index and value
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
```

#### While Loops
```python
count = 0
while count < 5:
    print(count)
    count += 1

# Infinite loop with break
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input == 'quit':
        break
    print(f"You entered: {user_input}")
```

#### Loop Control
```python
for i in range(10):
    if i == 3:
        continue  # Skip this iteration
    if i == 7:
        break     # Exit the loop
    print(i)
```

---

## Functions

### Basic Functions
```python
def greet(name):
    """Greet a person by name."""
    return f"Hello, {name}!"

# Calling the function
message = greet("Alice")
print(message)

# Function with multiple parameters
def add(a, b):
    return a + b

result = add(5, 3)
```

### Advanced Function Features
```python
# Default parameters
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Keyword arguments
def create_profile(name, age, city="Unknown"):
    return {
        "name": name,
        "age": age,
        "city": city
    }

profile = create_profile("Alice", age=25, city="New York")

# Variable arguments (*args)
def sum_all(*numbers):
    return sum(numbers)

total = sum_all(1, 2, 3, 4, 5)

# Keyword variable arguments (**kwargs)
def print_info(**info):
    for key, value in info.items():
        print(f"{key}: {value}")

print_info(name="Bob", age=30, job="Engineer")

# Lambda functions
square = lambda x: x ** 2
numbers = [1, 2, 3, 4, 5]
squared = list(map(square, numbers))
```

### Scope and Global Variables
```python
global_var = "I'm global"

def my_function():
    local_var = "I'm local"
    global global_var
    global_var = "Modified global"
    return local_var

# Nonlocal for nested functions
def outer_function():
    x = "outer"
    
    def inner_function():
        nonlocal x
        x = "modified by inner"
    
    inner_function()
    return x
```

---

## Data Structures

### Lists
```python
# Creating lists
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# List operations
fruits.append("grape")        # Add to end
fruits.insert(1, "kiwi")     # Insert at index
removed = fruits.pop()       # Remove and return last
fruits.remove("banana")      # Remove by value
fruits.sort()               # Sort in place
reversed_fruits = fruits[::-1]  # Reverse

# List comprehensions
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Nested lists
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

### Tuples
```python
# Creating tuples (immutable)
coordinates = (10, 20)
rgb = (255, 128, 0)

# Tuple unpacking
x, y = coordinates
r, g, b = rgb

# Named tuples
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)
```

### Dictionaries
```python
# Creating dictionaries
person = {
    "name": "Alice",
    "age": 25,
    "city": "New York"
}

# Dictionary operations
person["email"] = "alice@example.com"  # Add key-value
age = person.get("age", 0)             # Get with default
keys = person.keys()                   # Get all keys
values = person.values()               # Get all values
items = person.items()                 # Get key-value pairs

# Dictionary comprehensions
squares_dict = {x: x**2 for x in range(5)}

# Nested dictionaries
users = {
    "user1": {"name": "Alice", "age": 25},
    "user2": {"name": "Bob", "age": 30}
}
```

### Sets
```python
# Creating sets (unique elements)
numbers = {1, 2, 3, 4, 5}
colors = set(["red", "green", "blue", "red"])  # Duplicates removed

# Set operations
numbers.add(6)              # Add element
numbers.remove(3)           # Remove element
numbers.discard(10)         # Remove if exists (no error)

# Set mathematics
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

union = set1 | set2         # Union
intersection = set1 & set2  # Intersection
difference = set1 - set2    # Difference
```

---

## Object-Oriented Programming

### Classes and Objects
```python
class Person:
    """A simple Person class."""
    
    # Class variable
    species = "Homo sapiens"
    
    def __init__(self, name, age):
        """Constructor method."""
        self.name = name        # Instance variable
        self.age = age          # Instance variable
    
    def introduce(self):
        """Instance method."""
        return f"Hi, I'm {self.name} and I'm {self.age} years old"
    
    def have_birthday(self):
        """Modify instance state."""
        self.age += 1
        return f"Happy birthday! I'm now {self.age}"
    
    @staticmethod
    def is_adult(age):
        """Static method - doesn't need instance."""
        return age >= 18
    
    @classmethod
    def from_string(cls, person_str):
        """Class method - alternative constructor."""
        name, age = person_str.split('-')
        return cls(name, int(age))

# Creating objects
person1 = Person("Alice", 25)
person2 = Person.from_string("Bob-30")

print(person1.introduce())
print(Person.is_adult(17))
```

### Inheritance
```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass  # Abstract method
    
    def info(self):
        return f"I'm {self.name}"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # Call parent constructor
        self.breed = breed
    
    def speak(self):
        return f"{self.name} says Woof!"
    
    def info(self):
        return f"{super().info()} and I'm a {self.breed}"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

# Using inheritance
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers")

print(dog.speak())
print(dog.info())
```

### Special Methods (Magic Methods)
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        """String representation for users."""
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        """String representation for developers."""
        return f"Vector({self.x}, {self.y})"
    
    def __add__(self, other):
        """Addition operator."""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __len__(self):
        """Length/magnitude."""
        return int((self.x**2 + self.y**2)**0.5)
    
    def __getitem__(self, index):
        """Index access."""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Vector index out of range")

v1 = Vector(2, 3)
v2 = Vector(1, 4)
v3 = v1 + v2  # Uses __add__
print(v3)     # Uses __str__
```

### Properties and Encapsulation
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius  # Protected attribute
    
    @property
    def radius(self):
        """Getter for radius."""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Setter for radius with validation."""
        if value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value
    
    @property
    def area(self):
        """Computed property."""
        return 3.14159 * self._radius ** 2
    
    @property
    def diameter(self):
        return 2 * self._radius

circle = Circle(5)
print(circle.area)
circle.radius = 10  # Uses setter
print(circle.diameter)
```

---

## File Handling

### Reading Files
```python
# Method 1: Basic file reading
file = open("data.txt", "r")
content = file.read()
file.close()

# Method 2: Using with statement (recommended)
with open("data.txt", "r") as file:
    content = file.read()
    # File automatically closed

# Reading line by line
with open("data.txt", "r") as file:
    for line in file:
        print(line.strip())

# Reading all lines into a list
with open("data.txt", "r") as file:
    lines = file.readlines()
```

### Writing Files
```python
# Writing text
with open("output.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("This is a new file.")

# Appending to file
with open("output.txt", "a") as file:
    file.write("\nAppended text")

# Writing multiple lines
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open("output.txt", "w") as file:
    file.writelines(lines)
```

### Working with CSV Files
```python
import csv

# Reading CSV
with open("data.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# Reading CSV with headers
with open("data.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row["name"], row["age"])

# Writing CSV
data = [
    ["Name", "Age", "City"],
    ["Alice", "25", "New York"],
    ["Bob", "30", "San Francisco"]
]

with open("output.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
```

### JSON Handling
```python
import json

# Reading JSON
with open("data.json", "r") as file:
    data = json.load(file)

# Writing JSON
data = {
    "name": "Alice",
    "age": 25,
    "hobbies": ["reading", "swimming"]
}

with open("output.json", "w") as file:
    json.dump(data, file, indent=2)

# JSON string conversion
json_string = json.dumps(data)
parsed_data = json.loads(json_string)
```

---

## Error Handling

### Try-Except Blocks
```python
try:
    number = int(input("Enter a number: "))
    result = 10 / number
    print(f"Result: {result}")
except ValueError:
    print("Invalid input! Please enter a number.")
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An error occurred: {e}")
else:
    print("No errors occurred!")
finally:
    print("This always executes")
```

### Custom Exceptions
```python
class CustomError(Exception):
    """Custom exception class."""
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code

def validate_age(age):
    if age < 0:
        raise CustomError("Age cannot be negative", code="INVALID_AGE")
    if age > 150:
        raise CustomError("Age seems unrealistic", code="UNREALISTIC_AGE")

try:
    validate_age(-5)
except CustomError as e:
    print(f"Error: {e}, Code: {e.code}")
```

### Context Managers
```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print("Opening file...")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_value, traceback):
        print("Closing file...")
        if self.file:
            self.file.close()

# Using custom context manager
with FileManager("test.txt", "w") as f:
    f.write("Hello, World!")
```

---

## Modules and Packages

### Importing Modules
```python
# Different import styles
import math
from math import pi, sqrt
from math import *  # Import everything (not recommended)
import math as m    # Alias

# Using imported functions
print(math.pi)
print(sqrt(16))
print(m.cos(0))
```

### Creating Your Own Module
```python
# mymodule.py
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b

PI = 3.14159

if __name__ == "__main__":
    # This code only runs when module is executed directly
    print("Module is being run directly")

# main.py
import mymodule

print(mymodule.greet("Alice"))
print(mymodule.add(5, 3))
```

### Package Structure
```
mypackage/
    __init__.py
    module1.py
    module2.py
    subpackage/
        __init__.py
        submodule.py
```

### Virtual Environments
```bash
# Creating virtual environment
python -m venv myenv

# Activating (Windows)
myenv\Scripts\activate

# Activating (Mac/Linux)
source myenv/bin/activate

# Installing packages
pip install requests numpy pandas

# Saving dependencies
pip freeze > requirements.txt

# Installing from requirements
pip install -r requirements.txt

# Deactivating
deactivate
```

---

## Advanced Concepts

### Decorators
```python
# Basic decorator
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def greet(name):
    return f"Hello, {name}!"

# Decorator with parameters
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def say_hello():
    print("Hello!")

# Built-in decorators
class MyClass:
    def __init__(self, value):
        self._value = value
    
    @property
    def value(self):
        return self._value
    
    @staticmethod
    def utility_function():
        return "I don't need an instance"
    
    @classmethod
    def from_string(cls, string_value):
        return cls(int(string_value))
```

### Generators and Iterators
```python
# Generator function
def countdown(n):
    while n > 0:
        yield n
        n -= 1

# Using generator
for num in countdown(5):
    print(num)

# Generator expression
squares = (x**2 for x in range(10))

# Custom iterator
class Counter:
    def __init__(self, max_count):
        self.max_count = max_count
        self.count = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count < self.max_count:
            self.count += 1
            return self.count
        raise StopIteration

counter = Counter(5)
for num in counter:
    print(num)
```

### Context Managers
```python
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    print("Timer started")
    try:
        yield
    finally:
        end = time.time()
        print(f"Timer ended. Elapsed: {end - start:.2f} seconds")

with timer():
    time.sleep(1)
    print("Doing some work...")
```

### Metaclasses
```python
class MyMeta(type):
    def __new__(cls, name, bases, attrs):
        # Modify class creation
        attrs['class_id'] = f"{name}_{id(cls)}"
        return super().__new__(cls, name, bases, attrs)

class MyClass(metaclass=MyMeta):
    def __init__(self, value):
        self.value = value

obj = MyClass(42)
print(obj.class_id)
```

### Advanced Function Features
```python
from functools import wraps, lru_cache, partial

# functools.wraps preserves original function metadata
def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# Caching with lru_cache
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Partial functions
def multiply(x, y):
    return x * y

double = partial(multiply, 2)
print(double(5))  # 10
```

### Regular Expressions
```python
import re

text = "The phone number is 123-456-7890"

# Pattern matching
pattern = r'\d{3}-\d{3}-\d{4}'
match = re.search(pattern, text)
if match:
    print(f"Found: {match.group()}")

# Finding all matches
numbers = re.findall(r'\d+', text)
print(numbers)  # ['123', '456', '7890']

# Substitution
new_text = re.sub(r'\d{3}-\d{3}-\d{4}', 'XXX-XXX-XXXX', text)
print(new_text)

# Compiled patterns for reuse
phone_pattern = re.compile(r'\d{3}-\d{3}-\d{4}')
matches = phone_pattern.findall(text)
```

---

## Best Practices

### Code Style (PEP 8)
```python
# Good naming conventions
user_name = "alice"          # Snake case for variables
MAX_CONNECTIONS = 100        # Constants in uppercase
class UserProfile:           # Pascal case for classes
    pass

def calculate_total_price(): # Functions in snake case
    pass

# Proper spacing
x = 1
y = 2
total = x + y

# List comprehensions vs loops
# Good
squares = [x**2 for x in range(10)]

# Avoid when complex
# Bad (too complex for comprehension)
result = [complex_function(x) for x in items if complex_condition(x)]

# Better
result = []
for x in items:
    if complex_condition(x):
        result.append(complex_function(x))
```

### Error Handling Best Practices
```python
# Specific exception handling
try:
    with open('file.txt', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("File not found")
except json.JSONDecodeError:
    print("Invalid JSON format")
except Exception as e:
    print(f"Unexpected error: {e}")

# Using logging instead of print
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data(data):
    try:
        # Process data
        result = complex_processing(data)
        logger.info("Data processed successfully")
        return result
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### Performance Tips
```python
# Use list comprehensions for simple operations
# Fast
squares = [x**2 for x in range(1000)]

# Slower
squares = []
for x in range(1000):
    squares.append(x**2)

# Use generators for memory efficiency
def read_large_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

# Use sets for membership testing
# Fast
valid_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
if user_id in valid_ids:
    process_user()

# Slower
valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
if user_id in valid_ids:
    process_user()

# Use string join for concatenation
# Fast
words = ['hello', 'world', 'python']
sentence = ' '.join(words)

# Slower for many strings
sentence = ''
for word in words:
    sentence += word + ' '
```

### Testing
```python
import unittest
from unittest.mock import patch, MagicMock

class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()
    
    def test_add(self):
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)
    
    def test_divide_by_zero(self):
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)
    
    @patch('requests.get')
    def test_api_call(self, mock_get):
        mock_get.return_value.json.return_value = {'status': 'ok'}
        # Test code that makes API calls

if __name__ == '__main__':
    unittest.main()
```

---

## Real-World Applications

### Web Scraping
```python
import requests
from bs4 import BeautifulSoup

def scrape_quotes():
    url = "http://quotes.toscrape.com"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    quotes = []
    for quote in soup.find_all('div', class_='quote'):
        text = quote.find('span', class_='text').text
        author = quote.find('small', class_='author').text
        quotes.append({'text': text, 'author': author})
    
    return quotes

quotes = scrape_quotes()
for quote in quotes[:3]:
    print(f'"{quote["text"]}" - {quote["author"]}')
```

### API Development with Flask
```python
from flask import Flask, jsonify, request

app = Flask(__name__)

users = [
    {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'},
    {'id': 2, 'name': 'Bob', 'email': 'bob@example.com'}
]

@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify(users)

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = next((u for u in users if u['id'] == user_id), None)
    if user:
        return jsonify(user)
    return jsonify({'error': 'User not found'}), 404

@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json()
    new_user = {
        'id': len(users) + 1,
        'name': data['name'],
        'email': data['email']
    }
    users.append(new_user)
    return jsonify(new_user), 201

if __name__ == '__main__':
    app.run(debug=True)
```

### Data Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Reading data
df = pd.read_csv('sales_data.csv')

# Basic analysis
print(df.head())
print(df.describe())
print(df.info())

# Data cleaning
df = df.dropna()  # Remove missing values
df['date'] = pd.to_datetime(df['date'])  # Convert to datetime

# Grouping and aggregation
monthly_sales = df.groupby(df['date'].dt.to_period('M'))['amount'].sum()

# Visualization
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='bar')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Statistical analysis
correlation = df[['price', 'quantity', 'amount']].corr()
print(correlation)
```

### Database Operations
```python
import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db_connection(db_name):
    conn = sqlite3.connect(db_name)
    try:
        yield conn
    finally:
        conn.close()

def create_users_table():
    with get_db_connection('app.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def add_user(name, email):
    with get_db_connection('app.db') as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (name, email) VALUES (?, ?)',
            (name, email)
        )
        conn.commit()
        return cursor.lastrowid

def get_users():
    with get_db_connection('app.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users')
        return cursor.fetchall()

# Usage
create_users_table()
user_id = add_user('Alice', 'alice@example.com')
users = get_users()
print(users)
```

### Asynchronous Programming
```python
import asyncio
import aiohttp
import time

async def fetch_url(session, url):
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        return f"Error fetching {url}: {e}"

async def fetch_multiple_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

# Synchronous version
def fetch_sync(urls):
    import requests
    results = []
    for url in urls:
        try:
            response = requests.get(url)
            results.append(response.text)
        except Exception as e:
            results.append(f"Error: {e}")
    return results

# Comparing performance
urls = [
    'http://httpbin.org/delay/1',
    'http://httpbin.org/delay/1',
    'http://httpbin.org/delay/1'
]

# Async version
async def main():
    start = time.time()
    results = await fetch_multiple_urls(urls)
    print(f"Async took: {time.time() - start:.2f} seconds")

# Run async code
asyncio.run(main())
```

---

## Conclusion

This guide covers Python from basic syntax to advanced concepts and real-world applications. Key takeaways:

1. **Start with basics**: Master variables, data types, and control flow
2. **Practice regularly**: Write code daily to build muscle memory
3. **Learn by building**: Create projects that interest you
4. **Read others' code**: Study well-written Python projects on GitHub
5. **Use the community**: Python has excellent documentation and helpful communities
6. **Keep learning**: Python constantly evolves with new features and libraries

### Next Steps
- Explore specific libraries for your interests (Django/Flask for web, NumPy/Pandas for data science, etc.)
- Contribute to open-source projects
- Build portfolio projects
- Consider Python certifications
- Join Python communities and attend conferences

Remember: becoming proficient in Python is a journey, not a destination. Keep practicing, stay curious, and don't be afraid to experiment!