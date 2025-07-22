# Complete SQL Guide: Beginner to Advanced

## Table of Contents
1. [What is SQL?](#what-is-sql)
2. [MySQL vs PostgreSQL](#mysql-vs-postgresql)
3. [Setting Up Your Environment](#setting-up-your-environment)
4. [Beginner Level](#beginner-level)
5. [Intermediate Level](#intermediate-level)
6. [Advanced Level](#advanced-level)
7. [Real-World Use Cases](#real-world-use-cases)
8. [Best Practices](#best-practices)
9. [Performance Optimization](#performance-optimization)

## What is SQL?

**SQL (Structured Query Language)** is a programming language designed for managing and manipulating relational databases. It's the standard language for relational database management systems (RDBMS).

### Key SQL Operations:
- **DDL (Data Definition Language)**: CREATE, ALTER, DROP
- **DML (Data Manipulation Language)**: INSERT, UPDATE, DELETE
- **DQL (Data Query Language)**: SELECT
- **DCL (Data Control Language)**: GRANT, REVOKE

## MySQL vs PostgreSQL

### MySQL
- **Type**: Open-source relational database
- **Strengths**: 
  - Fast for read-heavy workloads
  - Simple to set up and use
  - Great for web applications
  - Excellent replication features
- **Use Cases**: Web applications, e-commerce, content management
- **Companies Using**: Facebook, Twitter, YouTube, Netflix

### PostgreSQL
- **Type**: Open-source object-relational database
- **Strengths**:
  - ACID compliant
  - Advanced features (JSON, arrays, custom types)
  - Better for complex queries
  - Extensible with custom functions
- **Use Cases**: Complex applications, data analytics, geospatial data
- **Companies Using**: Apple, Instagram, Spotify, Reddit

## Setting Up Your Environment

### MySQL Installation
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install mysql-server

# macOS (using Homebrew)
brew install mysql

# Windows: Download from mysql.com
```

### PostgreSQL Installation
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# macOS (using Homebrew)
brew install postgresql

# Windows: Download from postgresql.org
```

### Connecting to Databases
```bash
# MySQL
mysql -u username -p

# PostgreSQL
psql -U username -d database_name
```

## Beginner Level

### 1. Database and Table Creation

```sql
-- Create Database
CREATE DATABASE company_db;
USE company_db; -- MySQL
-- \c company_db -- PostgreSQL

-- Create Table
CREATE TABLE employees (
    id INT PRIMARY KEY AUTO_INCREMENT, -- MySQL
    -- id SERIAL PRIMARY KEY,          -- PostgreSQL
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    department VARCHAR(50),
    salary DECIMAL(10,2),
    hire_date DATE
);

CREATE TABLE departments (
    dept_id INT PRIMARY KEY AUTO_INCREMENT,
    dept_name VARCHAR(50) NOT NULL,
    budget DECIMAL(12,2)
);
```

### 2. Basic Data Operations

```sql
-- Insert Data
INSERT INTO employees (name, email, department, salary, hire_date)
VALUES 
    ('John Doe', 'john@company.com', 'Engineering', 75000, '2023-01-15'),
    ('Jane Smith', 'jane@company.com', 'Marketing', 65000, '2023-02-20'),
    ('Bob Johnson', 'bob@company.com', 'Engineering', 80000, '2023-01-10');

INSERT INTO departments (dept_name, budget)
VALUES 
    ('Engineering', 500000),
    ('Marketing', 250000),
    ('HR', 150000);

-- Select Data
SELECT * FROM employees;
SELECT name, salary FROM employees;
SELECT * FROM employees WHERE department = 'Engineering';
SELECT * FROM employees WHERE salary > 70000;

-- Update Data
UPDATE employees 
SET salary = salary * 1.10 
WHERE department = 'Engineering';

-- Delete Data
DELETE FROM employees WHERE name = 'Bob Johnson';
```

### 3. Basic Filtering and Sorting

```sql
-- WHERE clause conditions
SELECT * FROM employees WHERE salary BETWEEN 60000 AND 80000;
SELECT * FROM employees WHERE department IN ('Engineering', 'Marketing');
SELECT * FROM employees WHERE name LIKE 'J%';
SELECT * FROM employees WHERE email IS NOT NULL;

-- ORDER BY
SELECT * FROM employees ORDER BY salary DESC;
SELECT * FROM employees ORDER BY department, salary DESC;

-- LIMIT
SELECT * FROM employees ORDER BY salary DESC LIMIT 5;
```

## Intermediate Level

### 1. Joins

```sql
-- Add foreign key relationship
ALTER TABLE employees ADD COLUMN dept_id INT;
ALTER TABLE employees ADD FOREIGN KEY (dept_id) REFERENCES departments(dept_id);

-- Update employees with department IDs
UPDATE employees SET dept_id = 1 WHERE department = 'Engineering';
UPDATE employees SET dept_id = 2 WHERE department = 'Marketing';

-- INNER JOIN
SELECT e.name, e.salary, d.dept_name, d.budget
FROM employees e
INNER JOIN departments d ON e.dept_id = d.dept_id;

-- LEFT JOIN
SELECT e.name, e.salary, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;

-- RIGHT JOIN
SELECT e.name, d.dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id;
```

### 2. Aggregate Functions

```sql
-- Basic aggregates
SELECT COUNT(*) as total_employees FROM employees;
SELECT AVG(salary) as average_salary FROM employees;
SELECT MAX(salary) as highest_salary FROM employees;
SELECT MIN(salary) as lowest_salary FROM employees;
SELECT SUM(salary) as total_payroll FROM employees;

-- GROUP BY
SELECT department, COUNT(*) as employee_count, AVG(salary) as avg_salary
FROM employees
GROUP BY department;

-- HAVING (filtering after grouping)
SELECT department, COUNT(*) as employee_count
FROM employees
GROUP BY department
HAVING COUNT(*) > 1;
```

### 3. Subqueries

```sql
-- Scalar subquery
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- IN subquery
SELECT name, salary
FROM employees
WHERE dept_id IN (
    SELECT dept_id FROM departments WHERE budget > 200000
);

-- EXISTS subquery
SELECT d.dept_name
FROM departments d
WHERE EXISTS (
    SELECT 1 FROM employees e WHERE e.dept_id = d.dept_id
);
```

### 4. Common Table Expressions (CTEs)

```sql
-- PostgreSQL and MySQL 8.0+
WITH high_earners AS (
    SELECT name, salary, department
    FROM employees
    WHERE salary > 70000
),
dept_stats AS (
    SELECT department, COUNT(*) as count, AVG(salary) as avg_sal
    FROM high_earners
    GROUP BY department
)
SELECT * FROM dept_stats;
```

## Advanced Level

### 1. Window Functions

```sql
-- Row numbering
SELECT 
    name, 
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) as salary_rank
FROM employees;

-- Ranking within groups
SELECT 
    name,
    department,
    salary,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) as dept_rank
FROM employees;

-- Running totals
SELECT 
    name,
    salary,
    SUM(salary) OVER (ORDER BY hire_date) as running_total
FROM employees;

-- Lead and Lag
SELECT 
    name,
    salary,
    LAG(salary) OVER (ORDER BY hire_date) as prev_salary,
    LEAD(salary) OVER (ORDER BY hire_date) as next_salary
FROM employees;
```

### 2. Advanced Data Types and Functions

```sql
-- JSON in PostgreSQL
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    attributes JSON
);

INSERT INTO products (name, attributes)
VALUES ('Laptop', '{"brand": "Dell", "ram": "16GB", "storage": "512GB SSD"}');

SELECT name, attributes->'brand' as brand
FROM products;

-- Arrays in PostgreSQL
CREATE TABLE tags_example (
    id SERIAL PRIMARY KEY,
    title VARCHAR(100),
    tags TEXT[]
);

INSERT INTO tags_example (title, tags)
VALUES ('Article 1', ARRAY['tech', 'programming', 'sql']);
```

### 3. Stored Procedures and Functions

```sql
-- MySQL Stored Procedure
DELIMITER //
CREATE PROCEDURE GetEmployeesByDept(IN dept_name VARCHAR(50))
BEGIN
    SELECT * FROM employees WHERE department = dept_name;
END //
DELIMITER ;

-- Call procedure
CALL GetEmployeesByDept('Engineering');

-- PostgreSQL Function
CREATE OR REPLACE FUNCTION get_employee_count(dept_name VARCHAR)
RETURNS INTEGER AS $$
BEGIN
    RETURN (SELECT COUNT(*) FROM employees WHERE department = dept_name);
END;
$$ LANGUAGE plpgsql;

-- Use function
SELECT get_employee_count('Engineering');
```

### 4. Triggers

```sql
-- MySQL Trigger
CREATE TABLE employee_audit (
    audit_id INT AUTO_INCREMENT PRIMARY KEY,
    employee_id INT,
    old_salary DECIMAL(10,2),
    new_salary DECIMAL(10,2),
    change_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

DELIMITER //
CREATE TRIGGER salary_audit
AFTER UPDATE ON employees
FOR EACH ROW
BEGIN
    IF OLD.salary != NEW.salary THEN
        INSERT INTO employee_audit (employee_id, old_salary, new_salary)
        VALUES (NEW.id, OLD.salary, NEW.salary);
    END IF;
END //
DELIMITER ;
```

### 5. Indexes and Performance

```sql
-- Create indexes
CREATE INDEX idx_employee_department ON employees(department);
CREATE INDEX idx_employee_salary ON employees(salary);
CREATE UNIQUE INDEX idx_employee_email ON employees(email);

-- Composite index
CREATE INDEX idx_dept_salary ON employees(department, salary);

-- Analyze query performance
EXPLAIN SELECT * FROM employees WHERE department = 'Engineering';
```

## Real-World Use Cases

### 1. E-commerce Analytics

```sql
-- Sales analysis
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        SUM(total_amount) as revenue,
        COUNT(*) as order_count
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
),
growth_calc AS (
    SELECT 
        month,
        revenue,
        LAG(revenue) OVER (ORDER BY month) as prev_revenue,
        (revenue - LAG(revenue) OVER (ORDER BY month)) / 
        LAG(revenue) OVER (ORDER BY month) * 100 as growth_rate
    FROM monthly_sales
)
SELECT 
    month,
    revenue,
    ROUND(growth_rate, 2) as growth_percentage
FROM growth_calc
ORDER BY month;
```

### 2. User Engagement Tracking

```sql
-- Daily active users and retention
SELECT 
    date,
    COUNT(DISTINCT user_id) as daily_active_users,
    COUNT(DISTINCT CASE 
        WHEN last_login_date >= date - INTERVAL '7 days' 
        THEN user_id 
    END) as weekly_retained_users
FROM user_activity
GROUP BY date
ORDER BY date;
```

### 3. Inventory Management

```sql
-- Low stock alert with supplier information
SELECT 
    p.product_name,
    i.current_stock,
    i.reorder_level,
    s.supplier_name,
    s.contact_email
FROM inventory i
JOIN products p ON i.product_id = p.product_id
JOIN suppliers s ON p.supplier_id = s.supplier_id
WHERE i.current_stock <= i.reorder_level
ORDER BY (i.reorder_level - i.current_stock) DESC;
```

### 4. Financial Reporting

```sql
-- Quarterly financial summary
SELECT 
    YEAR(transaction_date) as year,
    QUARTER(transaction_date) as quarter,
    SUM(CASE WHEN type = 'revenue' THEN amount ELSE 0 END) as total_revenue,
    SUM(CASE WHEN type = 'expense' THEN amount ELSE 0 END) as total_expenses,
    SUM(CASE WHEN type = 'revenue' THEN amount ELSE -amount END) as net_income
FROM financial_transactions
GROUP BY YEAR(transaction_date), QUARTER(transaction_date)
ORDER BY year, quarter;
```

## Best Practices

### 1. Query Optimization
- Use indexes strategically
- Avoid SELECT * in production
- Use appropriate data types
- Normalize your database design
- Use EXPLAIN to analyze query plans

### 2. Security
```sql
-- Use parameterized queries to prevent SQL injection
-- Bad: SELECT * FROM users WHERE id = '" + userId + "'
-- Good: Use prepared statements with parameters

-- Create users with minimal privileges
CREATE USER 'app_user'@'localhost' IDENTIFIED BY 'secure_password';
GRANT SELECT, INSERT, UPDATE ON company_db.* TO 'app_user'@'localhost';
```

### 3. Maintainability
- Use consistent naming conventions
- Comment complex queries
- Keep queries readable with proper formatting
- Use aliases for clarity
- Document your database schema

## Performance Optimization

### 1. Query Optimization Techniques

```sql
-- Use EXISTS instead of IN for large datasets
SELECT name FROM employees e
WHERE EXISTS (SELECT 1 FROM departments d WHERE d.dept_id = e.dept_id);

-- Use UNION ALL instead of UNION when duplicates don't matter
SELECT name FROM employees WHERE department = 'Engineering'
UNION ALL
SELECT name FROM contractors WHERE department = 'Engineering';

-- Optimize JOIN order (smaller tables first)
-- Use appropriate WHERE clauses to reduce dataset size early
```

### 2. Index Strategy

```sql
-- Monitor index usage
-- MySQL
SELECT 
    TABLE_NAME,
    INDEX_NAME,
    CARDINALITY
FROM INFORMATION_SCHEMA.STATISTICS
WHERE TABLE_SCHEMA = 'your_database';

-- PostgreSQL
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read
FROM pg_stat_user_indexes;
```

### 3. Database Maintenance

```sql
-- Regular maintenance tasks
-- MySQL
ANALYZE TABLE employees;
OPTIMIZE TABLE employees;

-- PostgreSQL
VACUUM ANALYZE employees;
REINDEX TABLE employees;
```

This guide covers SQL from basic concepts to advanced techniques with practical examples for both MySQL and PostgreSQL. Practice these concepts with real datasets to build proficiency, and always consider performance implications in production environments.