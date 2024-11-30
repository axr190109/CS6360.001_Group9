-- Query 1: Simple SELECT
SELECT name, age FROM users WHERE age > 30;

-- Query 2: Join with multiple conditions
SELECT orders.id, users.name 
FROM orders 
INNER JOIN users ON orders.user_id = users.id 
WHERE orders.total > 100;

-- Query 3: Aggregate function
SELECT department, COUNT(*) as employee_count 
FROM employees 
GROUP BY department 
HAVING employee_count > 10;

-- Query 4: Nested subquery
SELECT product_name 
FROM products 
WHERE price > (SELECT AVG(price) FROM products);

-- Query 5: Complex query with multiple joins
SELECT o.id, u.name, p.product_name 
FROM orders o 
JOIN users u ON o.user_id = u.id 
JOIN products p ON o.product_id = p.id 
WHERE o.date > '2024-01-01' AND p.category = 'electronics';
