// db_init.js
const { Pool } = require('pg');

// Retrieve connection details from environment variables (DB_HOST, DB_USER, etc.)
const connectionString = process.env.DATABASE_URL || 
  `postgresql://${process.env.DB_USER}:${process.env.DB_PASSWORD}@${process.env.DB_HOST}:${process.env.DB_PORT}/${process.env.DB_DATABASE}`;

const pool = new Pool({
  connectionString: connectionString
});

// The SQL query to create the table based on your schema
const createTableQuery = `
  CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
  );
`;

async function createTable() {
  console.log('Attempting to connect and create the "users" table...');
  try {
    // Execute the query
    await pool.query(createTableQuery);
    console.log('✅ Table "users" created successfully or already exists.');
  } catch (err) {
    console.error('❌ Error creating table:', err.message);
    // Exit with a non-zero code to indicate a deployment failure
    process.exit(1); 
  } finally {
    // Close the connection pool
    await pool.end();
    console.log('Database connection closed.');
  }
}

createTable();