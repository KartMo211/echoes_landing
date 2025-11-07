const { Pool } = require('pg');

// Create a new connection pool using environment variables
const pool = new Pool({
  user: process.env.DB_USER,
  host: process.env.DB_HOST,
  database: process.env.DB_DATABASE,
  password: process.env.DB_PASSWORD,
  port: process.env.DB_PORT,
});

// Export a query function that will be used by controllers
module.exports = {
  query: (text, params) => pool.query(text, params),
};
