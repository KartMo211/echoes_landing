// Load environment variables
require('dotenv').config();

const express = require('express');
const cors = require('cors');
const authRoutes = require('./routes/authRoutes.js');

const app = express();
const PORT = process.env.PORT || 5000;

// --- Middleware ---
// Enable Cross-Origin Resource Sharing (CORS)
// This allows your React frontend (on a different port) to talk to this backend
app.use(cors());

// Parse incoming JSON request bodies
app.use(express.json());

// --- Routes ---
// Mount the authentication routes under the /api/auth prefix
app.use('/api/auth', authRoutes);

// Base route for testing
app.get('/', (req, res) => {
  res.send('Login Backend API is running!');
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is listening on port ${PORT}`);
});
