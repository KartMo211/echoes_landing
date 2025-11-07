const express = require('express');
const router = express.Router();
const authController = require('../controllers/authController');

// --- Authentication Routes ---

// POST /api/auth/register
// Route for creating a new user
router.post('/register', authController.registerUser);

// POST /api/auth/login
// Route for logging in an existing user
router.post('/login', authController.loginUser);

module.exports = router;
