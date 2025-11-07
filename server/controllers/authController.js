const db = require('../db/db.js');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

/**
 * --- Register a new user ---
 * Endpoint: POST /api/auth/register
 */
exports.registerUser = async (req, res) => {
  const { email, password } = req.body;

  // 1. Validate input
  if (!email || !password) {
    return res.status(400).json({ message: 'Email and password are required.' });
  }
  
  if (password.length < 6) {
     return res.status(400).json({ message: 'Password must be at least 6 characters long.' });
  }

  try {
    // 2. Check if user already exists
    const userCheck = await db.query('SELECT * FROM users WHERE email = $1', [email]);
    if (userCheck.rows.length > 0) {
      return res.status(409).json({ message: 'Email already in use.' });
    }

    // 3. Hash the password
    const salt = await bcrypt.genSalt(10);
    const passwordHash = await bcrypt.hash(password, salt);

    // 4. Insert new user into the database
    const newUser = await db.query(
      'INSERT INTO users (email, password_hash) VALUES ($1, $2) RETURNING id, email, created_at',
      [email, passwordHash]
    );

    // 5. Send success response
    res.status(201).json({
      message: 'User registered successfully.',
      user: newUser.rows[0],
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ message: 'Server error during registration.' });
  }
};


/**
 * --- Log in an existing user ---
 * Endpoint: POST /api/auth/login
 */
exports.loginUser = async (req, res) => {
  const { email, password } = req.body;

  // 1. Validate input
  if (!email || !password) {
    return res.status(400).json({ message: 'Email and password are required.' });
  }

  try {
    // 2. Find the user by email
    const result = await db.query('SELECT * FROM users WHERE email = $1', [email]);
    const user = result.rows[0];

    if (!user) {
      // User not found
      return res.status(401).json({ message: 'Invalid credentials.' });
    }

    // 3. Compare the provided password with the stored hash
    const isMatch = await bcrypt.compare(password, user.password_hash);

    if (!isMatch) {
      // Passwords do not match
      return res.status(401).json({ message: 'Invalid credentials.' });
    }

    // 4. Passwords match! Create a JWT token
    const payload = {
      user: {
        id: user.id,
        email: user.email,
      },
    };

    jwt.sign(
      payload,
      process.env.JWT_SECRET,
      { expiresIn: '1h' }, // Token expires in 1 hour
      (err, token) => {
        if (err) throw err;
        // 5. Send the token back to the client
        res.status(200).json({
          message: 'Login successful!',
          token: token,
        });
      }
    );
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ message: 'Server error during login.' });
  }
};
