import React, { useState } from 'react';
import axios from 'axios';

// We'll import useNavigate and Link from 'react-router-dom'
// In a real multi-file app, you'd use:
import { useNavigate, Link } from 'react-router-dom';
import NavBar from '../../components/NavBar/NavBar';
/**
 * Note: For this single-file environment, we assume 'axios' is available globally
 * or that your build setup handles it.
 * If not, you might need to load it from a CDN in your main index.html or App.jsx.
 * * We also assume `useNavigate` and `Link` are available if you're using the
 * CDN-based router setup from our previous chat. If so, you'd get them like:
 * const { useNavigate, Link } = window.ReactRouterDOM;
 */

// --- Re-usable Styles Component (same as LoginPage) ---
const AuthPageStyles = () => (
  <style>
    {`
    :root {
      --auth-bg: #0b0d11;
      --auth-text: #e8eef9;
      --auth-muted: #b7c0d8;
      --auth-accent: #7c5cff;
      --auth-accent-2: #00e0d1;
      --auth-card: #121621;
      --auth-glass: rgba(255, 255, 255, .06);
      --auth-border: rgba(255, 255, 255, .12);
      --auth-shadow: 0 10px 30px rgba(0, 0, 0, .35);
      --auth-radius: 20px;
    }
    .auth-wrapper {
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji';
      background: radial-gradient(1200px 800px at 80% -10%, rgba(124, 92, 255, .25), transparent 60%),
        radial-gradient(1000px 700px at -10% 10%, rgba(0, 224, 209, .2), transparent 55%),
        var(--auth-bg);
      color: var(--auth-text);
      line-height: 1.6;
      display: grid;
      place-items: center;
      min-height: 100vh;
      padding: 20px;
    }
    .auth-wrapper a {
      color: var(--auth-accent);
      text-decoration: none;
      cursor: pointer;
    }
    .auth-wrapper a:hover {
      text-decoration: underline;
    }
    .btn {
      padding: 14px 18px;
      border-radius: 14px;
      border: 1px solid var(--auth-border);
      background: var(--auth-glass);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      font-weight: 600;
      font-size: 16px;
      cursor: pointer;
      color: var(--auth-text);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .btn.primary {
      background: linear-gradient(135deg, var(--auth-accent), var(--auth-accent-2));
      border-color: transparent;
      color: white;
      box-shadow: 0 10px 25px rgba(124, 92, 255, .35);
    }
    .btn:hover {
      transform: translateY(-2px);
    }
    .btn:disabled {
      opacity: 0.7;
      cursor: not-allowed;
    }
    .auth-card {
      width: min(420px, 95vw);
      background: var(--auth-card);
      border: 1px solid var(--auth-border);
      border-radius: var(--auth-radius);
      padding: 28px;
      box-shadow: var(--auth-shadow);
    }
    .auth-card h2 {
      font-size: clamp(24px, 3.2vw, 32px);
      margin: 0 0 10px;
      text-align: center;
      font-weight: 700;
      color: var(--auth-text);
    }
    .auth-sub {
      color: var(--auth-muted);
      margin: 0 0 24px;
      text-align: center;
      font-size: 16px;
    }
    .auth-form {
      display: grid;
      gap: 16px;
    }
    .auth-form input {
      background: var(--auth-glass);
      border: 1px solid var(--auth-border);
      border-radius: 12px;
      padding: 14px 16px;
      color: var(--auth-text);
      outline: none;
      font-size: 16px;
      width: 100%;
    }
    .auth-form input:focus {
      border-color: var(--auth-accent);
      box-shadow: 0 0 10px rgba(124, 92, 255, .25);
    }
    .auth-form .btn {
      width: 100%;
      padding-top: 16px;
      padding-bottom: 16px;
    }
    .auth-links {
      margin-top: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      font-size: 14px;
      color: var(--auth-muted);
    }
    .message-box {
      padding: 12px;
      border-radius: 8px;
      margin-bottom: 16px;
      font-size: 14px;
      text-align: center;
    }
    .message-box.error {
      background-color: rgba(255, 100, 100, 0.2);
      color: rgba(255, 200, 200, 1);
      border: 1px solid rgba(255, 100, 100, 0.5);
    }
    .message-box.success {
      background-color: rgba(100, 255, 100, 0.2);
      color: rgba(200, 255, 200, 1);
      border: 1px solid rgba(100, 255, 100, 0.5);
    }
    `}
  </style>
);

function RegisterPage() {
  // Use hooks from React Router (assuming they are loaded/imported)
  const navigate = window.ReactRouterDOM?.useNavigate ? window.ReactRouterDOM.useNavigate() : () => console.warn('useNavigate not loaded');
  const Link = window.ReactRouterDOM?.Link ? window.ReactRouterDOM.Link : ({ to, children }) => <a href={to}>{children}</a>;

  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [message, setMessage] = useState('');
  const [isError, setIsError] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // --- Client-side validation ---
    if (formData.password !== formData.confirmPassword) {
      setIsError(true);
      setMessage('Passwords do not match.');
      return;
    }
    if (formData.password.length < 6) {
      // This matches the validation in your authController
      setIsError(true);
      setMessage('Password must be at least 6 characters long.');
      return;
    }

    setIsLoading(true);
    setMessage('');
    setIsError(false);

    try {
      // Destructure only the fields the backend schema expects
      const { email, password } = formData;

      // Send request to backend
      const response = await axios.post(
        `${import.meta.env.VITE_API_URL}api/auth/register`,
        { email, password } // Only send email and password
      );

      // Handle success
      console.log('Registration successful:', response.data);
      setIsError(false);
      setMessage(response.data.message || 'Registration successful! Redirecting to login...');

      // Redirect to login page after a short delay
      setTimeout(() => {
        if (navigate) navigate('/login');
      }, 1500);

    } catch (error) {
      // Handle errors
      console.error('Registration error:', error);
      setIsError(true);
      if (error.response && error.response.data && error.response.data.message) {
        // Display the specific error message from the backend
        setMessage(error.response.data.message);
      } else {
        // Generic fallback error
        setMessage('Registration failed. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <NavBar />
      <AuthPageStyles />
      <div className="auth-wrapper">
        <div className="auth-card">
          <h2>Create Account</h2>
          <p className="auth-sub">Get started by creating your new account.</p>

          <form className="auth-form" onSubmit={handleSubmit}>
            {/* Message box for errors or success */}
            {message && (
              <div className={`message-box ${isError ? 'error' : 'success'}`}>
                {message}
              </div>
            )}

            <input
              type="email"
              name="email"
              placeholder="Email address"
              required
              value={formData.email}
              onChange={handleChange}
              disabled={isLoading}
            />
            <input
              type="password"
              name="password"
              placeholder="Password"
              required
              value={formData.password}
              onChange={handleChange}
              disabled={isLoading}
            />
            <input
              type="password"
              name="confirmPassword"
              placeholder="Confirm Password"
              required
              value={formData.confirmPassword}
              onChange={handleChange}
              disabled={isLoading}
            />
            <button
              className="btn primary"
              type="submit"
              disabled={isLoading}
            >
              {isLoading ? 'Creating Account...' : 'Sign Up'}
            </button>
          </form>

          <div className="auth-links">
            <span>Already have an account? <Link to="/login">Sign In</Link></span>
          </div>
        </div>
      </div>
    </>
  );
}

// In a real multi-file app, you'd export this:
export default RegisterPage;
