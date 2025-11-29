import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { useDispatch, useSelector } from 'react-redux';
import { setCredentials } from '../../features/auth/authSlice';
import NavBar from '../../components/NavBar/NavBar';

// Import the external stylesheet
import './LoginPage.css';

function LoginPage() {
  const dispatch = useDispatch();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [message, setMessage] = useState('');
  const [isError, setIsError] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Handles changes for all inputs
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value,
    }));
  };

  // const { isAuthenticated } = useSelector((state) => state.auth);

  // useEffect(() => {
  //   console.log('[App.js] Authentication state changed:', isAuthenticated);
  // }, [isAuthenticated]);

  // --- UPDATED handleSubmit using axios ---
  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage('');
    setIsError(false);

    try {
      // Send request to backend on port 5001 using axios
      const response = await axios.post(
        `${import.meta.env.VITE_API_URL}api/auth/login`,
        formData
      );

      // Login successful!
      console.log('Login successful:', response.data.token);
      setIsError(false);
      setMessage(response.data.message || 'Login successful! Welcome back.');

      // Get data from response
      const token = response.data.token;
      // Get username from API response, or fall back to the email from the form
      const username = response.data.username || formData.email;

      // Dispatch action to save credentials to Redux store
      dispatch(setCredentials({ username, token }));

    } catch (error) {
      // axios throws an error for non-2xx responses
      console.error('Login error:', error);
      setIsError(true);

      // Get the error message from the backend's JSON response
      if (error.response && error.response.data && error.response.data.message) {
        setMessage(error.response.data.message);
      } else {
        // Fallback for network errors or other issues
        setMessage('Login failed. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };
  // --- End of updated handleSubmit ---

  return (
    <>
      <NavBar />
      {/* <LoginStyles /> <-- This is now in LoginPage.css */}
      <div className="login-wrapper">
        <div className="login-card">
          <h2>Login</h2>
          <p className="login-sub">Welcome back! Please enter your details.</p>

          <form className="login-form" onSubmit={handleSubmit}>

            {/* Message Box */}
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
            <button
              className="btn primary"
              type="submit"
              disabled={isLoading}
            >
              {isLoading ? 'Signing In...' : 'Sign In'}
            </button>
          </form>

          <div className="login-links">
            <a href="#">Forgot Password?</a>
            <span>No account? <Link to="/register">Sign Up</Link></span>
          </div>
        </div>
      </div>
    </>
  );
}

export default LoginPage;

