// All the GSAP/ScrollTrigger imports are kept as you had them
import './App.css';
import { useEffect, useRef }
from 'react';
import { useLayoutEffect } from 'react';

import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useSelector } from 'react-redux';

import Homepage from './pages/Homepage/Homepage.jsx';
import LoginPage from './pages/Login/LoginPage.jsx';
import Register from './pages/Register/Register.jsx';
import Dashboard from './pages/Dashboard/Dashboard.jsx';

import './components/Step/Step.css';

gsap.registerPlugin(ScrollTrigger);

function App() {
  const { isAuthenticated } = useSelector((state) => state.auth);

  useEffect(() => {
    console.log('[App.js] Authentication state changed:', isAuthenticated);
  }, [isAuthenticated]);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Homepage />} />
        <Route
          path="/login"
          element={isAuthenticated ? <Navigate to="/dashboard" /> : <LoginPage />}
        />
        <Route
          path="/register"
          element={isAuthenticated ? <Navigate to="/dashboard" /> : <Register />}
        />
        <Route
          path="/dashboard"
          element={isAuthenticated ? <Dashboard /> : <Navigate to="/login" />}
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;