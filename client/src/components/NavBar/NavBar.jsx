import React from 'react';
import './navBar.css'; // Import the component's dedicated CSS file

const NavBar = () => {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 w-full flex justify-center pt-4">
      <nav className="glass-nav max-w-lg w-full rounded-full">
        <div className="px-6 py-3 flex justify-between items-center">
          <h1 className="text-xl font-bold tracking-wider">Echoes</h1>
          <a
            href="#"
            className="bg-violet-600 hover:bg-violet-500 text-white font-semibold py-2 px-5 rounded-full text-sm transition-all duration-300"
          >
            Try Echoes
          </a>
        </div>
      </nav>
    </header>
  );
};

export default NavBar;