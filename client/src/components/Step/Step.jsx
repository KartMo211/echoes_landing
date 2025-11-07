// src/components/Step/Step.jsx

import React from 'react';
import './Step.css'; // We will create this CSS file next

const Step = ({ number, title, description }) => {
  return (
    <div className="step-container">
      {/* The span element for the number, positioned absolutely */}
      <span className="step-number">{number}</span>
      
      {/* The div containing the text content */}
      <div className="step-content">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
};

export default Step;