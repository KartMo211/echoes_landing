import React from 'react';
import './AnimatedTitle.css';
import LiquidEther from './LiquidEther';
import ScrollReveal from '../ScrollReveal/ScrollReveal.jsx';



const AnimatedTitle = () => {
  return (
    <>
      <div className="animation-container">
          <LiquidEther
              colors={[ '#5227FF', '#FF9FFC', '#B19EEF' ]}
              mouseForce={20}
              cursorSize={100}
              isViscous={false}
              viscous={30}
              iterationsViscous={32}
              iterationsPoisson={32}
              resolution={0.5}
              isBounce={false}
              autoDemo={true}
              autoSpeed={0.5}
              autoIntensity={2.2}
              takeoverDuration={0.25}
              autoResumeDelay={3000}
              autoRampDuration={0.6}
          />
        <span className="welcome-text">Welcome to</span>
        <h1 className="echoes-text">Echoes</h1>

        {/* NEW: Scroll Down Indicator */}
        <div className="scroll-indicator">
          <span>Scroll down</span>
          <span className="arrow">↓</span> {/* Downward arrow character */}
        </div>

      </div>
      {/* <div className="circular-gallery-container">
        <CircularGallery />
      </div>
      <ScrollReveal
        baseOpacity={0}
        enableBlur={true}
        baseRotation={5}
        blurStrength={10}
      >
        When does a man die? When he is hit by a bullet? No! When he suffers a disease?
        No! When he ate a soup made out of a poisonous mushroom?
        No! A man dies when he is forgotten!
      </ScrollReveal> */}
    </>
    
  );
};

export default AnimatedTitle;