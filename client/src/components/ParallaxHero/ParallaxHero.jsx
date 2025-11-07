import React, { useRef, useEffect } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import ParallaxHeroImage from '../../assets/ParallaxHero.jpg'; // Replace with your image path
import './ParallaxHero.css'; // Make sure to link the CSS

// Register ScrollTrigger plugin
gsap.registerPlugin(ScrollTrigger);

const ParallaxHero = () => {
  const sectionRef = useRef(null);
  const backgroundRef = useRef(null);
  const weTextRef = useRef(null);
  const designTextRef = useRef(null);
  const digitalTextRef = useRef(null);

  useEffect(() => {
    const section = sectionRef.current;
    const background = backgroundRef.current;

    if (!section || !background) return;

    // Pin the section while scrolling through it, and move the background
    // This creates the parallax effect
    gsap.to(background, {
      y: '0%', // Adjust for parallax intensity
      ease: 'none',
      scrollTrigger: {
        trigger: section,
        start: 'top bottom',
        end: 'bottom top',
        scrub: true,
        // markers: true, // For debugging
      },
    });

    // Optional: Animate the text elements to appear or move slightly during scroll
    // This makes them integrate with the parallax more dynamically
    gsap.fromTo(weTextRef.current, 
      { opacity: 0, y: 50 },
      { opacity: 1, y: 0, ease: 'power1.out', duration: 1, scrollTrigger: { trigger: section, start: 'top center', end: 'center center', scrub: 0.5 } }
    );
    gsap.fromTo(designTextRef.current, 
      { opacity: 0, y: 50 },
      { opacity: 1, y: 0, ease: 'power1.out', duration: 1, scrollTrigger: { trigger: section, start: 'top center+=100', end: 'center center', scrub: 0.5 } }
    );
    gsap.fromTo(digitalTextRef.current, 
      { opacity: 0, y: 50 },
      { opacity: 1, y: 0, ease: 'power1.out', duration: 1, scrollTrigger: { trigger: section, start: 'top center+=200', end: 'center center', scrub: 0.5 } }
    );


  }, []);

  return (
    <section className="parallax-hero-section" ref={sectionRef}>
      <div 
        className="parallax-background" 
        ref={backgroundRef} 
        style={{ backgroundImage: `url(${ParallaxHeroImage})` }}
      ></div>
      <div className="parallax-content-overlay">
        {/* We'll layer text elements */}
        <p className="large-text we-text" ref={weTextRef}>We</p>
        <p className="large-text design-text" ref={designTextRef}>Transform</p>
        <p className="large-text digital-text" ref={digitalTextRef}>Memories.</p>
      </div>
    </section>
  );
};

export default ParallaxHero;