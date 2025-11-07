// import Homepage from './pages/Homepage/homepage.jsx' // 👈 Remove this line
import '../../App.css'
import { useEffect, useRef } from 'react';
import { useLayoutEffect } from 'react';

import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

import AnimatedTitle from '../../components/AnimatedTitle/AnimatedTitle.jsx'
import ParallaxHero from '../../components/ParallaxHero/ParallaxHero.jsx';
import ScrollReveal from '../../components/ScrollReveal/ScrollReveal.jsx';
import Step from '../../components/Step/Step.jsx';
import DecryptedText from '../../components/Decrypted Text/DecryptedText.jsx';
import Silk from '../../components/Silk/Silk.jsx';


import '../../components/Step/Step.css'
import Footer from '../../components/Footer/Footer.jsx';
import ChromaGrid from '../../components/ChromaGrid/ChromaGrid.jsx';
import teamMembers from '../../data/team/teamMembers.js'
import NavBar from '../../components/NavBar/NavBar.jsx';

gsap.registerPlugin(ScrollTrigger);

function Homepage() {
  const scrollRevealContainer = useRef(null);

  // 2. Set up the animation inside a useEffect hook
  useEffect(() => {
    // Ensure the ref is connected to the element
    const el = scrollRevealContainer.current;

    // Use a GSAP timeline to animate the margin-top property
    gsap.to(el, {
      marginTop: '-100px', // The target value you want to animate to
      ease: 'none',        // No easing for a direct link to scroll
      scrollTrigger: {
        trigger: el,            // The element that triggers the animation
        start: 'top bottom',    // When the top of the trigger hits the bottom of the viewport
        end: 'top top',         // When the top of the trigger hits the top of the viewport
        scrub: true,            // This links the animation progress directly to scroll position
        // markers: true,          // Uncomment this for debugging to see start/end points
      },
    });
    
    // Note: GSAP's context handles cleanup automatically in React 18+,
    // so a manual cleanup function is often not needed for simple tweens.

  }, []); // Empty dependency array ensures this runs only once on mount
  // ... rest of your code is fine

  const main = useRef();
  
  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
        const panels = gsap.utils.toArray(".step-panel");

        // The timeline will control the animations that happen while the section is pinned
        const timeline = gsap.timeline();
      
        // 1. Animate Panel 2 (from the left) to cover Panel 1
        // This is the FIRST part of the scroll animation
        timeline.from(panels[1], { 
            xPercent: 100, 
            ease: "none" 
        }); 

        // 2. Animate Panel 3 (from the right) to cover Panel 2
        // This is the SECOND part of the scroll animation. It happens after the first one is complete.
        timeline.from(panels[2], { 
            xPercent: 200, 
            ease: "none" 
        });

        // The first panel (panels[0]) doesn't need a 'from' animation in the timeline
        // because it's the starting point of our pinned section. It's already visible.

        // Pin the container and link the animation timeline to the scrollbar
        ScrollTrigger.create({
            animation: timeline,
            trigger: main.current,
            pin: true,
            scrub: 1, // Using a number like 1 provides a little smoothing
            start: "top top",
            end: "+=3000", // Adjust this value to control how much scroll is needed
            // markers: true, 
        });

    }, main);

    // Cleanup function
    return () => ctx.revert();
}, []);

  return (
    <>
      <NavBar />
      <AnimatedTitle />
      <div className='ScrollRevealText' ref={scrollRevealContainer}>
        <h2 className="title">Turn memories into living stories.</h2>
        <ScrollReveal>
          Echoes blends AI and craft to revive your past—authentically and beautifully. Relive and share the moments that matter.
        </ScrollReveal>
      </div>
      <ParallaxHero />

       <section className="steps-section-wrapper" ref={main}>
        <div style={{marginLeft:"50px",marginBottom:"40px"}}>
          <DecryptedText
            text="Your Connection in"
            animateOn="view"
            speed={200}
            className='steps-section-title'
            encryptedClassName='steps-section-title'
            revealDirection="start"
          />
          <br /><br />
          <DecryptedText
            text="Three Simple Steps."
            animateOn="view"
            speed={200}
            className='steps-section-title second'
            encryptedClassName='steps-section-title second'
            revealDirection="start"
          />
        </div>
        
        
        {/* This container will be pinned and will scroll horizontally */}
        <div className="steps-panels-container" >
          
          {/* Panel 1 */}
          <div className="step-panel" style={{backgroundColor:"black", color:"white"}}>
            <div style={{position:"absolute",zIndex:1, top:0, left:0, width:"100%", height:"100%"}}>
              <Silk
                speed={5}
                scale={1}
                color="#7C5CFF"
                noiseIntensity={1.5}
                rotation={0}
              />

            </div>
            <Step 
              number="01" 
              title="Gather Memories"
              description="We start with your stories and photos, understanding the moments you want to bring back to life. So Users upload digital footprints." 
            />
          </div>

          {/* Panel 2 */}
          <div className="step-panel" style={{backgroundColor:"white", color:"black"}}>
            <div style={{position:"absolute",zIndex:1, top:0, left:0, width:"100%", height:"100%"}}>
                <Silk
                  speed={5}
                  scale={1}
                  color="#22D3EE"
                  noiseIntensity={1.5}
                  rotation={0}
                />
            </div>
            <Step 
              number="02" 
              title="AI-Powered Revival"
              description="Build Digital Twins using advanced AI that captures voices, mannerisms, and stories and transforms them into an interactive persona" 
            />
          </div>
          
          {/* Panel 3 */}
          <div className="step-panel" style={{backgroundColor:"grey", color:"white"}}>
            <div style={{position:"absolute",zIndex:1, top:0, left:0, width:"100%", height:"100%"}}>
              <Silk
                speed={5}
                scale={1}
                color="#FF8EDB"
                noiseIntensity={1.5}
                rotation={0}
              />

            </div>
            <Step 
              number="03" 
              title="Converse Forever"
              description="Engage in meaningful conversations with your loved ones' digital twins, preserving their essence for generations to come." 
            />
          </div>

        </div>
      </section>
          

      
      {/* <div style={{ height: '150vh', background: '#222', color: '#ccc', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '2rem', padding: '50px' }}>
        <p>This is placeholder content to make the page scrollable and demonstrate the parallax effect above. </p>
      </div>
       <div style={{ height: '100vh', background: '#0a0a0a', color: '#888', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '2rem', padding: '50px' }}>
        <p>Keep scrolling... The design possibilities are endless!</p>
      </div> */}

      <div style={{marginLeft:"50px",marginBottom:"40px"}}>
        <DecryptedText
          text="Meet the Team."
          animateOn="view"
          speed={100}
          className='steps-section-title second'
          encryptedClassName='steps-section-title second'
          revealDirection="start"
        />
      </div>
      <div style={{ height: '600px', position: 'relative' }}>
      <ChromaGrid 
        items={teamMembers}
        radius={300}
        damping={0.45}
        fadeOut={0.6}
        ease="power3.out"
      />
    </div>


      <Footer />
    </>
  )
}

export default Homepage