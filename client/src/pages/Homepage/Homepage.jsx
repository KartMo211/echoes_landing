import { useState } from 'react';
import CircularGallery from '../../components/circularGallery/CircularGallery.jsx';
import './Homepage.css';


function Homepage() {
  const [backgroundName, setBackgroundName] = useState("https://picsum.photos/seed/1/800/600?grayscale")

  return (
    <div>
      <div className="background" style={{backgroundImage: `url(${backgroundName})`}}>
      </div>
      <CircularGallery setBackgroundName={setBackgroundName} />
    </div>
  );
}

export default Homepage;