import React, { useState } from 'react';

const SplineBackground: React.FC = () => {
  const [loaded, setLoaded] = useState(false);

  return (
    <div className="fixed inset-0 z-0 w-full h-full overflow-hidden pointer-events-none">
       {/* Overlay to dim the spline slightly for text readability */}
      <div className="absolute inset-0 bg-black/40 z-10 pointer-events-none" />
      
      <div className={`w-full h-full transition-opacity duration-1000 ${loaded ? 'opacity-100' : 'opacity-0'}`}>
        <iframe 
          src='https://my.spline.design/interactiveaiwebsite-HqXdcCPoQnVLD2wraqAFhK3L/' 
          frameBorder='0' 
          width='100%' 
          height='100%'
          className="w-full h-full scale-110" // Slight scale to cover edges
          onLoad={() => setLoaded(true)}
          title="Spline 3D Background"
        />
      </div>
      
      {!loaded && (
        <div className="absolute inset-0 z-0 flex items-center justify-center bg-black">
          <div className="w-8 h-8 border-4 border-brand-accent border-t-transparent rounded-full animate-spin"></div>
        </div>
      )}
    </div>
  );
};

export default SplineBackground;