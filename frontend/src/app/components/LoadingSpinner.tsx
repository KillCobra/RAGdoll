import React from 'react';
import { bouncy } from 'ldrs'
// Import the quantum loader

// Register the quantum loader
bouncy.register()


interface LoadingSpinnerProps {
  text?: string;
}

export default function LoadingSpinner({ text = '' }: LoadingSpinnerProps) {
  return (
    <div className="flex flex-col items-center justify-center">
      <l-bouncy
        size="35"
        speed="1.75" 
        color="black" 
      ></l-bouncy>
      {text && <span className="mt-2 text-sm text-gray-600">{text}</span>}
    </div>
  );
}
