// BackgroundAnimation.tsx
import React from 'react';
import styles from './background.module.css';

const BackgroundAnimation: React.FC = () => {
  return (
    <div className={styles.backgroundContainer}>
      <div className={`${styles.dot} ${styles.dot1}`}>•</div>
      <div className={`${styles.dot} ${styles.dot2}`}>•</div>
      <div className={`${styles.dot} ${styles.dot3}`}>•</div>
      <div className={`${styles.dot} ${styles.dot4}`}>•</div>
      <div className={styles.title}>
        <span className={styles.titleFirstLine}></span>
      </div>
    </div>
  );
};

export default BackgroundAnimation;
