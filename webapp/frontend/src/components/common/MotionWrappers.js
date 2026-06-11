import React from 'react';
import { motion } from 'framer-motion';

// Page Transition Wrapper
export const PageTransition = ({ children }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -20 }}
    transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }} // Custom cubic bezier for "luxury" feel
    style={{ width: '100%' }}
  >
    {children}
  </motion.div>
);

// Stagger Container for Lists/Grids
export const StaggerContainer = ({ children, delay = 0, className = "", animationKey = "" }) => (
  <motion.div
    key={animationKey}
    className={className}
    initial="hidden"
    animate="visible"
    variants={{
      hidden: { opacity: 0 },
      visible: {
        opacity: 1,
        transition: {
          staggerChildren: 0.1,
          delayChildren: delay,
        },
      },
    }}
  >
    {children}
  </motion.div>
);

// Individual Item Fade Up
export const FadeUpItem = ({ children }) => (
  <motion.div
    variants={{
      hidden: { opacity: 0, y: 30, scale: 0.95 },
      visible: { 
        opacity: 1, 
        y: 0, 
        scale: 1,
        transition: { type: "spring", stiffness: 50, damping: 20 }
      },
    }}
  >
    {children}
  </motion.div>
);

// Interactive Card with Glow and Tilt-like effect
export const GlowCard = ({ children, className, onClick }) => (
  <motion.div
    className={className}
    onClick={onClick}
    whileHover={{ 
      y: -5, 
      scale: 1.02,
      boxShadow: "0 20px 40px -10px rgba(0, 242, 255, 0.3)" 
    }}
    whileTap={{ scale: 0.98 }}
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ type: "spring", stiffness: 500, damping: 25 }} // Increased stiffness for faster response
    style={{ cursor: 'pointer', position: 'relative', overflow: 'hidden' }}
  >
    <motion.div
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'radial-gradient(circle at var(--mouse-x, 50%) var(--mouse-y, 50%), rgba(255,255,255,0.1) 0%, transparent 50%)',
        opacity: 0,
        pointerEvents: 'none',
      }}
      whileHover={{ opacity: 1 }}
    />
    {children}
  </motion.div>
);

// Text Reveal Animation
export const TextReveal = ({ text, className }) => {
  const words = text.split(" ");
  return (
    <div className={className} style={{ overflow: 'hidden', display: 'flex', flexWrap: 'wrap', gap: '0.25em' }}>
      {words.map((word, i) => (
        <motion.span
          key={i}
          initial={{ y: 40, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: i * 0.05, ease: [0.22, 1, 0.36, 1] }}
          style={{ display: 'inline-block' }}
        >
          {word}
        </motion.span>
      ))}
    </div>
  );
};
