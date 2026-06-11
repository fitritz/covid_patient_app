import React from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import Navbar from './components/Navbar/Navbar';
import Dashboard from './components/Dashboard/Dashboard';
import PatientDetail from './components/PatientDetail/PatientDetail';
import PredictionPage from './components/PredictionPage/PredictionPage';
import BackgroundAnimation from './components/common/BackgroundAnimation';
import './App.css';

const AnimatedRoutes = () => {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/patients/:id" element={<PatientDetail />} />
        <Route path="/predict" element={<PredictionPage />} />
      </Routes>
    </AnimatePresence>
  );
};

function App() {
  return (
    <Router>
      <div className="App">
        <BackgroundAnimation />
        <Navbar />
        <main style={{ position: 'relative', zIndex: 1 }}>
          <AnimatedRoutes />
        </main>
      </div>
    </Router>
  );
}


export default App;
