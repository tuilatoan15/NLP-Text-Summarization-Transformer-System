import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import Sidebar from './Sidebar';
import Header from './Header';

const DashboardLayout = () => {
  const location = useLocation();

  return (
    <div className="flex h-screen overflow-hidden font-sans transition-colors duration-200 bg-[var(--surface-muted)] text-[var(--text)]">
      {/* Sidebar is fixed width */}
      <Sidebar />
      
      {/* Main content wrapper */}
      <div className="flex-1 flex flex-col ml-64 min-w-0">
        <Header />
        
        {/* Scrollable page content */}
        <main className="flex-1 overflow-y-auto p-6 scroll-smooth">
          <div className="max-w-7xl mx-auto w-full">
            <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
              >
                <Outlet />
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  );
};

export default DashboardLayout;
