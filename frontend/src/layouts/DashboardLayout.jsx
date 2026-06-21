import React, { Suspense } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import Sidebar from './Sidebar';
import Header from './Header';
import { useApp } from '../context/AppContext';

const PageLoader = () => (
  <div className="flex items-center justify-center py-24">
    <div className="flex flex-col items-center gap-3">
      <div className="w-8 h-8 rounded-full border-2 border-t-transparent animate-spin" style={{ borderColor: 'var(--border)', borderTopColor: 'transparent' }} />
      <span className="text-sm text-[var(--text-faint)]">Đang tải...</span>
    </div>
  </div>
);

const DashboardLayout = () => {
  const location = useLocation();
  const { sidebarCollapsed } = useApp();
  const [isMobile, setIsMobile] = React.useState(typeof window !== 'undefined' ? window.innerWidth < 1024 : false);

  React.useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 1024);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div
      className="flex h-screen overflow-hidden transition-colors duration-150"
      style={{
        fontFamily: 'var(--font-sans)',
        backgroundColor: 'var(--bg)',
        color: 'var(--text-primary)',
      }}
    >
      {/* Sidebar */}
      <Sidebar />

      {/* Main content wrapper */}
      <motion.div
        className="flex-1 flex flex-col min-w-0"
        animate={{ marginLeft: isMobile ? 0 : (sidebarCollapsed ? 68 : 260) }}
        transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
      >
        <Header />

        {/* Scrollable page content */}
        <main className="flex-1 overflow-y-auto scroll-smooth">
          <div className="max-w-7xl mx-auto w-full px-6 py-6">
            <Suspense fallback={<PageLoader />}>
              <AnimatePresence mode="wait">
                <motion.div
                  key={location.pathname}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                >
                  <Outlet />
                </motion.div>
              </AnimatePresence>
            </Suspense>
          </div>
        </main>
      </motion.div>
    </div>
  );
};

export default DashboardLayout;
