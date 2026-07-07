import React, { Suspense, lazy, memo, useCallback, useEffect, useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import Sidebar from './Sidebar';
import Header from './Header';
import { useApp } from '../context/AppContext';
import { usePrefersReducedMotion } from '../hooks/usePrefersReducedMotion';
import { throttle } from '../utils/throttle';

const CommandPalette = lazy(() => import('../components/CommandPalette'));

const EASE = [0.16, 1, 0.3, 1];
const TRANSITION_MS = 200;

const PageLoader = memo(() => (
  <div className="flex items-center justify-center py-24">
    <div className="flex flex-col items-center gap-3">
      <div className="w-8 h-8 rounded-full border-2 border-t-transparent animate-spin" style={{ borderColor: 'var(--border)', borderTopColor: 'transparent' }} />
      <span className="text-sm text-[var(--text-faint)]">Đang tải...</span>
    </div>
  </div>
));

const DashboardLayout = () => {
  const location = useLocation();
  const { sidebarCollapsed } = useApp();
  const reducedMotion = usePrefersReducedMotion();
  const [isMobile, setIsMobile] = useState(
    typeof window !== 'undefined' ? window.innerWidth < 1024 : false,
  );

  const handleResize = useCallback(
    throttle(() => setIsMobile(window.innerWidth < 1024), 150),
    [],
  );

  useEffect(() => {
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [handleResize]);

  const sidebarOffset = isMobile ? 0 : (sidebarCollapsed ? 68 : 260);

  return (
    <div
      className="flex h-screen overflow-hidden transition-colors duration-150"
      style={{
        fontFamily: 'var(--font-sans)',
        backgroundColor: 'var(--bg)',
        color: 'var(--text-primary)',
      }}
    >
      <Sidebar />

      <div
        className="flex-1 flex flex-col min-w-0 gpu-layer"
        style={{
          marginLeft: sidebarOffset,
          transition: reducedMotion ? 'none' : `margin-left ${TRANSITION_MS}ms cubic-bezier(0.16, 1, 0.3, 1)`,
        }}
      >
        <Header />

        <main className="flex-1 overflow-y-auto scroll-smooth">
          <div className="max-w-7xl mx-auto w-full px-6 py-6">
            <Suspense fallback={<PageLoader />}>
              {reducedMotion ? (
                <Outlet />
              ) : (
                <AnimatePresence mode="wait">
                  <motion.div
                    key={location.pathname}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: TRANSITION_MS / 1000, ease: EASE }}
                    className="gpu-layer"
                  >
                    <Outlet />
                  </motion.div>
                </AnimatePresence>
              )}
            </Suspense>
          </div>
        </main>
      </div>

      <Suspense fallback={null}>
        <CommandPalette />
      </Suspense>
    </div>
  );
};

export default memo(DashboardLayout);
