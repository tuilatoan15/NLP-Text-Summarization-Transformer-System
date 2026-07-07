import React, { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import { QueryProvider } from './providers/QueryProvider';
import DashboardLayout from './layouts/DashboardLayout';

// Lazy-loaded pages for code splitting
const Overview = lazy(() => import('./pages/Overview'));
const Playground = lazy(() => import('./pages/Playground'));
const Analytics = lazy(() => import('./pages/Analytics'));
const DatasetAnalytics = lazy(() => import('./pages/DatasetAnalytics'));
const Chat = lazy(() => import('./pages/Chat'));
const Compare = lazy(() => import('./pages/Compare'));
const Settings = lazy(() => import('./pages/Settings'));
const Benchmark = lazy(() => import('./pages/Benchmark'));
const DocumentWorkspace = lazy(() => import('./pages/documents/DocumentWorkspace'));

const App = () => {
  return (
    <QueryProvider>
    <AppProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<DashboardLayout />}>
            <Route path="/" element={<Overview />} />

            {/* Main features */}
            <Route path="/summarize" element={<Playground />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/dataset-analytics" element={<DatasetAnalytics />} />

            {/* AI features */}
            <Route path="/compare" element={<Compare />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="/search" element={<Chat />} />

            {/* Document routes */}
            <Route path="/documents" element={<DocumentWorkspace />} />

            {/* Settings page */}
            <Route path="/settings" element={<Settings />} />

            {/* Backward compatibility redirects */}
            <Route path="/playground" element={<Navigate to="/summarize" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppProvider>
    </QueryProvider>
  );
};

export default App;
