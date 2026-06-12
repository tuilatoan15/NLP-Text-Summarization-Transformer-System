import React, { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import { QueryProvider } from './providers/QueryProvider';
import DashboardLayout from './layouts/DashboardLayout';

// Lazy-loaded pages for code splitting
const Overview = lazy(() => import('./pages/Overview'));
const Playground = lazy(() => import('./pages/Playground'));
const Analytics = lazy(() => import('./pages/Analytics'));
const Chat = lazy(() => import('./pages/Chat'));
const Compare = lazy(() => import('./pages/Compare'));
const DocumentLayout = lazy(() => import('./pages/documents/DocumentLayout'));
const DocumentEvaluation = lazy(() => import('./pages/documents/DocumentEvaluation'));
const DocumentExplainability = lazy(() => import('./pages/documents/DocumentExplainability'));

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

            {/* AI features */}
            <Route path="/compare" element={<Compare />} />
            <Route path="/search" element={<Chat />} />

            {/* Document routes */}
            <Route path="/documents" element={<DocumentLayout />}>
              <Route index element={<Navigate to="evaluation" replace />} />
              <Route path="evaluation" element={<DocumentEvaluation />} />
              <Route path="explainability" element={<DocumentExplainability />} />
            </Route>

            {/* Settings placeholder */}
            <Route path="/settings" element={<Analytics />} />

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
