import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import DashboardLayout from './layouts/DashboardLayout';
import Overview from './pages/Overview';
import Playground from './pages/Playground';
import Analytics from './pages/Analytics';
import DocumentLayout from './pages/documents/DocumentLayout';
import DocumentEvaluation from './pages/documents/DocumentEvaluation';
import DocumentExplainability from './pages/documents/DocumentExplainability';
import Chat from './pages/Chat';

const App = () => {
  return (
    <AppProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<DashboardLayout />}>
            <Route path="/" element={<Overview />} />
            <Route path="/playground" element={<Playground />} />
            <Route path="/documents" element={<DocumentLayout />}>
              <Route index element={<Navigate to="evaluation" replace />} />
              <Route path="evaluation" element={<DocumentEvaluation />} />
              <Route path="explainability" element={<DocumentExplainability />} />
            </Route>
            <Route path="/chat" element={<Chat />} />
            <Route path="/analytics" element={<Analytics />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppProvider>
  );
};

export default App;
