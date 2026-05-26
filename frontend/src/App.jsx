import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import DashboardLayout from './layouts/DashboardLayout';
import Overview from './pages/Overview';
import Playground from './pages/Playground';
import ModelSettings from './pages/ModelSettings';
import Analytics from './pages/Analytics';
import DocumentLayout from './pages/documents/DocumentLayout';
import DocumentUpload from './pages/documents/DocumentUpload';
import DocumentAnalysis from './pages/documents/DocumentAnalysis';
import DocumentCompare from './pages/documents/DocumentCompare';
import DocumentEvaluation from './pages/documents/DocumentEvaluation';
import DocumentSearchPage from './pages/documents/DocumentSearchPage';
import DocumentExplainability from './pages/documents/DocumentExplainability';
import DocumentNotebook from './pages/documents/DocumentNotebook';

const App = () => {
  return (
    <AppProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<DashboardLayout />}>
            <Route path="/" element={<Overview />} />
            <Route path="/playground" element={<Playground />} />
            <Route path="/documents" element={<DocumentLayout />}>
              <Route index element={<Navigate to="upload" replace />} />
              <Route path="upload" element={<DocumentUpload />} />
              <Route path="analysis" element={<DocumentAnalysis />} />
              <Route path="compare" element={<DocumentCompare />} />
              <Route path="evaluation" element={<DocumentEvaluation />} />
              <Route path="search" element={<DocumentSearchPage />} />
              <Route path="explainability" element={<DocumentExplainability />} />
              <Route path="notebook" element={<DocumentNotebook />} />
            </Route>
            <Route path="/settings" element={<ModelSettings />} />
            <Route path="/analytics" element={<Analytics />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppProvider>
  );
};

export default App;
