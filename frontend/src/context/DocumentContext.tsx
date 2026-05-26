import React, { createContext, useContext, useMemo, useState } from 'react';
import type { DocumentPayload } from '../services/apiService';

type DocumentContextValue = {
  document: DocumentPayload | null;
  setDocument: (doc: DocumentPayload | null) => void;
  compareResult: Record<string, unknown> | null;
  setCompareResult: (value: Record<string, unknown> | null) => void;
};

const DocumentContext = createContext<DocumentContextValue | undefined>(undefined);

export function DocumentProvider({ children }: { children: React.ReactNode }) {
  const [document, setDocument] = useState<DocumentPayload | null>(null);
  const [compareResult, setCompareResult] = useState<Record<string, unknown> | null>(null);
  const value = useMemo(
    () => ({ document, setDocument, compareResult, setCompareResult }),
    [document, compareResult],
  );
  return <DocumentContext.Provider value={value}>{children}</DocumentContext.Provider>;
}

export function useDocumentContext() {
  const ctx = useContext(DocumentContext);
  if (!ctx) throw new Error('useDocumentContext must be used within DocumentProvider');
  return ctx;
}
