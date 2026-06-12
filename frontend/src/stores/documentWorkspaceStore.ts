import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { DocumentPayload } from '../services/apiService';
import { cacheLog } from '../lib/cacheLogger';

type DocumentWorkspaceState = {
  document: DocumentPayload | null;
  compareResult: Record<string, unknown> | null;
  searchResult: Record<string, unknown> | null;
  reference: string;
  selectedAlgorithms: string[];
  activeTab: string;
  setDocument: (doc: DocumentPayload | null) => void;
  setCompareResult: (result: Record<string, unknown> | null) => void;
  setSearchResult: (result: Record<string, unknown> | null) => void;
  setReference: (reference: string) => void;
  setSelectedAlgorithms: (algorithms: string[]) => void;
  setActiveTab: (tab: string) => void;
  resetSession: () => void;
};

const initialState = {
  document: null as DocumentPayload | null,
  compareResult: null as Record<string, unknown> | null,
  searchResult: null as Record<string, unknown> | null,
  reference: '',
  selectedAlgorithms: ['textrank', 'lexrank', 'lsa', 'tfidf'],
  activeTab: 'upload',
};

export const useDocumentWorkspaceStore = create<DocumentWorkspaceState>()(
  persist(
    (set) => ({
      ...initialState,
      setDocument: (document) => {
        cacheLog('SET', 'document workspace', document?.document_id);
        set({ document });
      },
      setCompareResult: (compareResult) => {
        cacheLog('SET', 'document compare result');
        set({ compareResult });
      },
      setSearchResult: (searchResult) => set({ searchResult }),
      setReference: (reference) => set({ reference }),
      setSelectedAlgorithms: (selectedAlgorithms) => set({ selectedAlgorithms }),
      setActiveTab: (activeTab) => set({ activeTab }),
      resetSession: () => {
        cacheLog('INVALIDATE', 'document workspace session');
        set({ ...initialState });
      },
    }),
    {
      name: 'aidh-document-workspace',
      storage: createJSONStorage(() => sessionStorage),
      onRehydrateStorage: () => () => {
        cacheLog('RESTORED', 'document workspace store');
      },
    },
  ),
);
