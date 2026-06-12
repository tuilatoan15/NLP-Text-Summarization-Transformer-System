import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { cacheLog } from '../lib/cacheLogger';

const SAMPLE_TEXT = `Tập đoàn Điện lực Việt Nam cho biết nhu cầu tiêu thụ điện trong mùa nắng nóng tiếp tục tăng cao tại nhiều địa phương. Các nhà máy thủy điện ở miền Bắc được yêu cầu vận hành thận trọng do mực nước một số hồ chứa chưa phục hồi hoàn toàn.`;

export type PlaygroundFileMeta = {
  name: string;
  size: number;
  lastModified: number;
  type: string;
};

export type PlaygroundRunState = Record<
  string,
  { status: string; result: Record<string, unknown> | null; error: string | null }
>;

type PlaygroundState = {
  text: string;
  reference: string;
  fileMetas: PlaygroundFileMeta[];
  selected: string[];
  summaryLength: string;
  result: Record<string, unknown> | null;
  runState: PlaygroundRunState;
  completedCount: number;
  lastExtractFingerprint: string | null;
  setText: (text: string) => void;
  setReference: (reference: string) => void;
  setFileMetas: (metas: PlaygroundFileMeta[]) => void;
  setSelected: (selected: string[]) => void;
  setSummaryLength: (length: string) => void;
  setResult: (result: Record<string, unknown> | null) => void;
  setRunState: (runState: PlaygroundRunState) => void;
  setCompletedCount: (count: number) => void;
  setLastExtractFingerprint: (fp: string | null) => void;
  clearUploadCache: () => void;
  resetSession: () => void;
};

const initialAlgorithms = ['textrank', 'lexrank', 'lsa', 'vit5', 'mt5', 'bartpho'];

const initialState = {
  text: SAMPLE_TEXT,
  reference: '',
  fileMetas: [] as PlaygroundFileMeta[],
  selected: initialAlgorithms,
  summaryLength: 'auto',
  result: null as Record<string, unknown> | null,
  runState: {} as PlaygroundRunState,
  completedCount: 0,
  lastExtractFingerprint: null as string | null,
};

export const usePlaygroundStore = create<PlaygroundState>()(
  persist(
    (set) => ({
      ...initialState,
      setText: (text) => set({ text }),
      setReference: (reference) => set({ reference }),
      setFileMetas: (fileMetas) => set({ fileMetas }),
  setSelected: (selected) => set((state) => ({
    selected: typeof selected === 'function' ? selected(state.selected) : selected,
  })),
  setSummaryLength: (summaryLength) => set({ summaryLength }),
  setResult: (result) => {
    cacheLog('SET', 'playground summarize result');
    set({ result: typeof result === 'function' ? result(usePlaygroundStore.getState().result) : result });
  },
  setRunState: (runState) => set((state) => ({
    runState: typeof runState === 'function' ? runState(state.runState) : runState,
  })),
      setCompletedCount: (completedCount) => set({ completedCount }),
      setLastExtractFingerprint: (lastExtractFingerprint) => set({ lastExtractFingerprint }),
      clearUploadCache: () => {
        cacheLog('INVALIDATE', 'playground upload cache');
        set({ fileMetas: [], lastExtractFingerprint: null });
      },
      resetSession: () => {
        cacheLog('INVALIDATE', 'playground session');
        set({ ...initialState });
      },
    }),
    {
      name: 'aidh-playground-session',
      storage: createJSONStorage(() => sessionStorage),
      onRehydrateStorage: () => () => {
        cacheLog('RESTORED', 'playground session store');
      },
      partialize: (state) => ({
        text: state.text,
        reference: state.reference,
        fileMetas: state.fileMetas,
        selected: state.selected,
        summaryLength: state.summaryLength,
        result: state.result,
        runState: state.runState,
        completedCount: state.completedCount,
        lastExtractFingerprint: state.lastExtractFingerprint,
      }),
    },
  ),
);

export function fileFingerprint(meta: PlaygroundFileMeta): string {
  return `${meta.name}:${meta.size}:${meta.lastModified}`;
}

export function filesFingerprint(metas: PlaygroundFileMeta[]): string {
  return metas.map(fileFingerprint).join('|');
}
