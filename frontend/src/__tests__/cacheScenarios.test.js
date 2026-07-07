import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClientProvider } from '@tanstack/react-query';
import { useOverviewBundleQuery, useAnalyticsDashboardQuery } from '../hooks/useApiQueries';
import { invalidateAllCaches, invalidateFileExtractCache } from '../lib/cacheInvalidation';
import { usePlaygroundStore } from '../stores/playgroundStore';
import { queryKeys } from '../lib/queryKeys';
import * as apiService from '../services/apiService';

vi.mock('../services/apiService', () => ({
  getHealth: vi.fn(async () => ({ status: 'ok' })),
  getMetrics: vi.fn(async () => ({ models_preloaded: true, model_load_times: { vit5: 1.2 } })),
  getAnalyticsDashboard: vi.fn(async (range) => ({
    metrics: { total_runs: range === '30d' ? 5 : 2 },
    visualization: { timeseries: [], model_performance: [] },
    recent_runs: [],
    overview: { document_count: 0 },
  })),
  getSystemGpu: vi.fn(async () => ({ available: false, status: 'unavailable' })),
  getSystemNode: vi.fn(async () => ({ status: 'healthy', node_id: 'test-node' })),
  getSystemModels: vi.fn(async () => ({ summarizers: [], rag_models: [], loaded_count: 0 })),
  getResearchLeaderboard: vi.fn(async () => ({ leaderboard: [{ key: 'vit5' }], metadata: {} })),
  getLeaderboardByCategory: vi.fn(async () => ({ leaderboard: [] })),
  getResearchHybridStudy: vi.fn(async () => ({ groups: {} })),
  getResearchReport: vi.fn(async () => ({ title: 'Report' })),
  getResearchBenchmarkSamples: vi.fn(async () => ({ items: [], pages: 1 })),
  getExplainability: vi.fn(async () => ({ keywords: [] })),
}));

function createWrapper(client) {
  return function Wrapper({ children }) {
    return React.createElement(QueryClientProvider, { client }, children);
  };
}

function createTestClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: Infinity,
        gcTime: Infinity,
        retry: false,
      },
    },
  });
}

describe('Cache Scenario 1: Summarize flow tab switching', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    usePlaygroundStore.getState().resetSession();
  });

  it('does not refetch overview when remounting after data loaded', async () => {
    const client = createTestClient();
    const wrapper = createWrapper(client);

    const { unmount } = renderHook(() => useOverviewBundleQuery(), { wrapper });
    await waitFor(() => expect(apiService.getHealth).toHaveBeenCalledTimes(1));

    unmount();
    renderHook(() => useOverviewBundleQuery(), { wrapper });

    await waitFor(() => expect(apiService.getHealth).toHaveBeenCalledTimes(1));
    expect(apiService.getMetrics).toHaveBeenCalledTimes(1);
    expect(apiService.getAnalyticsDashboard).toHaveBeenCalledTimes(1);
  });

  it('restores playground summarize result from session store without API', () => {
    usePlaygroundStore.getState().setResult({
      results: [{ key: 'vit5', algorithm: 'ViT5' }],
      ranking: [{ key: 'vit5', rank: 1 }],
    });

    const restored = usePlaygroundStore.getState().result;
    expect(restored?.results).toHaveLength(1);
    expect(restored?.results?.[0]?.algorithm).toBe('ViT5');
  });
});

describe('Cache Scenario 2: Analytics dashboard tab switching', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('serves analytics from cache when returning to Analytics page', async () => {
    const client = createTestClient();
    const wrapper = createWrapper(client);

    const first = renderHook(() => useAnalyticsDashboardQuery('30d', 20), { wrapper });
    await waitFor(() => expect(first.result.current.isSuccess).toBe(true));
    expect(apiService.getAnalyticsDashboard).toHaveBeenCalledTimes(1);

    first.unmount();
    renderHook(() => useAnalyticsDashboardQuery('30d', 20), { wrapper });

    expect(apiService.getAnalyticsDashboard).toHaveBeenCalledTimes(1);
    expect(client.getQueryData(queryKeys.analyticsDashboard('30d', 20))).toBeTruthy();
  });

  it('fetches once per time range then caches each range separately', async () => {
    const client = createTestClient();
    const wrapper = createWrapper(client);

    const hook = renderHook(
      ({ range }) => useAnalyticsDashboardQuery(range, 20),
      { wrapper, initialProps: { range: '30d' } },
    );

    await waitFor(() => expect(hook.result.current.isSuccess).toBe(true));
    hook.rerender({ range: '7d' });
    await waitFor(() => expect(apiService.getAnalyticsDashboard).toHaveBeenCalledTimes(2));

    hook.rerender({ range: '30d' });
    expect(apiService.getAnalyticsDashboard).toHaveBeenCalledTimes(2);
  });
});

describe('Cache Scenario 3: New upload clears old cache', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    usePlaygroundStore.getState().resetSession();
  });

  it('clears file extract cache and playground upload state on invalidation', () => {
    const client = createTestClient();
    const fp = 'doc.pdf:1000:123';
    client.setQueryData(queryKeys.fileExtract(fp), { text: 'old content' });
    usePlaygroundStore.getState().setFileMetas([{
      name: 'doc.pdf', size: 1000, lastModified: 123, type: 'application/pdf',
    }]);
    usePlaygroundStore.getState().setLastExtractFingerprint(fp);

    invalidateFileExtractCache(client);

    expect(client.getQueryData(queryKeys.fileExtract(fp))).toBeUndefined();
    expect(usePlaygroundStore.getState().fileMetas).toHaveLength(0);
    expect(usePlaygroundStore.getState().lastExtractFingerprint).toBeNull();
  });

  it('refresh clears all react-query and zustand session data', async () => {
    const client = createTestClient();
    client.setQueryData(queryKeys.overview, { health: { status: 'ok' } });
    usePlaygroundStore.getState().setResult({ results: [] });

    await invalidateAllCaches(client);

    expect(client.getQueryData(queryKeys.overview)).toBeUndefined();
    expect(usePlaygroundStore.getState().result).toBeNull();
  });
});
