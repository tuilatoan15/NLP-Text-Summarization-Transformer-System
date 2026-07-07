import { useQuery, useQueryClient, type UseQueryOptions } from '@tanstack/react-query';
import { cacheLog } from '../lib/cacheLogger';
import { BENCHMARK_SAMPLE_SIZE } from '../lib/benchmarkConfig';
import { queryKeys } from '../lib/queryKeys';
import {
  getAnalyticsDashboard,
  getDatasetAnalytics,
  getDatasetAnalyticsProgress,
  getDatasetCharts,
  rebuildDatasetAnalytics,
  getExplainability,
  getHealth,
  getLeaderboardByCategory,
  getMetrics,
  getResearchBenchmarkSamples,
  getResearchHybridStudy,
  getResearchLeaderboard,
  getResearchReport,
  getSystemConfig,
  getSystemGpu,
  getSystemModels,
  getSystemNode,
  searchDashboard,
} from '../services/apiService';

async function fetchWithLog<T>(label: string, fn: () => Promise<T>): Promise<T> {
  cacheLog('API', label);
  return fn();
}

type LoggedQueryExtras<T> = Pick<
  UseQueryOptions<T>,
  'refetchInterval' | 'staleTime' | 'gcTime'
>;

function useLoggedQuery<T>(
  key: readonly unknown[],
  label: string,
  queryFn: () => Promise<T>,
  enabled = true,
  extras?: LoggedQueryExtras<T>,
) {
  const queryClient = useQueryClient();
  const cached = queryClient.getQueryData<T>(key);

  return useQuery({
    queryKey: key,
    queryFn: async () => {
      if (cached !== undefined) {
        cacheLog('HIT', label, 'in-memory');
      } else {
        cacheLog('MISS', label);
      }
      return fetchWithLog(label, queryFn);
    },
    enabled,
    placeholderData: cached as any,
    ...extras,
  });
}

export function useHealthQuery() {
  return useLoggedQuery(queryKeys.health, 'GET /health', getHealth);
}

export function useMetricsQuery() {
  return useLoggedQuery(queryKeys.metrics, 'GET /metrics', getMetrics);
}

export function useOverviewBundleQuery() {
  return useLoggedQuery(queryKeys.overview, 'overview bundle', async () => {
    const [health, metrics, dashboard, gpu, node, models] = await Promise.all([
      getHealth(),
      getMetrics(),
      getAnalyticsDashboard('30d', 10),
      getSystemGpu().catch(() => ({ available: false, status: 'unavailable' })),
      getSystemNode().catch(() => ({ status: 'offline' })),
      getSystemModels().catch(() => ({ summarizers: [], rag_models: [] })),
    ]);
    return { health, metrics, dashboard, gpu, node, models };
  });
}

const SIDEBAR_POLL_MS = 30_000;

export function useSystemGpuQuery(pollInterval = false as number | false) {
  const interval = pollInterval === false ? undefined : pollInterval;
  return useLoggedQuery(
    queryKeys.systemGpu,
    'GET /system/gpu',
    getSystemGpu,
    true,
    interval
      ? { refetchInterval: interval, staleTime: interval }
      : undefined,
  );
}

export function useSystemNodeQuery(pollInterval = false as number | false) {
  const interval = pollInterval === false ? undefined : pollInterval;
  return useLoggedQuery(
    queryKeys.systemNode,
    'GET /system/node',
    getSystemNode,
    true,
    interval
      ? { refetchInterval: interval, staleTime: interval }
      : undefined,
  );
}

/** GPU/node polling cho sidebar — 30s interval, tránh spam nvidia-smi. */
export function useSidebarSystemQueries() {
  return {
    gpu: useSystemGpuQuery(SIDEBAR_POLL_MS),
    node: useSystemNodeQuery(SIDEBAR_POLL_MS),
  };
}

export function useSystemModelsQuery() {
  return useLoggedQuery(queryKeys.systemModels, 'GET /system/models', getSystemModels);
}

export function useSystemConfigQuery() {
  return useLoggedQuery(queryKeys.systemConfig, 'GET /config', getSystemConfig);
}

export function useDashboardSearchQuery(query: string, enabled = true) {
  return useLoggedQuery(
    queryKeys.dashboardSearch(query),
    `GET /search?q=${query}`,
    () => searchDashboard(query),
    enabled && query.trim().length >= 2,
  );
}

export function useAnalyticsDashboardQuery(timeRange: string, limit = 20) {
  return useLoggedQuery(
    queryKeys.analyticsDashboard(timeRange, limit),
    `GET /analytics/dashboard?time_range=${timeRange}`,
    () => getAnalyticsDashboard(timeRange, limit),
  );
}

export function useDatasetAnalyticsQuery() {
  return useLoggedQuery(
    queryKeys.datasetAnalytics,
    'GET /analytics/dataset',
    getDatasetAnalytics,
  );
}

export function useDatasetChartsQuery() {
  return useLoggedQuery(
    queryKeys.datasetCharts,
    'GET /analytics/charts',
    getDatasetCharts,
  );
}

export function useDatasetProgressQuery(enabled = true) {
  return useQuery({
    queryKey: queryKeys.datasetProgress,
    queryFn: () => fetchWithLog('GET /analytics/dataset/progress', getDatasetAnalyticsProgress),
    enabled,
    refetchInterval: enabled ? 5000 : false,
    staleTime: 3000,
  });
}

export function useRebuildDatasetAnalytics() {
  const queryClient = useQueryClient();
  return async () => {
    cacheLog('API', 'POST /analytics/dataset/rebuild');
    const result = await rebuildDatasetAnalytics();
    await queryClient.invalidateQueries({ queryKey: queryKeys.datasetAnalytics });
    await queryClient.invalidateQueries({ queryKey: queryKeys.datasetCharts });
    await queryClient.invalidateQueries({ queryKey: queryKeys.datasetProgress });
    return result;
  };
}

export function useResearchLeaderboardQuery(category = 'All', size = BENCHMARK_SAMPLE_SIZE, enabled = true) {
  return useLoggedQuery(
    queryKeys.researchLeaderboard(category, size),
    `GET /research/leaderboard${category !== 'All' ? `?category=${category}&` : '?'}size=${size}`,
    () => (category === 'All'
      ? getResearchLeaderboard(size)
      : getLeaderboardByCategory(category, size)),
    enabled,
  );
}

export function useResearchHybridStudyQuery(locale = 'vie', size = BENCHMARK_SAMPLE_SIZE, enabled = true) {
  return useLoggedQuery(
    queryKeys.researchHybridStudy(locale, size),
    `GET /research/hybrid-study?locale=${locale}&size=${size}`,
    () => getResearchHybridStudy(locale, size),
    enabled,
  );
}

export function useResearchReportQuery(locale = 'vie', size = BENCHMARK_SAMPLE_SIZE, enabled = true) {
  return useLoggedQuery(
    queryKeys.researchReport(locale, size),
    `GET /research/report?locale=${locale}&size=${size}`,
    () => getResearchReport(locale, size),
    enabled,
  );
}

export function useResearchBenchmarkSamplesQuery(
  page: number,
  limit: number,
  category: string,
  search: string,
  size = BENCHMARK_SAMPLE_SIZE,
  enabled = true,
) {
  return useLoggedQuery(
    queryKeys.researchBenchmarkSamples(page, limit, category, search, size),
    `GET /research/benchmark/samples?page=${page}&size=${size}`,
    () => getResearchBenchmarkSamples(page, limit, category, search, size),
    enabled,
  );
}

export function useExplainabilityQuery(
  documentId: string | undefined,
  algorithm: string,
) {
  return useLoggedQuery(
    queryKeys.explainability(documentId ?? '', algorithm),
    `GET /documents/${documentId}/explainability`,
    () => getExplainability(documentId!, algorithm),
    Boolean(documentId),
  );
}
