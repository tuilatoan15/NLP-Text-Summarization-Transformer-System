import { useQuery, useQueryClient } from '@tanstack/react-query';
import { cacheLog } from '../lib/cacheLogger';
import { queryKeys } from '../lib/queryKeys';
import {
  getAnalyticsDashboard,
  getExplainability,
  getHealth,
  getLeaderboardByCategory,
  getMetrics,
  getResearchBenchmarkSamples,
  getResearchHybridStudy,
  getResearchLeaderboard,
  getResearchReport,
} from '../services/apiService';

async function fetchWithLog<T>(label: string, fn: () => Promise<T>): Promise<T> {
  cacheLog('API', label);
  return fn();
}

function useLoggedQuery<T>(
  key: readonly unknown[],
  label: string,
  queryFn: () => Promise<T>,
  enabled = true,
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
    const [health, metrics, dashboard] = await Promise.all([
      getHealth(),
      getMetrics(),
      getAnalyticsDashboard('30d', 10),
    ]);
    return { health, metrics, dashboard };
  });
}

export function useAnalyticsDashboardQuery(timeRange: string, limit = 20) {
  return useLoggedQuery(
    queryKeys.analyticsDashboard(timeRange, limit),
    `GET /analytics/dashboard?time_range=${timeRange}`,
    () => getAnalyticsDashboard(timeRange, limit),
  );
}

export function useResearchLeaderboardQuery(category = 'All', size = 1000, enabled = true) {
  return useLoggedQuery(
    queryKeys.researchLeaderboard(category, size),
    `GET /research/leaderboard${category !== 'All' ? `?category=${category}&` : '?'}size=${size}`,
    () => (category === 'All'
      ? getResearchLeaderboard(size)
      : getLeaderboardByCategory(category, size)),
    enabled,
  );
}

export function useResearchHybridStudyQuery(locale = 'vie', size = 1000, enabled = true) {
  return useLoggedQuery(
    queryKeys.researchHybridStudy(locale, size),
    `GET /research/hybrid-study?locale=${locale}&size=${size}`,
    () => getResearchHybridStudy(locale, size),
    enabled,
  );
}

export function useResearchReportQuery(locale = 'vie', size = 1000, enabled = true) {
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
  size = 1000,
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
