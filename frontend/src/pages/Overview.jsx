import React, { memo, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import {
  Bot, Cpu, Activity, Clock, CheckCircle2,
  Sparkles, MessageSquare, GitCompareArrows, ArrowRight,
  TrendingUp, Server, FileText, BarChart3, AlertCircle
} from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { useApp } from '../context/AppContext';
import { useOverviewBundleQuery } from '../hooks/useApiQueries';
import { useCacheHitLogger } from '../hooks/useCacheHitLogger';
import { queryKeys } from '../lib/queryKeys';
import { getChartTheme, dateLocale } from '../theme/chartTheme';
import GpuMonitor from '../components/GpuMonitor';

const Skeleton = memo(({ className = '' }) => (
  <div className={`ui-skeleton ${className}`} />
));

const StatSkeleton = memo(() => (
  <div className="ui-card p-5 bg-[var(--bg-elevated)] border border-[var(--border)]">
    <Skeleton className="h-3 w-20 mb-3" />
    <Skeleton className="h-8 w-16 mb-2" />
    <Skeleton className="h-3 w-28" />
  </div>
));

const StatCard = memo(({ title, value, subtext, icon: Icon, color, delay = 0 }) => (
  <motion.div
    initial={{ opacity: 0, y: 8 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay }}
    whileHover={{ y: -2, scale: 1.01 }}
    className="ui-card p-5 relative overflow-hidden bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm hover:shadow-md transition-all duration-200"
  >
    <div className="flex justify-between items-start mb-3">
      <p className="text-[11px] font-bold uppercase tracking-wider text-[var(--text-muted)]">{title}</p>
      <div
        className="w-9 h-9 rounded-lg flex items-center justify-center"
        style={{ backgroundColor: `${color}10`, color }}
      >
        <Icon className="w-[18px] h-[18px]" />
      </div>
    </div>
    <div className="flex items-baseline gap-2 mb-1">
      <h3 className="text-2xl font-extrabold text-[var(--text-primary)] tracking-tight">{value}</h3>
    </div>
    {subtext && <p className="text-[11px] text-[var(--text-faint)] font-medium">{subtext}</p>}
  </motion.div>
));

const QuickAction = memo(({ icon: Icon, label, description, to, color }) => (
  <Link to={to} className="block">
    <motion.div
      whileHover={{ y: -3, borderColor: 'var(--accent)' }}
      whileTap={{ scale: 0.99 }}
      className="ui-card-interactive p-4 flex items-center gap-3.5 group bg-[var(--bg-elevated)] border border-[var(--border)] transition-all duration-200 shadow-sm"
    >
      <div
        className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0 shadow-sm"
        style={{ backgroundColor: `${color}10`, color }}
      >
        <Icon className="w-5 h-5" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-bold text-[var(--text-primary)] group-hover:text-[var(--accent)] transition-colors leading-tight">{label}</p>
        <p className="text-[11px] text-[var(--text-muted)] mt-1 font-medium truncate">{description}</p>
      </div>
      <ArrowRight className="w-4 h-4 text-[var(--text-faint)] group-hover:translate-x-1 group-hover:text-[var(--accent)] transition-all" />
    </motion.div>
  </Link>
));

const ACTIVITY_ICONS = {
  compare: GitCompareArrows,
  upload: FileText,
  chat: MessageSquare,
};

const Overview = () => {
  const { t, locale, isDark } = useApp();
  const { data, isLoading, isFetching, error, refetch } = useOverviewBundleQuery();
  useCacheHitLogger('overview bundle', data, isFetching);
  const queryClient = useQueryClient();

  React.useEffect(() => {
    if (!data) return;
    if (data.gpu) queryClient.setQueryData(queryKeys.systemGpu, data.gpu);
    if (data.node) queryClient.setQueryData(queryKeys.systemNode, data.node);
    if (data.models) queryClient.setQueryData(queryKeys.systemModels, data.models);
  }, [data, queryClient]);

  React.useEffect(() => {
    const prefetch = async () => {
      try {
        const { getResearchLeaderboard, getResearchHybridStudy, getResearchReport } = await import('../services/apiService');
        const { queryKeys } = await import('../lib/queryKeys');
        const { BENCHMARK_SAMPLE_SIZE } = await import('../lib/benchmarkConfig');
        queryClient.prefetchQuery({
          queryKey: queryKeys.researchLeaderboard('All', BENCHMARK_SAMPLE_SIZE),
          queryFn: () => getResearchLeaderboard(BENCHMARK_SAMPLE_SIZE),
          staleTime: 5 * 60 * 1000,
        });
        queryClient.prefetchQuery({
          queryKey: queryKeys.researchHybridStudy('vie', BENCHMARK_SAMPLE_SIZE),
          queryFn: () => getResearchHybridStudy('vie', BENCHMARK_SAMPLE_SIZE),
          staleTime: 5 * 60 * 1000,
        });
        queryClient.prefetchQuery({
          queryKey: queryKeys.researchReport('vie', BENCHMARK_SAMPLE_SIZE),
          queryFn: () => getResearchReport('vie', BENCHMARK_SAMPLE_SIZE),
          staleTime: 5 * 60 * 1000,
        });
      } catch (err) { console.warn('Failed to prefetch background queries:', err); }
    };
    const timer = setTimeout(prefetch, 800);
    return () => clearTimeout(timer);
  }, [queryClient]);

  const { health, metrics, dashboard, gpu, node, models } = data || {};
  const loading = isLoading && !data;
  const chart = getChartTheme(isDark);
  const dLocale = dateLocale(locale);
  const dashMetrics = dashboard?.metrics || {};
  const overview = dashboard?.overview || {};
  const timeseries = dashboard?.visualization?.timeseries || [];
  const recentActivity = dashboard?.recent_activity || dashboard?.recent_runs || [];

  const documentCount = overview.document_count ?? 0;
  const datasetDocs = overview.dataset_total_documents;
  const datasetVocab = overview.dataset_vocab_size;
  const sessionCount = overview.chat_session_count ?? dashMetrics.total_runs ?? 0;
  const modelCount = models?.loaded_count ?? (metrics?.models_preloaded ? Object.keys(metrics.model_load_times || {}).length : 0);
  const algorithmOutputs = dashMetrics.total_algorithm_outputs ?? overview.algorithm_output_count ?? 0;
  const avgLatency = dashMetrics.avg_processing_time_seconds ?? overview.avg_processing_time_seconds ?? 0;
  const avgRougeL = dashMetrics.avg_rouge?.rougeL ?? overview.avg_rouge_l ?? 0;

  const hour = new Date().getHours();
  const greeting = hour < 12 ? t('morningGreeting') : hour < 18 ? t('afternoonGreeting') : t('eveningGreeting');

  return (
    <div className="space-y-6 pb-12">
      {error && (
        <div className="flex items-center justify-between gap-3 text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/40 rounded-xl px-4 py-3">
          <div className="flex items-center gap-2">
            <AlertCircle size={16} />
            <span>{error.message || 'Không tải được dữ liệu dashboard'}</span>
          </div>
          <button type="button" onClick={() => refetch()} className="text-xs font-bold underline cursor-pointer">
            Thử lại
          </button>
        </div>
      )}

      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-2xl border border-[var(--border)] bg-gradient-to-br from-sky-500/5 via-sky-600/5 to-transparent p-6 md:p-8"
      >
        <div className="max-w-2xl relative z-10 space-y-3">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider bg-sky-100 dark:bg-sky-950/50 text-sky-700 dark:text-sky-400 border border-sky-200/50 dark:border-sky-800/40">
            <Sparkles size={11} className="animate-pulse" />
            AI Research Hub
          </span>
          <h1 className="text-3xl font-extrabold text-[var(--text-primary)] tracking-tight leading-tight md:text-4xl">
            {loading ? <Skeleton className="h-10 w-64" /> : `${greeting}!`}
          </h1>
          <p className="text-sm text-[var(--text-secondary)] font-medium leading-relaxed">
            Xây dựng và so sánh phương pháp tóm tắt lai kết hợp giữa tóm tắt trích rút và mô hình Transformer cho bài toán tóm tắt văn bản tiếng việt.
          </p>
          <div className="flex flex-wrap items-center gap-2 pt-2 text-xs font-semibold">
            {loading ? (
              <Skeleton className="h-4 w-40" />
            ) : health?.status === 'ok' ? (
              <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400">
                <CheckCircle2 size={15} />
                <span>{t('overviewSubtitleOk')}</span>
              </div>
            ) : (
              <span className="text-red-500">{t('overviewSubtitleError')}</span>
            )}
            <span className="text-[var(--text-faint)] hidden sm:inline">•</span>
            <span className="text-[var(--text-muted)] font-medium">
              Node: {node?.node_id || 'local'}
            </span>
          </div>
        </div>
        <div className="absolute right-0 bottom-0 top-0 w-1/3 hidden lg:flex items-center justify-center opacity-10">
          <Bot size={280} className="text-sky-600" />
        </div>
      </motion.div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {loading ? (
          Array.from({ length: 6 }).map((_, i) => <StatSkeleton key={i} />)
        ) : (
          <>
            <StatCard
              title="Tổng Tài Liệu"
              value={documentCount}
              subtext={datasetDocs != null ? `VietNews: ${datasetDocs.toLocaleString()} mẫu` : 'RAG + document index'}
              icon={FileText}
              color="#0284c7"
              delay={0}
            />
            <StatCard
              title="Phiên Phân Tích"
              value={dashMetrics.total_runs ?? 0}
              subtext={`${overview.chat_session_count ?? 0} chat · ${algorithmOutputs} thuật toán`}
              icon={Activity}
              color="#10b981"
              delay={0.04}
            />
            <StatCard
              title="Mô Hình Đã Nạp"
              value={modelCount}
              subtext={metrics?.gpu_name ? `VRAM: ${metrics.gpu_name}` : 'CPU Fallback'}
              icon={Cpu}
              color="#6366f1"
              delay={0.08}
            />
            <StatCard
              title="ROUGE-L TB"
              value={avgRougeL.toFixed(3)}
              subtext="Từ benchmark đã lưu"
              icon={Bot}
              color="#3b82f6"
              delay={0.12}
            />
            <StatCard
              title="Độ Trễ TB"
              value={avgLatency > 0 ? `${avgLatency.toFixed(2)}s` : '—'}
              subtext="Thời gian suy luận trung bình"
              icon={Clock}
              color="#fb7185"
              delay={0.16}
            />
            <StatCard
              title="Lượt Gọi AI"
              value={algorithmOutputs}
              subtext="Tổng output thuật toán"
              icon={Sparkles}
              color="#f59e0b"
              delay={0.2}
            />
          </>
        )}
      </div>

      <div className="space-y-3">
        <h2 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)]">Thao tác nhanh Workspace</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
          <QuickAction icon={Sparkles} label="Tóm tắt mới" description="So kè hiệu năng các thuật toán NLP" to="/summarize" color="#6366f1" />
          <QuickAction icon={MessageSquare} label="Chat tài liệu RAG" description="Trò chuyện ngữ nghĩa với file PDF" to="/chat" color="#0284c7" />
          <QuickAction icon={GitCompareArrows} label="So sánh mô hình" description="Thống kê BLEU, ROUGE, Latency" to="/compare" color="#10b981" />
          <QuickAction icon={BarChart3} label="Benchmark" description="Leaderboard 15 thuật toán" to="/benchmark" color="#a855f7" />
          <QuickAction icon={BarChart3} label="Dataset Analytics" description={datasetVocab ? `${datasetVocab.toLocaleString()} từ vựng VietNews` : 'Thống kê VietNews thực'} to="/dataset-analytics" color="#0ea5e9" />
          <QuickAction icon={TrendingUp} label="Xem báo cáo" description="Đánh giá chi tiết hiệu suất hệ thống" to="/analytics" color="#f59e0b" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="ui-card p-5 lg:col-span-1 flex flex-col justify-between bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)] mb-4 flex items-center gap-2">
              <Server size={14} className="text-sky-500" />
              Giám Sát Hạ Tầng AI
            </h3>
            <GpuMonitor gpu={gpu} node={node} models={models} loading={loading} />
          </div>
        </div>

        <div className="ui-card p-5 lg:col-span-2 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)]">{t('chartRunsTitle')}</h3>
              <p className="text-[11px] text-[var(--text-muted)] font-medium mt-0.5">{t('chartRunsSub')}</p>
            </div>
            <span className="text-xs font-bold text-sky-600 dark:text-sky-400 bg-sky-50 dark:bg-sky-950/30 px-2.5 py-1 rounded-lg border border-sky-200/30">
              30 ngày gần nhất
            </span>
          </div>
          <div className="h-56 w-full">
            {loading ? (
              <Skeleton className="w-full h-full rounded-xl" />
            ) : timeseries.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={timeseries} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={chart.accent} stopOpacity={0.2} />
                      <stop offset="95%" stopColor={chart.accent} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={chart.grid} />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 10, fill: chart.axis, fontWeight: 500 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis tick={{ fontSize: 10, fill: chart.axis, fontWeight: 500 }} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={chart.tooltipStyle} />
                  <Area type="monotone" dataKey="count" stroke={chart.accent} strokeWidth={2.5} fillOpacity={1} fill="url(#colorCount)" name={t('chartRunsName')} />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full">
                <p className="text-sm text-[var(--text-faint)] font-medium">{t('emptyChart')}</p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="ui-card p-5 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm">
        <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)] mb-5">{t('recentActivity')}</h3>
        <div className="relative pl-6 border-l border-[var(--border)] ml-3 space-y-6">
          {loading ? (
            <>
              <Skeleton className="h-12 w-full rounded-lg" />
              <Skeleton className="h-12 w-full rounded-lg" />
              <Skeleton className="h-12 w-full rounded-lg" />
            </>
          ) : recentActivity.length === 0 ? (
            <p className="text-sm text-[var(--text-faint)] font-medium pl-2">{t('emptyRecent')}</p>
          ) : (
            recentActivity.map((item, i) => {
              const Icon = ACTIVITY_ICONS[item.type] || Activity;
              const title = item.title || item.best_algorithm || t('runCompareLabel');
              const detail = item.detail || item.text_preview || '';
              return (
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  key={item.id || item.result_id || `act-${i}`}
                  className="relative group"
                >
                  <span className="absolute -left-[31px] top-1.5 w-2.5 h-2.5 rounded-full border-2 border-[var(--bg-elevated)] bg-sky-500 group-hover:scale-125 transition-transform" />
                  <Link to={item.link || '/analytics'} className="block">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 bg-[var(--bg-muted)]/30 hover:bg-[var(--bg-muted)]/60 border border-[var(--border)]/40 rounded-xl p-3 transition-colors duration-150">
                      <div className="min-w-0 space-y-1">
                        <div className="flex flex-wrap items-center gap-1.5 text-xs text-[var(--text-secondary)]">
                          <Icon size={12} className="text-sky-500" />
                          <span className="font-bold text-[var(--text-primary)]">{title}</span>
                          {item.meta?.algorithm_count != null && (
                            <>
                              <span className="text-[var(--text-faint)]">•</span>
                              <span>{t('runAlgorithms', { count: item.meta.algorithm_count })}</span>
                            </>
                          )}
                        </div>
                        {detail && (
                          <p className="text-[11px] text-[var(--text-muted)] leading-relaxed line-clamp-1 italic font-medium">
                            &quot;{detail}&quot;
                          </p>
                        )}
                      </div>
                      <div className="text-[10px] font-bold text-[var(--text-faint)] shrink-0 sm:text-right">
                        {item.created_at ? new Date(item.created_at).toLocaleString(dLocale) : ''}
                      </div>
                    </div>
                  </Link>
                </motion.div>
              );
            })
          )}
          {!loading && health?.status === 'ok' && (
            <div className="relative group">
              <span className="absolute -left-[31px] top-1.5 w-2.5 h-2.5 rounded-full border-2 border-[var(--bg-elevated)] bg-emerald-500" />
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 bg-[var(--bg-muted)]/30 border border-[var(--border)]/40 rounded-xl p-3">
                <div>
                  <p className="text-xs font-bold text-[var(--text-primary)] flex items-center gap-1.5">
                    <CheckCircle2 size={13} className="text-emerald-500" />
                    {t('apiReady')}
                  </p>
                  <p className="text-[11px] text-[var(--text-muted)] font-medium mt-0.5">
                    Các mô hình đã được {metrics?.models_preloaded ? t('apiPreload') : t('apiLazy')} và sẵn sàng xử lý.
                  </p>
                </div>
                <div className="text-[10px] font-bold text-[var(--text-faint)] shrink-0 sm:text-right">
                  {t('justChecked')}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default memo(Overview);
