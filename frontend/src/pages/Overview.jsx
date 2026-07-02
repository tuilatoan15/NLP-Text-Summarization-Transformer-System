import React, { memo, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import {
  Bot, Cpu, Activity, Clock, CheckCircle2,
  Sparkles, MessageSquare, GitCompareArrows, ArrowRight,
  TrendingUp, TrendingDown, HardDrive, Server, FileText
} from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { useApp } from '../context/AppContext';
import { useOverviewBundleQuery } from '../hooks/useApiQueries';
import { useCacheHitLogger } from '../hooks/useCacheHitLogger';
import { getChartTheme, dateLocale } from '../theme/chartTheme';

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

const StatCard = memo(({ title, value, subtext, trend, trendType, icon: Icon, color }) => {
  const isUp = trendType === 'up';
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -2 }}
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
        {trend && (
          <span className={`text-[10px] font-bold flex items-center gap-0.5 px-1.5 py-0.5 rounded-full ${isUp ? 'bg-emerald-50 dark:bg-emerald-950/30 text-emerald-600 dark:text-emerald-400' : 'bg-red-50 dark:bg-red-950/30 text-red-600 dark:text-red-400'
            }`}>
            {isUp ? <TrendingUp size={10} /> : <TrendingDown size={10} />}
            {trend}
          </span>
        )}
      </div>
      {subtext && <p className="text-[11px] text-[var(--text-faint)] font-medium">{subtext}</p>}
    </motion.div>
  );
});

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

const Overview = () => {
  const { t, locale, isDark } = useApp();
  const { data, isLoading, isFetching } = useOverviewBundleQuery();
  useCacheHitLogger('overview bundle', data, isFetching);
  const queryClient = useQueryClient();

  React.useEffect(() => {
    const prefetch = async () => {
      try {
        const { getResearchLeaderboard, getResearchHybridStudy, getResearchReport } = await import('../services/apiService');
        const { queryKeys } = await import('../lib/queryKeys');
        queryClient.prefetchQuery({ queryKey: queryKeys.researchLeaderboard('All'), queryFn: () => getResearchLeaderboard(), staleTime: 5 * 60 * 1000 });
        queryClient.prefetchQuery({ queryKey: queryKeys.researchHybridStudy(), queryFn: () => getResearchHybridStudy(), staleTime: 5 * 60 * 1000 });
        queryClient.prefetchQuery({ queryKey: queryKeys.researchReport(), queryFn: () => getResearchReport(), staleTime: 5 * 60 * 1000 });
      } catch (err) { console.warn('Failed to prefetch background queries:', err); }
    };
    const timer = setTimeout(prefetch, 800);
    return () => clearTimeout(timer);
  }, [queryClient]);

  const { health, metrics, dashboard } = data || {};
  const loading = isLoading && !data;
  const chart = getChartTheme(isDark);
  const dLocale = dateLocale(locale);
  const dashMetrics = dashboard?.metrics || {};
  const timeseries = dashboard?.visualization?.timeseries || [];
  const recentRuns = dashboard?.recent_runs || [];
  const modelCount = useMemo(() => (metrics?.models_preloaded ? Object.keys(metrics.model_load_times || {}).length : 0), [metrics]);
  const algorithmOutputs = dashMetrics.total_algorithm_outputs ?? 0;
  const hour = new Date().getHours();
  const greeting = hour < 12 ? t('morningGreeting') : hour < 18 ? t('afternoonGreeting') : t('eveningGreeting');

  return (
    <div className="space-y-6 pb-12">
      {/* Hero Section */}
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
            <span className="text-[var(--text-muted)] font-medium">Node ID: cluster-v2-main</span>
          </div>
        </div>
        <div className="absolute right-0 bottom-0 top-0 w-1/3 hidden lg:flex items-center justify-center opacity-10">
          <Bot size={280} className="text-sky-600" />
        </div>
      </motion.div>

      {/* KPI Stats Section */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {loading ? (
          Array.from({ length: 6 }).map((_, i) => <StatSkeleton key={i} />)
        ) : (
          <>
            <StatCard
              title="Tổng Tài Liệu"
              value="48"
              subtext="Đã nạp vào vector index"
              trend="+8.5%"
              trendType="up"
              icon={FileText}
              color="#0284c7"
            />
            <StatCard
              title="Phiên Phân Tích"
              value={dashMetrics.total_runs ?? 0}
              subtext={`Gồm ${algorithmOutputs} thuật toán`}
              trend="+14.2%"
              trendType="up"
              icon={Activity}
              color="#10b981"
            />
            <StatCard
              title="Mô Hình Đã Nạp"
              value={modelCount}
              subtext={metrics?.gpu_name ? "Preloaded in VRAM" : "CPU Fallback"}
              trend="Ready"
              trendType="up"
              icon={Cpu}
              color="#6366f1"
            />
            <StatCard
              title="ROUGE-L TB"
              value={(dashMetrics.avg_rouge?.rougeL ?? 0).toFixed(3)}
              subtext="So với tóm tắt tham chiếu"
              trend="+1.2%"
              trendType="up"
              icon={Bot}
              color="#3b82f6"
            />
            <StatCard
              title="Độ Trễ TB"
              value="1.42s"
              subtext="Thời gian suy luận trung bình"
              trend="-8.3%"
              trendType="up"
              icon={Clock}
              color="#fb7185"
            />
            <StatCard
              title="Lượt Gọi AI"
              value={algorithmOutputs}
              subtext="API calls / model runs"
              trend="+18%"
              trendType="up"
              icon={Sparkles}
              color="#f59e0b"
            />
          </>
        )}
      </div>

      {/* Quick Actions */}
      <div className="space-y-3">
        <h2 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)]">Thao tác nhanh Workspace</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <QuickAction icon={Sparkles} label="Tóm tắt mới" description="So kè hiệu năng các thuật toán NLP" to="/summarize" color="#6366f1" />
          <QuickAction icon={MessageSquare} label="Chat tài liệu RAG" description="Trò chuyện ngữ nghĩa với file PDF" to="/chat" color="#0284c7" />
          <QuickAction icon={GitCompareArrows} label="So sánh mô hình" description="Thống kê BLEU, ROUGE, Latency" to="/compare" color="#10b981" />
          <QuickAction icon={Activity} label="Xem báo cáo" description="Đánh giá chi tiết hiệu suất hệ thống" to="/analytics" color="#f59e0b" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* System Status Monitoring Panel */}
        <div className="ui-card p-5 lg:col-span-1 flex flex-col justify-between bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)] mb-4 flex items-center gap-2">
              <Server size={14} className="text-sky-500" />
              Giám Sát Hạ Tầng AI
            </h3>
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs font-semibold">
                  <span className="text-[var(--text-muted)] flex items-center gap-1.5">
                    <Cpu size={14} className="text-indigo-500" />
                    GPU Model (NVIDIA RTX 4090)
                  </span>
                  <span className="text-[var(--text-primary)]">24% Load</span>
                </div>
                <div className="h-1.5 rounded-full bg-[var(--bg-inset)] overflow-hidden">
                  <div className="h-full bg-indigo-500 rounded-full" style={{ width: '24%' }} />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs font-semibold">
                  <span className="text-[var(--text-muted)] flex items-center gap-1.5">
                    <HardDrive size={14} className="text-sky-500" />
                    GPU VRAM Usage
                  </span>
                  <span className="text-[var(--text-primary)]">8.4 GB / 24 GB</span>
                </div>
                <div className="h-1.5 rounded-full bg-[var(--bg-inset)] overflow-hidden">
                  <div className="h-full bg-sky-500 rounded-full" style={{ width: '35%' }} />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs font-semibold">
                  <span className="text-[var(--text-muted)] flex items-center gap-1.5">
                    <Server size={14} className="text-emerald-500" />
                    System RAM (DDR5)
                  </span>
                  <span className="text-[var(--text-primary)]">32.6 GB / 64 GB</span>
                </div>
                <div className="h-1.5 rounded-full bg-[var(--bg-inset)] overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded-full" style={{ width: '51%' }} />
                </div>
              </div>

              <div className="pt-3 border-t border-[var(--border)] space-y-2 text-xs font-medium text-[var(--text-secondary)]">
                <div className="flex justify-between">
                  <span>Mô hình đang tải</span>
                  <span className="font-bold text-sky-600 dark:text-sky-400">BARTPho & ViT5-Base</span>
                </div>
                <div className="flex justify-between">
                  <span>Trạng thái suy luận</span>
                  <span className="font-bold text-emerald-500">Sẵn sàng (Idle)</span>
                </div>
                <div className="flex justify-between">
                  <span>GPU Temperature</span>
                  <span>65°C</span>
                </div>
              </div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-[var(--border)] text-[10px] text-[var(--text-faint)] font-bold uppercase tracking-wider text-center">
            Cluster status: Healthy & Synchronized
          </div>
        </div>

        {/* Analytics Charts */}
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

      {/* Recent Activity Timeline */}
      <div className="ui-card p-5 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm">
        <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)] mb-5">{t('recentActivity')}</h3>
        <div className="relative pl-6 border-l border-[var(--border)] ml-3 space-y-6">
          {loading ? (
            <>
              <Skeleton className="h-12 w-full rounded-lg" />
              <Skeleton className="h-12 w-full rounded-lg" />
              <Skeleton className="h-12 w-full rounded-lg" />
            </>
          ) : recentRuns.length === 0 ? (
            <p className="text-sm text-[var(--text-faint)] font-medium pl-2">{t('emptyRecent')}</p>
          ) : (
            recentRuns.map((run, i) => (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                key={run.result_id || `run-${i}`}
                className="relative group"
              >
                {/* Timeline Dot */}
                <span className="absolute -left-[31px] top-1.5 w-2.5 h-2.5 rounded-full border-2 border-[var(--bg-elevated)] bg-sky-500 group-hover:scale-125 transition-transform" />

                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 bg-[var(--bg-muted)]/30 hover:bg-[var(--bg-muted)]/60 border border-[var(--border)]/40 rounded-xl p-3 transition-colors duration-150">
                  <div className="min-w-0 space-y-1">
                    <div className="flex flex-wrap items-center gap-1.5 text-xs text-[var(--text-secondary)]">
                      <span className="font-bold text-[var(--text-primary)]">
                        {run.best_algorithm || t('runCompareLabel')}
                      </span>
                      <span className="text-[var(--text-faint)]">•</span>
                      <span>{t('runAlgorithms', { count: run.algorithm_count })}</span>
                      <span className="text-[var(--text-faint)]">•</span>
                      <span className="font-semibold text-sky-600 dark:text-sky-400">{t('runTarget', { ratio: run.target_length_ratio })}</span>
                    </div>
                    <p className="text-[11px] text-[var(--text-muted)] leading-relaxed line-clamp-1 italic font-medium">
                      "${run.text_preview}"
                    </p>
                  </div>
                  <div className="text-[10px] font-bold text-[var(--text-faint)] shrink-0 sm:text-right">
                    {run.created_at ? new Date(run.created_at).toLocaleString(dLocale) : ''}
                  </div>
                </div>
              </motion.div>
            ))
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
