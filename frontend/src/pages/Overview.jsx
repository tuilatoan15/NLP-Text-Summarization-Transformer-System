import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import {
  Bot, Cpu, Activity, Clock, Loader2, CheckCircle2,
  Sparkles, MessageSquare, GitCompareArrows, ArrowRight,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { getAnalyticsDashboard, getHealth, getMetrics } from '../services/apiService';
import { getChartTheme, dateLocale } from '../theme/chartTheme';

/* ─── Skeleton ─── */
const Skeleton = ({ className = '' }) => (
  <div className={`ui-skeleton ${className}`} />
);

const StatSkeleton = () => (
  <div className="ui-card p-5">
    <Skeleton className="h-3 w-20 mb-3" />
    <Skeleton className="h-8 w-16 mb-2" />
    <Skeleton className="h-3 w-28" />
  </div>
);

/* ─── Stat Card ─── */
const StatCard = ({ title, value, subtext, icon: Icon, color }) => (
  <motion.div
    initial={{ opacity: 0, y: 8 }}
    animate={{ opacity: 1, y: 0 }}
    className="ui-card p-5"
  >
    <div className="flex justify-between items-start mb-3">
      <p className="ui-overline">{title}</p>
      <div
        className="w-8 h-8 rounded-lg flex items-center justify-center"
        style={{ backgroundColor: `${color}14`, color }}
      >
        <Icon className="w-4 h-4" />
      </div>
    </div>
    <h3 className="text-2xl font-bold text-[var(--text-primary)] mb-1">{value}</h3>
    {subtext && <p className="text-xs text-[var(--text-faint)]">{subtext}</p>}
  </motion.div>
);

/* ─── Quick Action ─── */
const QuickAction = ({ icon: Icon, label, to, color }) => (
  <Link to={to}>
    <motion.div
      whileHover={{ y: -2 }}
      whileTap={{ scale: 0.98 }}
      className="ui-card-interactive p-4 flex items-center gap-3 group"
    >
      <div
        className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0"
        style={{ backgroundColor: `${color}14`, color }}
      >
        <Icon className="w-4.5 h-4.5" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-[var(--text-primary)]">{label}</p>
      </div>
      <ArrowRight className="w-4 h-4 text-[var(--text-faint)] group-hover:text-[var(--text-muted)] transition-colors" />
    </motion.div>
  </Link>
);

/* ─── Main ─── */
const Overview = () => {
  const { t, locale, isDark, overviewCache, setOverviewCache } = useApp();
  const [health, setHealth] = useState(() => overviewCache?.health || null);
  const [metrics, setMetrics] = useState(() => overviewCache?.metrics || null);
  const [dashboard, setDashboard] = useState(() => overviewCache?.dashboard || null);
  const [loading, setLoading] = useState(!overviewCache);
  const chart = getChartTheme(isDark);
  const dLocale = dateLocale(locale);

  useEffect(() => {
    if (overviewCache) {
      setLoading(false);
      return;
    }
    async function fetchData() {
      try {
        const [healthData, metricsData, dash] = await Promise.all([
          getHealth(),
          getMetrics(),
          getAnalyticsDashboard('30d', 10),
        ]);
        setHealth(healthData);
        setMetrics(metricsData);
        setDashboard(dash);
        setOverviewCache({ health: healthData, metrics: metricsData, dashboard: dash });
      } catch (err) {
        console.error('Failed to fetch overview data:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [overviewCache, setOverviewCache]);

  const dashMetrics = dashboard?.metrics || {};
  const timeseries = dashboard?.visualization?.timeseries || [];
  const recentRuns = dashboard?.recent_runs || [];
  const modelCount = metrics?.models_preloaded
    ? Object.keys(metrics.model_load_times || {}).length
    : 0;
  const algorithmOutputs = dashMetrics.total_algorithm_outputs ?? 0;

  // Greeting based on time
  const hour = new Date().getHours();
  const greeting = hour < 12 ? '☀️ Chào buổi sáng' : hour < 18 ? '🌤️ Chào buổi chiều' : '🌙 Chào buổi tối';

  return (
    <div className="space-y-6 pb-12">
      {/* Greeting */}
      <div>
        <h1 className="ui-heading-1 mb-1 flex items-center gap-2">
          {loading ? <Skeleton className="h-8 w-48" /> : greeting}
        </h1>
        <p className="ui-text flex items-center gap-1.5">
          {loading ? (
            <Skeleton className="h-4 w-56" />
          ) : health?.status === 'ok' ? (
            <>
              <CheckCircle2 className="w-4 h-4" style={{ color: 'var(--success)' }} />
              <span>{t('overviewSubtitleOk')}</span>
            </>
          ) : (
            t('overviewSubtitleError')
          )}
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {loading ? (
          <>
            <StatSkeleton />
            <StatSkeleton />
            <StatSkeleton />
            <StatSkeleton />
          </>
        ) : (
          <>
            <StatCard
              title={t('statRunsTitle')}
              value={dashMetrics.total_runs ?? 0}
              subtext={t('statRunsSub', { count: algorithmOutputs })}
              icon={Activity}
              color="#10b981"
            />
            <StatCard
              title={t('statModelsTitle')}
              value={modelCount}
              subtext={metrics?.gpu_name || 'CPU'}
              icon={Cpu}
              color="#6366f1"
            />
            <StatCard
              title={t('statRougeTitle')}
              value={(dashMetrics.avg_rouge?.rougeL ?? 0).toFixed(2)}
              subtext={t('statRougeSub')}
              icon={Bot}
              color="#3b82f6"
            />
            <StatCard
              title={t('statLengthTitle')}
              value={`${dashMetrics.avg_target_length_ratio ?? 0}%`}
              subtext={t('statLengthSub', { actual: dashMetrics.avg_actual_length_ratio ?? 0 })}
              icon={Clock}
              color="#f59e0b"
            />
          </>
        )}
      </div>

      {/* Quick Actions */}
      <div>
        <h2 className="ui-overline mb-3">Thao tác nhanh</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <QuickAction icon={Sparkles} label="Tóm tắt mới" to="/summarize" color="#6366f1" />
          <QuickAction icon={MessageSquare} label="Chat tài liệu" to="/chat" color="#3b82f6" />
          <QuickAction icon={GitCompareArrows} label="So sánh mô hình" to="/compare" color="#10b981" />
          <QuickAction icon={Activity} label="Xem báo cáo" to="/analytics" color="#f59e0b" />
        </div>
      </div>

      {/* Charts & Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Chart */}
        <div className="ui-card p-5">
          <div className="mb-4">
            <h3 className="ui-heading-3 mb-0.5">{t('chartRunsTitle')}</h3>
            <p className="text-xs text-[var(--text-faint)]">{t('chartRunsSub')}</p>
          </div>
          <div className="h-48 w-full">
            {loading ? (
              <Skeleton className="w-full h-full rounded-lg" />
            ) : timeseries.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={timeseries} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={chart.grid} />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 10, fill: chart.axis }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis tick={{ fontSize: 10, fill: chart.axis }} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={chart.tooltipStyle} />
                  <Bar dataKey="count" fill={chart.accent} radius={[4, 4, 0, 0]} name={t('chartRunsName')} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full">
                <p className="text-sm text-[var(--text-faint)]">{t('emptyChart')}</p>
              </div>
            )}
          </div>
        </div>

        {/* Recent Activity */}
        <div className="ui-card p-5 flex flex-col">
          <h3 className="ui-heading-3 mb-4">{t('recentActivity')}</h3>
          <div className="flex-1 space-y-3 overflow-y-auto max-h-64">
            {loading ? (
              <>
                <Skeleton className="h-12 w-full" />
                <Skeleton className="h-12 w-full" />
                <Skeleton className="h-12 w-full" />
              </>
            ) : recentRuns.length === 0 ? (
              <p className="text-sm text-[var(--text-faint)]">{t('emptyRecent')}</p>
            ) : (
              recentRuns.map(run => (
                <div key={run.result_id} className="flex items-start gap-3 group">
                  <div
                    className="w-6 h-6 rounded-full flex items-center justify-center mt-0.5 shrink-0"
                    style={{ backgroundColor: 'var(--success-muted)' }}
                  >
                    <CheckCircle2 className="w-3.5 h-3.5" style={{ color: 'var(--success)' }} />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm text-[var(--text-secondary)]">
                      <span className="font-semibold text-[var(--text-primary)]">
                        {run.best_algorithm || t('runCompareLabel')}
                      </span>
                      {' '}— {t('runAlgorithms', { count: run.algorithm_count })}
                      {' · '}{t('runTarget', { ratio: run.target_length_ratio })}
                    </p>
                    <p className="text-xs text-[var(--text-faint)] mt-0.5 line-clamp-1">{run.text_preview}</p>
                    <p className="text-[10px] text-[var(--text-faint)] mt-1">
                      {run.created_at ? new Date(run.created_at).toLocaleString(dLocale) : ''}
                    </p>
                  </div>
                </div>
              ))
            )}
            {!loading && health?.status === 'ok' && (
              <div className="flex items-start gap-3">
                <div
                  className="w-6 h-6 rounded-full flex items-center justify-center mt-0.5 shrink-0"
                  style={{ backgroundColor: 'var(--success-muted)' }}
                >
                  <CheckCircle2 className="w-3.5 h-3.5" style={{ color: 'var(--success)' }} />
                </div>
                <div>
                  <p className="text-sm text-[var(--text-secondary)]">
                    <span className="font-semibold text-[var(--text-primary)]">{t('apiReady')}</span>
                    {' '}— models{' '}
                    {metrics?.models_preloaded ? t('apiPreload') : t('apiLazy')}
                  </p>
                  <p className="text-[10px] text-[var(--text-faint)] mt-1">{t('justChecked')}</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Overview;
