import React, { memo, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import {
  Loader2, Database, Award, Target, Minimize2, FileText,
  Activity, Calendar, Zap, AlertCircle, Layers
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { useAnalyticsDashboardQuery } from '../hooks/useApiQueries';
import { useCacheHitLogger } from '../hooks/useCacheHitLogger';
import { getChartTheme, dateLocale } from '../theme/chartTheme';

const COLORS = ['#6366f1', '#0ea5e9', '#10b981', '#fbbf24', '#f43f5e', '#a855f7'];

const CustomTooltip = memo(({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="backdrop-blur-md bg-[color-mix(in_srgb,var(--bg-elevated)_95%,transparent)] border border-[var(--border)] p-3 rounded-xl shadow-lg text-xs space-y-1.5 min-w-[140px]">
        <p className="font-semibold text-[var(--text-primary)] border-b border-[var(--border-subtle)] pb-1 mb-1">{label}</p>
        {payload.map((pld, index) => (
          <div key={index} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: pld.fill || pld.stroke }} />
              <span className="text-[var(--text-muted)] font-medium">{pld.name}:</span>
            </div>
            <span className="font-bold text-[var(--text-primary)]">
              {typeof pld.value === 'number' ? (pld.value < 1 ? pld.value.toFixed(3) : pld.value) : pld.value}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
});

const StatCard = memo(({ title, value, subtext, icon: Icon, color, delay }) => (
  <motion.div
    initial={{ opacity: 0, y: 12 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.35, delay }}
    whileHover={{ y: -2 }}
    className="ui-card p-5 relative overflow-hidden bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm hover:shadow-md transition-all duration-200"
  >
    <div className="flex justify-between items-start mb-3">
      <p className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">{title}</p>
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

const rangeKey = (range) => {
  if (range === '7d') return 'range7d';
  if (range === '30d') return 'range30d';
  if (range === '90d') return 'range90d';
  return 'rangeAll';
};

const Analytics = () => {
  const { t, locale, isDark } = useApp();
  const [timeRange, setTimeRange] = useState('30d');
  const { data, isLoading, isFetching, error } = useAnalyticsDashboardQuery(timeRange, 20);
  useCacheHitLogger(`analytics ${timeRange}`, data, isFetching);

  const loading = isLoading && !data;
  const chart = getChartTheme(isDark);
  const dLocale = dateLocale(locale);

  const metrics = data?.metrics || {};
  const modelRows = data?.visualization?.model_performance || [];
  const timeseries = data?.visualization?.timeseries || [];
  const recentRuns = data?.recent_runs || [];

  const pieData = useMemo(
    () => (metrics.top_models || []).map(row => ({ name: row.model, value: row.count })),
    [metrics.top_models],
  );

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-[var(--text-muted)] space-y-4">
        <Loader2 className="animate-spin w-8 h-8 text-[var(--accent)]" />
        <span className="text-sm font-medium tracking-wide">{t('loadingAnalytics')}</span>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-12">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="ui-page-title mb-1">{t('analyticsTitle')}</h1>
          <p className="ui-page-subtitle">{t('analyticsSubtitle')}</p>
        </div>

        {/* Segmented Filter Control */}
        <div className="inline-flex p-0.5 bg-[var(--bg-muted)] border border-[var(--border)] rounded-xl shrink-0 self-start sm:self-center shadow-sm">
          {['7d', '30d', '90d', 'all'].map(range => (
            <button
              key={range}
              type="button"
              onClick={() => setTimeRange(range)}
              className={`relative px-4 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 cursor-pointer ${
                timeRange === range
                  ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)] shadow-sm border border-[var(--border)]/40 font-bold'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
              }`}
            >
              {t(rangeKey(range))}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-3 text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/40 rounded-xl px-4 py-3.5">
          <AlertCircle size={18} className="shrink-0" />
          <span className="font-medium">{error.message || t('analyticsLoadError')}</span>
        </div>
      )}

      {/* KPI Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title={t('statSavedRuns')}
          value={metrics.total_runs ?? 0}
          subtext={locale === 'eng' ? 'Active analysis sessions' : 'Số phiên phân tích được lưu'}
          icon={Database}
          color="#6366f1"
          delay={0.0}
        />
        <StatCard
          title={t('statAvgRouge')}
          value={(metrics.avg_rouge?.rougeL ?? 0).toFixed(3)}
          subtext={t('statAvgRougeSub')}
          icon={Award}
          color="#fbbf24"
          delay={0.05}
        />
        <StatCard
          title={t('statAvgBert')}
          value={(metrics.avg_bertscore_f1 ?? 0).toFixed(3)}
          subtext={locale === 'eng' ? 'Semantic correlation score' : 'Đánh giá tương đồng ngữ nghĩa'}
          icon={Target}
          color="#0ea5e9"
          delay={0.1}
        />
        <StatCard
          title={t('statAvgLength')}
          value={`${metrics.avg_target_length_ratio ?? 0}%`}
          subtext={t('statActualLength', { actual: metrics.avg_actual_length_ratio ?? 0 })}
          icon={Minimize2}
          color="#10b981"
          delay={0.15}
        />
      </div>

      {metrics.total_runs === 0 ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          className="rounded-2xl border border-dashed border-[var(--border)] bg-[var(--bg-elevated)] p-12 text-center text-[var(--text-muted)] shadow-sm max-w-2xl mx-auto my-8"
        >
          <div className="w-12 h-12 rounded-full bg-sky-50 dark:bg-sky-950/30 flex items-center justify-center mx-auto mb-4 text-sky-600 dark:text-sky-400">
            <Activity className="w-6 h-6" />
          </div>
          <p className="text-sm font-medium leading-relaxed">
            {t('analyticsEmpty')}{' '}
            <Link to="/playground" className="font-bold text-sky-600 dark:text-sky-400 hover:underline">
              {t('analyticsEmptyLink')}
            </Link>
            , {t('analyticsEmptySuffix')}{' '}
            <code className="text-xs font-mono bg-[var(--bg-inset)] px-1.5 py-0.5 rounded text-[var(--text-secondary)] border border-[var(--border)]/40">
              storage/results
            </code>
            .
          </p>
        </motion.div>
      ) : (
        <>
          {/* Main Visualizations Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Model Metrics Benchmarks */}
            <div className="ui-card p-5 lg:col-span-2 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm flex flex-col justify-between">
              <div>
                <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)] mb-4 flex items-center gap-1.5">
                  <Layers size={14} className="text-indigo-500" />
                  {t('chartModelMetrics')}
                </h3>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={modelRows} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={chart.grid} />
                      <XAxis dataKey="model" tick={{ fontSize: 10, fill: chart.axis, fontWeight: 500 }} tickLine={false} axisLine={false} />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: chart.axis, fontWeight: 500 }} tickLine={false} axisLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11, pt: 10, color: chart.axis }} />
                      <Bar dataKey="rougeL" fill="#6366f1" radius={[4, 4, 0, 0]} name={t('colRougeL')} />
                      <Bar dataKey="bertScore" fill="#0ea5e9" radius={[4, 4, 0, 0]} name={t('colBert')} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Algorithm Freq Share */}
            <div className="ui-card p-5 lg:col-span-1 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm flex flex-col justify-between">
              <div>
                <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)] mb-4 flex items-center gap-1.5">
                  <Activity size={14} className="text-emerald-500" />
                  {t('chartAlgoFreq')}
                </h3>
                {pieData.length > 0 ? (
                  <div className="h-72 w-full flex items-center justify-center">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={pieData}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={65}
                          outerRadius={88}
                          paddingAngle={4}
                          label={false}
                        >
                          {pieData.map((_, i) => (
                            <Cell key={i} fill={COLORS[i % COLORS.length]} stroke="var(--bg-elevated)" strokeWidth={2} />
                          ))}
                        </Pie>
                        <Tooltip content={<CustomTooltip />} />
                        <Legend
                          iconType="circle"
                          iconSize={8}
                          layout="vertical"
                          align="right"
                          verticalAlign="middle"
                          wrapperStyle={{ fontSize: 11, color: chart.axis, paddingLeft: 10 }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-72 flex items-center justify-center text-[var(--text-faint)] text-xs">
                    No algorithm data
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Daily Quality & Runs Volume Trend */}
          <div className="ui-card p-5 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm">
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)] mb-4 flex items-center gap-1.5">
              <Calendar size={14} className="text-sky-500" />
              {t('chartTrend')}
            </h3>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeseries} margin={{ top: 10, right: -10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={chart.grid} />
                  <XAxis dataKey="date" tick={{ fontSize: 10, fill: chart.axis, fontWeight: 500 }} tickLine={false} axisLine={false} />
                  <YAxis yAxisId="left" domain={[0, 1]} tick={{ fontSize: 10, fill: chart.axis, fontWeight: 500 }} tickLine={false} axisLine={false} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10, fill: chart.axis, fontWeight: 500 }} tickLine={false} axisLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11, pt: 10, color: chart.axis }} />
                  <Line yAxisId="left" type="monotone" dataKey="avgRougeL" stroke="#fbbf24" strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} name={t('colRougeL')} />
                  <Line yAxisId="left" type="monotone" dataKey="avgBertScore" stroke="#6366f1" strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} name={t('colBert')} />
                  <Line yAxisId="right" type="monotone" dataKey="count" stroke="#10b981" strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} name={t('chartRunCount')} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Model Metrics Table */}
          <div className="ui-card overflow-hidden bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm">
            <div className="px-6 py-4 border-b border-[var(--border)] bg-[var(--bg-muted)]/20 flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)]">{t('tableModelDetail')}</h3>
              <span className="text-[10px] font-bold text-sky-600 dark:text-sky-400 bg-sky-50 dark:bg-sky-950/40 px-2 py-0.5 border border-sky-200/20 rounded-md">
                Live Data
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="ui-table-head border-b border-[var(--border)] text-left">
                    <th className="px-6 py-3.5 text-xs font-bold tracking-wider">{t('colModel')}</th>
                    <th className="px-6 py-3.5 text-xs font-bold tracking-wider text-center">{t('colRuns')}</th>
                    <th className="px-6 py-3.5 text-xs font-bold tracking-wider">{t('colAvgRouge')}</th>
                    <th className="px-6 py-3.5 text-xs font-bold tracking-wider">{t('colAvgBert')}</th>
                    <th className="px-6 py-3.5 text-xs font-bold tracking-wider text-center">{t('colAvgLength')}</th>
                    <th className="px-6 py-3.5 text-xs font-bold tracking-wider text-center">{t('colAvgTime')}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-subtle)]">
                  {modelRows.map((row, index) => (
                    <tr key={row.model || index} className="ui-table-row hover:bg-[var(--bg-muted)]/40 transition-colors">
                      <td className="px-6 py-4 font-bold text-[var(--text-primary)] text-xs flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-sky-500" />
                        {row.model}
                      </td>
                      <td className="px-6 py-4 text-center text-xs font-semibold text-[var(--text-secondary)]">{row.count}</td>
                      <td className="px-6 py-4">
                        <div className="space-y-1">
                          <div className="flex justify-between items-center text-xs font-bold text-sky-600 dark:text-sky-400">
                            <span>{row.rougeL.toFixed(3)}</span>
                          </div>
                          <div className="h-1 w-20 rounded-full bg-[var(--bg-inset)] overflow-hidden">
                            <div className="h-full bg-sky-500 rounded-full" style={{ width: `${row.rougeL * 100}%` }} />
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="space-y-1">
                          <div className="flex justify-between items-center text-xs font-bold text-indigo-600 dark:text-indigo-400">
                            <span>{row.bertScore.toFixed(3)}</span>
                          </div>
                          <div className="h-1 w-20 rounded-full bg-[var(--bg-inset)] overflow-hidden">
                            <div className="h-full bg-indigo-500 rounded-full" style={{ width: `${row.bertScore * 100}%` }} />
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-center text-xs font-semibold text-[var(--text-secondary)]">{row.avgLengthRatio}%</td>
                      <td className="px-6 py-4 text-center text-xs font-bold text-emerald-600 dark:text-emerald-400">{row.avgTime.toFixed(2)}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Recent Runs Audit Timeline Logs */}
          <div className="ui-card bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm">
            <div className="px-6 py-4 border-b border-[var(--border)] bg-[var(--bg-muted)]/20 flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)]">{t('tableHistory')}</h3>
              <span className="text-[10px] font-bold text-[var(--text-muted)] flex items-center gap-1">
                <FileText size={12} />
                {recentRuns.length} entries
              </span>
            </div>
            <ul className="divide-y divide-[var(--border-subtle)]">
              {recentRuns.map((run, i) => (
                <motion.li
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: i * 0.03 }}
                  key={run.result_id || `run-${i}`}
                  className="px-6 py-4 hover:bg-[var(--bg-muted)]/30 transition-colors"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-sky-50 dark:bg-sky-950/30 text-sky-700 dark:text-sky-400 border border-sky-200/30">
                        {run.best_algorithm || '—'}
                      </span>
                      <span className="text-xs font-semibold text-[var(--text-muted)]">
                        · {t('runAlgorithms', { count: run.algorithm_count })}
                      </span>
                    </div>
                    <span className="text-[10px] font-bold text-[var(--text-faint)] flex items-center gap-1">
                      <Calendar size={12} />
                      {run.created_at ? new Date(run.created_at).toLocaleString(dLocale) : '—'}
                    </span>
                  </div>
                  <p className="text-xs text-[var(--text-secondary)] leading-relaxed italic line-clamp-2 bg-[var(--bg-muted)]/30 rounded-lg p-2.5 border border-[var(--border)]/40">
                    "{run.text_preview || '—'}"
                  </p>
                  <div className="flex flex-wrap items-center gap-4 text-xs font-medium text-[var(--text-muted)] mt-2">
                    <span className="flex items-center gap-1">
                      <FileText size={13} className="text-sky-500" />
                      {t('historyWords', {
                        input: run.input_words,
                        target: run.target_length_ratio ?? '—',
                        targetWords: run.target_words ?? '—',
                      })}
                    </span>
                    {run.processing_time_seconds != null && (
                      <>
                        <span className="text-[var(--text-faint)]">•</span>
                        <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400">
                          <Zap size={13} />
                          {Number(run.processing_time_seconds).toFixed(1)}s
                        </span>
                      </>
                    )}
                  </div>
                </motion.li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  );
};

export default memo(Analytics);
