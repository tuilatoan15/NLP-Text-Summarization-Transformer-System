import React, { useEffect, useMemo, useState } from 'react';
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { Loader2 } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { getAnalyticsDashboard } from '../services/apiService';
import { getChartTheme, dateLocale } from '../theme/chartTheme';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#14b8a6'];

const StatBox = ({ title, value, sub }) => (
  <div className="ui-card p-6">
    <p className="text-sm text-[var(--text-muted)] mb-2">{title}</p>
    <span className="text-3xl font-bold text-[var(--text)]">{value}</span>
    {sub && <p className="text-xs text-[var(--text-faint)] mt-2">{sub}</p>}
  </div>
);

const rangeKey = (range) => {
  if (range === '7d') return 'range7d';
  if (range === '30d') return 'range30d';
  if (range === '90d') return 'range90d';
  return 'rangeAll';
};

const Analytics = () => {
  const { t, locale, isDark } = useApp();
  const [timeRange, setTimeRange] = useState('30d');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const chart = getChartTheme(isDark);
  const dLocale = dateLocale(locale);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError('');
      try {
        const payload = await getAnalyticsDashboard(timeRange, 20);
        if (!cancelled) setData(payload);
      } catch (err) {
        if (!cancelled) setError(err.message || t('analyticsLoadError'));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [timeRange]);

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
      <div className="flex items-center justify-center py-16 text-[var(--text-muted)]">
        <Loader2 className="animate-spin mr-2 w-5 h-5" />
        {t('loadingAnalytics')}
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-12">
      <div>
        <h1 className="ui-page-title mb-1">{t('analyticsTitle')}</h1>
        <p className="ui-page-subtitle">{t('analyticsSubtitle')}</p>
      </div>

      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex gap-2 flex-wrap">
          {['7d', '30d', '90d', 'all'].map(range => (
            <button
              key={range}
              type="button"
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                timeRange === range
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'ui-btn-secondary !py-2'
              }`}
            >
              {t(rangeKey(range))}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <p className="text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800/50 rounded-lg px-4 py-3">
          {error}
        </p>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatBox title={t('statSavedRuns')} value={metrics.total_runs ?? 0} />
        <StatBox
          title={t('statAvgRouge')}
          value={(metrics.avg_rouge?.rougeL ?? 0).toFixed(3)}
          sub={t('statAvgRougeSub')}
        />
        <StatBox
          title={t('statAvgBert')}
          value={(metrics.avg_bertscore_f1 ?? 0).toFixed(3)}
        />
        <StatBox
          title={t('statAvgLength')}
          value={`${metrics.avg_target_length_ratio ?? 0}%`}
          sub={t('statActualLength', { actual: metrics.avg_actual_length_ratio ?? 0 })}
        />
      </div>

      {metrics.total_runs === 0 ? (
        <div className="rounded-xl border border-dashed border-[var(--border)] bg-[var(--surface-elevated)] p-12 text-center text-[var(--text-muted)]">
          {t('analyticsEmpty')}{' '}
          <Link to="/playground" className="font-semibold text-blue-600 dark:text-blue-400 hover:underline">
            {t('analyticsEmptyLink')}
          </Link>
          , {t('analyticsEmptySuffix')}{' '}
          <code className="text-xs bg-[var(--surface-inset)] px-1.5 py-0.5 rounded text-[var(--text-secondary)]">
            storage/results
          </code>
          .
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="ui-card p-6">
              <h3 className="text-sm font-semibold text-[var(--text)] mb-4">{t('chartModelMetrics')}</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={modelRows}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
                  <XAxis dataKey="model" tick={{ fontSize: 11, fill: chart.axis }} />
                  <YAxis domain={[0, 1]} tick={{ fill: chart.axis }} />
                  <Tooltip contentStyle={chart.tooltipStyle} />
                  <Legend wrapperStyle={{ color: chart.axis }} />
                  <Bar dataKey="rougeL" fill="#f59e0b" name={t('colRougeL')} />
                  <Bar dataKey="bertScore" fill="#8b5cf6" name={t('colBert')} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="ui-card p-6">
              <h3 className="text-sm font-semibold text-[var(--text)] mb-4">{t('chartTrend')}</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={timeseries}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
                  <XAxis dataKey="date" tick={{ fontSize: 11, fill: chart.axis }} />
                  <YAxis yAxisId="left" domain={[0, 1]} tick={{ fill: chart.axis }} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fill: chart.axis }} />
                  <Tooltip contentStyle={chart.tooltipStyle} />
                  <Legend wrapperStyle={{ color: chart.axis }} />
                  <Line yAxisId="left" type="monotone" dataKey="avgRougeL" stroke="#f59e0b" name={t('colRougeL')} />
                  <Line yAxisId="left" type="monotone" dataKey="avgBertScore" stroke="#8b5cf6" name={t('colBert')} />
                  <Line yAxisId="right" type="monotone" dataKey="count" stroke="#3b82f6" name={t('chartRunCount')} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {pieData.length > 0 && (
              <div className="ui-card p-6">
                <h3 className="text-sm font-semibold text-[var(--text)] mb-4">{t('chartAlgoFreq')}</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label>
                      {pieData.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={chart.tooltipStyle} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          <div className="ui-card overflow-hidden">
            <div className="px-6 py-4 border-b border-[var(--border)] bg-[var(--surface-inset)]">
              <h3 className="text-sm font-semibold text-[var(--text)]">{t('tableModelDetail')}</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="ui-table-head border-b">
                  <tr>
                    <th className="px-6 py-3 text-left">{t('colModel')}</th>
                    <th className="px-6 py-3 text-center">{t('colRuns')}</th>
                    <th className="px-6 py-3 text-center">{t('colAvgRouge')}</th>
                    <th className="px-6 py-3 text-center">{t('colAvgBert')}</th>
                    <th className="px-6 py-3 text-center">{t('colAvgLength')}</th>
                    <th className="px-6 py-3 text-center">{t('colAvgTime')}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-subtle)]">
                  {modelRows.map(row => (
                    <tr key={row.model} className="ui-table-row">
                      <td className="px-6 py-3 font-medium text-[var(--text)]">{row.model}</td>
                      <td className="px-6 py-3 text-center text-[var(--text-secondary)]">{row.count}</td>
                      <td className="px-6 py-3 text-center text-blue-600 dark:text-blue-400 font-semibold">
                        {row.rougeL.toFixed(3)}
                      </td>
                      <td className="px-6 py-3 text-center text-emerald-600 dark:text-emerald-400 font-semibold">
                        {row.bertScore.toFixed(3)}
                      </td>
                      <td className="px-6 py-3 text-center text-[var(--text-secondary)]">{row.avgLengthRatio}%</td>
                      <td className="px-6 py-3 text-center text-[var(--text-secondary)]">{row.avgTime.toFixed(2)}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="ui-card overflow-hidden">
            <div className="px-6 py-4 border-b border-[var(--border)] bg-[var(--surface-inset)]">
              <h3 className="text-sm font-semibold text-[var(--text)]">{t('tableHistory')}</h3>
            </div>
            <ul className="divide-y divide-[var(--border-subtle)]">
              {recentRuns.map(run => (
                <li key={run.result_id} className="px-6 py-4 ui-table-row">
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
                    <span className="text-sm font-semibold text-[var(--text)]">
                      {run.best_algorithm || '—'} · {t('runAlgorithms', { count: run.algorithm_count })}
                    </span>
                    <span className="text-xs text-[var(--text-faint)]">
                      {run.created_at ? new Date(run.created_at).toLocaleString(dLocale) : '—'}
                    </span>
                  </div>
                  <p className="text-xs text-[var(--text-muted)] line-clamp-2">{run.text_preview || '—'}</p>
                  <p className="text-xs text-indigo-600 dark:text-indigo-400 mt-1">
                    {t('historyWords', {
                      input: run.input_words,
                      target: run.target_length_ratio ?? '—',
                      targetWords: run.target_words ?? '—',
                    })}
                    {run.processing_time_seconds != null && (
                      <> · {Number(run.processing_time_seconds).toFixed(1)}s</>
                    )}
                  </p>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  );
};

export default Analytics;
