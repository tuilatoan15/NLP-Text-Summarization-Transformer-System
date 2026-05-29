import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import {
  Bot, Cpu, Activity, Clock, Loader2, CheckCircle2,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { getAnalyticsDashboard, getHealth, getMetrics } from '../services/apiService';
import { getChartTheme, dateLocale } from '../theme/chartTheme';

const StatCard = ({ title, value, subtext, icon: Icon, color }) => (
  <div className="ui-card p-5 flex flex-col justify-between">
    <div className="flex justify-between items-start mb-4">
      <div>
        <p className="ui-stat-label">{title}</p>
        <h3 className="text-3xl font-bold text-[var(--text)]">{value}</h3>
      </div>
      <div className={`p-2.5 rounded-xl text-white ${color}`}>
        <Icon className="w-5 h-5" />
      </div>
    </div>
    {subtext && <p className="text-xs text-[var(--text-muted)] font-medium">{subtext}</p>}
  </div>
);

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

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-[var(--text-muted)]">
        <Loader2 className="animate-spin mr-2 w-5 h-5" />
        {t('loading')}
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-12">
      <div>
        <h1 className="ui-page-title mb-1">{t('overviewTitle')}</h1>
        <p className="ui-page-subtitle flex items-center gap-1.5">
          {health?.status === 'ok' ? (
            <>
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
              {t('overviewSubtitleOk')}
            </>
          ) : (
            t('overviewSubtitleError')
          )}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard
          title={t('statRunsTitle')}
          value={dashMetrics.total_runs ?? 0}
          subtext={t('statRunsSub', { count: algorithmOutputs })}
          icon={Activity}
          color="bg-emerald-500"
        />
        <StatCard
          title={t('statModelsTitle')}
          value={modelCount}
          subtext={metrics?.gpu_name || 'CPU'}
          icon={Cpu}
          color="bg-cyan-500"
        />
        <StatCard
          title={t('statRougeTitle')}
          value={(dashMetrics.avg_rouge?.rougeL ?? 0).toFixed(2)}
          subtext={t('statRougeSub')}
          icon={Bot}
          color="bg-blue-500"
        />
        <StatCard
          title={t('statLengthTitle')}
          value={`${dashMetrics.avg_target_length_ratio ?? 0}%`}
          subtext={t('statLengthSub', { actual: dashMetrics.avg_actual_length_ratio ?? 0 })}
          icon={Clock}
          color="bg-amber-500"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="ui-card p-5">
          <div className="mb-4">
            <h3 className="ui-stat-label">{t('chartRunsTitle')}</h3>
            <p className="text-[11px] text-[var(--text-faint)]">{t('chartRunsSub')}</p>
          </div>
          <div className="h-48 w-full">
            {timeseries.length > 0 ? (
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
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name={t('chartRunsName')} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-[var(--text-faint)] text-center py-16">{t('emptyChart')}</p>
            )}
          </div>
        </div>

        <div className="ui-card p-6 flex flex-col">
          <h3 className="text-sm font-bold text-[var(--text)] mb-4">{t('recentActivity')}</h3>
          <div className="flex-1 space-y-4 overflow-y-auto max-h-64">
            {recentRuns.length === 0 && (
              <p className="text-sm text-[var(--text-faint)]">{t('emptyRecent')}</p>
            )}
            {recentRuns.map(run => (
              <div key={run.result_id} className="flex items-start gap-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm text-[var(--text-secondary)]">
                    <span className="font-semibold text-[var(--text)]">
                      {run.best_algorithm || t('runCompareLabel')}
                    </span>
                    {' '}— {t('runAlgorithms', { count: run.algorithm_count })}
                    {' · '}{t('runTarget', { ratio: run.target_length_ratio })}
                  </p>
                  <p className="text-xs text-[var(--text-muted)] line-clamp-1 mt-0.5">{run.text_preview}</p>
                  <p className="text-xs text-[var(--text-faint)] mt-1">
                    {run.created_at ? new Date(run.created_at).toLocaleString(dLocale) : ''}
                  </p>
                </div>
              </div>
            ))}
            {health?.status === 'ok' && (
              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm text-[var(--text-secondary)]">
                    <span className="font-semibold text-[var(--text)]">{t('apiReady')}</span>
                    {' '}— models{' '}
                    {metrics?.models_preloaded ? t('apiPreload') : t('apiLazy')}
                  </p>
                  <p className="text-xs text-[var(--text-faint)] mt-1">{t('justChecked')}</p>
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
