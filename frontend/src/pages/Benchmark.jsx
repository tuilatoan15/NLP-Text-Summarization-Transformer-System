import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';
import {
  TrendingUp, Trophy, Zap, Clock, Target, Shield, AlertTriangle, BarChart3, Loader2, RefreshCw
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { useResearchLeaderboardQuery } from '../hooks/useApiQueries';

/* ────────────────────────────────────────────────────────────────── */
/*  Translation strings                                               */
/* ────────────────────────────────────────────────────────────────── */
const T = {
  vie: {
    title: 'Kết quả Benchmark',
    subtitle: 'So sánh hiệu năng trên các mô hình tóm tắt văn bản',
    loading: 'Đang tải dữ liệu benchmark...',
    error: 'Không thể tải dữ liệu benchmark.',
    retry: 'Thử lại',
    topPerformer: 'Mô hình tốt nhất',
    composite: 'Composite',
    latency: 'Độ trễ',
    type: 'Loại',
    modelRanking: 'Bảng xếp hạng mô hình',
    detailedComparison: 'So sánh chi tiết',
    model: 'Mô hình',
    group: 'Nhóm',
    qualityRadar: 'Radar chất lượng',
    latencyChart: 'Biểu đồ độ trễ',
    keyInsights: 'Nhận xét chính',
    insightAbstractive: 'Ưu thế Abstractive',
    insightAbstractiveDesc: 'Mô hình abstractive đạt ROUGE-1 cao hơn nhưng tiêu tốn nhiều tài nguyên tính toán hơn.',
    insightHybrid: 'Phương pháp Hybrid',
    insightHybridDesc: 'Kết hợp extractive + abstractive cho tốc độ nhanh hơn mà vẫn giữ chất lượng tốt.',
    insightSpeed: 'Tốc độ xử lý',
    insightSpeedDesc: 'Extractive hoàn thành trong mili-giây, phù hợp cho ứng dụng thời gian thực.',
    avgRouge: 'TB ROUGE-1',
    avgBertscore: 'TB BERTScore',
    fastestModel: 'Nhanh nhất',
    totalSamples: 'Tổng mẫu',
    abstractive: 'Abstractive',
    extractive: 'Extractive',
    hybrid: 'Hybrid',
    faithfulness: 'Faithfulness',
    hallucination: 'Hallucination',
    throughput: 'Throughput',
    fluency: 'Fluency',
    coverage: 'Coverage',
    infoRetention: 'Info Retention',
    semantic: 'Semantic',
    noData: 'Chưa có dữ liệu benchmark. Hãy chạy benchmark trước.',
  },
  eng: {
    title: 'Benchmark Results',
    subtitle: 'Compare performance across summarization models',
    loading: 'Loading benchmark data...',
    error: 'Unable to load benchmark data.',
    retry: 'Retry',
    topPerformer: 'Top Performer',
    composite: 'Composite',
    latency: 'Latency',
    type: 'Type',
    modelRanking: 'Model Ranking',
    detailedComparison: 'Detailed Comparison',
    model: 'Model',
    group: 'Group',
    qualityRadar: 'Quality Radar',
    latencyChart: 'Latency Chart',
    keyInsights: 'Key Insights',
    insightAbstractive: 'Abstractive Advantage',
    insightAbstractiveDesc: 'Abstractive models achieve higher ROUGE-1 but require more computational resources.',
    insightHybrid: 'Hybrid Approach',
    insightHybridDesc: 'Combining extractive + abstractive yields faster speed while maintaining good quality.',
    insightSpeed: 'Processing Speed',
    insightSpeedDesc: 'Extractive methods complete in milliseconds, ideal for real-time applications.',
    avgRouge: 'Avg ROUGE-1',
    avgBertscore: 'Avg BERTScore',
    fastestModel: 'Fastest',
    totalSamples: 'Total Samples',
    abstractive: 'Abstractive',
    extractive: 'Extractive',
    hybrid: 'Hybrid',
    faithfulness: 'Faithfulness',
    hallucination: 'Hallucination',
    throughput: 'Throughput',
    fluency: 'Fluency',
    coverage: 'Coverage',
    infoRetention: 'Info Retention',
    semantic: 'Semantic',
    noData: 'No benchmark data available. Run a benchmark first.',
  },
};

/* ────────────────────────────────────────────────────────────────── */
/*  Helper sub-components                                             */
/* ────────────────────────────────────────────────────────────────── */
const GROUP_COLORS = {
  extractive: '#10b981',
  abstractive: '#6366f1',
  hybrid: '#f59e0b',
};

const ModelRankCard = ({ model, rank, t }) => (
  <motion.div
    initial={{ opacity: 0, y: 8 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: rank * 0.04 }}
    className="ui-card p-4 flex items-center gap-4 group hover:shadow-lg transition-all"
  >
    <div
      className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0 font-bold"
      style={{
        backgroundColor: rank === 1 ? '#fbbf24' : rank === 2 ? '#a8adc1' : rank === 3 ? '#cd7f32' : 'var(--bg-muted)',
        color: rank <= 3 ? 'white' : 'var(--text-secondary)'
      }}
    >
      {rank}
    </div>
    <div className="flex-1 min-w-0">
      <h4 className="font-semibold text-[var(--text-primary)]">{model.name}</h4>
      <p className="text-xs text-[var(--text-muted)]">
        <span
          className="inline-block px-2 py-0.5 rounded-full text-white text-[10px] font-semibold mr-1"
          style={{ backgroundColor: GROUP_COLORS[model.group] || '#888' }}
        >
          {t(model.group) || model.group}
        </span>
      </p>
    </div>
    <div className="text-right space-y-0.5">
      <p className="text-lg font-bold text-[var(--accent)]">
        {(model.rouge1 * 100).toFixed(1)}%
      </p>
      <p className="text-[10px] text-[var(--text-muted)]">ROUGE-1</p>
    </div>
    <div className="text-right space-y-0.5">
      <p className="text-sm font-semibold text-[var(--text-primary)]">
        {(model.composite * 100).toFixed(1)}%
      </p>
      <p className="text-[10px] text-[var(--text-muted)]">Composite</p>
    </div>
  </motion.div>
);

const MetricCard = ({ icon: Icon, label, value, unit, color }) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.95 }}
    animate={{ opacity: 1, scale: 1 }}
    className="ui-card p-4"
  >
    <div className="flex items-start justify-between mb-3">
      <div
        className="w-8 h-8 rounded-lg flex items-center justify-center"
        style={{ backgroundColor: `${color}14`, color }}
      >
        <Icon className="w-4 h-4" />
      </div>
    </div>
    <p className="text-xs text-[var(--text-muted)] mb-1">{label}</p>
    <p className="text-2xl font-bold text-[var(--text-primary)]">
      {value}{unit}
    </p>
  </motion.div>
);

/* ────────────────────────────────────────────────────────────────── */
/*  Main Component                                                     */
/* ────────────────────────────────────────────────────────────────── */
const Benchmark = () => {
  const { isDark, locale } = useApp();
  const lang = locale === 'vie' ? 'vie' : 'eng';
  const t = (key) => T[lang][key] ?? key;

  const [benchmarkSize, setBenchmarkSize] = useState(1000);

  const leaderboardQuery = useResearchLeaderboardQuery('All', benchmarkSize, true);
  const { data, isLoading, isError, refetch } = leaderboardQuery;

  const chartTheme = {
    cartesianAxis: { stroke: isDark ? '#52525b' : '#e5e7eb' },
    tooltip: {
      contentStyle: {
        backgroundColor: isDark ? '#27272a' : '#ffffff',
        border: `1px solid ${isDark ? '#52525b' : '#e5e7eb'}`,
        borderRadius: '8px',
        color: isDark ? '#fafafa' : '#18181b'
      }
    }
  };

  /* Parse API response into sorted model array */
  const models = useMemo(() => {
    if (!data?.leaderboard) return [];
    const list = Array.isArray(data.leaderboard) ? data.leaderboard : Object.values(data.leaderboard);
    return list
      .filter(m => m && m.rouge1 !== undefined)
      .sort((a, b) => (b.composite || 0) - (a.composite || 0));
  }, [data]);

  const metadata = data?.metadata || {};
  const topModel = models[0];

  /* Aggregate metrics */
  const avgRouge = models.length
    ? (models.reduce((s, m) => s + (m.rouge1 || 0), 0) / models.length * 100).toFixed(1)
    : '—';
  const avgBertscore = models.length
    ? (models.reduce((s, m) => s + (m.bertscore || 0), 0) / models.length * 100).toFixed(1)
    : '—';
  const fastest = models.length
    ? Math.min(...models.map(m => m.latency || Infinity))
    : 0;
  const fastestName = models.find(m => m.latency === fastest)?.name || '—';

  /* Radar data */
  const radarData = useMemo(() => {
    if (models.length === 0) return [];
    const metrics = ['rouge1', 'bertscore', 'semantic', 'faithfulness', 'fluency', 'coverage'];
    const labels = ['ROUGE-1', 'BERTScore', 'Semantic', 'Faithfulness', 'Fluency', 'Coverage'];
    return metrics.map((key, i) => {
      const entry = { metric: labels[i] };
      models.slice(0, 6).forEach(m => {
        entry[m.name] = parseFloat(((m[key] || 0) * 100).toFixed(1));
      });
      return entry;
    });
  }, [models]);

  /* Latency bar data */
  const latencyData = useMemo(() => {
    return models.map(m => ({
      name: m.name,
      latency: parseFloat((m.latency || 0).toFixed(3)),
      group: m.group,
    }));
  }, [models]);

  /* ── Loading / Error states ── */
  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-4">
        <Loader2 className="w-10 h-10 animate-spin" style={{ color: 'var(--accent)' }} />
        <p className="text-[var(--text-muted)]">{t('loading')}</p>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-4">
        <AlertTriangle className="w-10 h-10" style={{ color: '#ef4444' }} />
        <p className="text-[var(--text-muted)]">{t('error')}</p>
        <button onClick={() => refetch()} className="ui-btn-primary px-4 py-2 rounded-lg flex items-center gap-2">
          <RefreshCw className="w-4 h-4" /> {t('retry')}
        </button>
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-4">
        <BarChart3 className="w-10 h-10" style={{ color: 'var(--text-muted)' }} />
        <p className="text-[var(--text-muted)]">{t('noData')}</p>
      </div>
    );
  }

  /* ── Radar colors ── */
  const radarColors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6'];

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="ui-heading-1 mb-1 flex items-center gap-2">
            <Trophy className="w-8 h-8" style={{ color: '#fbbf24' }} />
            {t('title')}
          </h1>
          <p className="ui-text text-[var(--text-muted)]">
            {t('subtitle')} — {metadata.total_samples?.toLocaleString() || '?'} {t('totalSamples').toLowerCase()} ({metadata.dataset_name || ''})
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs font-bold text-[var(--text-secondary)]">{lang === 'vie' ? 'Quy mô dữ liệu:' : 'Dataset size:'}</label>
          <select
            value={benchmarkSize}
            onChange={(e) => setBenchmarkSize(Number(e.target.value))}
            className="px-3 py-1.5 text-xs font-bold rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-primary)] focus:border-sky-500 outline-none transition-all cursor-pointer shadow-sm"
          >
            <option value={1000}>{lang === 'vie' ? '1.000 mẫu (Tiêu chuẩn)' : '1,000 samples (Standard)'}</option>
            <option value={2500}>{lang === 'vie' ? '2.500 mẫu (Quy mô vừa - 2 ngày)' : '2,500 samples (Medium-scale)'}</option>
            <option value={5000}>{lang === 'vie' ? '5.000 mẫu (Quy mô lớn)' : '5,000 samples (Large-scale)'}</option>
          </select>
        </div>
      </div>

      {/* Top Performer Card */}
      {topModel && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="ui-card p-6 border-2"
          style={{ borderColor: '#fbbf24' }}
        >
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-lg bg-yellow-100 dark:bg-yellow-900/30 flex items-center justify-center">
              <span className="text-3xl">👑</span>
            </div>
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-[var(--text-primary)]">{topModel.name}</h2>
              <p className="text-sm text-[var(--text-muted)]">{t('topPerformer')}</p>
            </div>
            <div className="text-right">
              <p className="text-4xl font-bold text-yellow-600">{(topModel.composite * 100).toFixed(1)}%</p>
              <p className="text-xs text-[var(--text-muted)]">{t('composite')}</p>
            </div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-[var(--text-muted)]">ROUGE-1</p>
              <p className="text-lg font-semibold text-[var(--text-primary)]">{(topModel.rouge1 * 100).toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-xs text-[var(--text-muted)]">BERTScore</p>
              <p className="text-lg font-semibold text-[var(--text-primary)]">{(topModel.bertscore * 100).toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-xs text-[var(--text-muted)]">{t('latency')}</p>
              <p className="text-lg font-semibold text-[var(--text-primary)]">{topModel.latency?.toFixed(3)}s</p>
            </div>
            <div>
              <p className="text-xs text-[var(--text-muted)]">{t('type')}</p>
              <p className="text-lg font-semibold text-[var(--text-primary)] capitalize">{t(topModel.group) || topModel.group}</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard icon={Target} label={t('avgRouge')} value={avgRouge} unit="%" color="#6366f1" />
        <MetricCard icon={TrendingUp} label={t('avgBertscore')} value={avgBertscore} unit="%" color="#10b981" />
        <MetricCard icon={Zap} label={t('fastestModel')} value={fastestName} unit={` (${fastest.toFixed(3)}s)`} color="#f59e0b" />
        <MetricCard icon={Clock} label={t('totalSamples')} value={metadata.total_samples?.toLocaleString() || '—'} unit="" color="#3b82f6" />
      </div>

      {/* Model Rankings */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-3"
      >
        <h2 className="text-lg font-semibold text-[var(--text-primary)] flex items-center gap-2">
          <Trophy size={20} style={{ color: 'var(--accent)' }} />
          {t('modelRanking')}
        </h2>
        <div className="space-y-2">
          {models.map((model, index) => (
            <ModelRankCard key={model.key || model.name} model={model} rank={index + 1} t={t} />
          ))}
        </div>
      </motion.div>

      {/* Radar Chart */}
      {radarData.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="ui-card p-6"
        >
          <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-4">{t('qualityRadar')}</h2>
          <ResponsiveContainer width="100%" height={380}>
            <RadarChart data={radarData}>
              <PolarGrid stroke={isDark ? '#3f3f46' : '#e5e7eb'} />
              <PolarAngleAxis dataKey="metric" tick={{ fill: isDark ? '#a1a1aa' : '#71717a', fontSize: 12 }} />
              <PolarRadiusAxis domain={[0, 100]} tick={{ fill: isDark ? '#71717a' : '#a1a1aa', fontSize: 10 }} />
              {models.slice(0, 6).map((m, i) => (
                <Radar
                  key={m.name}
                  name={m.name}
                  dataKey={m.name}
                  stroke={radarColors[i % radarColors.length]}
                  fill={radarColors[i % radarColors.length]}
                  fillOpacity={0.08}
                  strokeWidth={2}
                />
              ))}
              <Legend />
              <Tooltip {...chartTheme.tooltip} />
            </RadarChart>
          </ResponsiveContainer>
        </motion.div>
      )}

      {/* Latency Chart */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="ui-card p-6"
      >
        <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-4">{t('latencyChart')}</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={latencyData} layout="vertical" margin={{ left: 80 }}>
            <CartesianGrid {...chartTheme.cartesianAxis} />
            <XAxis type="number" stroke={isDark ? '#a1a1aa' : '#71717a'} unit="s" />
            <YAxis dataKey="name" type="category" stroke={isDark ? '#a1a1aa' : '#71717a'} width={90} tick={{ fontSize: 11 }} />
            <Tooltip {...chartTheme.tooltip} />
            <Bar
              dataKey="latency"
              radius={[0, 6, 6, 0]}
              fill="#6366f1"
              barSize={18}
            />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Detailed Comparison Table */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="ui-card p-6 overflow-x-auto"
      >
        <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-4">{t('detailedComparison')}</h2>
        <table className="w-full text-sm" style={{ minWidth: '900px' }}>
          <thead>
            <tr className="border-b" style={{ borderColor: 'var(--border)' }}>
              <th className="text-left py-2 px-2 font-semibold text-[var(--text-primary)]">{t('model')}</th>
              <th className="text-left py-2 px-2 font-semibold text-[var(--text-primary)]">{t('group')}</th>
              <th className="text-right py-2 px-2 font-semibold text-[var(--text-primary)]">ROUGE-1</th>
              <th className="text-right py-2 px-2 font-semibold text-[var(--text-primary)]">ROUGE-L</th>
              <th className="text-right py-2 px-2 font-semibold text-[var(--text-primary)]">BLEU</th>
              <th className="text-right py-2 px-2 font-semibold text-[var(--text-primary)]">BERTScore</th>
              <th className="text-right py-2 px-2 font-semibold text-[var(--text-primary)]">{t('semantic')}</th>
              <th className="text-right py-2 px-2 font-semibold text-[var(--text-primary)]">{t('faithfulness')}</th>
              <th className="text-right py-2 px-2 font-semibold text-[var(--text-primary)]">{t('hallucination')}</th>
              <th className="text-right py-2 px-2 font-semibold text-[var(--text-primary)]">{t('latency')} (s)</th>
              <th className="text-right py-2 px-2 font-semibold text-[var(--text-primary)]">{t('composite')}</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.key || m.name} className="border-b hover:bg-[var(--bg-muted)]" style={{ borderColor: 'var(--border-subtle)' }}>
                <td className="py-3 px-2">
                  <span className="font-medium text-[var(--text-primary)]">{m.name}</span>
                </td>
                <td className="py-3 px-2">
                  <span
                    className="text-xs px-2 py-1 rounded-full text-white font-semibold"
                    style={{ backgroundColor: GROUP_COLORS[m.group] || '#888' }}
                  >
                    {t(m.group) || m.group}
                  </span>
                </td>
                <td className="py-3 px-2 text-right font-semibold text-[var(--accent)]">{(m.rouge1 * 100).toFixed(2)}%</td>
                <td className="py-3 px-2 text-right">{(m.rougeL * 100).toFixed(2)}%</td>
                <td className="py-3 px-2 text-right">{(m.bleu * 100).toFixed(2)}%</td>
                <td className="py-3 px-2 text-right">{(m.bertscore * 100).toFixed(2)}%</td>
                <td className="py-3 px-2 text-right">{(m.semantic * 100).toFixed(1)}%</td>
                <td className="py-3 px-2 text-right">{((m.faithfulness || 0) * 100).toFixed(1)}%</td>
                <td className="py-3 px-2 text-right">
                  <span style={{ color: (m.hallucination_pct || 0) > 15 ? '#ef4444' : (m.hallucination_pct || 0) > 5 ? '#f59e0b' : '#10b981' }}>
                    {(m.hallucination_pct || 0).toFixed(1)}%
                  </span>
                </td>
                <td className="py-3 px-2 text-right text-[var(--text-muted)]">{(m.latency || 0).toFixed(3)}s</td>
                <td className="py-3 px-2 text-right font-bold text-[var(--text-primary)]">{((m.composite || 0) * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </motion.div>

      {/* Insights */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-4"
      >
        <h2 className="text-lg font-semibold text-[var(--text-primary)]">{t('keyInsights')}</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="ui-card p-4" style={{ borderLeft: '4px solid var(--success)' }}>
            <h4 className="font-semibold text-[var(--text-primary)] mb-2">{t('insightAbstractive')}</h4>
            <p className="text-sm text-[var(--text-muted)]">{t('insightAbstractiveDesc')}</p>
          </div>
          <div className="ui-card p-4" style={{ borderLeft: '4px solid #f59e0b' }}>
            <h4 className="font-semibold text-[var(--text-primary)] mb-2">{t('insightHybrid')}</h4>
            <p className="text-sm text-[var(--text-muted)]">{t('insightHybridDesc')}</p>
          </div>
          <div className="ui-card p-4" style={{ borderLeft: '4px solid var(--info)' }}>
            <h4 className="font-semibold text-[var(--text-primary)] mb-2">{t('insightSpeed')}</h4>
            <p className="text-sm text-[var(--text-muted)]">{t('insightSpeedDesc')}</p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Benchmark;
