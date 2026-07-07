import React, { useState, useMemo } from 'react';
import { BENCHMARK_SAMPLE_SIZE } from '../lib/benchmarkConfig';
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

  const getModelTypeStyles = (key) => {
    // 1. Extractive
    if (['textrank', 'lexrank', 'lsa'].includes(key)) {
      return {
        bg: 'bg-emerald-50 dark:bg-emerald-950/20',
        border: 'border-emerald-250 dark:border-emerald-900/30',
        text: 'text-emerald-700 dark:text-emerald-400',
        label: 'Extractive'
      };
    }
    // 2. Abstractive (Fine-tuned)
    if (['vit5', 'bartpho'].includes(key)) {
      return {
        bg: 'bg-amber-50 dark:bg-amber-950/20',
        border: 'border-amber-250 dark:border-amber-900/30',
        text: 'text-amber-700 dark:text-amber-400',
        label: 'Abstractive (FT)'
      };
    }
    // 3. Abstractive (Baseline)
    if (key === 'mt5') {
      return {
        bg: 'bg-rose-50 dark:bg-rose-950/20',
        border: 'border-rose-250 dark:border-rose-900/30',
        text: 'text-rose-700 dark:text-rose-400',
        label: 'Abstractive (Base)'
      };
    }
    // 4. Hybrid (Lai ghép)
    return {
      bg: 'bg-sky-50 dark:bg-sky-950/20',
      border: 'border-sky-250 dark:border-sky-900/30',
      text: 'text-sky-700 dark:text-sky-400',
      label: 'Hybrid'
    };
  };

  const benchmarkSize = BENCHMARK_SAMPLE_SIZE;

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
        <span className="px-3 py-1.5 text-xs font-bold rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-secondary)] shadow-sm">
          {lang === 'vie' ? '5.000 mẫu (benchmark thực nghiệm)' : '5,000 samples (research benchmark)'}
        </span>
      </div>



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
              <th className="text-left py-2 px-2 font-semibold text-[var(--text-primary)]">Phương pháp</th>
              <th className="text-center py-2 px-2 font-semibold text-[var(--text-primary)]">ROUGE-1 (R1)</th>
              <th className="text-center py-2 px-2 font-semibold text-[var(--text-primary)]">ROUGE-2 (R2)</th>
              <th className="text-center py-2 px-2 font-semibold text-[var(--text-primary)]">ROUGE-L (RL)</th>
              <th className="text-center py-2 px-2 font-semibold text-[var(--text-primary)]">ROUGE-LSum</th>
              <th className="text-center py-2 px-2 font-semibold text-[var(--text-primary)]">BERT P</th>
              <th className="text-center py-2 px-2 font-semibold text-[var(--text-primary)]">BERT R</th>
              <th className="text-center py-2 px-2 font-semibold text-[var(--text-primary)]">BERT F1</th>
              <th className="text-center py-2 px-2 font-semibold text-[var(--text-primary)] bg-sky-500/5 text-sky-600 dark:text-sky-400">Latency (s)</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.key || m.name} className="border-b hover:bg-[var(--bg-muted)]" style={{ borderColor: 'var(--border-subtle)' }}>
                <td className="py-3 px-2">
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold border whitespace-nowrap ${getModelTypeStyles(m.key).bg} ${getModelTypeStyles(m.key).border} ${getModelTypeStyles(m.key).text}`}>
                      {getModelTypeStyles(m.key).label}
                    </span>
                    <span className="font-semibold text-[var(--text-primary)]">{m.name}</span>
                  </div>
                </td>
                <td className="py-3 px-2 text-center font-mono">{(m.rouge1 ?? 0).toFixed(4)}</td>
                <td className="py-3 px-2 text-center font-mono">{(m.rouge2 ?? 0).toFixed(4)}</td>
                <td className="py-3 px-2 text-center font-mono">{(m.rougeL ?? 0).toFixed(4)}</td>
                <td className="py-3 px-2 text-center font-mono">{(m.rougeLsum ?? m.rougeL ?? 0).toFixed(4)}</td>
                <td className="py-3 px-2 text-center font-mono">{(m.bert_p ?? 0).toFixed(4)}</td>
                <td className="py-3 px-2 text-center font-mono">{(m.bert_r ?? 0).toFixed(4)}</td>
                <td className="py-3 px-2 text-center font-mono">{(m.bertscore ?? 0).toFixed(4)}</td>
                <td className="py-3 px-2 bg-sky-500/5 text-sky-700 dark:text-sky-300 font-bold text-center font-mono">
                  {(m.latency ?? 0).toFixed(4)}
                </td>
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
