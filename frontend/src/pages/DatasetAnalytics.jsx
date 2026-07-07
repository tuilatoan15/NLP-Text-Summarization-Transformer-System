import React, { memo, useMemo } from 'react';
import { motion, useSpring, useTransform } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis,
} from 'recharts';
import {
  Database, BarChart3, Cloud, GitBranch, AlertTriangle,
  Layers, BookOpen, Target, Loader2, AlertCircle, TrendingDown,
  FileText, Hash, Activity, Clock, HardDrive,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import {
  useDatasetAnalyticsQuery,
  useDatasetChartsQuery,
} from '../hooks/useApiQueries';
import { getChartTheme } from '../theme/chartTheme';
import { buildDatasetAnalyticsMetaLine } from '../utils/formatDuration';

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const COLORS = ['#6366f1', '#0ea5e9', '#10b981', '#fbbf24', '#f43f5e', '#a855f7'];

const AnimatedNumber = memo(({ value, decimals = 0 }) => {
  const num = Number(value) || 0;
  const spring = useSpring(0, { stiffness: 60, damping: 20 });
  React.useEffect(() => {
    spring.set(num);
  }, [num, spring]);
  const display = useTransform(spring, (v) =>
    decimals > 0 ? v.toFixed(decimals) : Math.round(v).toLocaleString(),
  );
  return <motion.span>{display}</motion.span>;
});

const Skeleton = memo(({ className = '' }) => (
  <div className={`ui-skeleton ${className}`} />
));

const Section = memo(({ title, icon: Icon, children, delay = 0 }) => (
  <motion.section
    initial={{ opacity: 0, y: 16 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay, duration: 0.35 }}
    className="ui-card p-5 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm space-y-4"
  >
    <h2 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)] flex items-center gap-2">
      {Icon && <Icon size={14} className="text-sky-500" />}
      {title}
    </h2>
    {children}
  </motion.section>
));

const StatCard = memo(({ title, value, sub, icon: Icon, color, delay = 0, numeric }) => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay }}
    className="ui-card p-4 bg-[var(--bg-elevated)] border border-[var(--border)]"
  >
    <div className="flex justify-between mb-2">
      <p className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">{title}</p>
      <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ backgroundColor: `${color}15`, color }}>
        <Icon size={16} />
      </div>
    </div>
    <p className="text-2xl font-extrabold text-[var(--text-primary)]">
      {numeric !== false ? <AnimatedNumber value={value} decimals={typeof value === 'number' && value < 10 ? 3 : 0} /> : value}
    </p>
    {sub && <p className="text-[11px] text-[var(--text-faint)] mt-1">{sub}</p>}
  </motion.div>
));

const ChartImage = memo(({ chart }) => {
  if (!chart?.url && !chart?.filename) return null;
  const src = chart.url?.startsWith('http') ? chart.url : `${API}${chart.url || `/analytics/charts/file/${chart.filename}`}`;
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      className="rounded-xl overflow-hidden border border-[var(--border)] bg-white"
    >
      <img src={src} alt={chart.filename || 'chart'} className="w-full h-auto" loading="lazy" />
    </motion.div>
  );
});

const DatasetAnalytics = () => {
  const { isDark, locale, t } = useApp();
  const { data, isLoading, error, refetch } = useDatasetAnalyticsQuery();
  const { data: chartsData } = useDatasetChartsQuery();
  const chart = getChartTheme(isDark);

  const overview = data?.overview || {};
  const docStats = data?.document_stats || {};
  const sumStats = data?.summary_stats || {};
  const vocab = data?.vocabulary || {};
  const quality = data?.quality || {};
  const compression = data?.compression_statistics || {};
  const correlation = data?.correlation || {};
  const training = data?.training_statistics || {};
  const rouge = data?.rouge_baseline || {};
  const rougeLead = rouge.lead_words_proportional || rouge;
  const extractive = data?.extractive_metrics || {};
  const lengthDist = data?.length_distribution || {};
  const charts = chartsData?.charts || data?.charts || {};
  const meta = data?.metadata || chartsData?.metadata || {};

  const metaLine = useMemo(
    () => buildDatasetAnalyticsMetaLine(meta, locale, t),
    [meta, locale, t],
  );

  const splitPie = useMemo(() => {
    const splits = overview.splits || {};
    return Object.entries(splits).map(([name, value]) => ({ name, value }));
  }, [overview.splits]);

  const histArticle = useMemo(() => {
    const d = lengthDist.article_words;
    if (!d?.bins?.length) return [];
    return (d.counts || []).map((c, i) => ({
      bin: Math.round((d.bins[i] + d.bins[i + 1]) / 2),
      count: c,
    }));
  }, [lengthDist]);

  const zipfData = vocab.zipf || [];
  const vocabGrowth = vocab.vocab_growth || [];
  const scatterData = (compression.scatter_sample || []).slice(0, 300);
  const topWords = (vocab.top_100_words || []).slice(0, 15);
  const bigrams = (vocab.top_30_bigrams || []).slice(0, 12);

  const corrLabels = correlation.labels || [];
  const corrMatrix = correlation.matrix || [];

  if (isLoading && !data) {
    return (
      <div className="space-y-6 pb-12">
        <Skeleton className="h-24 w-full rounded-2xl" />
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-28 rounded-xl" />)}
        </div>
        <Skeleton className="h-64 rounded-xl" />
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-12">
      {error && (
        <div className="flex items-center justify-between gap-3 text-sm text-red-600 bg-red-50 dark:bg-red-950/20 border border-red-200 rounded-xl px-4 py-3">
          <div className="flex items-center gap-2">
            <AlertCircle size={16} />
            <span>{error.message || 'Không tải được dữ liệu dataset'}</span>
          </div>
          <button type="button" onClick={() => refetch()} className="text-xs font-bold underline">Thử lại</button>
        </div>
      )}

      <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider bg-indigo-100 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-300 border border-indigo-200/50">
                <Database size={11} />
                VietNews Analytics
              </span>
              {(meta.source === 'colab' || overview.source === 'colab') && (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider bg-orange-100 dark:bg-orange-950/40 text-orange-700 dark:text-orange-300 border border-orange-200/50">
                  <Cloud size={11} />
                  Nguồn: Colab
                </span>
              )}
            </div>
            <h1 className="text-2xl md:text-3xl font-extrabold text-[var(--text-primary)] tracking-tight">
              Phân tích Dataset & Tiền xử lý
            </h1>
            <p className="text-sm text-[var(--text-secondary)] max-w-3xl">
              Thống kê thực tế từ <strong>{overview.dataset_name || 'nam194/vietnews'}</strong>
              {meta.full_dataset || overview.full_dataset
                ? ' — toàn bộ dataset'
                : meta.limit_per_split
                  ? ` (giới hạn ${meta.limit_per_split}/split)`
                  : ''}.
              {metaLine ? ` ${metaLine}` : ''}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-3 text-[11px] text-[var(--text-muted)]">
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-[var(--bg-muted)]">
            <HardDrive size={12} /> Cache: {meta.cache_enabled !== false ? 'bật' : 'tắt'}
          </span>
          {meta.chart_count != null && (
            <span className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-[var(--bg-muted)]">
              <Cloud size={12} /> {meta.chart_count} biểu đồ PNG
            </span>
          )}
          {overview.total_raw_samples != null && (
            <span className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-[var(--bg-muted)]">
              <Clock size={12} /> Raw HF: {overview.total_raw_samples?.toLocaleString?.()} mẫu
            </span>
          )}
        </div>
      </motion.div>

      {/* Overview */}
      <div className="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-6 gap-4">
        <StatCard title="Tổng mẫu" value={overview.total_documents || 0} sub="Cặp article–abstract" icon={FileText} color="#0ea5e9" delay={0} />
        <StatCard title="Từ vựng" value={overview.vocab_size || 0} sub="Unique tokens" icon={Hash} color="#6366f1" delay={0.05} />
        <StatCard title="TB bài viết" value={overview.avg_article_words || 0} sub="từ / bài" icon={BookOpen} color="#10b981" delay={0.1} />
        <StatCard title="TB tóm tắt" value={overview.avg_summary_words || 0} sub="từ / summary" icon={Target} color="#f59e0b" delay={0.15} />
        <StatCard title="Tỷ lệ nén" value={overview.avg_compression_ratio || 0} sub={`${overview.avg_reduction_pct || 0}% giảm`} icon={TrendingDown} color="#f43f5e" delay={0.2} numeric />
        <StatCard title="ROUGE-L" value={rougeLead.rougeL ?? '—'} sub="Lead proportional" icon={Activity} color="#a855f7" delay={0.25} numeric={rougeLead.rougeL != null} />
      </div>

      {/* Split + Training */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Section title="Phân chia Dataset" icon={Layers} delay={0.1}>
          {splitPie.length > 0 ? (
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={splitPie} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                    {splitPie.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-sm text-[var(--text-muted)]">No Data</p>
          )}
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            {Object.entries(overview.splits || {}).map(([k, v]) => (
              <div key={k} className="p-2 rounded-lg bg-[var(--bg-muted)]">
                <p className="font-bold text-[var(--text-primary)]">{v?.toLocaleString?.() ?? v}</p>
                <p className="text-[var(--text-faint)] uppercase">{k}</p>
              </div>
            ))}
          </div>
        </Section>

        <Section title="Cấu hình Huấn luyện" icon={GitBranch} delay={0.15}>
          <div className="grid grid-cols-2 gap-3 text-sm">
            {[
              ['Batch size', training.train_batch_size ?? 'No Data'],
              ['Eval batch', training.eval_batch_size ?? 'No Data'],
              ['Epochs', training.num_epochs ?? 'No Data'],
              ['Learning rate', training.learning_rate ?? 'No Data'],
              ['Max input tok', training.max_input_tokens ?? 'No Data'],
              ['Max target tok', training.max_target_tokens ?? 'No Data'],
              ['Grad accum', training.gradient_accumulation_steps ?? 'No Data'],
              ['Model', training.default_model ?? 'No Data'],
            ].map(([k, v]) => (
              <div key={k} className="p-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg)]">
                <p className="text-[10px] uppercase text-[var(--text-faint)] font-bold">{k}</p>
                <p className="font-semibold text-[var(--text-primary)] truncate" title={String(v)}>{String(v)}</p>
              </div>
            ))}
          </div>
        </Section>
      </div>

      {/* Statistics */}
      <Section title="Thống kê độ dài" icon={BarChart3} delay={0.2}>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={[
                { metric: 'min', article: docStats.words?.min, summary: sumStats.words?.min },
                { metric: 'mean', article: docStats.words?.mean, summary: sumStats.words?.mean },
                { metric: 'median', article: docStats.words?.median, summary: sumStats.words?.median },
                { metric: 'max', article: docStats.words?.max, summary: sumStats.words?.max },
              ]}>
                <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
                <XAxis dataKey="metric" tick={{ fill: chart.axis, fontSize: 11 }} />
                <YAxis tick={{ fill: chart.axis, fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="article" name="Bài viết" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                <Bar dataKey="summary" name="Tóm tắt" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          {histArticle.length > 0 && (
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={histArticle}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
                  <XAxis dataKey="bin" tick={{ fill: chart.axis, fontSize: 10 }} />
                  <YAxis tick={{ fill: chart.axis, fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#6366f1" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </Section>

      {/* Compression */}
      <Section title="Tỷ lệ nén & Hồi quy" icon={TrendingDown} delay={0.25}>
        {scatterData.length > 0 ? (
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
                <XAxis type="number" dataKey="article_words" name="Article words" tick={{ fill: chart.axis, fontSize: 10 }} />
                <YAxis type="number" dataKey="compression_ratio" name="Ratio" tick={{ fill: chart.axis, fontSize: 10 }} />
                <ZAxis range={[20, 20]} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter data={scatterData} fill="#6366f1" fillOpacity={0.5} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="text-sm text-[var(--text-muted)]">No Data</p>
        )}
        {compression.regression?.equation && (
          <p className="text-xs font-mono text-[var(--text-muted)]">{compression.regression.equation}</p>
        )}
      </Section>

      {/* Vocabulary & N-grams */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Section title="Từ vựng Top 15" icon={BookOpen} delay={0.3}>
          {topWords.length > 0 ? (
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topWords} layout="vertical" margin={{ left: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} horizontal={false} />
                  <XAxis type="number" tick={{ fill: chart.axis, fontSize: 10 }} />
                  <YAxis type="category" dataKey="word" tick={{ fill: chart.axis, fontSize: 10 }} width={55} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <p className="text-sm text-[var(--text-muted)]">No Data</p>}
        </Section>
        <Section title="Bigrams Top 12" icon={Hash} delay={0.35}>
          {bigrams.length > 0 ? (
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={bigrams} layout="vertical" margin={{ left: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} horizontal={false} />
                  <XAxis type="number" tick={{ fill: chart.axis, fontSize: 10 }} />
                  <YAxis type="category" dataKey="ngram" tick={{ fill: chart.axis, fontSize: 9 }} width={75} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#a855f7" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <p className="text-sm text-[var(--text-muted)]">No Data</p>}
        </Section>
      </div>

      {/* Zipf & Vocab growth */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Section title="Zipf (log-log)" icon={Activity} delay={0.4}>
          {zipfData.length > 0 ? (
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={zipfData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
                  <XAxis dataKey="rank" scale="log" domain={['auto', 'auto']} tick={{ fill: chart.axis, fontSize: 10 }} />
                  <YAxis scale="log" domain={['auto', 'auto']} tick={{ fill: chart.axis, fontSize: 10 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="frequency" stroke="#10b981" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : <p className="text-sm text-[var(--text-muted)]">No Data</p>}
        </Section>
        <Section title="Tăng trưởng từ vựng" icon={Layers} delay={0.45}>
          {vocabGrowth.length > 0 ? (
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={vocabGrowth}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
                  <XAxis dataKey="tokens_seen" tick={{ fill: chart.axis, fontSize: 10 }} />
                  <YAxis tick={{ fill: chart.axis, fontSize: 10 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="unique_vocab" stroke="#0ea5e9" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : <p className="text-sm text-[var(--text-muted)]">No Data</p>}
        </Section>
      </div>

      {/* Correlation */}
      <Section title="Ma trận tương quan" icon={GitBranch} delay={0.5}>
        {corrLabels.length > 0 && corrMatrix.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr>
                  <th className="p-2" />
                  {corrLabels.map((l) => <th key={l} className="p-2 text-[var(--text-muted)] font-semibold">{l}</th>)}
                </tr>
              </thead>
              <tbody>
                {corrMatrix.map((row, i) => (
                  <tr key={corrLabels[i]}>
                    <td className="p-2 font-semibold text-[var(--text-muted)]">{corrLabels[i]}</td>
                    {row.map((val, j) => {
                      const v = Number(val);
                      const bg = v > 0 ? `rgba(16,185,129,${Math.abs(v) * 0.5})` : `rgba(244,63,94,${Math.abs(v) * 0.5})`;
                      return (
                        <td key={j} className="p-2 text-center font-mono rounded" style={{ backgroundColor: bg }}>
                          {v.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="text-sm text-[var(--text-muted)]">No Data</p>}
      </Section>

      {/* Quality & Outliers */}
      <Section title="Chất lượng & Outliers" icon={AlertTriangle} delay={0.55}>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {[
            ['Trùng lặp', quality.duplicates],
            ['Rỗng (article)', quality.empty_articles],
            ['Rỗng (summary)', quality.empty_summaries],
            ['Quá ngắn', quality.very_short_articles],
            ['Quá dài', quality.very_long_articles],
            ['Outliers 3σ', quality.outliers_3sigma],
          ].map(([label, val]) => (
            <div key={label} className="p-3 rounded-xl border border-[var(--border)] text-center">
              <p className="text-lg font-bold text-[var(--text-primary)]">{val ?? 0}</p>
              <p className="text-[10px] text-[var(--text-faint)] uppercase font-semibold">{label}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* PNG Charts gallery */}
      <Section title="Biểu đồ (PNG)" icon={Cloud} delay={0.6}>
        {Object.keys(charts).length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(charts).map(([key, chart]) => (
              <div key={key} className="space-y-2">
                <p className="text-[10px] font-bold uppercase text-[var(--text-faint)]">{key.replace(/_/g, ' ')}</p>
                <ChartImage chart={typeof chart === 'string' ? { url: `/analytics/charts/file/${chart.split(/[/\\]/).pop()}` } : chart} />
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 py-8 text-[var(--text-muted)]">
            <Loader2 className="animate-spin" size={24} />
            <p className="text-sm">Chưa có biểu đồ. Chạy: <code className="text-xs bg-[var(--bg-muted)] px-2 py-1 rounded">python scripts/run_dataset_analytics.py</code></p>
          </div>
        )}
      </Section>

      <p className="text-xs text-[var(--text-faint)] text-center">
        Dữ liệu từ pipeline thực — không mock.{' '}
        <Link to="/" className="text-sky-600 underline">Về Dashboard</Link>
      </p>
    </div>
  );
};

export default DatasetAnalytics;
