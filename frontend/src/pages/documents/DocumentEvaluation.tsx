import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend, LineChart, Line,
} from 'recharts';
import { BarChart3, Clock, Zap, GitCompare, TrendingUp, FileText } from 'lucide-react';
import { useDocumentContext } from '../../context/DocumentContext';

const ALGO_COLORS: Record<string, string> = {
  textrank: '#14b8a6', lexrank: '#38bdf8', lsa: '#84cc16', tfidf: '#a78bfa',
  vit5: '#f59e0b', mt5: '#e879f9', bartpho: '#fb7185',
};

const pct = (v: any) => Number(v ?? 0);
const pctLabel = (v: any) => `${(pct(v) * 100).toFixed(1)}%`;

function Panel({ title, icon: Icon, children }: { title: string; icon: React.ElementType; children: React.ReactNode }) {
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="ui-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <div className="p-1.5 rounded-lg bg-blue-50 dark:bg-blue-900/30">
          <Icon className="w-4 h-4 text-blue-600 dark:text-blue-400" />
        </div>
        <h3 className="text-sm font-bold text-[var(--text)]">{title}</h3>
      </div>
      {children}
    </motion.div>
  );
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[var(--surface-elevated)] border border-[var(--border)] rounded-xl p-3 shadow-xl text-xs">
      <p className="font-bold text-[var(--text)] mb-2">{label}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' && p.value <= 1 ? pctLabel(p.value) : p.value}
        </p>
      ))}
    </div>
  );
};

export default function DocumentEvaluation() {
  const { compareResult } = useDocumentContext();
  const rows = (compareResult?.results as Array<Record<string, any>>) ?? [];

  const metricsData = useMemo(() =>
    rows.map(row => ({
      model: row.algorithm ?? row.key,
      key: row.key,
      rouge1:   pct(row.metrics?.rouge1),
      rouge2:   pct(row.metrics?.rouge2),
      rougeL:   pct(row.metrics?.rougeL),
      bertscore: pct(row.metrics?.bertscore_f1),
      semantic:  pct(row.metrics?.semantic_similarity),
      bleu:      pct(row.metrics?.bleu),
      latency:   Number(row.processing_time ?? row.metrics?.processing_time ?? 0),
      wordCount: Number(row.word_count ?? 0),
      group:     row.group ?? '',
    })), [rows]);

  const radarData = useMemo(() => {
    const metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore', 'Semantic'];
    return metrics.map(m => {
      const entry: Record<string, any> = { metric: m };
      rows.forEach(row => {
        const key = row.key;
        const map: Record<string, string> = {
          'ROUGE-1': 'rouge1', 'ROUGE-2': 'rouge2', 'ROUGE-L': 'rougeL',
          'BERTScore': 'bertscore_f1', 'Semantic': 'semantic_similarity',
        };
        entry[key] = pct(row.metrics?.[map[m]]) * 100;
      });
      return entry;
    });
  }, [rows]);

  const compressionData = useMemo(() =>
    rows.map(row => ({
      model: row.algorithm ?? row.key,
      key: row.key,
      ratio: Number(row.length_ratio_percent ?? 0),
      words: Number(row.word_count ?? 0),
    })), [rows]);

  if (!compareResult || rows.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <BarChart3 className="w-16 h-16 text-[var(--text-faint)] mb-4" />
        <p className="text-[var(--text-muted)] font-medium">Chạy so sánh ở tab Compare trước.</p>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* Summary stats */}
      <motion.div
        initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-2 md:grid-cols-4 gap-3"
      >
        {[
          { label: 'Thuật toán', value: rows.length, color: 'text-blue-600 dark:text-blue-400' },
          {
            label: 'ROUGE-L tốt nhất',
            value: pctLabel(Math.max(...metricsData.map(d => d.rougeL))),
            color: 'text-emerald-600 dark:text-emerald-400',
          },
          {
            label: 'BERTScore cao nhất',
            value: pctLabel(Math.max(...metricsData.map(d => d.bertscore))),
            color: 'text-amber-600 dark:text-amber-400',
          },
          {
            label: 'Latency thấp nhất',
            value: `${Math.min(...metricsData.map(d => d.latency)).toFixed(2)}s`,
            color: 'text-violet-600 dark:text-violet-400',
          },
        ].map(s => (
          <div key={s.label} className="ui-card p-4">
            <p className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-faint)] mb-1">{s.label}</p>
            <p className={`text-2xl font-bold ${s.color}`}>{s.value}</p>
          </div>
        ))}
      </motion.div>

      {/* ROUGE grouped bar */}
      <Panel title="ROUGE-1 / ROUGE-2 / ROUGE-L so sánh" icon={BarChart3}>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={metricsData} margin={{ left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
              <XAxis dataKey="model" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
              <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text-muted)' }} />
              <Bar dataKey="rouge1" name="ROUGE-1" radius={[3, 3, 0, 0]}>
                {metricsData.map(d => <Cell key={d.key} fill={ALGO_COLORS[d.key] ?? '#64748b'} fillOpacity={0.6} />)}
              </Bar>
              <Bar dataKey="rouge2" name="ROUGE-2" radius={[3, 3, 0, 0]}>
                {metricsData.map(d => <Cell key={d.key} fill={ALGO_COLORS[d.key] ?? '#64748b'} fillOpacity={0.8} />)}
              </Bar>
              <Bar dataKey="rougeL" name="ROUGE-L" radius={[3, 3, 0, 0]}>
                {metricsData.map(d => <Cell key={d.key} fill={ALGO_COLORS[d.key] ?? '#64748b'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      {/* Radar + BERTScore */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <Panel title="Radar: Tất cả metrics" icon={TrendingUp}>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="var(--border)" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                {rows.map(row => (
                  <Radar
                    key={row.key}
                    name={row.algorithm ?? row.key}
                    dataKey={row.key}
                    stroke={ALGO_COLORS[row.key] ?? '#64748b'}
                    fill={ALGO_COLORS[row.key] ?? '#64748b'}
                    fillOpacity={0.15}
                  />
                ))}
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text-muted)' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="BERTScore F1 & Semantic Similarity" icon={Zap}>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metricsData} margin={{ left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                <XAxis dataKey="model" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text-muted)' }} />
                <Bar dataKey="bertscore" name="BERTScore F1" fill="#8b5cf6" radius={[3, 3, 0, 0]}>
                  {metricsData.map(d => <Cell key={d.key} fill={ALGO_COLORS[d.key] ?? '#8b5cf6'} />)}
                </Bar>
                <Bar dataKey="semantic" name="Semantic Sim" fill="#10b981" radius={[3, 3, 0, 0]} fillOpacity={0.7}>
                  {metricsData.map(d => <Cell key={d.key} fill={ALGO_COLORS[d.key] ?? '#10b981'} fillOpacity={0.6} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      {/* Latency scatter + Compression */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <Panel title="Latency (giây)" icon={Clock}>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metricsData} margin={{ left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                <XAxis dataKey="model" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} unit="s" />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="latency" name="Latency (s)" radius={[3, 3, 0, 0]}>
                  {metricsData.map(d => <Cell key={d.key} fill={ALGO_COLORS[d.key] ?? '#64748b'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Tỷ lệ nén (% so với gốc)" icon={FileText}>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={compressionData} margin={{ left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                <XAxis dataKey="model" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} unit="%" />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="ratio" name="Compression %" radius={[3, 3, 0, 0]}>
                  {compressionData.map(d => <Cell key={d.key} fill={ALGO_COLORS[d.key] ?? '#64748b'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      {/* Full comparison table */}
      <Panel title="Bảng so sánh chi tiết" icon={GitCompare}>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)] bg-[var(--surface-inset)]">
                {['Thuật toán', 'Nhóm', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore', 'Semantic', 'Latency', 'Số từ'].map(h => (
                  <th key={h} className="px-3 py-2.5 text-left font-semibold text-[var(--text-muted)] uppercase tracking-wide whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border-subtle)]">
              {metricsData.map(d => (
                <tr key={d.key} className="hover:bg-[var(--surface-inset)] transition">
                  <td className="px-3 py-2.5 font-semibold text-[var(--text)]">
                    <span className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full" style={{ background: ALGO_COLORS[d.key] ?? '#64748b' }} />
                      {d.model}
                    </span>
                  </td>
                  <td className="px-3 py-2.5">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                      d.group?.toLowerCase() === 'abstractive'
                        ? 'bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-300'
                        : 'bg-teal-50 dark:bg-teal-900/20 text-teal-700 dark:text-teal-300'
                    }`}>{d.group}</span>
                  </td>
                  <td className="px-3 py-2.5 text-blue-600 dark:text-blue-400 font-mono">{pctLabel(d.rouge1)}</td>
                  <td className="px-3 py-2.5 text-blue-600 dark:text-blue-400 font-mono">{pctLabel(d.rouge2)}</td>
                  <td className="px-3 py-2.5 text-emerald-600 dark:text-emerald-400 font-mono font-bold">{pctLabel(d.rougeL)}</td>
                  <td className="px-3 py-2.5 text-violet-600 dark:text-violet-400 font-mono">{pctLabel(d.bertscore)}</td>
                  <td className="px-3 py-2.5 text-amber-600 dark:text-amber-400 font-mono">{pctLabel(d.semantic)}</td>
                  <td className="px-3 py-2.5 text-[var(--text-secondary)] font-mono">{d.latency.toFixed(2)}s</td>
                  <td className="px-3 py-2.5 text-[var(--text-secondary)]">{d.wordCount}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
