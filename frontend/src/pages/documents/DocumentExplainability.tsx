import React, { memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, BarChart3, Network, CheckCircle2, XCircle, Info } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';
import { useDocumentContext } from '../../context/DocumentContext';
import { useExplainabilityQuery } from '../../hooks/useApiQueries';
import { useCacheHitLogger } from '../../hooks/useCacheHitLogger';

const ALGO_OPTIONS = [
  { key: 'textrank', label: 'TextRank', color: '#14b8a6' },
  { key: 'lexrank',  label: 'LexRank',  color: '#38bdf8' },
  { key: 'lsa',     label: 'LSA',       color: '#84cc16' },
  { key: 'tfidf',   label: 'TF-IDF',   color: '#a78bfa' },
];

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

function DocumentExplainability() {
  const { document } = useDocumentContext();
  const [algorithm, setAlgorithm] = React.useState('textrank');
  const docId = document?.document_id as string | undefined;

  const { data: payload, isLoading, isFetching } = useExplainabilityQuery(docId, algorithm);
  useCacheHitLogger(`explainability ${algorithm}`, payload, isFetching);

  const current = ALGO_OPTIONS.find(a => a.key === algorithm)!;

  if (!document) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <Brain className="w-16 h-16 text-[var(--text-faint)] mb-4" />
        <p className="text-[var(--text-muted)] font-medium">Upload tài liệu trước để xem explainability.</p>
      </div>
    );
  }

  const graph = (payload?.ranking_graph as Record<string, any>) ?? {};
  const nodes: Array<Record<string, any>> = graph.nodes ?? [];
  const keywords: Array<{ term: string; score: number }> = payload?.keywords ?? [];
  const selectedCount = nodes.filter(n => n.selected).length;

  const barData = nodes
    .slice(0, 15)
    .map((n, i) => ({
      name: `S${n.index ?? i + 1}`,
      score: Number(n.rank_score ?? n.score ?? 0),
      selected: !!n.selected,
      text: String(n.label ?? '').slice(0, 80),
    }));

  const kwData = keywords.slice(0, 12).map(k => ({
    term: k.term,
    score: Number(k.score ?? 0),
  }));

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap gap-2">
        {ALGO_OPTIONS.map(a => (
          <button
            key={a.key}
            type="button"
            onClick={() => setAlgorithm(a.key)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold border transition-all ${
              algorithm === a.key
                ? 'text-white border-transparent shadow-sm'
                : 'border-[var(--border)] text-[var(--text-muted)] hover:border-blue-300'
            }`}
            style={algorithm === a.key ? { background: a.color } : {}}
          >
            <span className="w-2 h-2 rounded-full" style={{ background: a.color }} />
            {a.label}
          </button>
        ))}
        {isLoading && (
          <span className="text-xs text-[var(--text-muted)] self-center animate-pulse">Đang tải...</span>
        )}
      </div>

      {nodes.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="grid grid-cols-3 gap-3"
        >
          {[
            { label: 'Tổng câu', value: nodes.length, color: 'text-blue-600 dark:text-blue-400' },
            { label: 'Câu được chọn', value: selectedCount, color: 'text-emerald-600 dark:text-emerald-400' },
            { label: 'Tỷ lệ chọn', value: `${Math.round((selectedCount / nodes.length) * 100)}%`, color: 'text-amber-600 dark:text-amber-400' },
          ].map(s => (
            <div key={s.label} className="ui-card p-4">
              <p className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-faint)] mb-1">{s.label}</p>
              <p className={`text-2xl font-bold ${s.color}`}>{s.value}</p>
            </div>
          ))}
        </motion.div>
      )}

      {barData.length > 0 && (
        <Panel title={`Sentence Ranking — ${current.label}`} icon={BarChart3}>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} domain={[0, 'auto']} />
                <Tooltip
                  content={({ active, payload: tipPayload }) => {
                    if (!active || !tipPayload?.length) return null;
                    const d = tipPayload[0].payload;
                    return (
                      <div className="bg-[var(--surface-elevated)] border border-[var(--border)] rounded-xl p-3 shadow-xl max-w-xs text-xs">
                        <p className="font-bold text-[var(--text)] mb-1">{d.name}</p>
                        <p style={{ color: current.color }}>Score: {d.score.toFixed(4)}</p>
                        <p className={d.selected ? 'text-emerald-500' : 'text-[var(--text-faint)]'}>
                          {d.selected ? '✓ Được chọn' : '✗ Bị loại'}
                        </p>
                        <p className="text-[var(--text-muted)] mt-1 line-clamp-3">{d.text}...</p>
                      </div>
                    );
                  }}
                />
                <Bar dataKey="score" name="Score" radius={[3, 3, 0, 0]}>
                  {barData.map((d, i) => (
                    <Cell
                      key={i}
                      fill={d.selected ? current.color : 'var(--border)'}
                      opacity={d.selected ? 1 : 0.5}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}

      {kwData.length > 0 && (
        <Panel title="Keyword Importance" icon={Network}>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={kwData} layout="vertical" margin={{ left: 0, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--border)" />
                <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} domain={[0, 1]} />
                <YAxis type="category" dataKey="term" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} width={80} />
                <Tooltip
                  contentStyle={{ background: 'var(--surface-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
                  formatter={(v: number) => [`${(v * 100).toFixed(1)}%`]}
                />
                <Bar dataKey="score" name="Importance" radius={[0, 3, 3, 0]} fill={current.color} fillOpacity={0.85} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}

      {nodes.length > 0 && (
        <Panel title="Chi tiết câu — Selected vs Rejected" icon={Brain}>
          <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
            {nodes.map((node, i) => (
              <motion.div
                key={node.id ?? i}
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                transition={{ delay: i * 0.02 }}
                className={`flex gap-3 p-3 rounded-xl border text-xs transition-all ${
                  node.selected
                    ? 'border-emerald-200 dark:border-emerald-800 bg-emerald-50/50 dark:bg-emerald-900/10'
                    : 'border-[var(--border)] bg-[var(--surface-inset)] opacity-60'
                }`}
              >
                <div className="shrink-0 mt-0.5">
                  {node.selected
                    ? <CheckCircle2 size={14} className="text-emerald-500" />
                    : <XCircle size={14} className="text-[var(--text-faint)]" />}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-bold text-[var(--text-muted)]">#{node.index ?? i + 1}</span>
                    <span
                      className="font-mono px-1.5 py-0.5 rounded text-[10px]"
                      style={{ background: `${current.color}20`, color: current.color }}
                    >
                      {Number(node.rank_score ?? node.score ?? 0).toFixed(4)}
                    </span>
                  </div>
                  <p className="text-[var(--text-secondary)] line-clamp-2 leading-relaxed">
                    {node.label ?? ''}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </Panel>
      )}

      {!payload && !isLoading && (
        <div className="flex items-center gap-2 text-sm text-[var(--text-muted)] bg-[var(--surface-inset)] rounded-xl px-4 py-3">
          <Info size={15} />
          Không có dữ liệu explainability cho thuật toán này.
        </div>
      )}
    </div>
  );
}

export default memo(DocumentExplainability);
