import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  GitCompare, Loader2, FileText, Zap, Layers3,
  ChevronDown, ChevronUp, CheckCircle2, AlertTriangle,
} from 'lucide-react';
import { compareDocumentSummaries, hierarchicalSummarize } from '../../services/apiService';
import { useDocumentContext } from '../../context/DocumentContext';

const ALGO_META: Record<string, { label: string; color: string; group: string }> = {
  textrank:  { label: 'TextRank',  color: '#14b8a6', group: 'Extractive' },
  lexrank:   { label: 'LexRank',   color: '#38bdf8', group: 'Extractive' },
  lsa:       { label: 'LSA',       color: '#84cc16', group: 'Extractive' },
  tfidf:     { label: 'TF-IDF',   color: '#a78bfa', group: 'Extractive' },
  vit5:      { label: 'ViT5',      color: '#f59e0b', group: 'Abstractive' },
  mt5:       { label: 'mT5',       color: '#e879f9', group: 'Abstractive' },
  bartpho:   { label: 'BARTPho',   color: '#fb7185', group: 'Abstractive' },
};

const pct = (v: any) => `${Math.round((Number(v) || 0) * 100)}%`;

function AlgoChip({ k, selected, onClick }: { k: string; selected: boolean; onClick: () => void }) {
  const m = ALGO_META[k] ?? { label: k, color: '#64748b', group: 'Other' };
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
        selected
          ? 'text-white border-transparent shadow-sm'
          : 'border-[var(--border)] text-[var(--text-muted)] hover:border-blue-300 dark:hover:border-blue-700'
      }`}
      style={selected ? { background: m.color } : {}}
    >
      <span className="w-2 h-2 rounded-full" style={{ background: m.color }} />
      {m.label}
      <span className="text-[10px] opacity-70">{m.group}</span>
    </button>
  );
}

function SummaryCard({
  label, summary, group, metrics, color, expanded, onToggle,
}: {
  label: string; summary: string; group: string;
  metrics?: Record<string, any>; color: string;
  expanded: boolean; onToggle: () => void;
}) {
  const isAbstractive = group?.toLowerCase() === 'abstractive';
  return (
    <motion.div
      layout
      className="ui-card overflow-hidden"
      style={{ borderTopWidth: 3, borderTopColor: color }}
    >
      <div className="px-4 pt-4 pb-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
            <span className="text-sm font-bold text-[var(--text)]">{label}</span>
            <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${
              isAbstractive
                ? 'bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-300'
                : 'bg-teal-50 dark:bg-teal-900/20 text-teal-700 dark:text-teal-300'
            }`}>
              {group}
            </span>
          </div>
          <button type="button" onClick={onToggle} className="text-[var(--text-faint)] hover:text-[var(--text)] transition">
            {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          </button>
        </div>

        <p className={`text-sm text-[var(--text-secondary)] leading-relaxed border-l-2 pl-3 ${expanded ? '' : 'line-clamp-4'}`}
           style={{ borderColor: color }}>
          {summary || '(Trống)'}
        </p>
      </div>

      {metrics && (
        <div className="px-4 pb-3 border-t border-[var(--border)] mt-1 pt-3 flex flex-wrap gap-3">
          {[
            { k: 'rouge1',           label: 'R-1' },
            { k: 'rouge2',           label: 'R-2' },
            { k: 'rougeL',           label: 'R-L' },
            { k: 'bertscore_f1',     label: 'BERT-F' },
            { k: 'semantic_similarity', label: 'Semantic' },
          ].map(({ k, label }) =>
            metrics[k] != null ? (
              <div key={k} className="text-center">
                <p className="text-[10px] text-[var(--text-faint)] uppercase tracking-wide">{label}</p>
                <p className="text-sm font-bold" style={{ color }}>{pct(metrics[k])}</p>
              </div>
            ) : null
          )}
          {metrics.processing_time != null && (
            <div className="text-center">
              <p className="text-[10px] text-[var(--text-faint)] uppercase tracking-wide">Time</p>
              <p className="text-sm font-bold text-[var(--text-muted)]">{Number(metrics.processing_time).toFixed(2)}s</p>
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}

export default function DocumentCompare() {
  const { document, setCompareResult } = useDocumentContext();
  const [selected, setSelected] = useState<string[]>(['textrank', 'lexrank', 'lsa', 'vit5']);
  const [reference, setReference] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingHier, setLoadingHier] = useState(false);
  const [results, setResults] = useState<Array<Record<string, any>>>([]);
  const [hierarchical, setHierarchical] = useState<Record<string, any> | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);

  if (!document) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <GitCompare className="w-16 h-16 text-[var(--text-faint)] mb-4" />
        <p className="text-[var(--text-muted)] font-medium">Upload tài liệu trước để so sánh.</p>
      </div>
    );
  }

  const toggle = (k: string) =>
    setSelected(c => c.includes(k) ? c.filter(x => x !== k) : [...c, k]);

  async function runCompare() {
    if (!selected.length) return;
    setLoading(true);
    setError(null);
    try {
      const result = await compareDocumentSummaries(document.document_id as string, {
        reference: reference || null,
        algorithms: selected,
        targetLengthRatio: 35,
        extractiveSentences: 5,
        maxAbstractiveLength: 180,
      });
      const rows = (result as any)?.results ?? [];
      setResults(rows);
      setCompareResult(result as Record<string, unknown>);
      const initExp: Record<string, boolean> = {};
      rows.forEach((r: any) => { initExp[r.key] = false; });
      setExpanded(initExp);
    } catch (e: any) {
      setError(e?.message ?? 'So sánh thất bại');
    } finally {
      setLoading(false);
    }
  }

  async function runHierarchical() {
    setLoadingHier(true);
    try {
      const result = await hierarchicalSummarize(document.document_id as string, 'vit5', true);
      setHierarchical(result as Record<string, any>);
    } finally {
      setLoadingHier(false);
    }
  }

  const extractiveRows = results.filter(r => r.group?.toLowerCase() === 'extractive');
  const abstractiveRows = results.filter(r => r.group?.toLowerCase() === 'abstractive');

  return (
    <div className="space-y-5">
      {/* Config panel */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="ui-card p-5">
        <h3 className="text-sm font-bold text-[var(--text)] mb-4 flex items-center gap-2">
          <GitCompare className="w-4 h-4 text-blue-500" />
          Cấu hình so sánh
        </h3>

        <div className="space-y-4">
          <div>
            <p className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-2">
              Extractive
            </p>
            <div className="flex flex-wrap gap-2">
              {['textrank', 'lexrank', 'lsa', 'tfidf'].map(k => (
                <AlgoChip key={k} k={k} selected={selected.includes(k)} onClick={() => toggle(k)} />
              ))}
            </div>
          </div>
          <div>
            <p className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-2">
              Abstractive
            </p>
            <div className="flex flex-wrap gap-2">
              {['vit5', 'mt5', 'bartpho'].map(k => (
                <AlgoChip key={k} k={k} selected={selected.includes(k)} onClick={() => toggle(k)} />
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-1.5">
              Reference summary (tùy chọn)
            </label>
            <textarea
              className="ui-textarea min-h-20 text-sm"
              value={reference}
              onChange={e => setReference(e.target.value)}
              placeholder="Dán bản tóm tắt tham chiếu để tính ROUGE/BLEU..."
            />
          </div>

          <div className="flex gap-3">
            <button
              type="button"
              className="ui-btn-primary flex-1"
              disabled={loading || !selected.length}
              onClick={runCompare}
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <GitCompare className="w-4 h-4" />}
              {loading ? 'Đang so sánh...' : `So sánh ${selected.length} thuật toán`}
            </button>
            <button
              type="button"
              className="ui-btn-secondary"
              disabled={loadingHier}
              onClick={runHierarchical}
            >
              {loadingHier ? <Loader2 className="w-4 h-4 animate-spin" /> : <Layers3 className="w-4 h-4" />}
              Hierarchical
            </button>
          </div>
        </div>
      </motion.div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-xl px-4 py-3"
          >
            <AlertTriangle size={16} /> {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Hierarchical result */}
      <AnimatePresence>
        {hierarchical && (
          <motion.div
            initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            className="ui-card p-5 border-t-4 border-t-purple-500"
          >
            <div className="flex items-center gap-2 mb-3">
              <Layers3 className="w-4 h-4 text-purple-500" />
              <h3 className="text-sm font-bold">Hierarchical Summary (Map-Reduce)</h3>
              <CheckCircle2 className="w-4 h-4 text-emerald-500 ml-auto" />
            </div>
            <p className="text-sm text-[var(--text-secondary)] leading-relaxed border-l-4 border-purple-400 pl-3">
              {String(hierarchical.global_summary ?? '')}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results: 2-column layout Extractive | Abstractive */}
      {results.length > 0 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="w-4 h-4 text-emerald-500" />
            <h3 className="text-sm font-bold text-[var(--text)]">
              Kết quả — {results.length} thuật toán
            </h3>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {/* Extractive column */}
            {extractiveRows.length > 0 && (
              <div className="space-y-3">
                <p className="text-xs font-bold text-teal-600 dark:text-teal-400 uppercase tracking-wider flex items-center gap-1.5">
                  <FileText size={12} /> Extractive ({extractiveRows.length})
                </p>
                {extractiveRows.map(row => {
                  const m = ALGO_META[row.key] ?? { label: row.algorithm, color: '#64748b', group: 'Extractive' };
                  return (
                    <SummaryCard
                      key={row.key}
                      label={m.label}
                      summary={row.summary}
                      group={row.group}
                      metrics={row.metrics}
                      color={m.color}
                      expanded={!!expanded[row.key]}
                      onToggle={() => setExpanded(e => ({ ...e, [row.key]: !e[row.key] }))}
                    />
                  );
                })}
              </div>
            )}

            {/* Abstractive column */}
            {abstractiveRows.length > 0 && (
              <div className="space-y-3">
                <p className="text-xs font-bold text-amber-600 dark:text-amber-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Zap size={12} /> Abstractive ({abstractiveRows.length})
                </p>
                {abstractiveRows.map(row => {
                  const m = ALGO_META[row.key] ?? { label: row.algorithm, color: '#64748b', group: 'Abstractive' };
                  return (
                    <SummaryCard
                      key={row.key}
                      label={m.label}
                      summary={row.summary}
                      group={row.group}
                      metrics={row.metrics}
                      color={m.color}
                      expanded={!!expanded[row.key]}
                      onToggle={() => setExpanded(e => ({ ...e, [row.key]: !e[row.key] }))}
                    />
                  );
                })}
              </div>
            )}
          </div>
        </motion.div>
      )}
    </div>
  );
}
