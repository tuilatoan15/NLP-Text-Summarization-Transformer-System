import React, { useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Loader2, Sparkles, Hash, ExternalLink, AlertCircle } from 'lucide-react';
import { searchDocument } from '../../services/apiService';
import { useDocumentContext } from '../../context/DocumentContext';

const SAMPLE_QUERIES = [
  'Phương pháp nghiên cứu chính',
  'Kết quả thực nghiệm',
  'Kết luận và đề xuất',
  'Mô hình và thuật toán sử dụng',
];

function HighlightedText({ text, query }: { text: string; query: string }) {
  if (!query.trim()) return <span>{text}</span>;
  const terms = query.trim().split(/\s+/).filter(t => t.length > 2);
  if (!terms.length) return <span>{text}</span>;

  const regex = new RegExp(`(${terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
  const parts = text.split(regex);

  return (
    <span>
      {parts.map((part, i) =>
        regex.test(part) ? (
          <mark key={i} className="bg-yellow-200 dark:bg-yellow-900/60 text-yellow-900 dark:text-yellow-200 rounded px-0.5">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </span>
  );
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  let color = '#ef4444';
  if (pct >= 70) color = '#10b981';
  else if (pct >= 50) color = '#f59e0b';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-[var(--surface-inset)] overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-[10px] font-bold font-mono" style={{ color }}>{pct}%</span>
    </div>
  );
}

export default function DocumentSearchPage() {
  const { document } = useDocumentContext();
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<Record<string, any> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [topK, setTopK] = useState(5);
  const inputRef = useRef<HTMLInputElement>(null);

  if (!document) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <Search className="w-16 h-16 text-[var(--text-faint)] mb-4" />
        <p className="text-[var(--text-muted)] font-medium">Upload tài liệu trước để tìm kiếm ngữ nghĩa.</p>
      </div>
    );
  }

  async function onSearch(q?: string) {
    const searchQ = q ?? query;
    if (!searchQ.trim()) return;
    if (q) setQuery(q);
    setLoading(true);
    setError(null);
    try {
      const r = await searchDocument(document.document_id as string, searchQ, topK);
      setResult(r as Record<string, any>);
    } catch (e: any) {
      setError(e?.message ?? 'Tìm kiếm thất bại');
    } finally {
      setLoading(false);
    }
  }

  const results: Array<Record<string, any>> = result?.results ?? [];
  const backend: string = result?.retrieval_backend ?? '';

  return (
    <div className="space-y-5">
      {/* Search box */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="ui-card p-5">
        <div className="flex gap-3 mb-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-faint)]" />
            <input
              ref={inputRef}
              className="ui-input pl-9"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Nhập câu truy vấn ngữ nghĩa..."
              onKeyDown={e => e.key === 'Enter' && onSearch()}
            />
          </div>
          <select
            className="ui-input w-24 text-xs"
            value={topK}
            onChange={e => setTopK(Number(e.target.value))}
          >
            {[3, 5, 8, 10].map(n => <option key={n} value={n}>Top {n}</option>)}
          </select>
          <button
            type="button"
            className="ui-btn-primary px-5"
            onClick={() => onSearch()}
            disabled={loading || !query.trim()}
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
          </button>
        </div>

        {/* Sample queries */}
        <div className="flex flex-wrap gap-2">
          <span className="text-xs text-[var(--text-faint)] self-center">Gợi ý:</span>
          {SAMPLE_QUERIES.map(q => (
            <button
              key={q}
              type="button"
              onClick={() => onSearch(q)}
              className="text-xs px-3 py-1 rounded-full border border-[var(--border)] text-[var(--text-muted)] hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400 transition"
            >
              {q}
            </button>
          ))}
        </div>
      </motion.div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-xl px-4 py-3"
          >
            <AlertCircle size={15} /> {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results */}
      <AnimatePresence>
        {results.length > 0 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-blue-500" />
                <span className="text-sm font-bold text-[var(--text)]">
                  {results.length} kết quả
                </span>
                {backend && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-800">
                    {backend}
                  </span>
                )}
              </div>
            </div>

            {results.map((item, idx) => {
              const chunk = item.chunk ?? {};
              const text: string = item.highlight ?? chunk.text ?? chunk.content ?? '';
              const score: number = Number(item.score ?? 0);
              const page: number | null = chunk.page ?? chunk.metadata?.page ?? null;
              const section: string = chunk.section ?? chunk.parent_section ?? '';

              return (
                <motion.div
                  key={chunk.chunk_id ?? idx}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="ui-card p-4 hover:shadow-md transition-all"
                >
                  {/* Header */}
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="w-6 h-6 rounded-full bg-blue-600 text-white text-xs font-bold flex items-center justify-center shrink-0">
                        {idx + 1}
                      </span>
                      {section && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-[var(--surface-inset)] text-[var(--text-muted)] border border-[var(--border)]">
                          {section}
                        </span>
                      )}
                      {page != null && (
                        <span className="text-xs text-[var(--text-faint)] flex items-center gap-1">
                          <Hash size={11} /> Trang {page}
                        </span>
                      )}
                    </div>
                    <div className="w-32 shrink-0">
                      <ScoreBar score={score} />
                    </div>
                  </div>

                  {/* Text content with highlight */}
                  <p className="text-sm text-[var(--text-secondary)] leading-relaxed border-l-2 border-blue-200 dark:border-blue-800 pl-3">
                    <HighlightedText text={text} query={query} />
                  </p>

                  {/* Keywords */}
                  {(chunk.keywords?.length > 0) && (
                    <div className="flex flex-wrap gap-1 mt-3">
                      {(chunk.keywords as string[]).slice(0, 6).map((kw: string) => (
                        <span key={kw} className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--surface-inset)] text-[var(--text-faint)] border border-[var(--border)]">
                          {kw}
                        </span>
                      ))}
                    </div>
                  )}
                </motion.div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>

      {!loading && results.length === 0 && result && (
        <div className="text-center py-12 text-[var(--text-muted)]">
          <Search className="w-10 h-10 mx-auto mb-3 opacity-30" />
          <p>Không tìm thấy kết quả phù hợp.</p>
        </div>
      )}
    </div>
  );
}
