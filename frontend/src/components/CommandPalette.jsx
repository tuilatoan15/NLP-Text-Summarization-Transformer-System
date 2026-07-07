import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Loader2, FileText, Bot, MessageSquare, BarChart3, History } from 'lucide-react';
import { useApp } from '../context/AppContext';
import { searchDashboard } from '../services/apiService';

const TYPE_ICONS = {
  algorithm: Bot,
  benchmark_run: BarChart3,
  summarize_history: History,
  document: FileText,
  conversation: MessageSquare,
  leaderboard: BarChart3,
};

const MATCH_FIELD_KEYS = {
  source: 'searchMatchSource',
  reference: 'searchMatchReference',
  summary: 'searchMatchSummary',
  algorithm: 'searchMatchAlgorithm',
};

function highlightSnippet(snippet, query) {
  if (!snippet || !query?.trim()) return snippet;
  const needle = query.trim();
  const lower = snippet.toLowerCase();
  const idx = lower.indexOf(needle.toLowerCase());
  if (idx < 0) return snippet;
  return (
    <>
      {snippet.slice(0, idx)}
      <mark className="rounded bg-amber-200/80 px-0.5 text-[var(--text-primary)] dark:bg-amber-500/30">
        {snippet.slice(idx, idx + needle.length)}
      </mark>
      {snippet.slice(idx + needle.length)}
    </>
  );
}

function useDebouncedValue(value, delay = 300) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return debounced;
}

const CommandPalette = () => {
  const { commandPaletteOpen, setCommandPaletteOpen, t } = useApp();
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const debounced = useDebouncedValue(query, 280);

  useEffect(() => {
    if (!commandPaletteOpen) {
      setQuery('');
      setResults([]);
      setError(null);
    }
  }, [commandPaletteOpen]);

  useEffect(() => {
    if (!debounced || debounced.trim().length < 2) {
      setResults([]);
      return undefined;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    searchDashboard(debounced, 15)
      .then((data) => {
        if (!cancelled) setResults(data?.results || []);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [debounced]);

  const staticRoutes = useMemo(() => [
    { type: 'route', title: 'Dashboard', link: '/', keywords: 'overview bảng điều khiển' },
    { type: 'route', title: 'Tóm tắt / Summarize', link: '/summarize', keywords: 'playground summarize' },
    { type: 'route', title: 'Chat RAG', link: '/chat', keywords: 'chat rag hỏi đáp' },
    { type: 'route', title: 'So sánh mô hình', link: '/compare', keywords: 'compare rouge' },
    { type: 'route', title: 'Benchmark', link: '/benchmark', keywords: 'leaderboard benchmark' },
    { type: 'route', title: 'Analytics / Performance', link: '/analytics', keywords: 'analytics performance' },
    { type: 'route', title: 'Dataset Analytics', link: '/dataset-analytics', keywords: 'vietnews dataset thống kê' },
    { type: 'route', title: 'Cấu hình', link: '/settings', keywords: 'settings config' },
  ], []);

  const filteredRoutes = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return staticRoutes.slice(0, 5);
    return staticRoutes.filter(
      (r) => r.title.toLowerCase().includes(q) || r.keywords.includes(q),
    );
  }, [query, staticRoutes]);

  const handleSelect = useCallback((link) => {
    setCommandPaletteOpen(false);
    if (link) navigate(link);
  }, [navigate, setCommandPaletteOpen]);

  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setCommandPaletteOpen((v) => !v);
      }
      if (e.key === 'Escape') setCommandPaletteOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setCommandPaletteOpen]);

  if (!commandPaletteOpen) return null;

  const displayItems = query.trim().length >= 2 ? results : filteredRoutes;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-start justify-center pt-[12vh] px-4 bg-black/40 backdrop-blur-sm"
        onClick={() => setCommandPaletteOpen(false)}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: -8 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96, y: -8 }}
          className="w-full max-w-lg rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] shadow-2xl overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center gap-3 px-4 py-3 border-b border-[var(--border)]">
            <Search size={18} className="text-sky-500 shrink-0" />
            <input
              autoFocus
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t('searchPlaceholder')}
              className="flex-1 bg-transparent text-sm outline-none text-[var(--text-primary)] placeholder:text-[var(--text-faint)]"
            />
            {loading && <Loader2 size={16} className="animate-spin text-[var(--text-muted)]" />}
          </div>

          <div className="max-h-80 overflow-y-auto py-2">
            {error && (
              <p className="px-4 py-3 text-xs text-red-500">{error}</p>
            )}
            {displayItems.length === 0 && !loading && query.trim().length >= 2 && (
              <p className="px-4 py-6 text-center text-sm text-[var(--text-faint)]">Không có kết quả</p>
            )}
            {displayItems.map((item, i) => {
              const Icon = TYPE_ICONS[item.type] || Search;
              const link = item.link;
              const title = item.title;
              const subtitle = item.subtitle || item.keywords || '';
              const matchLabel = MATCH_FIELD_KEYS[item.match_field]
                ? t(MATCH_FIELD_KEYS[item.match_field])
                : null;
              const isHistoryHit = item.type === 'summarize_history' || item.type === 'benchmark_run';
              return (
                <button
                  key={`${item.type || 'route'}-${item.id || item.result_id || title}-${i}`}
                  type="button"
                  onClick={() => handleSelect(link)}
                  className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-[var(--bg-muted)] transition-colors"
                >
                  <Icon size={16} className="text-sky-500 shrink-0" />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold text-[var(--text-primary)] truncate">{title}</p>
                    {matchLabel && (
                      <p className="text-[10px] font-medium uppercase tracking-wide text-sky-600/90 dark:text-sky-400/90">
                        {matchLabel}
                      </p>
                    )}
                    {subtitle && (
                      <p className="text-[11px] text-[var(--text-muted)] truncate">
                        {isHistoryHit ? highlightSnippet(subtitle, debounced) : subtitle}
                      </p>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default CommandPalette;
