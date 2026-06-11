import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Play, RefreshCcw, Check, Loader2, Clock, SlidersHorizontal, Activity, Terminal, AlertCircle, FileText, UploadCloud, BookOpen, Sliders
} from 'lucide-react';
import { useApp } from '../context/AppContext';

const getAPI = () => import.meta.env.VITE_API_BASE || 'http://localhost:8000';

const ALGORITHMS = [
  { key: 'textrank', name: 'TextRank', group: 'extractive', color: '#14b8a6' },
  { key: 'lexrank', name: 'LexRank', group: 'extractive', color: '#38bdf8' },
  { key: 'lsa', name: 'LSA', group: 'extractive', color: '#84cc16' },
  { key: 'vit5', name: 'ViT5', group: 'abstractive', color: '#f59e0b' },
  { key: 'mt5', name: 'mT5', group: 'abstractive', color: '#e879f9' },
  { key: 'bartpho', name: 'BARTPho', group: 'abstractive', color: '#fb7185' },
];

const SAMPLE_TEXT = `Tập đoàn Điện lực Việt Nam cho biết nhu cầu tiêu thụ điện trong mùa nắng nóng tiếp tục tăng cao tại nhiều địa phương. Các nhà máy thủy điện ở miền Bắc được yêu cầu vận hành thận trọng do mực nước một số hồ chứa chưa phục hồi hoàn toàn.`;

const STATUS = {
  idle: 'idle',
  running: 'running',
  done: 'done',
  error: 'error',
};

function pct(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function metric(row, key) {
  return Number(row?.metrics?.[key] ?? row?.[key] ?? 0);
}

function byKey(key) {
  return ALGORITHMS.find(item => item.key === key) || { key, name: key, group: 'extractive', color: '#64748b' };
}

function initialRunState(keys) {
  return Object.fromEntries(keys.map(key => [key, { status: STATUS.idle, result: null, error: null }]));
}

async function consumeCompareStream(response, onEvent) {
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `HTTP ${response.status}`);
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error('Streaming not supported by browser');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split('\n\n');
    buffer = chunks.pop() || '';
    for (const chunk of chunks) {
      for (const line of chunk.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        onEvent(JSON.parse(trimmed.slice(5).trim()));
      }
    }
  }
}

const AlgorithmSelector = ({ selected, setSelected, disabled }) => {
  const { t } = useApp();
  const groupLabel = (group) => (group === 'abstractive' ? t('groupAbstractive') : t('groupExtractive'));

  const toggle = (key) => {
    if (disabled) return;
    setSelected(current =>
      current.includes(key) ? current.filter(item => item !== key) : [...current, key]
    );
  };

  return (
    <div className="space-y-4">
      {['extractive', 'abstractive'].map(group => (
        <section key={group} className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              {groupLabel(group)}
            </h3>
            <span className="text-xs text-[var(--text-faint)]">
              {selected.filter(key => byKey(key).group === group).length}/3
            </span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {ALGORITHMS.filter(item => item.group === group).map(item => {
              const isSelected = selected.includes(item.key);
              return (
                <button
                  key={item.key}
                  type="button"
                  disabled={disabled}
                  onClick={() => toggle(item.key)}
                  className={`flex flex-col items-center justify-center p-2.5 rounded-xl border text-center transition cursor-pointer ${
                    isSelected
                      ? 'bg-blue-50 dark:bg-blue-950/20 border-blue-400 dark:border-blue-800 text-blue-600 dark:text-blue-400 font-bold'
                      : 'bg-[var(--surface-elevated)] border-[var(--border)] text-[var(--text-muted)] hover:border-blue-300 dark:hover:border-blue-700'
                  } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <span className="w-2.5 h-2.5 rounded-full mb-1" style={{ background: item.color }} />
                  <span className="text-xs truncate w-full">{item.name}</span>
                </button>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
};

const StatusBadge = ({ status }) => {
  const { t } = useApp();
  const map = {
    [STATUS.idle]: { label: t('statusIdle'), className: 'bg-gray-100 dark:bg-slate-800 text-gray-600 dark:text-slate-400' },
    [STATUS.running]: { label: t('statusRunning'), className: 'bg-blue-100 dark:bg-blue-950/50 text-blue-700 dark:text-blue-300 algo-badge-pulse' },
    [STATUS.done]: { label: t('statusDone'), className: 'bg-emerald-100 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 font-bold' },
    [STATUS.error]: { label: t('statusError'), className: 'bg-red-100 dark:bg-red-950/40 text-red-700 dark:text-red-300 font-bold' },
  };
  const item = map[status] || map[STATUS.idle];
  return (
    <span className={`ui-badge text-[10px] ${item.className}`}>
      {status === STATUS.running && <Loader2 size={10} className="animate-spin mr-1" />}
      {item.label}
    </span>
  );
};

// Motion variants for staggered elements
const itemVariants = {
  hidden: { opacity: 0, y: 12, scale: 0.98 },
  show: { 
    opacity: 1, 
    y: 0, 
    scale: 1,
    transition: { type: 'spring', stiffness: 350, damping: 25 }
  }
};

const AlgorithmCard = ({ algoKey, state, rank }) => {
  const { t } = useApp();
  const meta = byKey(algoKey);
  const row = state.result;
  const isRunning = state.status === STATUS.running;
  const isDone = state.status === STATUS.done;

  return (
    <motion.article
      variants={itemVariants}
      animate={isRunning ? { scale: 1.015, borderColor: 'var(--accent)' } : { scale: 1 }}
      className={`algo-card ui-card p-4 transition-all duration-300 ${
        isRunning ? 'algo-card-running border-blue-300 dark:border-blue-600 shadow-md shadow-blue-100 dark:shadow-blue-900/30' : ''
      } ${isDone ? 'algo-card-done' : ''}`}
      style={{ '--algo-color': meta.color }}
    >
      <header className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className={`w-3 h-3 rounded-full shrink-0 ${isRunning ? 'algo-dot-pulse animate-pulse' : ''}`}
            style={{ background: meta.color }}
          />
          <div className="min-w-0">
            <h3 className="font-bold text-sm text-[var(--text)] truncate">{meta.name}</h3>
            <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider font-semibold">{meta.group}</p>
          </div>
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <StatusBadge status={state.status} />
          {rank != null && isDone && (
            <span className="text-xs text-amber-600 dark:text-amber-400 font-bold bg-amber-50 dark:bg-amber-950/20 border border-amber-200/50 dark:border-amber-800/40 rounded px-1.5 py-0.5">
              {t('pgRank', { rank })}
            </span>
          )}
        </div>
      </header>

      {isRunning && (
        <div className="mb-3 rounded-xl bg-blue-50 dark:bg-blue-950/40 border border-blue-100 dark:border-blue-900 px-3 py-2 text-xs text-blue-800 dark:text-blue-200 flex items-center gap-2">
          <Loader2 size={13} className="animate-spin shrink-0" />
          <span>{t('pgSummarizing')}</span>
        </div>
      )}

      {state.status === STATUS.idle && (
        <p className="text-xs text-[var(--text-faint)] italic">{t('pgClickRun')}</p>
      )}

      {state.status === STATUS.error && (
        <p className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/30 rounded-xl p-3 border border-red-200 dark:border-red-900">
          {state.error || t('pgAlgoError')}
        </p>
      )}

      {isDone && row && (
        <div className="space-y-3 algo-summary-reveal">
          <p className="text-sm text-[var(--text-secondary)] leading-relaxed border-l-4 pl-3 py-0.5 whitespace-pre-wrap font-sans" style={{ borderColor: meta.color }}>
            {row.summary || '—'}
          </p>
          <div className="flex flex-wrap gap-1.5 text-[10px] font-semibold">
            <span className="px-2 py-1 rounded-lg bg-[var(--surface-inset)] text-[var(--text-secondary)] flex items-center gap-1">
              <Clock size={11} />
              {(row.processing_time ?? row.time_seconds ?? 0).toFixed(2)}s
            </span>
            <span className="px-2 py-1 rounded-lg bg-[var(--surface-inset)] text-[var(--text-secondary)]">
              ROUGE-L {pct(metric(row, 'rougeL'))}
            </span>
            <span className="px-2 py-1 rounded-lg bg-[var(--surface-inset)] text-[var(--text-secondary)]">
              BERT {pct(metric(row, 'bertscore_f1'))}
            </span>
            <span className="px-2 py-1 rounded-lg bg-[var(--surface-inset)] text-[var(--text-secondary)]">
              {t('pgWords', { count: row.word_count ?? 0 })}
            </span>
            {row.length_ratio_percent != null && (
              <span className="px-2 py-1 rounded-lg bg-indigo-50 dark:bg-indigo-950/40 text-indigo-800 dark:text-indigo-300 font-bold border border-indigo-200/30 dark:border-indigo-900/30">
                {t('pgLengthOfSource', { pct: row.length_ratio_percent })}
              </span>
            )}
          </div>
          {row.warning_badge && (
            <p className="text-[10px] text-amber-700 dark:text-amber-400 flex items-center gap-1 bg-amber-50 dark:bg-amber-950/10 border border-amber-200/40 dark:border-amber-800/40 px-2 py-1 rounded-lg font-bold w-fit">
              <AlertCircle size={12} /> {row.warning_badge}
            </p>
          )}
        </div>
      )}
    </motion.article>
  );
};

const RunProgress = ({ runningKey, completed, total, loading }) => {
  const { t } = useApp();
  const pctDone = total ? Math.round((completed / total) * 100) : 0;
  const current = runningKey ? byKey(runningKey).name : null;

  return (
    <div className="rounded-2xl border border-blue-200 dark:border-blue-800 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 p-4 space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-bold text-blue-900 dark:text-blue-100">
          {loading ? <Loader2 size={16} className="animate-spin" /> : <Activity size={16} className="text-blue-500" />}
          {loading
            ? (current ? t('pgProgressRunning', { name: current }) : t('pgProgressInit'))
            : t('pgProgressReady')}
        </div>
        <span className="text-xs font-bold text-blue-700 dark:text-blue-300 bg-blue-100 dark:bg-blue-900/40 px-2 py-0.5 rounded-full">{completed}/{total}</span>
      </div>
      <div className="h-2 rounded-full bg-blue-100 dark:bg-blue-900/50 overflow-hidden">
        <div
          className="h-full rounded-full bg-blue-600 dark:bg-blue-500 transition-all duration-500 ease-out algo-progress-bar"
          style={{ width: `${loading ? Math.max(pctDone, runningKey ? 8 : 4) : 0}%` }}
        />
      </div>
      <p className="text-[11px] text-blue-800/80 dark:text-blue-200/70">
        {loading ? t('pgProgressHint') : t('pgProgressHintIdle')}
      </p>
    </div>
  );
};

const ComparisonTable = ({ rows }) => {
  const { t } = useApp();
  if (!rows.length) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15 }}
      className="ui-card overflow-hidden"
    >
      <div className="border-b border-[var(--border)] px-4 py-3 bg-[var(--surface-inset)] flex items-center gap-2">
        <FileText size={15} className="text-blue-500" />
        <h3 className="text-xs font-bold text-[var(--text)] uppercase tracking-wider">{t('pgMetricsTable')}</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="ui-table-head border-b">
            <tr>
              <th className="px-4 py-3 text-left">{t('colModel')}</th>
              <th className="px-4 py-3">{t('colRouge1')}</th>
              <th className="px-4 py-3">{t('colRouge2')}</th>
              <th className="px-4 py-3">{t('colRougeL')}</th>
              <th className="px-4 py-3">{t('colBert')}</th>
              <th className="px-4 py-3">{t('colLength')}</th>
              <th className="px-4 py-3">{t('colTime')}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-subtle)] font-medium">
            {rows.map(row => (
              <tr key={row.key} className="ui-table-row">
                <td className="px-4 py-3 font-semibold text-[var(--text)]">
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: byKey(row.key).color }} />
                    {row.algorithm}
                  </span>
                </td>
                <td className="px-4 py-3 text-center text-[var(--text-secondary)] font-mono">{pct(metric(row, 'rouge1'))}</td>
                <td className="px-4 py-3 text-center text-[var(--text-secondary)] font-mono">{pct(metric(row, 'rouge2'))}</td>
                <td className="px-4 py-3 text-center text-[var(--text-secondary)] font-mono">{pct(metric(row, 'rougeL'))}</td>
                <td className="px-4 py-3 text-center text-[var(--text-secondary)] font-mono">{pct(metric(row, 'bertscore_f1'))}</td>
                <td className="px-4 py-3 text-center text-[var(--text-secondary)] font-mono">
                  {row.length_ratio_percent != null ? `${row.length_ratio_percent}%` : '—'}
                </td>
                <td className="px-4 py-3 text-center text-[var(--text-secondary)] font-mono">
                  {(row.processing_time ?? 0).toFixed(2)}s
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
};

const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.05 }
  }
};

const Playground = () => {
  const { t, addNotification } = useApp();
  const [text, setText] = useState(SAMPLE_TEXT);
  const [reference, setReference] = useState('');
  const [files, setFiles] = useState([]);
  const [selected, setSelected] = useState(ALGORITHMS.map(item => item.key));
  const [summaryLength, setSummaryLength] = useState('auto');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [runState, setRunState] = useState({});
  const [runningKey, setRunningKey] = useState(null);
  const [completedCount, setCompletedCount] = useState(0);

  const rows = useMemo(() => result?.results || [], [result]);
  
  const rankByKey = useMemo(() => {
    const map = {};
    (result?.ranking || []).forEach(item => { map[item.key] = item.rank; });
    return map;
  }, [result]);

  const orderedKeys = useMemo(() => {
    const fromResult = rows.map(r => r.key);
    if (fromResult.length) return fromResult;
    return selected;
  }, [rows, selected]);

  const sourceWordCount = useMemo(() => {
    if (!text.trim()) return 0;
    return text.trim().split(/\s+/).filter(Boolean).length;
  }, [text]);

  // Dynamic values that handle either uploaded files or text inputs seamlessly
  const displaySourceWords = useMemo(() => {
    if (files.length > 0 && !text.trim()) {
      return result?.meta?.input_words || '—';
    }
    return sourceWordCount || '—';
  }, [files, sourceWordCount, result, text]);

  const displayTargetWords = useMemo(() => {
    if (files.length > 0 && !text.trim()) {
      return result?.meta?.target_words || '—';
    }
    let mode = summaryLength;
    if (mode === 'auto') {
      if (sourceWordCount < 500) mode = 'short';
      else if (sourceWordCount <= 3000) mode = 'standard';
      else mode = 'detailed';
    }
    if (mode === 'short') return '50 - 80';
    if (mode === 'standard') return '100 - 150';
    if (mode === 'detailed') return '200 - 300';
    return '—';
  }, [files, summaryLength, sourceWordCount, result, text]);

  const handleFileChange = async (e) => {
    const selectedFiles = Array.from(e.target.files || []);
    setFiles(selectedFiles);
    setError('');

    if (selectedFiles.length > 0) {
      const file = selectedFiles[0];
      const suffix = file.name.split('.').pop()?.toLowerCase();
      
      // If it is standard txt/md, read client side
      if (suffix === 'txt' || suffix === 'md') {
        const reader = new FileReader();
        reader.onload = (event) => {
          if (event.target?.result) {
            setText(event.target.result);
          }
        };
        reader.readAsText(file);
      } else {
        // PDF, Word, Doc, Docx: Extract from backend immediately
        setLoading(true);
        try {
          const form = new FormData();
          form.append('files', file);
          const API = getAPI();
          const response = await fetch(`${API}/summarize/files/extract`, { method: 'POST', body: form });
          if (!response.ok) {
            const body = await response.text();
            throw new Error(body || `HTTP ${response.status}`);
          }
          const { text: extractedText } = await response.json();
          setText(extractedText);
          
          addNotification({
            title: t('notifyFileExtracted', 'Đã trích xuất thành công'),
            message: t('notifyFileExtractedMsg', 'Nội dung tài liệu đã được tải vào khung Văn bản.'),
            type: 'success',
            link: '/playground',
            showBrowser: false,
          });
        } catch (err) {
          setError(err.message || String(err));
        } finally {
          setLoading(false);
        }
      }
    }
  };

  function applyBatchResults(data, keys) {
    const next = { ...initialRunState(keys) };
    for (const row of data.results || []) {
      next[row.key] = { status: STATUS.done, result: row, error: null };
    }
    setRunState(next);
    setCompletedCount((data.results || []).length);
    setRunningKey(null);
    setResult(data);
  }

  async function runTextStream(textToUse = text, docMetadata = null) {
    setRunState(initialRunState(selected));
    setCompletedCount(0);
    setRunningKey(null);
    setResult(null);

    const API = getAPI();
    const response = await fetch(`${API}/summarize/compare/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: textToUse,
        reference: reference || null,
        algorithms: selected,
        summary_length: summaryLength,
        save_result: true,
      }),
    });

    await consumeCompareStream(response, (evt) => {
      if (evt.event === 'start') {
        const keys = evt.algorithms || selected;
        setRunState(initialRunState(keys));
      } else if (evt.event === 'running') {
        setRunningKey(evt.algorithm);
        setRunState(prev => ({
          ...prev,
          [evt.algorithm]: { ...prev[evt.algorithm], status: STATUS.running },
        }));
      } else if (evt.event === 'done') {
        setRunningKey(null);
        setCompletedCount(evt.completed ?? 0);
        setRunState(prev => ({
          ...prev,
          [evt.algorithm]: { status: STATUS.done, result: evt.result, error: null },
        }));
      } else if (evt.event === 'finished') {
        setRunningKey(null);
        const finalData = docMetadata ? { ...evt.data, documents: docMetadata } : evt.data;
        setResult(finalData);
        setCompletedCount((evt.data?.results || []).length);
        const best = evt.data?.best_model?.algorithm || evt.data?.ranking?.[0]?.algorithm || '—';
        addNotification({
          title: t('notifyCompareDone'),
          message: t('notifyCompareDoneMsg', {
            count: (evt.data?.results || []).length,
            best,
          }),
          type: 'success',
          link: '/playground',
          showBrowser: true,
        });
      } else if (evt.event === 'error') {
        throw new Error(evt.error || 'Stream failed');
      }
    });
  }

  async function runFilesBatch() {
    const form = new FormData();
    files.forEach(file => form.append('files', file));

    setRunState(prev => {
      const next = { ...prev };
      selected.forEach(key => { next[key] = { ...next[key], status: STATUS.running }; });
      return next;
    });
    setRunningKey(selected[0] || null);

    const API = getAPI();
    // 1. Extract text from uploaded files
    const response = await fetch(`${API}/summarize/files/extract`, { method: 'POST', body: form });
    if (!response.ok) {
      const body = await response.text();
      throw new Error(body || `HTTP ${response.status}`);
    }
    const { text: extractedText, documents } = await response.json();
    
    // Save to textbox so user can see it
    setText(extractedText);
    
    // 2. Run streaming comparison over the extracted text
    await runTextStream(extractedText, documents);
  }

  async function runComparison(event) {
    event.preventDefault();
    setLoading(true);
    setError('');
    setRunState(initialRunState(selected));
    setCompletedCount(0);
    setRunningKey(null);
    setResult(null);

    addNotification({
      title: t('notifyCompareStart'),
      message: t('notifyCompareStartMsg', { count: selected.length }),
      type: 'info',
      link: '/playground',
      showBrowser: false,
    });

    try {
      await runTextStream();
    } catch (err) {
      setError(err.message || String(err));
      addNotification({
        title: t('notifyCompareError'),
        message: t('notifyCompareErrorMsg'),
        type: 'error',
        link: '/playground',
        showBrowser: true,
      });
      setRunState(prev => {
        const next = { ...prev };
        Object.keys(next).forEach(key => {
          if (next[key].status !== STATUS.done) {
            next[key] = { status: STATUS.error, result: null, error: err.message };
          }
        });
        return next;
      });
    } finally {
      setLoading(false);
      setRunningKey(null);
    }
  }

  return (
    <div className="space-y-5 pb-16">
      {/* Title Header with Modern Micro-interactions */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[var(--border-subtle)] pb-2">
        <div>
          <h1 className="ui-page-title mb-1 flex items-center gap-2 text-2xl font-black bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400 bg-clip-text text-transparent">
            {t('playgroundTitle')}
            <SlidersHorizontal className="w-5 h-5 text-blue-500 shrink-0 animate-pulse" />
          </h1>
          <p className="ui-page-subtitle text-xs text-[var(--text-muted)] font-medium">{t('playgroundSubtitle')}</p>
        </div>
      </div>

      {/* Horizontal Top-row Control Panel (3 Symmetrical Columns) */}
      <motion.div
        initial={{ opacity: 0, y: -15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: 'spring', stiffness: 260, damping: 20 }}
        className="ui-card p-6 bg-[var(--surface-elevated)] border-[var(--border)] shadow-xl relative overflow-hidden"
      >
        {/* Subtle top ambient glow */}
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500" />
        
        <form onSubmit={runComparison} className="space-y-6">
          
          {/* STEP 1: Sleek Horizontal Document Uploader (Full-width) */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="ui-label text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
                <UploadCloud size={12} className="text-blue-500 animate-pulse" />
                {t('pgFile')}
              </label>
              <span className="text-[9px] text-[var(--text-faint)] font-semibold">Định dạng hỗ trợ: PDF, DOCX, TXT</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center bg-[var(--surface-inset)] p-4 rounded-2xl border border-[var(--border-subtle)] shadow-inner">
              <div className="md:col-span-2 space-y-1">
                <h3 className="text-xs font-bold text-[var(--text)]">Tải tệp tin văn bản lên hệ thống</h3>
                <p className="text-[10px] text-[var(--text-muted)] leading-relaxed">
                  Văn bản sẽ được tự động trích xuất nội dung ngay lập tức và điền vào khung soạn thảo phía dưới. Rất tiện lợi cho các tài liệu dài.
                </p>
              </div>

              <div className="relative">
                <input
                  type="file"
                  multiple
                  disabled={loading}
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                  accept=".pdf,.docx,.txt,.doc"
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer text-xs text-[var(--text-secondary)] font-bold py-3 px-4 rounded-xl border-2 border-dashed border-[var(--border)] hover:border-blue-500 dark:hover:border-blue-500 hover:bg-blue-50/10 transition-all flex items-center justify-center gap-2 bg-[var(--surface-elevated)] shadow-sm hover:shadow"
                >
                  <UploadCloud size={16} className="text-blue-500" />
                  {files.length > 0
                    ? t('pgFileSelected', { count: files.length })
                    : t('pgFilePick')}
                </label>
              </div>
            </div>

            {files.length > 0 && (
              <div className="flex flex-wrap gap-2 pt-1">
                {files.map((f, i) => (
                  <div
                    key={i}
                    className="text-[10px] text-[var(--text-secondary)] flex items-center gap-2 px-3 py-1.5 bg-[var(--surface-elevated)] border border-[var(--border)] rounded-full shadow-sm"
                  >
                    <span className="truncate max-w-[200px] font-medium">{f.name}</span>
                    <button
                      type="button"
                      onClick={() => {
                        setFiles(files.filter((_, idx) => idx !== i));
                        if (files.length === 1) setText(SAMPLE_TEXT);
                      }}
                      className="text-red-500 hover:text-red-600 font-bold text-xs cursor-pointer focus:outline-none transition-colors"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* STEP 2: Full-Width Rich Text Editor */}
          <div className="space-y-2 flex flex-col w-full">
            <div className="flex items-center justify-between">
              <label className="ui-label text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
                <FileText size={12} className="text-blue-500" />
                {t('pgText')}
              </label>
              <span className="text-[10px] text-[var(--text-muted)] font-semibold bg-[var(--surface-inset)] px-2 py-0.5 rounded-md">
                {t('pgWords', { count: sourceWordCount })}
              </span>
            </div>
            <textarea
              value={text}
              disabled={loading}
              onChange={(e) => setText(e.target.value)}
              className="ui-textarea text-xs min-h-[420px] resize-y focus:ring-2 focus:ring-blue-500/20 w-full font-sans leading-relaxed p-4 bg-[var(--surface-elevated)] border border-[var(--border)] rounded-2xl shadow-sm"
              placeholder={t('pgTextPlaceholder')}
            />
          </div>

          {/* ROW 2: Two Columns for Reference Options (Left) and Algorithms / Trigger (Right) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-5 border-t border-[var(--border-subtle)]">
            
            {/* Left Column (Reference, Length Ratio slider & Advanced Settings) */}
            <div className="space-y-4 bg-[var(--surface-inset)] p-4 rounded-2xl border border-[var(--border-subtle)] flex flex-col justify-between">
              <section className="space-y-2 flex flex-col">
                <label className="ui-label text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
                  <BookOpen size={12} className="text-blue-500" />
                  {t('pgReference')}
                </label>
                <textarea
                  value={reference}
                  disabled={loading}
                  onChange={(e) => setReference(e.target.value)}
                  className="ui-textarea text-xs min-h-[90px] resize-none focus:ring-2 focus:ring-blue-500/20 w-full"
                  placeholder={t('pgReferencePlaceholder')}
                />
              </section>

              <section className="space-y-3">
                <label className="ui-label text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
                  <Sliders size={12} className="text-blue-500" />
                  Độ dài tóm tắt
                </label>
                <div className="grid grid-cols-2 gap-2.5">
                  {[
                    { key: 'auto', label: 'Tự động', desc: 'Đề xuất bởi AI' },
                    { key: 'short', label: 'Ngắn', desc: '~50-80 từ' },
                    { key: 'standard', label: 'Tiêu chuẩn', desc: '~100-150 từ' },
                    { key: 'detailed', label: 'Chi tiết', desc: '~200-300 từ' },
                  ].map((option) => (
                    <button
                      key={option.key}
                      type="button"
                      disabled={loading}
                      onClick={() => setSummaryLength(option.key)}
                      className={`flex flex-col items-start p-3 rounded-xl border text-left transition-all cursor-pointer select-none ${
                        summaryLength === option.key
                          ? 'border-blue-500 bg-blue-50/10 shadow-sm ring-1 ring-blue-500/20'
                          : 'border-[var(--border)] bg-[var(--surface-elevated)] hover:border-[var(--border-strong)]'
                      }`}
                    >
                      <span className="text-xs font-bold text-[var(--text)]">{option.label}</span>
                      <span className="text-[9px] text-[var(--text-muted)] mt-0.5">{option.desc}</span>
                    </button>
                  ))}
                </div>
                <p className="text-[10px] text-[var(--text-muted)] font-medium leading-relaxed mt-1">
                  Mục tiêu đầu ra: <span className="font-bold text-[var(--text)]">{displayTargetWords} từ</span>.
                  {summaryLength === 'auto' && (
                    <span className="text-blue-500 dark:text-blue-400 font-semibold ml-1">
                      (Tự động điều chỉnh theo kích thước văn bản đầu vào)
                    </span>
                  )}
                </p>
              </section>
            </div>

            {/* Right Column (Algorithm Selection & Large Submit button) */}
            <div className="space-y-4 flex flex-col justify-between">
              <section className="space-y-2 flex-1">
                <label className="ui-label text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
                  <Activity size={12} className="text-blue-500" />
                  {t('pgAlgorithms')}
                </label>
                <AlgorithmSelector selected={selected} setSelected={setSelected} disabled={loading} />
              </section>

              <div className="space-y-3">
                <button
                  type="submit"
                  disabled={loading || selected.length === 0 || (!text.trim() && !files.length)}
                  className={`ui-btn-primary w-full py-3.5 rounded-xl font-bold shadow-lg transition-all duration-300 relative overflow-hidden group cursor-pointer ${
                    loading 
                      ? 'opacity-80' 
                      : 'hover:scale-[1.01] hover:shadow-blue-500/20 active:scale-[0.99]'
                  }`}
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 opacity-100 transition-opacity" />
                  <span className="relative z-10 flex items-center justify-center gap-1.5 text-xs uppercase tracking-wider">
                    {loading ? <RefreshCcw className="animate-spin" size={14} /> : <Play className="animate-pulse" size={14} />}
                    {loading ? t('running') : t('runCompare')}
                  </span>
                </button>

                {error && (
                  <div className="text-red-600 dark:text-red-400 text-[10px] bg-red-50 dark:bg-red-950/30 p-2.5 rounded-lg flex gap-2 border border-red-200 dark:border-red-900/50">
                    <AlertCircle size={12} className="shrink-0" />
                    <span>{error}</span>
                  </div>
                )}
              </div>
            </div>
            
          </div>
        </form>
      </motion.div>

      {/* Stacked Results Section (Vertical below top inputs) */}
      <div className="space-y-6">
        
        {/* Full-width elegant Progress indicator */}
        <RunProgress
          loading={loading}
          runningKey={runningKey}
          completed={completedCount}
          total={selected.length}
        />

        {/* 3-column Symmetrical Wide Grid for Algorithm Result Cards */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="show"
          className="grid gap-6 grid-cols-1 md:grid-cols-2 lg:grid-cols-3"
        >
          {orderedKeys.map(key => (
            <motion.div
              key={key}
              whileHover={loading ? {} : { y: -6, transition: { duration: 0.2 } }}
              layout
              className="relative rounded-2xl"
            >
              <AlgorithmCard
                algoKey={key}
                state={runState[key] || { status: STATUS.idle, result: null, error: null }}
                rank={rankByKey[key]}
              />
            </motion.div>
          ))}
        </motion.div>

        {result?.warning && (
          <div className="text-xs text-amber-800 dark:text-amber-200 bg-amber-50/50 dark:bg-amber-950/10 border border-amber-200 dark:border-amber-800/40 rounded-xl px-4 py-3 flex gap-2 items-center">
            <AlertCircle className="w-4 h-4 text-amber-500 shrink-0" />
            <span>{result.warning}</span>
          </div>
        )}

        {/* Full-width Symmetrical Metrics Table */}
        {rows.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <ComparisonTable rows={rows} />
          </motion.div>
        )}

        {/* Sleek Empty state */}
        {!loading && !rows.length && Object.keys(runState).length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="rounded-2xl border border-dashed border-[var(--border)] bg-[var(--surface-elevated)] p-16 text-center shadow-sm"
          >
            <Terminal className="mx-auto text-[var(--text-faint)] mb-3 animate-pulse" size={32} />
            <p className="text-xs text-[var(--text-muted)] font-bold">{t('pgEmptyState')}</p>
          </motion.div>
        )}
        
      </div>
    </div>
  );
};

export default Playground;
