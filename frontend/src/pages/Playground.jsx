import React, { useMemo, useState, useCallback, memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQueryClient, useQuery, useMutation } from '@tanstack/react-query';
import {
  Play, RefreshCcw, Check, Loader2, Clock, SlidersHorizontal, Activity, Terminal, AlertCircle, FileText, UploadCloud, BookOpen, Award, CheckCircle2, Sparkles, Trash2
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { usePlaygroundStore, filesFingerprint } from '../stores/playgroundStore';
import { extractFilesFromUpload, streamCompareSummaries, fetchCompareHistoryList, fetchCompareHistoryDetail, deleteCompareHistoryRecord } from '../services/cachedApi';
import { invalidateAfterSummarization, invalidateFileExtractCache } from '../lib/cacheInvalidation';
import { queryKeys } from '../lib/queryKeys';
import { cacheLog } from '../lib/cacheLogger';

const ALGORITHMS = [
  { key: 'textrank', name: 'TextRank', group: 'extractive', color: '#14b8a6' },
  { key: 'lexrank', name: 'LexRank', group: 'extractive', color: '#38bdf8' },
  { key: 'lsa', name: 'LSA', group: 'extractive', color: '#84cc16' },
  { key: 'vit5', name: 'ViT5', group: 'abstractive', color: '#f59e0b' },
  { key: 'mt5', name: 'mT5', group: 'abstractive', color: '#e879f9' },
  { key: 'bartpho', name: 'BARTPho', group: 'abstractive', color: '#fb7185' },
];

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
            <h3 className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-faint)]">
              {groupLabel(group)}
            </h3>
            <span className="text-[10px] font-bold text-sky-600 dark:text-sky-400 bg-sky-50 dark:bg-sky-950/20 px-2 py-0.5 rounded-full">
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
                  className={`flex flex-col items-center justify-center p-3 rounded-xl border text-center transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-sky-50 dark:bg-sky-950/20 border-sky-400 dark:border-sky-800 text-sky-600 dark:text-sky-400 font-bold shadow-sm'
                      : 'bg-[var(--bg-elevated)] border-[var(--border)] text-[var(--text-secondary)] hover:border-sky-300 dark:hover:border-sky-800'
                  } ${disabled ? 'opacity-40 cursor-not-allowed' : ''}`}
                >
                  <span className="w-2.5 h-2.5 rounded-full mb-1.5" style={{ background: item.color }} />
                  <span className="text-[11px] font-bold truncate w-full">{item.name}</span>
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
    [STATUS.idle]: { label: t('statusIdle'), className: 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400' },
    [STATUS.running]: { label: t('statusRunning'), className: 'bg-sky-50 dark:bg-sky-950/30 text-sky-600 dark:text-sky-400 animate-pulse border border-sky-200/20' },
    [STATUS.done]: { label: t('statusDone'), className: 'bg-emerald-50 dark:bg-emerald-950/25 text-emerald-600 dark:text-emerald-400 border border-emerald-250/20 font-bold' },
    [STATUS.error]: { label: t('statusError'), className: 'bg-red-50 dark:bg-red-950/25 text-red-600 dark:text-red-400 border border-red-250/20 font-bold' },
  };
  const item = map[status] || map[STATUS.idle];
  return (
    <span className={`ui-badge text-[10px] font-bold ${item.className}`}>
      {status === STATUS.running && <Loader2 size={10} className="animate-spin mr-1 text-sky-500" />}
      {item.label}
    </span>
  );
};

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
      className={`algo-card ui-card p-5 transition-all duration-300 border bg-[var(--bg-elevated)] border-[var(--border)] shadow-sm hover:shadow-md ${
        isRunning ? 'border-sky-400 dark:border-sky-600' : ''
      }`}
      style={{ borderLeftColor: meta.color, borderLeftWidth: '4px' }}
    >
      <header className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <div className="min-w-0">
            <h3 className="font-extrabold text-sm text-[var(--text-primary)] truncate">{meta.name}</h3>
            <p className="text-[9px] text-[var(--text-faint)] uppercase tracking-wider font-bold">{meta.group}</p>
          </div>
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <StatusBadge status={state.status} />
          {rank != null && isDone && (
            <span className={`text-[10px] font-extrabold px-2 py-0.5 rounded border flex items-center gap-1 ${
              rank === 1
                ? 'bg-amber-50 dark:bg-amber-950/20 text-amber-700 dark:text-amber-400 border-amber-200/50'
                : 'bg-slate-50 dark:bg-slate-900 text-slate-600 dark:text-slate-400 border-slate-200'
            }`}>
              <Award size={10} />
              {t('pgRank', { rank })}
            </span>
          )}
        </div>
      </header>

      {isRunning && (
        <div className="mb-3 rounded-xl bg-sky-50 dark:bg-sky-950/30 border border-sky-100 dark:border-sky-900/50 px-3 py-2 text-xs text-sky-800 dark:text-sky-300 flex items-center gap-2 font-medium">
          <Loader2 size={13} className="animate-spin shrink-0 text-sky-500" />
          <span>{t('pgSummarizing')}</span>
        </div>
      )}

      {state.status === STATUS.idle && (
        <p className="text-xs text-[var(--text-faint)] italic font-medium">{t('pgClickRun')}</p>
      )}

      {state.status === STATUS.error && (
        <p className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/20 rounded-xl p-3 border border-red-200/30 dark:border-red-900/30 font-semibold">
          {state.error || t('pgAlgoError')}
        </p>
      )}

      {isDone && row && (
        <div className="space-y-4 algo-summary-reveal">
          <p className="text-sm text-[var(--text-secondary)] leading-relaxed py-1 whitespace-pre-wrap font-sans">
            {row.summary || '—'}
          </p>
          <div className="flex flex-wrap gap-1.5 text-[10px] font-bold pt-2 border-t border-[var(--border)]/60">
            <span className="px-2 py-1 rounded-lg bg-[var(--bg-muted)] text-[var(--text-muted)] flex items-center gap-1">
              <Clock size={11} className="text-sky-500" />
              {(row.processing_time ?? row.time_seconds ?? 0).toFixed(2)}s
            </span>
            <span className="px-2 py-1 rounded-lg bg-emerald-50 dark:bg-emerald-950/10 text-emerald-600 dark:text-emerald-400 border border-emerald-250/20">
              ROUGE-L {pct(metric(row, 'rougeL'))}
            </span>
            <span className="px-2 py-1 rounded-lg bg-sky-50 dark:bg-sky-950/10 text-sky-600 dark:text-sky-400 border border-sky-250/20">
              BERTScore {pct(metric(row, 'bertscore_f1'))}
            </span>
            <span className="px-2 py-1 rounded-lg bg-[var(--bg-muted)] text-[var(--text-muted)]">
              {t('pgWords', { count: row.word_count ?? 0 })} từ
            </span>
            {row.length_ratio_percent != null && (
              <span className="px-2 py-1 rounded-lg bg-indigo-50 dark:bg-indigo-950/10 text-indigo-700 dark:text-indigo-400 border border-indigo-250/20">
                {t('pgLengthOfSource', { pct: row.length_ratio_percent })}
              </span>
            )}
          </div>
          {row.warning_badge && (
            <p className="text-[10px] text-amber-700 dark:text-amber-400 flex items-center gap-1 bg-amber-50 dark:bg-amber-950/10 border border-amber-250/30 dark:border-amber-900/30 px-2 py-1 rounded-lg font-bold w-fit">
              <AlertCircle size={12} className="text-amber-500" /> {row.warning_badge}
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
    <div className="rounded-2xl border border-sky-200/50 dark:border-sky-850/50 bg-gradient-to-r from-sky-500/5 to-indigo-500/5 p-4 space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-bold text-sky-950 dark:text-sky-100">
          {loading ? <Loader2 size={16} className="animate-spin text-sky-500" /> : <Activity size={16} className="text-sky-500" />}
          {loading
            ? (current ? t('pgProgressRunning', { name: current }) : t('pgProgressInit'))
            : t('pgProgressReady')}
        </div>
        <span className="text-xs font-bold text-sky-700 dark:text-sky-400 bg-sky-100 dark:bg-sky-950/40 px-2.5 py-0.5 rounded-full border border-sky-200/30">{completed}/{total}</span>
      </div>
      <div className="h-1.5 rounded-full bg-sky-100 dark:bg-sky-950/40 overflow-hidden">
        <div
          className="h-full rounded-full bg-sky-500 transition-all duration-500 ease-out algo-progress-bar"
          style={{ width: `${loading ? Math.max(pctDone, runningKey ? 8 : 4) : 0}%` }}
        />
      </div>
      <p className="text-[10px] text-sky-800/80 dark:text-sky-300/80 font-medium">
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
      className="ui-card overflow-hidden bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm"
    >
      <div className="border-b border-[var(--border)] px-4 py-3 bg-[var(--bg-muted)]/40 flex items-center gap-2">
        <FileText size={15} className="text-sky-500" />
        <h3 className="text-xs font-bold text-[var(--text-primary)] uppercase tracking-wider">{t('pgMetricsTable')}</h3>
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
              <th className="px-4 py-3 text-sky-600 dark:text-sky-400 font-bold bg-sky-500/5">{t('colCombined')}</th>
              <th className="px-4 py-3">{t('colLength')}</th>
              <th className="px-4 py-3">{t('colTime')}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-subtle)] font-medium">
            {rows.map(row => (
              <tr key={row.key} className="ui-table-row">
                <td className="px-4 py-3 font-semibold text-[var(--text-primary)]">
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: byKey(row.key).color }} />
                    {row.algorithm}
                  </span>
                </td>
                <td className="px-4 py-3 text-center text-[var(--text-secondary)] font-mono">{pct(metric(row, 'rouge1'))}</td>
                <td className="px-4 py-3 text-center text-[var(--text-secondary)] font-mono">{pct(metric(row, 'rouge2'))}</td>
                <td className="px-4 py-3 text-center text-[var(--text-secondary)] font-mono">{pct(metric(row, 'rougeL'))}</td>
                <td className="px-4 py-3 text-center text-[var(--text-secondary)] font-mono">{pct(metric(row, 'bertscore_f1'))}</td>
                <td className="px-4 py-3 text-center text-sky-700 dark:text-sky-300 font-mono font-bold bg-sky-500/5">{pct(metric(row, 'combined_score'))}</td>
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

export default function Playground() {
  const { t } = useApp();
  const queryClient = useQueryClient();

  const textInput = usePlaygroundStore(state => state.text);
  const setTextInput = usePlaygroundStore(state => state.setText);
  const refInput = usePlaygroundStore(state => state.reference);
  const setRefInput = usePlaygroundStore(state => state.setReference);
  const [lengthRatio, setLengthRatio] = useState(20);
  const selectedAlgorithmsRaw = usePlaygroundStore(state => state.selected || []);
  const selectedAlgorithms = useMemo(() => {
    return selectedAlgorithmsRaw.filter(key => ALGORITHMS.some(a => a.key === key));
  }, [selectedAlgorithmsRaw]);
  const setSelectedAlgorithms = usePlaygroundStore(state => state.setSelected);

  const [loading, setLoading] = useState(false);
  const [runningKey, setRunningKey] = useState(null);

  const runStates = usePlaygroundStore(state => state.runState);
  const setRunStates = usePlaygroundStore(state => state.setRunState);
  const completedCount = usePlaygroundStore(state => state.completedCount);
  const setCompletedCount = usePlaygroundStore(state => state.setCompletedCount);
  const loadFromHistoryRecord = usePlaygroundStore(state => state.loadFromHistoryRecord);

  // History states
  const [showHistory, setShowHistory] = useState(false);
  const [activeHistoryId, setActiveHistoryId] = useState(null);
  const [fetchingDetailId, setFetchingDetailId] = useState(null);

  // File states
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState([]);
  const [extracting, setExtracting] = useState(false);
  const [extractProgress, setExtractProgress] = useState('');

  const wordCounts = useMemo(() => {
    const clean = (txt) => (txt || '').trim().replace(/\s+/g, ' ');
    const srcWords = clean(textInput).split(' ').filter(Boolean).length;
    const refWords = clean(refInput).split(' ').filter(Boolean).length;
    const targetWords = Math.round(srcWords * (lengthRatio / 100));
    return { source: srcWords, reference: refWords, target: targetWords };
  }, [textInput, refInput, lengthRatio]);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  }, []);

  const processFiles = useCallback(async (uploadedFiles) => {
    if (!uploadedFiles?.length) return;
    setExtracting(true);
    setExtractProgress('Đang phân tích cấu trúc file...');
    try {
      const { text, reference } = await extractFilesFromUpload(uploadedFiles, (prog) => {
        setExtractProgress(`Đang trích xuất văn bản: ${prog}%`);
      });
      if (text) setTextInput(text);
      if (reference) setRefInput(reference);
      setFiles(uploadedFiles);
    } catch (err) {
      console.error(err);
      alert('Không thể trích xuất nội dung file. Vui lòng thử lại với file có định dạng chuẩn.');
    } finally {
      setExtracting(false);
      setExtractProgress('');
    }
  }, [setTextInput, setRefInput]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.length) processFiles(Array.from(e.dataTransfer.files));
  }, [processFiles]);

  const handleFileSelect = useCallback((e) => {
    if (e.target.files?.length) processFiles(Array.from(e.target.files));
  }, [processFiles]);

  // History query and handlers
  const { data: historyList, isLoading: isLoadingHistory } = useQuery({
    queryKey: queryKeys.compareHistory,
    queryFn: () => fetchCompareHistoryList(20),
    enabled: showHistory,
  });

  const handleSelectHistory = async (resultId) => {
    if (fetchingDetailId) return;
    setFetchingDetailId(resultId);
    try {
      const detail = await fetchCompareHistoryDetail(resultId);
      loadFromHistoryRecord(detail);
      setActiveHistoryId(resultId);
      setShowHistory(false);
    } catch (err) {
      console.error('Failed to fetch history detail:', err);
      alert('Không thể tải chi tiết lịch sử chạy này. Vui lòng thử lại.');
    } finally {
      setFetchingDetailId(null);
    }
  };

  const deleteMutation = useMutation({
    mutationFn: (resultId) => deleteCompareHistoryRecord(resultId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.compareHistory });
    },
  });

  const handleDeleteHistory = async (e, resultId) => {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm('Bạn có chắc chắn muốn xóa bản ghi lịch sử này không?')) return;
    try {
      await deleteMutation.mutateAsync(resultId);
      if (activeHistoryId === resultId) {
        setActiveHistoryId(null);
      }
    } catch (err) {
      console.error('Failed to delete history record:', err);
      alert('Không thể xóa bản ghi lịch sử này.');
    }
  };

  const clearFiles = useCallback(() => {
    if (files.length > 0) {
      invalidateFileExtractCache(queryClient);
    }
    setFiles([]);
    setTextInput('');
    setRefInput('');
  }, [files, queryClient, setTextInput, setRefInput]);

  const handleRun = async () => {
    if (loading || selectedAlgorithms.length === 0 || !textInput.trim()) return;

    setLoading(true);
    setCompletedCount(0);
    setRunStates(initialRunState(selectedAlgorithms));
    setActiveHistoryId(null);

    try {
      await streamCompareSummaries({
        text: textInput,
        reference: refInput || null,
        algorithms: selectedAlgorithms,
        ratio: lengthRatio / 100,
      }, (event) => {
        const status = event.status || event.event;
        const { algorithm, result, error } = event;
        setRunStates(prev => ({
          ...prev,
          [algorithm]: {
            status: status === 'running' ? STATUS.running : status === 'done' ? STATUS.done : STATUS.error,
            result: result || prev[algorithm]?.result || null,
            error: error || null,
          }
        }));

        if (status === 'running') setRunningKey(algorithm);
        if (status === 'done' || status === 'error') {
          setCompletedCount(c => c + 1);
          setRunningKey(null);
        }
      });

      await invalidateAfterSummarization(queryClient);
    } catch (err) {
      console.error('Comparison stream failed:', err);
    } finally {
      setLoading(false);
      setRunningKey(null);
    }
  };

  // Compute leaderboard ranks
  const rankedAlgorithms = useMemo(() => {
    const completed = selectedAlgorithms
      .map(key => ({ key, result: runStates[key]?.result }))
      .filter(item => item.result);
    completed.sort((a, b) => (b.result.combined_score ?? 0) - (a.result.combined_score ?? 0));
    return Object.fromEntries(completed.map((item, idx) => [item.key, idx + 1]));
  }, [selectedAlgorithms, runStates]);

  const tableRows = useMemo(() => {
    return selectedAlgorithms
      .map(key => {
        const state = runStates[key];
        if (state?.status !== STATUS.done || !state.result) return null;
        return { key, algorithm: byKey(key).name, ...state.result };
      })
      .filter(Boolean);
  }, [selectedAlgorithms, runStates]);

  return (
    <div className="space-y-6 pb-12">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="ui-heading-1 flex items-center gap-2">
            <Sparkles className="text-sky-500 animate-pulse" />
            Tóm tắt & So kè Thuật toán
          </h1>
          <p className="ui-page-subtitle">So sánh trực quan kết quả và các chỉ số ROUGE, BERTScore của các giải thuật NLP thời gian thực.</p>
        </div>
        <button
          onClick={() => setShowHistory(true)}
          className="flex items-center gap-2 px-4 py-2.5 text-xs font-bold bg-[var(--bg-elevated)] hover:bg-[var(--bg-muted)]/50 border border-[var(--border)] rounded-xl text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-all cursor-pointer shadow-sm active:scale-95 shrink-0 self-start md:self-auto"
        >
          <Clock size={14} className="text-sky-500" />
          <span>Lịch sử chạy</span>
        </button>
      </div>

      {/* Top Control Panel (Cấu hình thử nghiệm) */}
      <div className="ui-card p-5 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm space-y-5">
        <div className="flex items-center justify-between border-b border-[var(--border)]/60 pb-3">
          <h2 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)] flex items-center gap-1.5">
            <SlidersHorizontal size={14} className="text-sky-500" />
            Cấu hình thử nghiệm
          </h2>
          
          <div className="flex gap-2.5 items-center">
            <button
              type="button"
              disabled={loading}
              onClick={() => setSelectedAlgorithms(ALGORITHMS.map(item => item.key))}
              className="text-[10px] font-extrabold text-sky-600 hover:text-sky-700 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            >
              Chọn tất cả
            </button>
            <span className="text-[10px] text-[var(--text-faint)] font-bold">|</span>
            <button
              type="button"
              disabled={loading}
              onClick={() => setSelectedAlgorithms([])}
              className="text-[10px] font-extrabold text-red-500 hover:text-red-600 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            >
              Bỏ chọn tất cả
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Column Left (9 spans): File upload and algorithm selection */}
          <div className="lg:col-span-9 grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Left Box: File Upload */}
            <div className="space-y-2">
              <label className="text-[10px] font-extrabold uppercase tracking-wider text-[var(--text-secondary)]">
                Tài liệu đầu vào
              </label>
              
              {files.length === 0 ? (
                <div
                  onDragEnter={handleDrag}
                  onDragOver={handleDrag}
                  onDragLeave={handleDrag}
                  onDrop={handleDrop}
                  className={`border border-dashed rounded-xl p-5 text-center cursor-pointer transition-all relative flex flex-col items-center justify-center group ${
                    dragActive
                      ? 'border-sky-500 bg-sky-500/5 ring-4 ring-sky-500/10'
                      : 'border-[var(--border)] bg-[var(--bg-muted)]/10 hover:border-sky-400 hover:bg-sky-500/5'
                  }`}
                  style={{ minHeight: '140px' }}
                >
                  <input
                    type="file"
                    multiple
                    accept=".pdf,.docx,.txt"
                    onChange={handleFileSelect}
                    className="absolute inset-0 opacity-0 cursor-pointer"
                  />
                  <div className="p-3 bg-sky-50 dark:bg-sky-950/40 rounded-full text-sky-500 mb-2 transition-transform group-hover:scale-110 duration-300">
                    <UploadCloud size={24} className="animate-bounce" />
                  </div>
                  <span className="text-xs font-bold text-[var(--text-primary)]">Tải lên tài liệu của bạn</span>
                  <span className="text-[10px] text-[var(--text-muted)] mt-1">Kéo thả hoặc click chọn file PDF, DOCX, TXT</span>
                </div>
              ) : (
                <div className="border rounded-xl p-4 text-center bg-gradient-to-br from-emerald-500/5 to-teal-500/5 border-emerald-500/20 dark:border-emerald-500/10 relative flex flex-col items-center justify-center min-h-[140px] shadow-inner">
                  <div className="absolute top-2.5 right-2.5">
                    <span className="text-[8px] uppercase tracking-wider font-extrabold text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-250/20 px-2 py-0.5 rounded-full">
                      Đã tải lên
                    </span>
                  </div>
                  <div className="p-2.5 bg-emerald-50 dark:bg-emerald-950/40 rounded-full text-emerald-500 mb-2">
                    <FileText size={20} className="animate-pulse" />
                  </div>
                  <span className="text-xs font-bold text-[var(--text-primary)] truncate max-w-full px-4">
                    {files[0].name}
                  </span>
                  <span className="text-[9px] text-[var(--text-faint)] mt-0.5 font-bold">
                    {(files[0].size / 1024).toFixed(1)} KB
                  </span>
                  <button
                    onClick={clearFiles}
                    className="mt-3 px-3 py-1 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-600 dark:text-red-400 text-[10px] font-extrabold transition-all cursor-pointer border border-red-500/20 active:scale-95"
                  >
                    Hủy tải lên
                  </button>
                </div>
              )}
              {extracting && (
                <div className="flex items-center gap-2 text-[10px] font-bold text-sky-600 dark:text-sky-400 bg-sky-50 dark:bg-sky-950/20 px-3 py-2 rounded-xl border border-sky-100 dark:border-sky-900/50">
                  <Loader2 size={12} className="animate-spin" />
                  <span>{extractProgress}</span>
                </div>
              )}
            </div>

            {/* Right Box: Algorithm Selector */}
            <div className="space-y-2">
              <label className="text-[10px] font-extrabold uppercase tracking-wider text-[var(--text-secondary)]">
                Lựa chọn thuật toán
              </label>
              <AlgorithmSelector selected={selectedAlgorithms} setSelected={setSelectedAlgorithms} disabled={loading} />
            </div>
          </div>

          {/* Column Right (3 spans): Action & Stats Summary */}
          <div className="lg:col-span-3 flex flex-col justify-between min-h-[140px] gap-4 pt-6 lg:pt-0">
            {/* Stats Summary Card */}
            <div className="ui-card p-3 bg-[var(--bg-muted)]/30 border-[var(--border)] rounded-xl space-y-2 flex-grow flex flex-col justify-center">
              <div className="text-[9px] font-extrabold text-[var(--text-faint)] uppercase tracking-wider mb-1">
                Tóm tắt cấu hình
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-[var(--text-secondary)] font-medium">Đã chọn:</span>
                  <span className="font-extrabold text-sky-600 dark:text-sky-400 bg-sky-50 dark:bg-sky-950/20 px-2 py-0.5 rounded-md">
                    {selectedAlgorithms.length} thuật toán
                  </span>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-[var(--text-secondary)] font-medium">Nguồn:</span>
                  <span className="font-extrabold text-[var(--text-primary)]">
                    {files.length > 0 ? 'File tài liệu' : textInput.trim() ? `${wordCounts.source} từ` : 'Trống'}
                  </span>
                </div>
              </div>
            </div>

            {/* Run Button */}
            <button
              onClick={handleRun}
              disabled={loading || selectedAlgorithms.length === 0 || !textInput.trim()}
              className="w-full py-3.5 rounded-xl font-extrabold text-xs bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-600 hover:to-indigo-700 text-white shadow-md hover:shadow-lg hover:shadow-sky-500/10 transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2 cursor-pointer active:scale-[0.98]"
            >
              {loading ? (
                <>
                  <Loader2 size={14} className="animate-spin" />
                  <span>Đang xử lý NLP...</span>
                </>
              ) : (
                <>
                  <Play size={14} className="fill-white/10" />
                  <span>Chạy so sánh ({selectedAlgorithms.length})</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Bottom Workspace Panel */}
      <div className="space-y-6">
        {/* Progress panel */}
        {(loading || completedCount > 0) && (
          <RunProgress runningKey={runningKey} completed={completedCount} total={selectedAlgorithms.length} loading={loading} />
        )}

        {/* Core editor text inputs */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="text-[10px] font-bold text-[var(--text-faint)] uppercase tracking-wider px-1">Văn bản gốc</label>
            <textarea
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Dán văn bản tiếng Việt của bạn vào đây..."
              disabled={loading}
              className="ui-textarea min-h-[320px] text-xs leading-relaxed font-medium"
            />
          </div>
          <div className="space-y-1">
            <label className="text-[10px] font-bold text-[var(--text-faint)] uppercase tracking-wider px-1">Tóm tắt tham chiếu (Tùy chọn)</label>
            <textarea
              value={refInput}
              onChange={(e) => setRefInput(e.target.value)}
              placeholder="Nhập bản tóm tắt tham chiếu do con người viết để đo lường ROUGE/BERTScore chính xác..."
              disabled={loading}
              className="ui-textarea min-h-[320px] text-xs leading-relaxed font-medium"
            />
          </div>
        </div>

        {/* Results Grid */}
        <div className="space-y-4">
          <h2 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)] flex items-center gap-1.5">
            <BookOpen size={14} className="text-sky-500" />
            Kết quả tóm tắt & Phân tích
          </h2>

          {selectedAlgorithms.length === 0 ? (
            <div className="ui-card p-8 text-center text-[var(--text-faint)] bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm rounded-xl">
              <Sparkles size={28} className="mx-auto mb-2 text-sky-500/60" />
              <p className="text-xs font-bold">{t('pgEmptyState')}</p>
              <p className="text-[10px] mt-1 font-medium">Vui lòng chọn thuật toán và nhấn Chạy so sánh.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {selectedAlgorithms.map(key => (
                <AlgorithmCard
                  key={key}
                  algoKey={key}
                  state={runStates[key] || { status: STATUS.idle, result: null, error: null }}
                  rank={rankedAlgorithms[key]}
                />
              ))}
            </div>
          )}
        </div>

        {/* Table summary of metrics */}
        <ComparisonTable rows={tableRows} />
      </div>

      {/* Loading Overlay for Detail Fetching */}
      {fetchingDetailId && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-[2px] z-50 flex items-center justify-center">
          <div className="ui-card p-6 bg-[var(--bg-elevated)] border border-[var(--border)] rounded-2xl flex items-center gap-3 shadow-2xl">
            <Loader2 className="animate-spin text-sky-500" size={20} />
            <span className="text-xs font-bold text-[var(--text-primary)]">Đang tải kết quả chi tiết...</span>
          </div>
        </div>
      )}

      {/* History Drawer */}
      <AnimatePresence>
        {showHistory && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.4 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowHistory(false)}
              className="fixed inset-0 bg-black z-40"
            />

            {/* Drawer */}
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed right-0 top-0 bottom-0 w-80 sm:w-96 bg-[var(--bg-elevated)] border-l border-[var(--border)] shadow-2xl z-50 flex flex-col"
            >
              {/* Header */}
              <div className="p-4 border-b border-[var(--border)] flex items-center justify-between bg-[var(--bg-muted)]/30">
                <div className="flex items-center gap-2 font-extrabold text-xs text-[var(--text-primary)] uppercase tracking-wider">
                  <Clock size={14} className="text-sky-500" />
                  Lịch sử so sánh
                </div>
                <button
                  onClick={() => setShowHistory(false)}
                  className="text-xs font-bold text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer"
                >
                  Đóng
                </button>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {isLoadingHistory ? (
                  <div className="flex flex-col items-center justify-center py-12 text-[var(--text-faint)] gap-2">
                    <Loader2 size={20} className="animate-spin text-sky-500" />
                    <span className="text-[10px] font-bold">Đang tải lịch sử...</span>
                  </div>
                ) : !historyList || historyList.length === 0 ? (
                  <div className="text-center py-12 text-[var(--text-faint)] text-xs font-medium">
                    Không có lịch sử chạy so sánh nào.
                  </div>
                ) : (
                  historyList.map((item) => {
                    const isActive = activeHistoryId === item.result_id;
                    const date = new Date(item.created_at);
                    const formattedDate = date.toLocaleString('vi-VN', {
                      hour: '2-digit',
                      minute: '2-digit',
                      day: '2-digit',
                      month: '2-digit',
                      year: 'numeric',
                    });

                    return (
                      <div
                        key={item.result_id}
                        onClick={(e) => {
                          if (e.target.closest('.btn-delete-history')) return;
                          handleSelectHistory(item.result_id);
                        }}
                        className={`p-3.5 rounded-xl border text-left cursor-pointer transition-all hover:border-sky-400 hover:bg-sky-500/5 ${
                          isActive
                            ? 'border-sky-500 bg-sky-500/5 ring-1 ring-sky-500/10'
                            : 'border-[var(--border)] bg-[var(--bg-elevated)]'
                        }`}
                      >
                        <div className="flex justify-between items-start gap-2 mb-1.5">
                          <span className="text-[9px] font-bold text-[var(--text-faint)]">{formattedDate}</span>
                          <button
                            type="button"
                            onClick={(e) => handleDeleteHistory(e, item.result_id)}
                            className="btn-delete-history text-[9px] font-bold text-red-500 hover:text-red-750 opacity-60 hover:opacity-100 transition-opacity p-0.5 rounded cursor-pointer"
                          >
                            Xóa
                          </button>
                        </div>
                        <p className="text-[11px] text-[var(--text-secondary)] line-clamp-2 mb-2.5 leading-relaxed font-medium">
                          {item.text_preview}
                        </p>
                        <div className="flex items-center justify-between text-[10px] font-extrabold text-[var(--text-faint)]">
                          <span>{item.algorithm_count} thuật toán</span>
                          {item.best_algorithm && (
                            <span className="text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/20 px-1.5 py-0.5 rounded border border-amber-250/20">
                              Tốt nhất: {byKey(item.best_algorithm).name}
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
