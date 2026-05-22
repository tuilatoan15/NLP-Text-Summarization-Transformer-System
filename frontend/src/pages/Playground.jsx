import React, { useMemo, useState } from 'react';
import {
  Play, RefreshCcw, Check, Loader2, Clock, Sparkles, AlertCircle,
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
  queued: 'queued',
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

function groupLabel(group) {
  return group === 'abstractive' ? 'Abstractive' : 'Extractive';
}

function initialRunState(keys) {
  return Object.fromEntries(keys.map(key => [key, { status: STATUS.queued, result: null, error: null }]));
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
            <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-600">{groupLabel(group)}</h3>
            <span className="text-xs text-gray-500">{selected.filter(key => byKey(key).group === group).length}/3</span>
          </div>
          <div className="grid gap-2">
            {ALGORITHMS.filter(item => item.group === group).map(item => {
              const isSelected = selected.includes(item.key);
              return (
                <button
                  key={item.key}
                  type="button"
                  disabled={disabled}
                  onClick={() => toggle(item.key)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium border transition ${
                    isSelected
                      ? 'bg-blue-50 border-blue-200 text-blue-600'
                      : 'bg-white border-gray-200 text-gray-600 hover:border-gray-300'
                  } ${disabled ? 'opacity-60 cursor-not-allowed' : ''}`}
                >
                  <span className="w-2 h-2 rounded-full" style={{ background: item.color }} />
                  <span className="flex-1 text-left">{item.name}</span>
                  {isSelected && <Check size={15} />}
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
  const map = {
    [STATUS.idle]: { label: 'Chờ chạy', className: 'bg-gray-100 text-gray-600' },
    [STATUS.queued]: { label: 'Đang chờ', className: 'bg-slate-100 text-slate-600' },
    [STATUS.running]: { label: 'Đang chạy', className: 'bg-blue-100 text-blue-700 algo-badge-pulse' },
    [STATUS.done]: { label: 'Hoàn tất', className: 'bg-emerald-100 text-emerald-700' },
    [STATUS.error]: { label: 'Lỗi', className: 'bg-red-100 text-red-700' },
  };
  const item = map[status] || map[STATUS.idle];
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${item.className}`}>
      {status === STATUS.running && <Loader2 size={12} className="animate-spin" />}
      {item.label}
    </span>
  );
};

const AlgorithmCard = ({ algoKey, state, rank }) => {
  const meta = byKey(algoKey);
  const row = state.result;
  const isRunning = state.status === STATUS.running;
  const isDone = state.status === STATUS.done;

  return (
    <article
      className={`algo-card rounded-xl border bg-white dark:bg-slate-900 p-4 transition-all duration-300 ${
        isRunning ? 'algo-card-running border-blue-300 dark:border-blue-600 shadow-md shadow-blue-100 dark:shadow-blue-900/30' : 'border-gray-200 dark:border-slate-700'
      } ${isDone ? 'algo-card-done' : ''}`}
      style={{ '--algo-color': meta.color }}
    >
      <header className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className={`w-3 h-3 rounded-full shrink-0 ${isRunning ? 'algo-dot-pulse' : ''}`}
            style={{ background: meta.color }}
          />
          <div className="min-w-0">
            <h3 className="font-semibold text-gray-900 truncate">{meta.name}</h3>
            <p className="text-xs text-gray-500 capitalize">{meta.group}</p>
          </div>
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <StatusBadge status={state.status} />
          {rank != null && isDone && (
            <span className="text-xs text-amber-600 font-medium">#{rank} xếp hạng</span>
          )}
        </div>
      </header>

      {isRunning && (
        <div className="mb-3 rounded-lg bg-blue-50 border border-blue-100 px-3 py-2 text-sm text-blue-800 flex items-center gap-2">
          <Loader2 size={16} className="animate-spin shrink-0" />
          <span>Đang tóm tắt văn bản…</span>
        </div>
      )}

      {state.status === STATUS.queued && (
        <p className="text-sm text-gray-400 italic">Sẽ chạy sau các thuật toán trước…</p>
      )}

      {state.status === STATUS.error && (
        <p className="text-sm text-red-600 bg-red-50 rounded-lg p-2">{state.error || 'Không thể chạy thuật toán này.'}</p>
      )}

      {isDone && row && (
        <div className="space-y-3 algo-summary-reveal">
          <p className="text-sm text-gray-800 leading-relaxed border-l-4 pl-3" style={{ borderColor: meta.color }}>
            {row.summary || '—'}
          </p>
          <div className="flex flex-wrap gap-2 text-xs">
            <span className="px-2 py-1 rounded-md bg-gray-100 text-gray-700">
              <Clock size={12} className="inline mr-1 -mt-0.5" />
              {(row.processing_time ?? row.time_seconds ?? 0).toFixed(2)}s
            </span>
            <span className="px-2 py-1 rounded-md bg-gray-100 text-gray-700">ROUGE-L {pct(metric(row, 'rougeL'))}</span>
            <span className="px-2 py-1 rounded-md bg-gray-100 text-gray-700">BERT {pct(metric(row, 'bertscore_f1'))}</span>
            <span className="px-2 py-1 rounded-md bg-gray-100 text-gray-700">{row.word_count ?? 0} từ</span>
          </div>
          {row.warning_badge && (
            <p className="text-xs text-amber-700 flex items-center gap-1">
              <AlertCircle size={14} /> {row.warning_badge}
            </p>
          )}
        </div>
      )}

      {state.status === STATUS.idle && (
        <p className="text-sm text-gray-400">Nhấn &quot;Chạy so sánh&quot; để xem bản tóm tắt.</p>
      )}
    </article>
  );
};

const RunProgress = ({ runningKey, completed, total, loading }) => {
  const pctDone = total ? Math.round((completed / total) * 100) : 0;
  const current = runningKey ? byKey(runningKey).name : null;

  return (
    <div className="rounded-xl border border-blue-200 dark:border-blue-800 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/40 dark:to-indigo-950/40 p-4 space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-medium text-blue-900">
          {loading ? <Loader2 size={18} className="animate-spin" /> : <Sparkles size={18} />}
          {loading
            ? (current ? `Đang chạy: ${current}` : 'Đang khởi tạo…')
            : 'Sẵn sàng chạy so sánh'}
        </div>
        <span className="text-sm font-semibold text-blue-700">{completed}/{total}</span>
      </div>
      <div className="h-2 rounded-full bg-blue-100 overflow-hidden">
        <div
          className="h-full rounded-full bg-blue-600 transition-all duration-500 ease-out algo-progress-bar"
          style={{ width: `${loading ? Math.max(pctDone, runningKey ? 8 : 4) : 0}%` }}
        />
      </div>
      <p className="text-xs text-blue-800/80">
        {loading
          ? 'Kết quả hiển thị ngay khi từng thuật toán hoàn thành — không cần chờ toàn bộ.'
          : 'Chọn thuật toán bên trái và bấm Chạy so sánh.'}
      </p>
    </div>
  );
};

const ComparisonTable = ({ rows }) => {
  if (!rows.length) return null;

  return (
    <div className="rounded-xl border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-900 overflow-hidden">
      <div className="border-b border-gray-200 px-4 py-3 bg-gray-50">
        <h3 className="text-sm font-semibold text-gray-900">Bảng metrics (sau khi hoàn tất)</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-xs uppercase text-gray-600 border-b border-gray-200">
            <tr>
              <th className="px-4 py-3 text-left">Model</th>
              <th className="px-4 py-3">ROUGE-1</th>
              <th className="px-4 py-3">ROUGE-2</th>
              <th className="px-4 py-3">ROUGE-L</th>
              <th className="px-4 py-3">BERTScore</th>
              <th className="px-4 py-3">Thời gian</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {rows.map(row => (
              <tr key={row.key} className="hover:bg-gray-50">
                <td className="px-4 py-3 font-medium">
                  <span className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full" style={{ background: byKey(row.key).color }} />
                    {row.algorithm}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">{pct(metric(row, 'rouge1'))}</td>
                <td className="px-4 py-3 text-center">{pct(metric(row, 'rouge2'))}</td>
                <td className="px-4 py-3 text-center">{pct(metric(row, 'rougeL'))}</td>
                <td className="px-4 py-3 text-center">{pct(metric(row, 'bertscore_f1'))}</td>
                <td className="px-4 py-3 text-center">{(row.processing_time ?? 0).toFixed(2)}s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const Playground = () => {
  const { t, addNotification } = useApp();
  const [text, setText] = useState(SAMPLE_TEXT);
  const [reference, setReference] = useState('');
  const [files, setFiles] = useState([]);
  const [selected, setSelected] = useState(ALGORITHMS.map(item => item.key));
  const [sentenceCount, setSentenceCount] = useState(5);
  const [maxLength, setMaxLength] = useState(150);
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

  async function runTextStream() {
    setRunState(initialRunState(selected));
    setCompletedCount(0);
    setRunningKey(null);
    setResult(null);

    const API = getAPI();
    const response = await fetch(`${API}/summarize/compare/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        reference: reference || null,
        algorithms: selected,
        extractive_sentences: sentenceCount,
        max_abstractive_length: maxLength,
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
        setResult(evt.data);
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
    form.append('reference', reference);
    form.append('algorithms', JSON.stringify(selected));
    form.append('extractive_sentences', String(sentenceCount));
    form.append('max_abstractive_length', String(maxLength));

    setRunState(prev => {
      const next = { ...prev };
      selected.forEach(key => { next[key] = { ...next[key], status: STATUS.running }; });
      return next;
    });
    setRunningKey(selected[0] || null);

    const API = getAPI();
    const response = await fetch(`${API}/summarize/files/compare`, { method: 'POST', body: form });
    if (!response.ok) {
      const body = await response.text();
      throw new Error(body || `HTTP ${response.status}`);
    }
    const data = await response.json();
    applyBatchResults(data, selected);
    const best = data?.best_model?.algorithm || data?.ranking?.[0]?.algorithm || '—';
    addNotification({
      title: t('notifyCompareDone'),
      message: t('notifyCompareDoneMsg', { count: (data?.results || []).length, best }),
      type: 'success',
      link: '/playground',
      showBrowser: true,
    });
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
      if (files.length) {
        await runFilesBatch();
      } else {
        await runTextStream();
      }
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
    <div className="space-y-6 pb-12">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-slate-100 mb-1">{t('playgroundTitle')}</h1>
        <p className="text-sm text-gray-500 dark:text-slate-400">{t('playgroundSubtitle')}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-gray-200 dark:border-slate-700 p-6 h-fit lg:sticky lg:top-4">
          <form onSubmit={runComparison} className="space-y-5">
            <section className="space-y-2">
              <label className="block text-sm font-semibold text-gray-900">Văn bản</label>
              <textarea
                value={text}
                disabled={files.length > 0 || loading}
                onChange={(e) => setText(e.target.value)}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                placeholder="Dán văn bản tiếng Việt..."
                rows={6}
              />
            </section>

            <section className="space-y-2">
              <label className="block text-sm font-semibold text-gray-900">Hoặc tải file (PDF, DOCX, TXT)</label>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-blue-500 transition">
                <input
                  type="file"
                  multiple
                  disabled={loading}
                  onChange={(e) => setFiles(Array.from(e.target.files || []))}
                  className="hidden"
                  id="file-upload"
                  accept=".pdf,.docx,.txt,.doc"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  {files.length > 0
                    ? `${files.length} file(s) selected`
                    : 'Nhấp để chọn file hoặc kéo thả'}
                </label>
              </div>
              {files.length > 0 && (
                <div className="space-y-2">
                  {files.map((f, i) => (
                    <div key={i} className="text-sm text-gray-600 flex justify-between items-center p-2 bg-gray-50 rounded">
                      <span>{f.name}</span>
                      <button
                        type="button"
                        onClick={() => setFiles(files.filter((_, idx) => idx !== i))}
                        className="text-red-600 hover:text-red-700"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </section>

            <section className="space-y-2">
              <label className="block text-sm font-semibold text-gray-900">Tham khảo (tùy chọn)</label>
              <textarea
                value={reference}
                disabled={loading}
                onChange={(e) => setReference(e.target.value)}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                placeholder="Dùng để tính ROUGE..."
                rows={3}
              />
            </section>

            <section className="space-y-2">
              <label className="block text-sm font-semibold text-gray-900">Thuật toán</label>
              <AlgorithmSelector selected={selected} setSelected={setSelected} disabled={loading} />
            </section>

            <section className="grid grid-cols-2 gap-3">
              <label className="space-y-2">
                <span className="block text-xs font-semibold text-gray-600 uppercase">Sentences</span>
                <input
                  type="number"
                  min="1"
                  max="20"
                  disabled={loading}
                  value={sentenceCount}
                  onChange={(e) => setSentenceCount(Number(e.target.value))}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm disabled:opacity-50"
                />
              </label>
              <label className="space-y-2">
                <span className="block text-xs font-semibold text-gray-600 uppercase">Max Tokens</span>
                <input
                  type="number"
                  min="24"
                  max="512"
                  disabled={loading}
                  value={maxLength}
                  onChange={(e) => setMaxLength(Number(e.target.value))}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm disabled:opacity-50"
                />
              </label>
            </section>

            <button
              type="submit"
              disabled={loading || selected.length === 0 || (!text.trim() && !files.length)}
              className="w-full flex items-center justify-center gap-2 bg-blue-600 text-white py-2.5 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              {loading ? <RefreshCcw className="animate-spin" size={17} /> : <Play size={17} />}
              {loading ? t('running') : t('runCompare')}
            </button>

            {error && (
              <div className="text-red-600 text-sm bg-red-50 p-3 rounded-lg flex gap-2">
                <AlertCircle size={18} className="shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </form>
        </div>

        <div className="lg:col-span-2 space-y-5">
          <RunProgress
            loading={loading}
            runningKey={runningKey}
            completed={completedCount}
            total={selected.length}
          />

          <div className="grid gap-4 sm:grid-cols-2">
            {orderedKeys.map(key => (
              <AlgorithmCard
                key={key}
                algoKey={key}
                state={runState[key] || { status: STATUS.idle, result: null, error: null }}
                rank={rankByKey[key]}
              />
            ))}
          </div>

          {result?.warning && (
            <p className="text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-4 py-3">
              {result.warning}
            </p>
          )}

          {rows.length > 0 && <ComparisonTable rows={rows} />}

          {!loading && !rows.length && Object.keys(runState).length === 0 && (
            <div className="rounded-xl border border-dashed border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-900 p-12 text-center">
              <Sparkles className="mx-auto text-gray-300 mb-3" size={32} />
              <p className="text-gray-500">Chạy so sánh để xem bản tóm tắt từng thuật toán</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Playground;
