import React, { useMemo, useState, useRef, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  FileText,
  Gauge,
  Highlighter,
  Loader2,
  UploadCloud,
  Copy,
  Download,
  Moon,
  Sun,
  Home,
  X,
  Info,
} from 'lucide-react';
import './styles.css';

import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar, Line, Pie, Radar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, LineElement, ArcElement, RadialLinearScale, Title, Tooltip, Legend);

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function App() {
  const [files, setFiles] = useState([]);
  const [text, setText] = useState('');
  const [urls, setUrls] = useState('');
  const [algorithms, setAlgorithms] = useState(['textrank', 'lsa', 'lexrank', 'vit5', 't5', 'bart', 'pegasus']);
  const [lengthControl, setLengthControl] = useState('100_words');
  const [modelName, setModelName] = useState('vit5');
  const [analysisMode, setAnalysisMode] = useState('fast');
  const [compareResults, setCompareResults] = useState({});
  const [progress, setProgress] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [chartsData, setChartsData] = useState(null);
  const streamController = useRef(null);
  const [dark, setDark] = useState(() => localStorage.getItem('dark') === '1');
  const [view, setView] = useState('summarize'); // or 'dashboard'
  const [toasts, setToasts] = useState([]);

  useEffect(() => {
    try {
      if (dark) {
        document.documentElement.classList.add('dark');
        document.body.classList.add('dark');
        localStorage.setItem('dark', '1');
      } else {
        document.documentElement.classList.remove('dark');
        document.body.classList.remove('dark');
        localStorage.removeItem('dark');
      }
    } catch (e) {}
  }, [dark]);

  const pushToast = useCallback((msg, type = 'default', ms = 3000) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, msg, type }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), ms);
  }, []);

  const hasFiles = files.length > 0;

  async function submit(event) {
    event.preventDefault();
    setError('');
    setCompareResults({});
    setChartsData(null);
    setProgress([]);
    setLoading(true);

    try {
      if (hasFiles) {
        await submitFilesStream(files);
      } else {
        await submitTextStream(text, urls);
      }
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
      streamController.current = null;
    }
  }

  // Streaming POST helper: read text/event-stream styled chunks
  async function fetchStream(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    streamController.current = reader;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let parts = buffer.split('\n\n');
      buffer = parts.pop();
      for (const part of parts) {
        if (!part.trim()) continue;
        // we expect lines like: data: {json}\n
        const m = part.match(/data:\s*(.*)/s);
        if (!m) continue;
        try {
          const obj = JSON.parse(m[1]);
          handleEvent(obj);
        } catch (e) {
          console.warn('Invalid event payload', e, part);
        }
      }
    }
    // process leftover
    if (buffer.trim()) {
      const m = buffer.match(/data:\s*(.*)/s);
      if (m) {
        try {
          const obj = JSON.parse(m[1]);
          handleEvent(obj);
        } catch (e) {
          console.warn('Invalid final payload', e);
        }
      }
    }
  }

  function handleEvent(obj) {
    if (!obj || !obj.event) return;
    setProgress((p) => [...p, obj]);

    // Running status for an algorithm
    if (obj.event === 'running') {
      if (obj.algorithm) pushToast(`Đang chạy ${obj.algorithm}…`, 'default', 2000);
      return;
    }

    // Single algorithm finished
    if (obj.event === 'done') {
      const data = obj.result || obj.data || obj;
      if (obj.algorithm) {
        setCompareResults((prev) => ({ ...prev, [obj.algorithm]: obj.result || obj.data }));
        const t = data?.time_seconds ? ` (${data.time_seconds}s)` : '';
        pushToast(`${obj.algorithm} hoàn thành${t}`, 'success', 2800);
      } else {
        pushToast('Một thuật toán hoàn thành', 'success', 2000);
      }
      return;
    }

    // Final aggregated results
    if (obj.event === 'finished') {
      const data = obj.result || obj.data || obj;
      const all = data.results || [];
      const map = {};
      all.forEach((r) => { map[r.algorithm] = r; });
      setCompareResults(map);
      buildCharts(all);
      pushToast('So sánh hoàn tất', 'success', 3500);
      return;
    }

    // Error events
    if (obj.event === 'error') {
      const msg = obj.error || 'Lỗi khi chạy thuật toán';
      setError(msg);
      pushToast(`Lỗi: ${msg}`, 'error', 6000);
      return;
    }
  }

  function buildCharts(results) {
    if (!results || !results.length) return;
    const labels = results.map((r) => r.algorithm);
    const rouge = results.map((r) => (r.rouge?.rougeL ? r.rouge.rougeL * 100 : 0));
    const times = results.map((r) => r.time_seconds || 0);
    const lengths = results.map((r) => r.length_words || 0);
    setChartsData({ labels, rouge, times, lengths });
  }

  async function submitTextStream(textValue, urlsValue) {
    const payload = {
      text: textValue.trim() || null,
      urls: urlsValue.split('\n').map((u) => u.trim()).filter(Boolean),
      algorithms,
      extractive_sentences: 5,
      max_abstractive_length: 150,
      save_result: true,
    };

    await fetchStream(`${API_BASE}/summarize/compare/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    pushToast('Streaming started', 'success');
  }

  async function submitFilesStream(files) {
    const form = new FormData();
    files.forEach((f) => form.append('files', f));
    form.append('algorithms', JSON.stringify(algorithms));
    // note: backend streaming for files uses the same compare/stream route
    await fetchStream(`${API_BASE}/summarize/compare/stream`, {
      method: 'POST',
      body: form,
    });
    pushToast('Files uploaded; streaming started', 'success');
  }

  function copyToClipboard(text) {
    if (!text) return pushToast('Không có nội dung để copy', 'error');
    navigator.clipboard?.writeText(text).then(() => pushToast('Đã copy vào clipboard', 'success')).catch(() => pushToast('Copy thất bại', 'error'));
  }

  function downloadText(text, filename = 'summary.txt') {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    pushToast('Download started', 'success');
  }

  function exportJSON(obj, filename = 'summary.json') {
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    pushToast('Exported JSON', 'success');
  }

  return (
    <main className="app">
      <section className="workspace">
        <aside className="inputPane">
          <div className="brand">
            <div className="brandMark"><FileText size={22} /></div>
            <div>
              <div className="brandHeader">
                <h1>AI Summarization Dashboard</h1>
                <div style={{ marginLeft: 8 }}>
                  <button className="secondaryButton" onClick={() => setDark((d) => !d)} title="Toggle dark mode">
                    {dark ? <Sun size={14} /> : <Moon size={14} />}
                  </button>
                </div>
              </div>
              <p>So sánh nhiều thuật toán. Thực thi song song, realtime progress.</p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
            <button className={`secondaryButton ${view === 'summarize' ? 'active' : ''}`} onClick={() => setView('summarize')}><Home size={14} /> Summarize</button>
            <button className={`secondaryButton ${view === 'dashboard' ? 'active' : ''}`} onClick={() => setView('dashboard')}><BarChart3 size={14} /> Dashboard</button>
          </div>

          {view === 'summarize' && (
            <form onSubmit={submit} className="controlStack">
              <label className="dropZone">
                <UploadCloud size={28} />
                <span>{hasFiles ? `${files.length} file đã chọn` : 'Upload TXT, PDF, DOCX'}</span>
                <input
                  type="file"
                  multiple
                  accept=".txt,.pdf,.docx"
                  onChange={(event) => setFiles(Array.from(event.target.files || []))}
                />
              </label>

              <textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                placeholder="Hoặc nhập văn bản trực tiếp tại đây"
                rows={7}
                disabled={hasFiles}
              />

              <textarea
                value={urls}
                onChange={(event) => setUrls(event.target.value)}
                placeholder="URL bài viết, mỗi dòng một URL"
                rows={4}
                disabled={hasFiles}
              />

              <div className="fieldGrid">
                <label>
                  Algorithms
                  <select value={algorithms.join(',')} onChange={(e) => setAlgorithms(e.target.value.split(',').map(s => s.trim()))}>
                    <option value={["textrank","lsa","lexrank","vit5","t5","bart","pegasus"]}>All (TextRank, LSA, LexRank, ViT5, T5, BART, Pegasus)</option>
                  </select>
                </label>
              </div>

              <button className="primaryButton" disabled={loading || (!hasFiles && !text.trim() && !urls.trim())}>
                {loading ? <Loader2 className="spin" size={18} /> : <Gauge size={18} />}
                <span>{loading ? 'Đang xử lý...' : 'Chạy & So sánh thuật toán'}</span>
              </button>

              {error && <div className="errorBox">{error}</div>}
            </form>
          )}
        </aside>

        <section className="resultPane">
          {view === 'dashboard' && <Dashboard pushToast={pushToast} />}

          {view === 'summarize' && (
            <>
              {!loading && !Object.keys(compareResults).length && <EmptyState />}
              {loading && <LoadingState />}

              {Object.keys(compareResults).length > 0 && (
                <div>
                  <div className="cardsRow">
                    {Object.entries(compareResults).map(([alg, data]) => (
                      <div key={alg} className="card">
                        <div className="cardHeader">
                          <h3>{alg.toUpperCase()}</h3>
                          <span className="time">{(data.time_seconds || 0) + 's'}</span>
                        </div>
                        <div className="cardBody">
                          <p className="summaryText">{data.summary || '—'}</p>
                        </div>
                        <div className="cardFooter" style={{ display: 'flex', gap: 8, justifyContent: 'space-between', alignItems: 'center' }}>
                          <div>
                            <small>ROUGE-L: {data.rouge?.rougeL ? Math.round(data.rouge.rougeL * 100) + '%' : '—'}</small>
                            <small style={{ marginLeft: 8 }}>BLEU: {data.bleu ?? '—'}</small>
                            <small style={{ marginLeft: 8 }}>Sim: {data.semantic_similarity ?? '—'}</small>
                          </div>
                          <div style={{ display: 'flex', gap: 8 }}>
                            <button className="secondaryButton" onClick={() => copyToClipboard(data.summary)} title="Copy summary"><Copy size={14} /></button>
                            <button className="secondaryButton" onClick={() => exportJSON(data, `${alg}_summary.json`)} title="Export JSON"><FileText size={14} /></button>
                            <button className="secondaryButton" onClick={() => downloadText(data.summary || '', `${alg}_summary.txt`)} title="Download"><Download size={14} /></button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {chartsData && (
                    <div className="chartsRow">
                      <div className="chartCard">
                        <h4>ROUGE-L Comparison</h4>
                        <Bar data={{ labels: chartsData.labels, datasets: [{ label: 'ROUGE-L %', data: chartsData.rouge }] }} />
                      </div>

                      <div className="chartCard">
                        <h4>Processing Time (s)</h4>
                        <Line data={{ labels: chartsData.labels, datasets: [{ label: 'Time (s)', data: chartsData.times, borderColor: '#3b82f6' }] }} />
                      </div>

                      <div className="chartCard">
                        <h4>Summary Length</h4>
                        <Pie data={{ labels: chartsData.labels, datasets: [{ data: chartsData.lengths }] }} />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          <div className="toastWrap">
            {toasts.map((t) => (
              <div key={t.id} className={`toast ${t.type}`}>
                {t.msg}
              </div>
            ))}
          </div>
        </section>
      </section>
    </main>
  );
}

function EmptyState() {
  return (
    <div className="emptyState">
      <Highlighter size={34} />
      <h2>Upload nhiều tài liệu để bắt đầu</h2>
      <p>Kết quả sẽ có summary tổng hợp, summary từng file, câu nguồn được highlight và các nhận định nghi vấn.</p>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="emptyState">
      <Loader2 className="spin" size={34} />
      <h2>Đang chạy pipeline NLP</h2>
      <p>Hệ thống đang trích xuất, tóm tắt, so sánh ROUGE và kiểm chứng consistency.</p>
    </div>
  );
}

function Dashboard({ pushToast }) {
  const [metrics, setMetrics] = useState(null);
  const [viz, setViz] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    try {
      const [mRes, vRes] = await Promise.all([
        axios.get(`${API_BASE}/dashboard/metrics`),
        axios.get(`${API_BASE}/dashboard/visualization`),
      ]);
      setMetrics(mRes.data || {});
      setViz(vRes.data || {});
      pushToast('Dashboard updated', 'success');
    } catch (e) {
      pushToast('Không thể tải dashboard', 'error');
    } finally {
      setLoading(false);
    }
  }, [pushToast]);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  if (loading) {
    return (
      <div className="panel">
        <div className="metricRow">
          <div className="metric glassCard">
            <div className="skeleton text" style={{ height: 22, width: '50%' }} />
            <div className="skeleton line" />
          </div>
          <div className="metric glassCard">
            <div className="skeleton text" style={{ height: 22, width: '40%' }} />
            <div className="skeleton line" />
          </div>
          <div className="metric glassCard">
            <div className="skeleton text" style={{ height: 22, width: '30%' }} />
            <div className="skeleton line" />
          </div>
        </div>
      </div>
    );
  }

  const top = metrics?.top_models || [];
  return (
    <div className="panel">
      <div className="panelHeader">
        <h2>Dashboard</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="secondaryButton" onClick={fetchAll}>Refresh</button>
        </div>
      </div>

      <div className="metricRow">
        <div className="metric">
          <span>Tổng văn bản</span>
          <div className="counter">{metrics.total_summaries}</div>
        </div>
        <div className="metric">
          <span>Tổng thời gian (s)</span>
          <div className="counter">{metrics.total_processing_time_seconds}</div>
        </div>
        <div className="metric">
          <span>Avg ROUGE-L</span>
          <div className="counter">{Math.round((metrics.avg_rouge?.rougeL || 0) * 100)}%</div>
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        <h4>Top models</h4>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {top.map((m) => (
            <div key={m.model} className="card" style={{ padding: 8 }}>
              <strong>{m.model}</strong>
              <div style={{ fontSize: 12, color: '#6c7480' }}>{m.count} runs</div>
            </div>
          ))}
        </div>
      </div>

      {viz && viz.labels && (
        <div className="chartsRow" style={{ marginTop: 20 }}>
          <div className="chartCard">
            <h4>ROUGE-L by Model</h4>
            <Bar data={{ labels: viz.labels, datasets: [{ label: 'ROUGE-L', data: viz.rougeL_avg.map(v => v * 100) }] }} />
          </div>
          <div className="chartCard">
            <h4>Processing Time</h4>
            <Line data={{ labels: viz.labels, datasets: [{ label: 'Time (s)', data: viz.time_avg }] }} />
          </div>
          <div className="chartCard">
            <h4>Summary Length</h4>
            <Pie data={{ labels: viz.labels, datasets: [{ data: viz.length_avg }] }} />
          </div>
        </div>
      )}
    </div>
  );
}

function Metric({ icon, label, value }) {
  return (
    <div className="metric">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatusPill({ status }) {
  const ok = status === 'consistent';
  return (
    <span className={`statusPill ${ok ? 'ok' : 'warn'}`}>
      {ok ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
      {status}
    </span>
  );
}

function HighlightedSummary({ summary, spans = [] }) {
  if (!spans.length) return <p className="summaryText">{summary}</p>;
  const suspicious = new Set(spans.map((span) => span.text));
  const sentences = splitSentences(summary);
  return (
    <p className="summaryText">
      {sentences.map((sentence, index) => (
        suspicious.has(sentence)
          ? <mark key={`${sentence}-${index}`} className="suspicious">{sentence}</mark>
          : <span key={`${sentence}-${index}`}>{sentence} </span>
      ))}
    </p>
  );
}

function ConsistencyList({ checks = [] }) {
  return (
    <div className="checkList">
      {checks.map((check, index) => (
        <details key={`${check.summary_sentence}-${index}`} className={`checkItem ${check.status}`}>
          <summary>
            <strong>{check.summary_sentence}</strong>
            <span>{check.support_percent ?? Math.round(check.support_score * 100)}%</span>
          </summary>
          <p>{check.reason}</p>
          {(check.evidence || []).map((item) => (
            <small key={`${item.index}-${item.score}`}>Evidence {item.index + 1} ({Math.round(item.score * 100)}%): {item.sentence}</small>
          ))}
        </details>
      ))}
    </div>
  );
}

function DocumentDetail({ document }) {
  const highlighted = new Set(document.explainability?.highlighted_sentence_indexes || []);
  const sentences = document.explainability?.sentences || [];
  const highlights = document.explainability?.highlights || [];
  const reasons = useMemo(() => new Map(highlights.map((item) => [item.source_index, item.reason])), [highlights]);

  return (
    <div className="documentDetail">
      <div className="documentSummary">
        <h3>{document.summary_type} summary</h3>
        <p>{document.summary}</p>
        <StatusPill status={document.consistency_status === 'consistent' ? 'consistent' : 'needs_review'} />
      </div>
      <div className="sourceSentences">
        {sentences.map((sentence, index) => (
          <div key={`${sentence}-${index}`} className={highlighted.has(index) ? 'sourceSentence selected' : 'sourceSentence'}>
            <span>{index + 1}</span>
            <p>{sentence}</p>
            {highlighted.has(index) && <small>{reasons.get(index)}</small>}
          </div>
        ))}
      </div>
    </div>
  );
}

function splitSentences(text) {
  return (text.match(/[^.!?]+[.!?]?/g) || [text]).map((item) => item.trim()).filter(Boolean);
}

function getBestRouge(result) {
  const best = result?.scores?.[result.best_type];
  if (!best || (best.rougeL !== 0 && !best.rougeL)) return result.best_type;
  return `${Math.round(best.rougeL * 100)}%`;
}

createRoot(document.getElementById('root')).render(<App />);
