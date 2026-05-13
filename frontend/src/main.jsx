import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { BarChart3, CheckCircle2, AlertTriangle, FileText, Loader2,
  UploadCloud, Copy, Download, Moon, Sun, Home, Star, Zap, Brain, BookOpen } from 'lucide-react';
import { Bar, Radar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement,
  RadialLinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import './styles.css';

ChartJS.register(CategoryScale, LinearScale, BarElement, RadialLinearScale,
  PointElement, LineElement, Title, Tooltip, Legend);

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

const ALG_META = {
  textrank: { label: 'TextRank', type: 'ext', color: '#10b981' },
  lsa:      { label: 'LSA',      type: 'ext', color: '#06b6d4' },
  lexrank:  { label: 'LexRank',  type: 'ext', color: '#3b82f6' },
  vit5:     { label: 'ViT5',     type: 'abs', color: '#8b5cf6' },
  t5:       { label: 'T5',       type: 'abs', color: '#6366f1' },
  bart:     { label: 'BART',     type: 'abs', color: '#ec4899' },
  pegasus:  { label: 'Pegasus',  type: 'abs', color: '#f59e0b' },
};

// ─── Toast ────────────────────────────────────────────────────────────────────
function useToasts() {
  const [toasts, setToasts] = useState([]);
  const push = useCallback((msg, type = 'default', ms = 3000) => {
    const id = Date.now();
    setToasts(t => [...t, { id, msg, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), ms);
  }, []);
  return { toasts, push };
}

// ─── Algorithm selector ───────────────────────────────────────────────────────
function AlgSelector({ selected, onChange }) {
  const toggle = key => onChange(
    selected.includes(key) ? selected.filter(k => k !== key) : [...selected, key]
  );
  return (
    <div className="algGrid">
      {Object.entries(ALG_META).map(([key, m]) => (
        <label key={key} className={`algCheck ${selected.includes(key) ? 'selected' : ''}`}
          onClick={() => toggle(key)}>
          <span className="algDot" />
          <span>{m.label}</span>
          <span className={`badge badge-${m.type}`} style={{ marginLeft: 'auto' }}>
            {m.type === 'ext' ? 'Ext' : 'Abs'}
          </span>
        </label>
      ))}
    </div>
  );
}

// ─── Metric bar ───────────────────────────────────────────────────────────────
function MetricBar({ value, max = 1, green = false }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div className="metricBar">
      <span className="metricVal">{pct}%</span>
      <div className="bar">
        <div className={`barFill ${green ? 'green' : ''}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// ─── Comparison Table ─────────────────────────────────────────────────────────
function CompareTable({ results }) {
  const rows = Object.values(results).filter(r => r && r.summary);
  if (!rows.length) return null;
  const best = rows.reduce((a, b) =>
    (b.rouge?.rougeL || 0) > (a.rouge?.rougeL || 0) ? b : a, rows[0]);

  return (
    <div className="tableWrap fadeIn">
      <div className="tableHeader">
        <h3>📊 Bảng So Sánh Metrics</h3>
        <span className="tag tag-best"><Star size={11} /> Best: {best.algorithm}</span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="metricsTable">
          <thead>
            <tr>
              <th>Thuật toán</th>
              <th>ROUGE-1</th>
              <th>ROUGE-2</th>
              <th>ROUGE-L</th>
              <th>BLEU</th>
              <th>BERTScore F1</th>
              <th>Sem. Sim</th>
              <th>Thời gian</th>
              <th>Số từ</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => {
              const m = ALG_META[r.algorithm?.toLowerCase()] || {};
              const isBest = r.algorithm === best.algorithm;
              return (
                <tr key={r.algorithm} className={isBest ? 'best-row' : ''}>
                  <td>
                    <span style={{ display:'flex', alignItems:'center', gap:6, fontWeight:600 }}>
                      <span style={{ width:10, height:10, borderRadius:'50%',
                        background: m.color || '#888', display:'inline-block' }} />
                      {m.label || r.algorithm}
                      {isBest && <span className="tag tag-best" style={{fontSize:'0.65rem'}}>★ Best</span>}
                    </span>
                  </td>
                  <td><MetricBar value={r.rouge?.rouge1} /></td>
                  <td><MetricBar value={r.rouge?.rouge2} /></td>
                  <td><MetricBar value={r.rouge?.rougeL} green /></td>
                  <td><span className="metricVal">{r.bleu?.toFixed?.(4) ?? '—'}</span></td>
                  <td><MetricBar value={r.bertscore?.f1} green /></td>
                  <td><MetricBar value={r.semantic_similarity} /></td>
                  <td><span className="metricVal">{r.time_seconds}s</span></td>
                  <td><span className="metricVal">{r.length_words}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ResultCards({ results, push }) {
  const rows = Object.values(results).filter(r => r?.summary);
  const best = rows.reduce((a, b) =>
    (b.rouge?.rougeL || 0) > (a.rouge?.rougeL || 0) ? b : a, rows[0] || {});

  function copy(text) {
    navigator.clipboard?.writeText(text)
      .then(() => push('Đã copy!', 'success'))
      .catch(() => push('Copy thất bại', 'error'));
  }
  function dl(text, name) {
    const a = Object.assign(document.createElement('a'), {
      href: URL.createObjectURL(new Blob([text], { type: 'text/plain' })),
      download: name,
    });
    a.click();
    push('Đã tải xuống', 'success');
  }

  return (
    <div className="cardsGrid fadeIn">
      {rows.map(r => {
        const m = ALG_META[r.algorithm?.toLowerCase()] || {};
        const isBest = r.algorithm === best.algorithm;
        return (
          <div key={r.algorithm} 
               className={`resultCard ${isBest ? 'best-card' : ''}`}
               onClick={() => push && push('detail', r)}
               style={{ cursor: 'pointer' }}>
            <div className="cardTop">
              <span className="cardAlgo">
                <span style={{ width:10, height:10, borderRadius:'50%',
                  background: m.color || '#888', display:'inline-block' }} />
                {m.label || r.algorithm}
                <span className={`badge badge-${m.type || 'ext'}`}>
                  {m.type === 'abs' ? 'Abstractive' : 'Extractive'}
                </span>
              </span>
              {isBest && <span className="tag tag-best"><Star size={10} /> Tốt nhất</span>}
            </div>
            <div className="cardBody">
              <p className="summaryText">{r.summary || '—'}</p>
            </div>
            <div className="cardFooter" onClick={e => e.stopPropagation()}>
              <div className="miniMetrics">
                <span>R-L <strong>{Math.round((r.rouge?.rougeL||0)*100)}%</strong></span>
                <span>BS <strong>{Math.round((r.bertscore?.f1||0)*100)}%</strong></span>
                <span><strong>{r.time_seconds}s</strong></span>
              </div>
              <div className="cardActions">
                <button className="btn-icon" title="Copy" onClick={() => copy(r.summary)}><Copy size={13}/></button>
                <button className="btn-icon" title="Tải TXT" onClick={() => dl(r.summary, `${r.algorithm}.txt`)}><Download size={13}/></button>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Sentence Highlight ────────────────────────────────────────────────────────
function SentenceHighlight({ result, onClose }) {
  if (!result) return null;
  const m = ALG_META[result.algorithm?.toLowerCase()] || { label: result.algorithm };
  const sourceSents = result.source_sentences || [];
  const selectedIdx = result.details?.highlighted_sentence_indexes || [];

  return (
    <div className="highlightPanel fadeIn">
      <div className="highlightHeader">
        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
          <div style={{ width:12, height:12, borderRadius:'50%', background: m.color || '#888' }} />
          <h4>{m.label} — Chi tiết trích xuất</h4>
          <span className="badge badge-ext" style={{fontSize:'0.6rem'}}>Explainability</span>
        </div>
        <button className="btn-icon" onClick={onClose}>✕</button>
      </div>
      <div className="sentenceList">
        {sourceSents.length > 0 ? (
          sourceSents.map((s, i) => {
            const isSelected = selectedIdx.includes(i);
            const detail = result.details?.selected_sentences?.find(d => d.sentence_index === i);
            return (
              <div key={i} className={`sentence ${isSelected ? 'highlighted' : ''}`}>
                <div className="sentNum">{i + 1}</div>
                <div className="sentContent">
                  {s}
                  {isSelected && detail && (
                    <div className="sentReason">
                      Score: <strong>{detail.sentence_score}</strong> · 
                      Match: <strong>{Math.round(detail.match_similarity*100)}%</strong>
                    </div>
                  )}
                </div>
              </div>
            );
          })
        ) : (
          <div className="emptyState" style={{padding:'20px'}}>Không có dữ liệu câu nguồn để hiển thị highlight.</div>
        )}
      </div>
    </div>
  );
}

// ─── Progress Panel ────────────────────────────────────────────────────────────
function ProgressPanel({ progress, algs }) {
  const statusOf = alg => {
    const events = progress.filter(e => e.algorithm === alg.toLowerCase());
    if (events.some(e => e.event === 'error')) return 'error';
    if (events.some(e => e.event === 'done')) return 'done';
    if (events.some(e => e.event === 'running')) return 'running';
    return 'waiting';
  };
  const timeOf = alg => {
    const done = progress.find(e => e.algorithm === alg.toLowerCase() && e.event === 'done');
    return done?.result?.time_seconds ? `${done.result.time_seconds}s` : '';
  };
  return (
    <div className="progressPanel fadeIn">
      <div className="progressTitle">⚡ Tiến trình xử lý</div>
      <div className="progressList">
        {algs.map(alg => {
          const m = ALG_META[alg] || { label: alg };
          const st = statusOf(alg);
          return (
            <div key={alg} className="progressItem">
              <span className={`pStatus ${st}`} />
              <span className="pName">{m.label}</span>
              <span className="tag tag-info" style={{fontSize:'0.65rem'}}>
                {st === 'running' ? '🔄 Đang chạy...' :
                 st === 'done'    ? '✅ Hoàn thành' :
                 st === 'error'   ? '❌ Lỗi' : '⏳ Chờ'}
              </span>
              <span className="pTime">{timeOf(alg)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Charts ────────────────────────────────────────────────────────────────────
function Charts({ results }) {
  const rows = Object.values(results).filter(r => r?.summary);
  if (rows.length < 2) return null;
  const labels = rows.map(r => ALG_META[r.algorithm?.toLowerCase()]?.label || r.algorithm);
  const isDark = document.documentElement.classList.contains('dark');
  const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.06)';
  const textColor = isDark ? '#94a3b8' : '#6b7280';

  const barData = {
    labels,
    datasets: [
      { label: 'ROUGE-1',    data: rows.map(r => Math.round((r.rouge?.rouge1||0)*100)), backgroundColor: 'rgba(99,102,241,.7)' },
      { label: 'ROUGE-2',    data: rows.map(r => Math.round((r.rouge?.rouge2||0)*100)), backgroundColor: 'rgba(139,92,246,.7)' },
      { label: 'ROUGE-L',    data: rows.map(r => Math.round((r.rouge?.rougeL||0)*100)), backgroundColor: 'rgba(16,185,129,.7)' },
      { label: 'BERTScore',  data: rows.map(r => Math.round((r.bertscore?.f1||0)*100)), backgroundColor: 'rgba(245,158,11,.7)' },
    ],
  };
  const opts = {
    responsive: true,
    plugins: { legend: { labels: { color: textColor, font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: textColor }, grid: { color: gridColor } },
      y: { ticks: { color: textColor }, grid: { color: gridColor }, max: 100 },
    },
  };

  const radarData = {
    labels: ['ROUGE-1','ROUGE-2','ROUGE-L','BERTScore','Sem.Sim','Speed'],
    datasets: rows.slice(0,4).map((r, i) => {
      const colors = ['#6366f1','#10b981','#f59e0b','#ec4899'];
      return {
        label: ALG_META[r.algorithm?.toLowerCase()]?.label || r.algorithm,
        data: [
          Math.round((r.rouge?.rouge1||0)*100),
          Math.round((r.rouge?.rouge2||0)*100),
          Math.round((r.rouge?.rougeL||0)*100),
          Math.round((r.bertscore?.f1||0)*100),
          Math.round((r.semantic_similarity||0)*100),
          Math.max(0, 100 - Math.round((r.time_seconds||0)*10)),
        ],
        borderColor: colors[i],
        backgroundColor: colors[i]+'22',
        pointBackgroundColor: colors[i],
      };
    }),
  };
  const radarOpts = {
    responsive: true,
    scales: { r: { ticks: { color: textColor, font:{size:9}, backdropColor:'transparent' }, grid: { color: gridColor }, pointLabels: { color: textColor, font:{size:11} } } },
    plugins: { legend: { labels: { color: textColor, font:{size:11} } } },
  };

  return (
    <div className="chartsGrid fadeIn">
      <div className="chartCard" style={{ gridColumn: 'span 2' }}>
        <h4>Comparison — ROUGE + BERTScore (%)</h4>
        <Bar data={barData} options={opts} />
      </div>
      <div className="chartCard">
        <h4>Radar — Đánh giá tổng hợp</h4>
        <Radar data={radarData} options={radarOpts} />
      </div>
    </div>
  );
}

// ─── Dashboard ─────────────────────────────────────────────────────────────────
function Dashboard({ push }) {
  const [metrics, setMetrics] = useState(null);
  const [viz, setViz] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [m, v] = await Promise.all([
        fetch(`${API}/dashboard/metrics`).then(r => r.json()),
        fetch(`${API}/dashboard/visualization`).then(r => r.json()),
      ]);
      setMetrics(m); setViz(v);
      push('Dashboard đã cập nhật', 'success');
    } catch { push('Không tải được dashboard', 'error'); }
    finally { setLoading(false); }
  }, [push]);

  useEffect(() => { load(); }, [load]);

  if (loading) return <div className="emptyState"><Loader2 className="spin" size={32}/><p>Đang tải...</p></div>;

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <h2 style={{ fontSize:'1rem', fontWeight:700 }}>📈 Dashboard Thống kê</h2>
        <button className="btn btn-ghost" onClick={load} style={{ padding:'6px 12px', fontSize:'0.8rem' }}>Refresh</button>
      </div>
      <div className="statGrid">
        {[
          { label: 'Tổng văn bản', value: metrics?.total_summaries ?? 0, icon: <FileText size={16}/> },
          { label: 'Thời gian (s)', value: metrics?.total_processing_time_seconds ?? 0, icon: <Zap size={16}/> },
          { label: 'Avg ROUGE-L', value: `${Math.round((metrics?.avg_rouge?.rougeL||0)*100)}%`, icon: <Brain size={16}/> },
        ].map(s => (
          <div key={s.label} className="statCard">
            <div style={{ color:'var(--accent)', marginBottom:4 }}>{s.icon}</div>
            <div className="statLabel">{s.label}</div>
            <div className="statValue">{s.value}</div>
          </div>
        ))}
      </div>
      {viz?.labels && (
        <div className="chartCard">
          <h4>ROUGE-L theo Model</h4>
          <Bar data={{
            labels: viz.labels,
            datasets: [{ label:'ROUGE-L %', data: viz.rougeL_avg?.map(v=>Math.round(v*100)), backgroundColor:'rgba(99,102,241,.7)' }]
          }} options={{ responsive:true, plugins:{ legend:{display:false} } }} />
        </div>
      )}
    </div>
  );
}

// ─── Benchmark View ────────────────────────────────────────────────────────────
function BenchmarkView({ push }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/benchmark/results`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => { push('Không tải được benchmark', 'error'); setLoading(false); });
  }, [push]);

  if (loading) return <div className="emptyState"><Loader2 className="spin" size={28}/><p>Đang tải...</p></div>;
  if (!data?.benchmark_results?.length)
    return <div className="emptyState"><BookOpen size={36} strokeWidth={1.2}/><h2>Chưa có kết quả benchmark</h2><p>Chạy: <code>python -m src.benchmark --samples 100</code></p></div>;

  const bench = data.benchmark_results[0];
  const rows  = bench.comparison || [];
  const isDark = document.documentElement.classList.contains('dark');
  const gc     = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.06)';
  const tc     = isDark ? '#94a3b8' : '#6b7280';

  const barData = {
    labels: rows.map(r => r.algorithm.replace(' (fine-tuned, 5000 samples)', '*').replace(' (pretrained)','').replace('BART-large-CNN','BART')),
    datasets: [
      { label:'ROUGE-1',   data: rows.map(r => Math.round(r.avg_rouge1*100)),      backgroundColor:'rgba(99,102,241,.75)' },
      { label:'ROUGE-2',   data: rows.map(r => Math.round(r.avg_rouge2*100)),      backgroundColor:'rgba(139,92,246,.75)' },
      { label:'ROUGE-L',   data: rows.map(r => Math.round(r.avg_rougeL*100)),      backgroundColor:'rgba(16,185,129,.75)' },
      { label:'BERTScore', data: rows.map(r => Math.round(r.avg_bertscore_f1*100)),backgroundColor:'rgba(245,158,11,.75)' },
    ],
  };
  const opts = {
    responsive:true,
    plugins:{ legend:{ labels:{ color:tc, font:{size:11} } }, title:{display:true, text:'Benchmark — VnExpress 100 samples', color:tc} },
    scales:{ x:{ticks:{color:tc},grid:{color:gc}}, y:{ticks:{color:tc},grid:{color:gc},max:100,title:{display:true,text:'Score (%)',color:tc}} },
  };

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <h2 style={{ fontSize:'1rem', fontWeight:700 }}>📋 Kết quả Benchmark</h2>
        <div style={{ fontSize:'0.75rem', color:'var(--text3)' }}>
          Dataset: <strong>{bench.dataset}</strong> · Samples: <strong>{bench.samples}</strong>
        </div>
      </div>

      {/* Chart */}
      <div className="chartCard">
        <Bar data={barData} options={opts}/>
      </div>

      {/* Table */}
      <div className="tableWrap">
        <div className="tableHeader"><h3>So sánh chi tiết</h3></div>
        <div style={{ overflowX:'auto' }}>
          <table className="metricsTable">
            <thead>
              <tr>
                <th>Thuật toán</th><th>Loại</th>
                <th>ROUGE-1</th><th>ROUGE-2</th><th>ROUGE-L</th>
                <th>BLEU</th><th>BERTScore F1</th><th>Sem.Sim</th>
                <th>Time avg</th><th>Len avg</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => {
                const bestRl = Math.max(...rows.map(x => x.avg_rougeL));
                const isBest = r.avg_rougeL === bestRl;
                return (
                  <tr key={i} className={isBest ? 'best-row' : ''}>
                    <td style={{ fontWeight:600 }}>
                      {r.algorithm.replace(' (fine-tuned, 5000 samples)','*').replace(' (pretrained)','')}
                      {isBest && <span className="tag tag-best" style={{marginLeft:6,fontSize:'0.65rem'}}>★ Best</span>}
                    </td>
                    <td><span className={`badge badge-${r.type==='abstractive'?'abs':'ext'}`}>{r.type==='abstractive'?'Abstractive':'Extractive'}</span></td>
                    <td><MetricBar value={r.avg_rouge1}/></td>
                    <td><MetricBar value={r.avg_rouge2}/></td>
                    <td><MetricBar value={r.avg_rougeL} green/></td>
                    <td><span className="metricVal">{r.avg_bleu?.toFixed(4)??'—'}</span></td>
                    <td><MetricBar value={r.avg_bertscore_f1} green/></td>
                    <td><MetricBar value={r.avg_semantic_sim}/></td>
                    <td><span className="metricVal">{r.avg_time_seconds}s</span></td>
                    <td><span className="metricVal">{r.avg_length_words} từ</span></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Key findings */}
      {bench.key_findings?.length > 0 && (
        <div className="highlightPanel">
          <div className="highlightHeader"><h4>💡 Nhận xét chính</h4></div>
          <div style={{ padding:'12px 16px', display:'flex', flexDirection:'column', gap:6 }}>
            {bench.key_findings.map((f, i) => (
              <div key={i} style={{ display:'flex', gap:8, fontSize:'0.83rem', lineHeight:1.55 }}>
                <span style={{ color:'var(--green)', fontWeight:700, flexShrink:0 }}>✓</span>
                <span>{f}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


// ─── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [files, setFiles]           = useState([]);
  const [text, setText]             = useState('');
  const [urls, setUrls]             = useState('');
  const [algs, setAlgs]             = useState(['textrank','lsa','lexrank','vit5']);
  const [results, setResults]       = useState({});
  const [progress, setProgress]     = useState([]);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState('');
  const [dark, setDark]             = useState(() => localStorage.getItem('dark')==='1');
  const [view, setView]             = useState('summarize'); // 'summarize' | 'dashboard' | 'benchmark'
  const [selectedResult, setSelectedResult] = useState(null);
  const { toasts, push }            = useToasts();
  const ctrl                        = useRef(null);

  useEffect(() => {
    if (dark) { document.documentElement.classList.add('dark'); localStorage.setItem('dark','1'); }
    else      { document.documentElement.classList.remove('dark'); localStorage.removeItem('dark'); }
  }, [dark]);

  function handleEvent(obj) {
    if (!obj?.event) return;
    setProgress(p => [...p, obj]);
    if (obj.event === 'done' && obj.algorithm && obj.result)
      setResults(prev => ({ ...prev, [obj.algorithm]: obj.result }));
    if (obj.event === 'finished') {
      (obj.data?.results || obj.result?.results || []).forEach(r => {
        setResults(prev => ({ ...prev, [r.algorithm]: r }));
      });
      push('So sánh hoàn tất! 🎉', 'success', 4000);
    }
    if (obj.event === 'error') push(`Lỗi: ${obj.error}`, 'error', 6000);
  }

  async function stream(url, init) {
    const res = await fetch(url, init);
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status}: ${body}`);
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    ctrl.current = reader;
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const part of parts) {
        const m = part.match(/data:\s*(.*)/s);
        if (m) { try { handleEvent(JSON.parse(m[1])); } catch {} }
      }
    }
  }

  async function submit(e) {
    e.preventDefault();
    setError(''); setResults({}); setProgress([]); setLoading(true);
    try {
      if (files.length > 0) {
        const fd = new FormData();
        files.forEach(f => fd.append('files', f));
        fd.append('algorithms', JSON.stringify(algs));
        await stream(`${API}/summarize/files/compare/stream`, { method:'POST', body: fd });
      } else {
        await stream(`${API}/summarize/compare/stream`, {
          method: 'POST',
          headers: { 'Content-Type':'application/json' },
          body: JSON.stringify({
            text: text.trim() || null,
            urls: urls.split('\n').map(u=>u.trim()).filter(Boolean),
            algorithms: algs,
            extractive_sentences: 5,
            max_abstractive_length: 150,
          }),
        });
      }
    } catch (err) { setError(err.message || String(err)); }
    finally { setLoading(false); ctrl.current = null; }
  }

  const hasInput = files.length > 0 || text.trim() || urls.trim();

  return (
    <div className="app">
      {/* Top Bar */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="topbar-logo">NLP</div>
          <div>
            <div className="topbar-title">Vietnamese Text Summarization</div>
            <div className="topbar-sub">So sánh 7 thuật toán · ROUGE · BERTScore · Realtime Streaming</div>
          </div>
        </div>
        <div className="topbar-nav">
          <button className={`btn btn-ghost ${view==='summarize'?'active':''}`}
            onClick={() => setView('summarize')} style={{gap:5}}>
            <Home size={14}/> Tóm tắt
          </button>
          <button className={`btn btn-ghost ${view==='dashboard'?'active':''}`}
            onClick={() => setView('dashboard')} style={{gap:5}}>
            <BarChart3 size={14}/> Dashboard
          </button>
          <button className={`btn btn-ghost ${view==='benchmark'?'active':''}`}
            onClick={() => setView('benchmark')} style={{gap:5}}>
            <BookOpen size={14}/> Benchmark
          </button>
          <button className="btn btn-ghost" onClick={() => setDark(d=>!d)} style={{width:36,padding:0}}>
            {dark ? <Sun size={14}/> : <Moon size={14}/>}
          </button>
        </div>
      </header>

      <div className="workspace">
        {/* Sidebar */}
        <aside className="sidePanel">
          {view === 'summarize' && (
            <form onSubmit={submit} style={{ display:'flex', flexDirection:'column', gap:14 }}>
              {/* Upload */}
              <div className="panelSection">
                <span className="sectionLabel">📂 Upload tài liệu</span>
                <label className={`dropZone ${files.length ? 'active' : ''}`}>
                  <UploadCloud size={22}/>
                  <span>{files.length ? `${files.length} file đã chọn` : 'TXT / PDF / DOCX'}</span>
                  <span style={{fontSize:'0.72rem', color:'var(--text3)'}}>Kéo thả hoặc click để chọn</span>
                  <input type="file" multiple accept=".txt,.pdf,.docx"
                    onChange={e => setFiles(Array.from(e.target.files||[]))}/>
                </label>
                {files.length > 0 && (
                  <button type="button" className="btn btn-ghost" style={{fontSize:'0.78rem',padding:'5px'}}
                    onClick={() => setFiles([])}>✕ Xóa files</button>
                )}
              </div>

              <div className="divider"/>

              {/* Text / URLs */}
              <div className="panelSection">
                <span className="sectionLabel">✍️ Hoặc nhập văn bản</span>
                <textarea value={text} onChange={e=>setText(e.target.value)}
                  placeholder="Dán văn bản tiếng Việt vào đây..." rows={6}
                  disabled={files.length > 0}/>
              </div>

              <div className="panelSection">
                <span className="sectionLabel">🔗 Hoặc URL bài báo</span>
                <textarea value={urls} onChange={e=>setUrls(e.target.value)}
                  placeholder="Mỗi dòng một URL..." rows={3}
                  disabled={files.length > 0}/>
              </div>

              <div className="divider"/>

              {/* Algorithm selector */}
              <div className="panelSection">
                <span className="sectionLabel">⚙️ Chọn thuật toán ({algs.length}/7)</span>
                <AlgSelector selected={algs} onChange={setAlgs}/>
                <div style={{display:'flex',gap:6,marginTop:4}}>
                  <button type="button" className="btn btn-ghost" style={{flex:1,fontSize:'0.75rem',padding:'5px'}}
                    onClick={()=>setAlgs(Object.keys(ALG_META))}>Tất cả</button>
                  <button type="button" className="btn btn-ghost" style={{flex:1,fontSize:'0.75rem',padding:'5px'}}
                    onClick={()=>setAlgs(['textrank','vit5'])}>Mặc định</button>
                </div>
              </div>

              <div className="divider"/>

              <button className="btn btn-primary" disabled={loading || !hasInput || algs.length===0}>
                {loading ? <><Loader2 className="spin" size={16}/> Đang xử lý...</> : <><Zap size={16}/> Chạy & So sánh</>}
              </button>

              {error && <div className="errorBox">⚠️ {error}</div>}
            </form>
          )}

          {view === 'dashboard' && (
            <div style={{padding:'4px 0', color:'var(--text3)', fontSize:'0.82rem'}}>
              Thống kê lịch sử các lần tóm tắt
            </div>
          )}
        </aside>

        {/* Result Pane */}
        <main className="resultPane">
          {view === 'dashboard' && <Dashboard push={push}/>}
          {view === 'benchmark' && <BenchmarkView push={push}/>}

          {view === 'summarize' && (
            <>
              {!loading && !Object.keys(results).length && !progress.length && (
                <div className="emptyState">
                  <Brain size={40} strokeWidth={1.2}/>
                  <h2>Sẵn sàng phân tích văn bản</h2>
                  <p>Upload file, nhập văn bản hoặc URL. Chọn thuật toán rồi nhấn "Chạy & So sánh" để xem kết quả ROUGE, BERTScore và so sánh trực quan.</p>
                </div>
              )}

              {(loading || progress.length > 0) && (
                <ProgressPanel progress={progress} algs={algs}/>
              )}

              {Object.keys(results).length > 0 && (
                <>
                  <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                    <h3 style={{ fontSize:'0.9rem', fontWeight:700 }}>✨ Kết quả phân tích</h3>
                    <button className="btn btn-ghost" style={{padding:'4px 10px', fontSize:'0.75rem'}} 
                            onClick={() => { setResults({}); setProgress([]); }}>Xóa kết quả</button>
                  </div>
                  <CompareTable results={results}/>
                  <Charts results={results}/>
                  <ResultCards results={results} push={(type, data) => {
                    if (type === 'detail') setSelectedResult(data);
                    else push(type, data);
                  }}/>
                  {selectedResult && (
                    <SentenceHighlight result={selectedResult} onClose={() => setSelectedResult(null)} />
                  )}
                </>
              )}
            </>
          )}
        </main>
      </div>

      {/* Toasts */}
      <div className="toastWrap">
        {toasts.map(t => (
          <div key={t.id} className={`toast ${t.type}`}>
            {t.type==='success' ? <CheckCircle2 size={13}/> : t.type==='error' ? <AlertTriangle size={13}/> : null}
            {t.msg}
          </div>
        ))}
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App/>);
