import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  BarChart3,
  Brain,
  Check,
  ChevronDown,
  Copy,
  FileText,
  Moon,
  Play,
  RefreshCcw,
  Sun,
  UploadCloud,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import './styles.css';

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

const ALGORITHMS = [
  { key: 'textrank', name: 'TextRank', group: 'extractive', color: '#14b8a6' },
  { key: 'lexrank', name: 'LexRank', group: 'extractive', color: '#38bdf8' },
  { key: 'lsa', name: 'LSA Summarizer', group: 'extractive', color: '#84cc16' },
  { key: 'vit5', name: 'ViT5', group: 'abstractive', color: '#f59e0b' },
  { key: 'mt5', name: 'mT5', group: 'abstractive', color: '#e879f9' },
  { key: 'bartpho', name: 'BARTPho', group: 'abstractive', color: '#fb7185' },
];

const SAMPLE_TEXT = `Tập đoàn Điện lực Việt Nam cho biết nhu cầu tiêu thụ điện trong mùa nắng nóng tiếp tục tăng cao tại nhiều địa phương. Các nhà máy thủy điện ở miền Bắc được yêu cầu vận hành thận trọng do mực nước một số hồ chứa chưa phục hồi hoàn toàn. Bộ Công Thương đề nghị các đơn vị cung ứng điện xây dựng kịch bản điều độ linh hoạt, đồng thời khuyến khích doanh nghiệp và hộ gia đình tiết kiệm điện trong giờ cao điểm. Một số dự án năng lượng tái tạo cũng được rà soát để bổ sung nguồn cung cho hệ thống. Các chuyên gia cho rằng việc kết hợp tiết kiệm năng lượng, nâng cấp lưới truyền tải và đa dạng hóa nguồn phát là giải pháp quan trọng nhằm bảo đảm an ninh năng lượng trong dài hạn.`;

function pct(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function metric(row, key) {
  return Number(row?.metrics?.[key] ?? row?.[key] ?? 0);
}

function byKey(key) {
  return ALGORITHMS.find((item) => item.key === key) || { key, name: key, group: 'extractive', color: '#64748b' };
}

function groupLabel(group) {
  return group === 'abstractive' ? 'Abstractive' : 'Extractive';
}

function pillClass(group, selected = false) {
  const base = 'inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm transition';
  if (selected && group === 'extractive') return `${base} border-teal-400/70 bg-teal-400/10 text-teal-100`;
  if (selected) return `${base} border-amber-300/70 bg-amber-300/10 text-amber-100`;
  return `${base} border-slate-700 bg-slate-900/40 text-slate-400 hover:border-slate-500`;
}

function AlgorithmSelector({ selected, setSelected }) {
  const toggle = (key) => {
    setSelected((current) => (current.includes(key) ? current.filter((item) => item !== key) : [...current, key]));
  };

  return (
    <div className="space-y-4">
      {['extractive', 'abstractive'].map((group) => (
        <section key={group} className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{groupLabel(group)}</h3>
            <span className="text-xs text-slate-500">{selected.filter((key) => byKey(key).group === group).length}/3</span>
          </div>
          <div className="grid grid-cols-1 gap-2">
            {ALGORITHMS.filter((item) => item.group === group).map((item) => {
              const isSelected = selected.includes(item.key);
              return (
                <button type="button" key={item.key} onClick={() => toggle(item.key)} className={pillClass(group, isSelected)}>
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: item.color }} />
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
}

function StatStrip({ result }) {
  const rows = result?.results || [];
  const best = result?.best_model;
  const ext = result?.group_summary?.extractive;
  const abs = result?.group_summary?.abstractive;
  const stats = [
    { label: 'Models', value: rows.length || 0 },
    { label: 'Best', value: best?.algorithm || '-' },
    { label: 'Ext ROUGE-L', value: ext ? pct(ext.avg_rougeL) : '-' },
    { label: 'Abs ROUGE-L', value: abs ? pct(abs.avg_rougeL) : '-' },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {stats.map((item) => (
        <div key={item.label} className="rounded-lg border border-slate-800 bg-slate-950/70 p-4">
          <div className="text-xs uppercase tracking-[0.16em] text-slate-500">{item.label}</div>
          <div className="mt-2 truncate text-xl font-semibold text-slate-100">{item.value}</div>
        </div>
      ))}
    </div>
  );
}

function ComparisonTable({ rows, onSelect, selectedKey }) {
  if (!rows.length) return null;
  const bestByMetric = (key) => Math.max(...rows.map((row) => metric(row, key)));
  const best = {
    rouge1: bestByMetric('rouge1'),
    rouge2: bestByMetric('rouge2'),
    rougeL: bestByMetric('rougeL'),
    bleu: bestByMetric('bleu'),
    bertscore_f1: bestByMetric('bertscore_f1'),
    semantic_similarity: bestByMetric('semantic_similarity'),
  };

  const scoreCell = (row, key) => {
    const value = metric(row, key);
    const isBest = value === best[key] && value > 0;
    return <td className={isBest ? 'best-cell' : ''}>{pct(value)}</td>;
  };

  return (
    <div className="overflow-hidden rounded-lg border border-slate-800 bg-slate-950/70">
      <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
          <BarChart3 size={16} /> Metrics
        </div>
        <div className="text-xs text-slate-500">Best score highlighted</div>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-slate-900/80 text-xs uppercase tracking-[0.12em] text-slate-500">
            <tr>
              <th className="px-4 py-3">Model</th>
              <th className="px-4 py-3">Group</th>
              <th className="px-4 py-3">ROUGE-1</th>
              <th className="px-4 py-3">ROUGE-2</th>
              <th className="px-4 py-3">ROUGE-L</th>
              <th className="px-4 py-3">BLEU</th>
              <th className="px-4 py-3">BERTScore</th>
              <th className="px-4 py-3">Semantic</th>
              <th className="px-4 py-3">Compression</th>
              <th className="px-4 py-3">Time</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800 text-slate-300">
            {rows.map((row) => {
              const meta = byKey(row.key);
              return (
                <tr
                  key={row.key}
                  onClick={() => onSelect(row.key)}
                  className={`cursor-pointer transition hover:bg-slate-900/80 ${selectedKey === row.key ? 'bg-slate-900' : ''}`}
                >
                  <td className="px-4 py-3 font-medium text-slate-100">
                    <span className="inline-flex items-center gap-2">
                      <span className="h-2.5 w-2.5 rounded-full" style={{ background: meta.color }} />
                      {row.algorithm}
                    </span>
                  </td>
                  <td className="px-4 py-3">{groupLabel(row.group)}</td>
                  {scoreCell(row, 'rouge1')}
                  {scoreCell(row, 'rouge2')}
                  {scoreCell(row, 'rougeL')}
                  {scoreCell(row, 'bleu')}
                  {scoreCell(row, 'bertscore_f1')}
                  {scoreCell(row, 'semantic_similarity')}
                  <td className="px-4 py-3">{pct(row.compression_ratio)}</td>
                  <td className="px-4 py-3">{Number(row.processing_time || 0).toFixed(2)}s</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Charts({ rows }) {
  if (rows.length < 2) return null;
  const barData = rows.map((row) => ({
    name: row.algorithm,
    rouge1: metric(row, 'rouge1'),
    rouge2: metric(row, 'rouge2'),
    rougeL: metric(row, 'rougeL'),
    bertscore: metric(row, 'bertscore_f1'),
  }));
  const radarData = ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1', 'semantic_similarity'].map((key) => {
    const item = { metric: key.replace('bertscore_f1', 'BERT').replace('semantic_similarity', 'Semantic') };
    rows.forEach((row) => {
      item[row.algorithm] = metric(row, key);
    });
    return item;
  });
  const timeData = rows.map((row) => ({ name: row.algorithm, seconds: row.processing_time, group: row.group }));

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <ChartShell title="ROUGE và BERTScore">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={barData}>
            <CartesianGrid stroke="#1e293b" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 1]} />
            <Tooltip contentStyle={{ background: '#020617', border: '1px solid #1e293b', borderRadius: 8 }} />
            <Legend />
            <Bar dataKey="rouge1" name="ROUGE-1" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            <Bar dataKey="rouge2" name="ROUGE-2" fill="#14b8a6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="rougeL" name="ROUGE-L" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            <Bar dataKey="bertscore" name="BERTScore" fill="#fb7185" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartShell>

      <ChartShell title="Radar tổng hợp">
        <ResponsiveContainer width="100%" height={280}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="#334155" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
            <PolarRadiusAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} />
            {rows.slice(0, 4).map((row) => (
              <Radar
                key={row.key}
                name={row.algorithm}
                dataKey={row.algorithm}
                stroke={byKey(row.key).color}
                fill={byKey(row.key).color}
                fillOpacity={0.14}
              />
            ))}
            <Legend />
            <Tooltip contentStyle={{ background: '#020617', border: '1px solid #1e293b', borderRadius: 8 }} />
          </RadarChart>
        </ResponsiveContainer>
      </ChartShell>

      <ChartShell title="Thời gian xử lý">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={timeData}>
            <CartesianGrid stroke="#1e293b" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <Tooltip contentStyle={{ background: '#020617', border: '1px solid #1e293b', borderRadius: 8 }} />
            <Line type="monotone" dataKey="seconds" name="Seconds" stroke="#f59e0b" strokeWidth={2.5} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </ChartShell>

      <Ranking rows={rows} />
    </div>
  );
}

function ChartShell({ title, children }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-4">
      <h3 className="mb-3 text-sm font-semibold text-slate-200">{title}</h3>
      {children}
    </div>
  );
}

function Ranking({ rows }) {
  const ranked = [...rows].sort((a, b) => metric(b, 'combined_score') - metric(a, 'combined_score'));
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-4">
      <h3 className="mb-3 text-sm font-semibold text-slate-200">Ranking</h3>
      <div className="space-y-2">
        {ranked.map((row, index) => (
          <div key={row.key} className="flex items-center gap-3 rounded-md bg-slate-900/60 px-3 py-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-slate-800 text-sm font-semibold text-slate-200">
              {index + 1}
            </div>
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-medium text-slate-100">{row.algorithm}</div>
              <div className="text-xs text-slate-500">{groupLabel(row.group)}</div>
            </div>
            <div className="font-mono text-sm text-amber-300">{metric(row, 'combined_score').toFixed(3)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SummaryGrid({ rows, onSelect }) {
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      {rows.map((row) => {
        const meta = byKey(row.key);
        return (
          <article key={row.key} className="rounded-lg border border-slate-800 bg-slate-950/70">
            <header className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
              <button onClick={() => onSelect(row.key)} className="flex items-center gap-2 text-left text-sm font-semibold text-slate-100">
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: meta.color }} />
                {row.algorithm}
              </button>
              <span className="rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-400">{groupLabel(row.group)}</span>
            </header>
            <p className="max-h-56 overflow-auto px-4 py-4 text-sm leading-7 text-slate-300">{row.summary || '-'}</p>
            <footer className="grid grid-cols-3 border-t border-slate-800 text-xs text-slate-400">
              <span className="px-4 py-3">R-L {pct(metric(row, 'rougeL'))}</span>
              <span className="px-4 py-3">BERT {pct(metric(row, 'bertscore_f1'))}</span>
              <span className="px-4 py-3">{Number(row.processing_time || 0).toFixed(2)}s</span>
            </footer>
          </article>
        );
      })}
    </div>
  );
}

function Explainability({ row }) {
  if (!row) return null;
  if (row.group === 'extractive') {
    const source = row.explainability?.extractive?.source_sentences || row.source_sentences || [];
    const selected = new Set(row.explainability?.extractive?.highlighted_sentence_indexes || []);
    const details = row.explainability?.extractive?.selected_sentences || [];
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-950/70">
        <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
          <h3 className="text-sm font-semibold text-slate-100">Extractive Explainability</h3>
          <span className="text-xs text-slate-500">{row.algorithm}</span>
        </div>
        <div className="max-h-96 space-y-2 overflow-auto p-4">
          {source.map((sentence, index) => {
            const detail = details.find((item) => item.sentence_index === index);
            return (
              <div key={`${index}-${sentence.slice(0, 20)}`} className={`rounded-md border px-3 py-2 text-sm leading-6 ${selected.has(index) ? 'border-teal-400/50 bg-teal-400/10 text-teal-50' : 'border-slate-800 bg-slate-900/40 text-slate-400'}`}>
                <div className="flex gap-3">
                  <span className="font-mono text-xs text-slate-500">{index + 1}</span>
                  <span className="flex-1">{sentence}</span>
                  {detail && <span className="font-mono text-xs text-teal-200">{Number(detail.sentence_score || 0).toFixed(3)}</span>}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  const tokens = row.explainability?.abstractive?.token_importance || [];
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/70">
      <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
        <h3 className="text-sm font-semibold text-slate-100">Abstractive Token Importance</h3>
        <span className="text-xs text-slate-500">{row.algorithm}</span>
      </div>
      <div className="flex flex-wrap gap-2 p-4">
        {tokens.length ? tokens.map((item, index) => (
          <span
            key={`${item.token}-${index}`}
            className="rounded-md border border-amber-300/20 px-2.5 py-1 text-sm text-amber-50"
            style={{ background: `rgba(245, 158, 11, ${0.08 + Math.min(0.35, Number(item.importance || 0) * 0.35)})` }}
          >
            {item.token}
          </span>
        )) : <span className="text-sm text-slate-500">Attention/token importance chưa khả dụng cho model này.</span>}
      </div>
    </div>
  );
}

function App() {
  const [dark, setDark] = useState(true);
  const [text, setText] = useState(SAMPLE_TEXT);
  const [reference, setReference] = useState('');
  const [files, setFiles] = useState([]);
  const [selected, setSelected] = useState(ALGORITHMS.map((item) => item.key));
  const [sentenceCount, setSentenceCount] = useState(5);
  const [maxLength, setMaxLength] = useState(150);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [selectedKey, setSelectedKey] = useState('textrank');

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
  }, [dark]);

  const rows = useMemo(() => result?.results || [], [result]);
  const selectedRow = rows.find((row) => row.key === selectedKey) || rows[0];

  async function runComparison(event) {
    event.preventDefault();
    setLoading(true);
    setError('');
    try {
      let response;
      if (files.length) {
        const form = new FormData();
        files.forEach((file) => form.append('files', file));
        form.append('reference', reference);
        form.append('algorithms', JSON.stringify(selected));
        form.append('extractive_sentences', String(sentenceCount));
        form.append('max_abstractive_length', String(maxLength));
        response = await fetch(`${API}/summarize/files/compare`, { method: 'POST', body: form });
      } else {
        response = await fetch(`${API}/summarize/compare`, {
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
      }
      if (!response.ok) {
        const body = await response.text();
        throw new Error(body || `HTTP ${response.status}`);
      }
      const data = await response.json();
      setResult(data);
      setSelectedKey(data?.ranking?.[0]?.key || data?.results?.[0]?.key || 'textrank');
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  function copySelected() {
    if (selectedRow?.summary) navigator.clipboard?.writeText(selectedRow.summary);
  }

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="sticky top-0 z-20 border-b border-slate-800 bg-slate-950/95 backdrop-blur">
        <div className="flex h-16 items-center justify-between px-5">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-teal-400/15 text-teal-200">
              <Brain size={20} />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-slate-100">Vietnamese Summarization Research</h1>
              <p className="text-xs text-slate-500">TextRank · LexRank · LSA · ViT5 · mT5 · BARTPho</p>
            </div>
          </div>
          <button className="icon-button" onClick={() => setDark((value) => !value)} title="Toggle theme">
            {dark ? <Sun size={17} /> : <Moon size={17} />}
          </button>
        </div>
      </header>

      <main className="grid min-h-[calc(100vh-4rem)] grid-cols-1 lg:grid-cols-[380px_1fr]">
        <aside className="border-b border-slate-800 bg-slate-950 p-5 lg:border-b-0 lg:border-r">
          <form onSubmit={runComparison} className="space-y-5">
            <section className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="section-title">Input</label>
                <button type="button" className="text-xs text-slate-500 hover:text-slate-200" onClick={() => setText(SAMPLE_TEXT)}>
                  Sample
                </button>
              </div>
              <textarea
                value={text}
                disabled={files.length > 0}
                onChange={(event) => setText(event.target.value)}
                className="input min-h-44"
                placeholder="Dán văn bản tiếng Việt..."
              />
              <label className="flex cursor-pointer items-center gap-3 rounded-lg border border-dashed border-slate-700 bg-slate-900/40 px-4 py-3 text-sm text-slate-400 hover:border-teal-400/60 hover:text-teal-100">
                <UploadCloud size={18} />
                <span className="flex-1 truncate">{files.length ? `${files.length} file selected` : 'TXT / DOCX / PDF'}</span>
                <input type="file" multiple accept=".txt,.docx,.pdf" className="hidden" onChange={(event) => setFiles(Array.from(event.target.files || []))} />
              </label>
              {files.length > 0 && (
                <button type="button" className="ghost-button w-full" onClick={() => setFiles([])}>
                  Clear files
                </button>
              )}
            </section>

            <section className="space-y-3">
              <label className="section-title">Reference Summary</label>
              <textarea
                value={reference}
                onChange={(event) => setReference(event.target.value)}
                className="input min-h-24"
                placeholder="Tùy chọn, dùng để đánh giá ROUGE/BLEU/BERTScore..."
              />
            </section>

            <AlgorithmSelector selected={selected} setSelected={setSelected} />

            <section className="grid grid-cols-2 gap-3">
              <label className="space-y-2">
                <span className="section-title">Sentences</span>
                <input className="input" type="number" min="1" max="20" value={sentenceCount} onChange={(event) => setSentenceCount(Number(event.target.value))} />
              </label>
              <label className="space-y-2">
                <span className="section-title">Max Tokens</span>
                <input className="input" type="number" min="24" max="512" value={maxLength} onChange={(event) => setMaxLength(Number(event.target.value))} />
              </label>
            </section>

            <button disabled={loading || selected.length === 0 || (!text.trim() && !files.length)} className="primary-button">
              {loading ? <RefreshCcw className="animate-spin" size={17} /> : <Play size={17} />}
              Run Comparison
            </button>
            {error && <div className="rounded-lg border border-rose-400/30 bg-rose-500/10 p-3 text-sm text-rose-100">{error}</div>}
          </form>
        </aside>

        <section className="space-y-5 p-5">
          {!result && (
            <div className="flex min-h-[50vh] items-center justify-center rounded-lg border border-slate-800 bg-slate-950/70 p-8 text-center">
              <div>
                <FileText className="mx-auto mb-3 text-slate-600" size={42} />
                <h2 className="text-lg font-semibold text-slate-200">Research dashboard ready</h2>
                <p className="mt-2 max-w-md text-sm text-slate-500">Chạy một batch so sánh để xem metrics, ranking, biểu đồ và explainability.</p>
              </div>
            </div>
          )}

          {result && (
            <>
              <StatStrip result={result} />
              <ComparisonTable rows={rows} onSelect={setSelectedKey} selectedKey={selectedKey} />
              <Charts rows={rows} />
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-slate-200">Summaries</h2>
                <button className="ghost-button" onClick={copySelected}>
                  <Copy size={15} /> Copy selected
                </button>
              </div>
              <SummaryGrid rows={rows} onSelect={setSelectedKey} />
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-slate-200">Explainability</span>
                <ChevronDown size={16} className="text-slate-500" />
              </div>
              <Explainability row={selectedRow} />
            </>
          )}
        </section>
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
