import React, { useMemo, useState } from 'react';
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
} from 'lucide-react';
import './styles.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function App() {
  const [files, setFiles] = useState([]);
  const [text, setText] = useState('');
  const [urls, setUrls] = useState('');
  const [lengthControl, setLengthControl] = useState('100_words');
  const [modelName, setModelName] = useState('vit5');
  const [analysisMode, setAnalysisMode] = useState('fast');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeDoc, setActiveDoc] = useState(0);

  const hasFiles = files.length > 0;
  const suspiciousCount = result?.consistency?.suspicious_spans?.length || 0;

  async function submit(event) {
    event.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = hasFiles
        ? await submitFiles(files, lengthControl, modelName, analysisMode)
        : await submitText(text, urls, lengthControl, modelName, analysisMode);

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || 'Không xử lý được yêu cầu.');
      }

      const payload = await response.json();
      setResult(payload);
      setActiveDoc(0);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const activeDocument = result?.documents?.[activeDoc];

  return (
    <main className="app">
      <section className="workspace">
        <aside className="inputPane">
          <div className="brand">
            <div className="brandMark"><FileText size={22} /></div>
            <div>
              <h1>Consistency Summarizer</h1>
              <p>Tóm tắt đa tài liệu có kiểm chứng nhất quán</p>
            </div>
          </div>

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
                Độ dài
                <select value={lengthControl} onChange={(event) => setLengthControl(event.target.value)}>
                  <option value="20_percent">20%</option>
                  <option value="50_percent">50%</option>
                  <option value="100_words">100 từ</option>
                  <option value="200_words">200 từ</option>
                  <option value="auto">Tự động</option>
                </select>
              </label>

              <label>
                Model
                <select value={modelName} onChange={(event) => setModelName(event.target.value)}>
                  <option value="vit5">ViT5 / T5</option>
                  <option value="bart">BART</option>
                </select>
              </label>
            </div>

            <div className="segmented">
              <button type="button" className={analysisMode === 'fast' ? 'active' : ''} onClick={() => setAnalysisMode('fast')}>
                Fast Mode
              </button>
              <button type="button" className={analysisMode === 'full' ? 'active' : ''} onClick={() => setAnalysisMode('full')}>
                Full Analysis
              </button>
            </div>

            <button className="primaryButton" disabled={loading || (!hasFiles && !text.trim() && !urls.trim())}>
              {loading ? <Loader2 className="spin" size={18} /> : <Gauge size={18} />}
              <span>{loading ? 'Đang xử lý' : 'Tóm tắt & kiểm chứng'}</span>
            </button>

            {error && <div className="errorBox">{error}</div>}
          </form>
        </aside>

        <section className="resultPane">
          {!result && !loading && <EmptyState />}
          {loading && <LoadingState />}
          {result && (
            <>
              <div className="metricRow">
                <Metric icon={<Gauge size={18} />} label="Consistency" value={`${result.consistency.consistency_percent ?? Math.round(result.consistency.consistency_score * 100)}%`} />
                <Metric icon={<BarChart3 size={18} />} label="ROUGE-L" value={getBestRouge(result)} />
                <Metric icon={<AlertTriangle size={18} />} label="Nghi vấn" value={suspiciousCount} />
                <Metric icon={<CheckCircle2 size={18} />} label="Thời gian" value={`${result.processing_time_seconds}s`} />
              </div>

              <section className="panel">
                <div className="panelHeader">
                  <h2>Summary tổng hợp</h2>
                  <StatusPill status={result.consistency.status} />
                </div>
                <HighlightedSummary summary={result.best} spans={result.consistency.suspicious_spans} />
              </section>

              <section className="panel">
                <div className="panelHeader">
                  <h2>Fact-check / Consistency</h2>
                  <span>{result.processing_time_seconds}s</span>
                </div>
                <ConsistencyList checks={result.consistency.checks} />
              </section>

              <section className="panel">
                <div className="panelHeader">
                  <h2>Tài liệu batch</h2>
                  <span>{result.documents.length} nguồn</span>
                </div>
                <div className="docTabs">
                  {result.documents.map((doc, index) => (
                    <button
                      key={`${doc.name}-${index}`}
                      className={activeDoc === index ? 'active' : ''}
                      onClick={() => setActiveDoc(index)}
                    >
                      {doc.name}
                    </button>
                  ))}
                </div>
                {activeDocument && <DocumentDetail document={activeDocument} />}
              </section>
            </>
          )}
        </section>
      </section>
    </main>
  );
}

async function submitFiles(files, lengthControl, modelName, analysisMode) {
  const form = new FormData();
  files.forEach((file) => form.append('files', file));
  form.append('length_control', lengthControl);
  form.append('model_name', modelName);
  form.append('analysis_mode', analysisMode);
  return fetch(`${API_BASE}/summarize/files`, { method: 'POST', body: form });
}

async function submitText(text, urls, lengthControl, modelName, analysisMode) {
  return fetch(`${API_BASE}/summarize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: text.trim() || null,
      urls: urls.split('\n').map((url) => url.trim()).filter(Boolean),
      length_control: lengthControl,
      model_name: modelName,
      analysis_mode: analysisMode,
      save_result: true,
    }),
  });
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
