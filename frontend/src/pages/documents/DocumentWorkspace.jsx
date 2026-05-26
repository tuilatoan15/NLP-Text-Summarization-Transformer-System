import React, { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, Cell,
} from 'recharts';
import {
  UploadCloud, Search, Network, FileText, Brain, GitBranch, Loader2, ShieldCheck, Quote,
  Presentation, Radio, HelpCircle, Layers3, ChartColumn, GitCompare, Map,
} from 'lucide-react';
import {
  compareDocumentSummaries, ingestDocument, searchDocument,
} from '../../services/apiService';

const algorithmOptions = [
  { key: 'textrank', label: 'TextRank', group: 'Extractive' },
  { key: 'lexrank', label: 'LexRank', group: 'Extractive' },
  { key: 'lsa', label: 'LSA', group: 'Extractive' },
  { key: 'tfidf', label: 'TF-IDF', group: 'Extractive' },
  { key: 'vit5', label: 'ViT5', group: 'Abstractive' },
  { key: 'bartpho', label: 'BARTPho', group: 'Abstractive' },
];

const TABS = [
  { id: 'upload', label: 'Upload', icon: UploadCloud },
  { id: 'analysis', label: 'Analysis', icon: Brain },
  { id: 'compare', label: 'Compare', icon: GitCompare },
  { id: 'evaluation', label: 'Evaluation', icon: ChartColumn },
  { id: 'search', label: 'Semantic Search', icon: Search },
  { id: 'citations', label: 'Citations', icon: ShieldCheck },
  { id: 'visualize', label: 'Visualize', icon: Network },
  { id: 'notebook', label: 'NotebookLM', icon: Map },
];

const palette = ['#2563eb', '#059669', '#d97706', '#7c3aed', '#dc2626', '#0891b2'];
const scorePercent = value => `${Math.round((Number(value) || 0) * 100)}%`;

function Panel({ title, icon: Icon, children, actions }) {
  return (
    <motion.section initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="ui-card p-5">
      <div className="flex items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2 min-w-0">
          {Icon && <Icon className="w-4 h-4 text-blue-500 shrink-0" />}
          <h2 className="text-sm font-bold text-[var(--text)] truncate">{title}</h2>
        </div>
        {actions}
      </div>
      {children}
    </motion.section>
  );
}

export default function DocumentWorkspace() {
  const [tab, setTab] = useState('upload');
  const [file, setFile] = useState(null);
  const [documentState, setDocumentState] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResult, setSearchResult] = useState(null);
  const [compareResult, setCompareResult] = useState(null);
  const [reference, setReference] = useState('');
  const [selectedAlgorithms, setSelectedAlgorithms] = useState(['textrank', 'lexrank', 'lsa', 'tfidf']);
  const [uploading, setUploading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState(null);

  const assets = documentState?.analysis_assets || {};
  const overview = assets.overview || {};
  const visualization = documentState?.visualization || {};
  const compareRows = compareResult?.results || [];
  const metricsData = useMemo(() => compareRows.map(row => ({
    model: row.algorithm,
    rougeL: Number(row.metrics?.rougeL || 0),
    bertscore: Number(row.metrics?.bertscore_f1 || 0),
    semantic: Number(row.metrics?.semantic_similarity || 0),
    latency: Number(row.metrics?.processing_time || 0),
  })), [compareRows]);

  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const payload = await ingestDocument(file, { includeEmbeddings: true, embeddingModel: 'hash' });
      setDocumentState(payload);
      setTab('analysis');
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleSearch() {
    if (!documentState || !searchQuery.trim()) return;
    setSearching(true);
    try {
      setSearchResult(await searchDocument(documentState.document_id, searchQuery, 5));
    } catch (err) {
      setError(err.message);
    } finally {
      setSearching(false);
    }
  }

  async function handleCompare() {
    if (!documentState) return;
    setComparing(true);
    try {
      const payload = await compareDocumentSummaries(documentState.document_id, {
        reference: reference || null,
        algorithms: selectedAlgorithms,
        targetLengthRatio: 35,
        extractiveSentences: 4,
        maxAbstractiveLength: 160,
      });
      setCompareResult(payload);
      setTab('evaluation');
    } catch (err) {
      setError(err.message);
    } finally {
      setComparing(false);
    }
  }

  function toggleAlgorithm(key) {
    setSelectedAlgorithms(c => (c.includes(key) ? c.filter(x => x !== key) : [...c, key]));
  }

  return (
    <div className="space-y-6 pb-12">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="ui-page-title mb-1">AI Document Intelligence</h1>
          <p className="ui-page-subtitle">Extractive vs abstractive · Vietnamese NLP · citation grounding</p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {TABS.map(item => {
            const Icon = item.icon;
            const disabled = item.id !== 'upload' && !documentState;
            return (
              <button
                key={item.id}
                type="button"
                disabled={disabled}
                onClick={() => setTab(item.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${
                  tab === item.id
                    ? 'bg-[var(--accent)] text-white border-transparent'
                    : 'bg-[var(--surface-elevated)] text-[var(--text-muted)] border-[var(--border)]'
                } ${disabled ? 'opacity-40 cursor-not-allowed' : ''}`}
              >
                <Icon className="w-3.5 h-3.5" />
                {item.label}
              </button>
            );
          })}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-950/30 dark:border-red-900 dark:text-red-300">
          {error}
        </div>
      )}

      {tab === 'upload' && (
        <Panel title="Upload tài liệu" icon={UploadCloud}>
          <label className="block border border-dashed border-[var(--border)] rounded-xl p-6 bg-[var(--surface-inset)] cursor-pointer">
            <input type="file" accept=".pdf,.docx,.txt" className="hidden" onChange={e => setFile(e.target.files?.[0] || null)} />
            <p className="text-sm font-semibold">{file?.name || 'PDF, DOCX, TXT'}</p>
            <p className="text-xs text-[var(--text-muted)] mt-1">Structured parsing · semantic chunks · embeddings</p>
          </label>
          <button type="button" disabled={!file || uploading} onClick={handleUpload} className="ui-btn-primary w-full mt-4">
            {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileText className="w-4 h-4" />}
            Phân tích tài liệu
          </button>
        </Panel>
      )}

      {tab === 'analysis' && documentState && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <Panel title="AI Overview" icon={Brain}>
            <p className="text-sm text-[var(--text-secondary)]">{overview.document_overview}</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {(overview.keywords || []).slice(0, 8).map(k => (
                <span key={k.term} className="ui-badge">{k.term}</span>
              ))}
            </div>
          </Panel>
          <Panel title="Key insights" icon={Quote}>
            {(overview.key_insights || []).map((item, i) => (
              <p key={item} className="text-sm text-[var(--text-secondary)] mb-2">
                <span className="font-bold">{i + 1}.</span> {item}
              </p>
            ))}
          </Panel>
          <Panel title="Ingest quality" icon={Layers3}>
            <p className="text-sm">Quality: {scorePercent(documentState.quality?.extraction?.score)}</p>
            <p className="text-sm">Chunks: {documentState.quality?.chunk_count}</p>
            <p className="text-xs text-[var(--text-faint)] mt-2 break-all">{documentState.document_id}</p>
          </Panel>
        </div>
      )}

      {tab === 'compare' && documentState && (
        <Panel title="So sánh Extractive vs Abstractive" icon={GitCompare}>
          <div className="flex flex-wrap gap-2 mb-4">
            {algorithmOptions.map(item => (
              <button
                key={item.key}
                type="button"
                onClick={() => toggleAlgorithm(item.key)}
                className={`px-3 py-1 rounded-lg text-xs font-semibold border ${
                  selectedAlgorithms.includes(item.key) ? 'bg-blue-600 text-white border-transparent' : 'border-[var(--border)]'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
          <textarea className="ui-textarea min-h-24 mb-3" value={reference} onChange={e => setReference(e.target.value)} placeholder="Reference summary (optional)" />
          <button type="button" onClick={handleCompare} disabled={comparing} className="ui-btn-primary">
            {comparing ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Chạy so sánh'}
          </button>
          {compareRows.length > 0 && (
            <div className="mt-6 grid gap-3 md:grid-cols-2">
              {compareRows.map(row => (
                <div key={row.key} className="ui-card-muted p-4">
                  <p className="text-xs font-bold mb-2">{row.algorithm} · {row.group}</p>
                  <p className="text-sm text-[var(--text-secondary)] line-clamp-6">{row.summary}</p>
                </div>
              ))}
            </div>
          )}
        </Panel>
      )}

      {tab === 'evaluation' && compareResult && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
          <Panel title="ROUGE / BERTScore / Semantic" icon={ChartColumn}>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                  <XAxis dataKey="model" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="rougeL" fill="#2563eb" />
                  <Bar dataKey="bertscore" fill="#059669" />
                  <Bar dataKey="semantic" fill="#d97706" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
          <Panel title="Research matrix" icon={ChartColumn}>
            <pre className="text-xs overflow-auto max-h-72 text-[var(--text-muted)]">
              {JSON.stringify(compareResult.research_matrix?.aggregate || {}, null, 2)}
            </pre>
          </Panel>
        </div>
      )}

      {tab === 'search' && documentState && (
        <Panel title="Semantic search" icon={Search}>
          <div className="flex gap-2">
            <input className="ui-input" value={searchQuery} onChange={e => setSearchQuery(e.target.value)} placeholder="Truy vấn ngữ nghĩa..." />
            <button type="button" onClick={handleSearch} disabled={searching} className="ui-btn-secondary">
              {searching ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            </button>
          </div>
          <div className="mt-4 space-y-3">
            {(searchResult?.results || []).map(item => (
              <div key={item.chunk.chunk_id} className="ui-card-muted p-3">
                <p className="text-xs font-bold">#{item.rank} · {scorePercent(item.score)} · {searchResult?.retrieval_backend}</p>
                <p className="text-xs mt-2 line-clamp-4">{item.highlight || item.chunk.text}</p>
              </div>
            ))}
          </div>
        </Panel>
      )}

      {tab === 'citations' && compareResult && (
        <Panel title="Citation viewer" icon={ShieldCheck}>
          {compareRows.map(row => (
            <div key={row.key} className="mb-4 ui-card-muted p-3">
              <p className="text-xs font-bold mb-2">{row.algorithm}</p>
              {(row.citations || []).map(c => (
                <div key={c.sentence_index} className="border-t border-[var(--border)] py-2 text-xs">
                  <p>{c.sentence}</p>
                  <p className="text-[var(--text-faint)]">{c.status} · {scorePercent(c.best_support_score)}</p>
                </div>
              ))}
            </div>
          ))}
        </Panel>
      )}

      {tab === 'visualize' && documentState && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
          <Panel title="Embedding map (PCA)" icon={GitBranch}>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="x" type="number" />
                  <YAxis dataKey="y" type="number" />
                  <Tooltip />
                  <Scatter data={visualization.embedding_map || []}>
                    {(visualization.embedding_map || []).map((e, i) => (
                      <Cell key={e.chunk_id} fill={palette[i % palette.length]} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </Panel>
          <Panel title="Chunk hierarchy" icon={Network}>
            <pre className="text-xs overflow-auto max-h-80">{JSON.stringify(visualization.chunk_hierarchy, null, 2)}</pre>
          </Panel>
        </div>
      )}

      {tab === 'notebook' && documentState && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
          <Panel title="Quiz" icon={HelpCircle}>
            {(assets.quiz || []).slice(0, 4).map(q => (
              <p key={q.id} className="text-xs mb-2">{q.question}</p>
            ))}
          </Panel>
          <Panel title="Podcast" icon={Radio}>
            {(assets.podcast?.turns || []).slice(0, 5).map((t, i) => (
              <p key={i} className="text-xs mb-1"><b>{t.speaker}:</b> {t.text}</p>
            ))}
          </Panel>
          <Panel title="Presentation" icon={Presentation}>
            {(assets.presentation || []).map(s => (
              <p key={s.slide} className="text-xs mb-2">{s.slide}. {s.title}</p>
            ))}
          </Panel>
        </div>
      )}
    </div>
  );
}
