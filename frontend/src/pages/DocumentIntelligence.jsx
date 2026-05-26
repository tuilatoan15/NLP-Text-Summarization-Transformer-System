import React, { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Scatter, ScatterChart,
  Tooltip, XAxis, YAxis,
} from 'recharts';
import {
  UploadCloud, Search, Network, FileText, Brain, GitBranch, Loader2,
  ShieldCheck, Quote, Presentation, Radio, HelpCircle, Layers3, ChartColumn,
} from 'lucide-react';
import {
  compareDocumentSummaries,
  ingestDocument,
  searchDocument,
} from '../services/apiService';

const algorithmOptions = [
  { key: 'textrank', label: 'TextRank', group: 'Extractive' },
  { key: 'lexrank', label: 'LexRank', group: 'Extractive' },
  { key: 'lsa', label: 'LSA', group: 'Extractive' },
  { key: 'vit5', label: 'ViT5', group: 'Abstractive' },
  { key: 'bartpho', label: 'BARTPho', group: 'Abstractive' },
];

const palette = ['#2563eb', '#059669', '#d97706', '#7c3aed', '#dc2626', '#0891b2'];

function MetricCard({ title, value, icon: Icon, tone = 'bg-blue-500' }) {
  return (
    <div className="ui-card p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="ui-stat-label">{title}</p>
          <p className="text-2xl font-bold text-[var(--text)]">{value}</p>
        </div>
        <div className={`p-2 rounded-lg text-white ${tone}`}>
          <Icon className="w-4 h-4" />
        </div>
      </div>
    </div>
  );
}

function Panel({ title, icon: Icon, children, actions }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="ui-card p-5"
    >
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

const scorePercent = value => `${Math.round((Number(value) || 0) * 100)}%`;

export default function DocumentIntelligence() {
  const [file, setFile] = useState(null);
  const [documentState, setDocumentState] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResult, setSearchResult] = useState(null);
  const [compareResult, setCompareResult] = useState(null);
  const [reference, setReference] = useState('');
  const [selectedAlgorithms, setSelectedAlgorithms] = useState(['textrank', 'lexrank', 'lsa']);
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
    setCompareResult(null);
    setSearchResult(null);
    try {
      const payload = await ingestDocument(file, { includeEmbeddings: true, embeddingModel: 'hash' });
      setDocumentState(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleSearch() {
    if (!documentState || !searchQuery.trim()) return;
    setSearching(true);
    setError(null);
    try {
      const payload = await searchDocument(documentState.document_id, searchQuery, 5);
      setSearchResult(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setSearching(false);
    }
  }

  async function handleCompare() {
    if (!documentState) return;
    setComparing(true);
    setError(null);
    try {
      const payload = await compareDocumentSummaries(documentState.document_id, {
        reference: reference || null,
        algorithms: selectedAlgorithms,
        targetLengthRatio: 35,
        extractiveSentences: 4,
        maxAbstractiveLength: 160,
      });
      setCompareResult(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setComparing(false);
    }
  }

  function toggleAlgorithm(key) {
    setSelectedAlgorithms(current =>
      current.includes(key) ? current.filter(item => item !== key) : [...current, key],
    );
  }

  return (
    <div className="space-y-6 pb-12">
      <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4">
        <div>
          <h1 className="ui-page-title mb-1">AI Document Intelligence</h1>
          <p className="ui-page-subtitle">NLP laboratory cho tóm tắt trích rút, diễn giải, retrieval và citation grounding.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          {algorithmOptions.map(item => (
            <button
              key={item.key}
              type="button"
              onClick={() => toggleAlgorithm(item.key)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${
                selectedAlgorithms.includes(item.key)
                  ? 'bg-[var(--accent)] text-white border-transparent'
                  : 'bg-[var(--surface-elevated)] text-[var(--text-muted)] border-[var(--border)]'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-950/30 dark:border-red-900 dark:text-red-300">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-5">
        <div className="space-y-5">
          <Panel title="Upload tài liệu" icon={UploadCloud}>
            <label className="block border border-dashed border-[var(--border)] rounded-xl p-4 bg-[var(--surface-inset)] cursor-pointer">
              <input
                type="file"
                accept=".pdf,.docx,.txt"
                className="hidden"
                onChange={event => setFile(event.target.files?.[0] || null)}
              />
              <div className="flex items-center gap-3">
                <UploadCloud className="w-8 h-8 text-blue-500" />
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-[var(--text)] truncate">
                    {file?.name || 'PDF, DOCX, TXT'}
                  </p>
                  <p className="text-xs text-[var(--text-muted)]">Parsing cấu trúc, chunking và embedding.</p>
                </div>
              </div>
            </label>
            <button
              type="button"
              disabled={!file || uploading}
              onClick={handleUpload}
              className="ui-btn-primary w-full mt-4"
            >
              {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileText className="w-4 h-4" />}
              Phân tích tài liệu
            </button>
          </Panel>

          {documentState && (
            <Panel title="Thông tin ingest" icon={ShieldCheck}>
              <div className="grid grid-cols-2 gap-3">
                <MetricCard title="Quality" value={scorePercent(documentState.quality?.extraction?.score)} icon={ShieldCheck} tone="bg-emerald-500" />
                <MetricCard title="Chunks" value={documentState.quality?.chunk_count || 0} icon={Layers3} tone="bg-cyan-500" />
                <MetricCard title="Words" value={documentState.quality?.extraction?.word_count || 0} icon={FileText} tone="bg-amber-500" />
                <MetricCard title="Pages" value={documentState.metadata?.pages || 1} icon={GitBranch} tone="bg-violet-500" />
              </div>
              <div className="mt-4 text-xs text-[var(--text-muted)] break-all">
                {documentState.document_id}
              </div>
            </Panel>
          )}

          {documentState && (
            <Panel title="Semantic search" icon={Search}>
              <div className="flex gap-2">
                <input
                  className="ui-input"
                  value={searchQuery}
                  onChange={event => setSearchQuery(event.target.value)}
                  placeholder="Tìm nội dung trong tài liệu..."
                />
                <button
                  type="button"
                  onClick={handleSearch}
                  disabled={searching || !searchQuery.trim()}
                  className="ui-btn-secondary !px-3"
                >
                  {searching ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                </button>
              </div>
              <div className="mt-4 space-y-3">
                {(searchResult?.results || []).map(item => (
                  <div key={item.chunk.chunk_id} className="ui-card-muted p-3">
                    <div className="flex items-center justify-between gap-3 mb-2">
                      <span className="text-xs font-bold text-[var(--text)]">#{item.rank} · {scorePercent(item.score)}</span>
                      <span className="text-[10px] text-[var(--text-faint)]">page {item.chunk.page_start || '-'}</span>
                    </div>
                    <p className="text-xs text-[var(--text-secondary)] line-clamp-4">{item.highlight || item.chunk.text}</p>
                  </div>
                ))}
              </div>
            </Panel>
          )}
        </div>

        <div className="space-y-5 min-w-0">
          {documentState ? (
            <>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
                <Panel title="AI Overview" icon={Brain}>
                  <p className="text-sm leading-6 text-[var(--text-secondary)]">{overview.document_overview}</p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {(overview.keywords || []).slice(0, 8).map(item => (
                      <span key={item.term} className="ui-badge bg-blue-50 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300">
                        {item.term}
                      </span>
                    ))}
                  </div>
                </Panel>

                <Panel title="Key insights" icon={Quote}>
                  <div className="space-y-2">
                    {(overview.key_insights || []).map((item, idx) => (
                      <p key={item} className="text-sm text-[var(--text-secondary)]">
                        <span className="font-bold text-[var(--text)]">{idx + 1}.</span> {item}
                      </p>
                    ))}
                  </div>
                </Panel>

                <Panel title="Research compare" icon={ChartColumn}>
                  <textarea
                    className="ui-textarea min-h-24"
                    value={reference}
                    onChange={event => setReference(event.target.value)}
                    placeholder="Reference summary tùy chọn..."
                  />
                  <button
                    type="button"
                    onClick={handleCompare}
                    disabled={comparing || selectedAlgorithms.length === 0}
                    className="ui-btn-primary w-full mt-3"
                  >
                    {comparing ? <Loader2 className="w-4 h-4 animate-spin" /> : <ChartColumn className="w-4 h-4" />}
                    So sánh mô hình
                  </button>
                </Panel>
              </div>

              {compareResult && (
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
                  <Panel title="Evaluation metrics" icon={ChartColumn}>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={metricsData} margin={{ left: -20, right: 10 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                          <XAxis dataKey="model" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                          <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                          <Tooltip />
                          <Bar dataKey="rougeL" name="ROUGE-L" fill="#2563eb" radius={[4, 4, 0, 0]} />
                          <Bar dataKey="bertscore" name="BERTScore" fill="#059669" radius={[4, 4, 0, 0]} />
                          <Bar dataKey="semantic" name="Semantic" fill="#d97706" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </Panel>

                  <Panel title="Citation grounding" icon={ShieldCheck}>
                    <div className="space-y-3 max-h-64 overflow-y-auto pr-1">
                      {compareRows.slice(0, 3).map(row => (
                        <div key={row.key} className="ui-card-muted p-3">
                          <p className="text-xs font-bold text-[var(--text)] mb-2">{row.algorithm}</p>
                          {(row.citations || []).slice(0, 3).map(citation => (
                            <div key={`${row.key}-${citation.sentence_index}`} className="border-t border-[var(--border)] py-2 first:border-t-0 first:pt-0">
                              <p className="text-xs text-[var(--text-secondary)]">{citation.sentence}</p>
                              <p className="text-[10px] mt-1 text-[var(--text-faint)]">
                                {citation.status} · support {scorePercent(citation.best_support_score)}
                              </p>
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  </Panel>
                </div>
              )}

              <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
                <Panel title="Chunk graph" icon={Network}>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-80 overflow-y-auto pr-1">
                    {(documentState.chunks || []).slice(0, 12).map((chunk, idx) => (
                      <div key={chunk.chunk_id} className="ui-card-muted p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-bold text-[var(--text)]">Chunk {idx + 1}</span>
                          <span className="text-[10px] text-[var(--text-faint)]">{chunk.token_count} tokens</span>
                        </div>
                        <div className="h-1.5 rounded-full bg-[var(--border)] overflow-hidden mb-2">
                          <div
                            className="h-full bg-blue-500"
                            style={{ width: `${Math.min(100, (chunk.token_count || 0) / 7)}%` }}
                          />
                        </div>
                        <p className="text-xs text-[var(--text-muted)] line-clamp-3">{chunk.text}</p>
                      </div>
                    ))}
                  </div>
                </Panel>

                <Panel title="Embedding map" icon={GitBranch}>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis type="number" dataKey="x" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                        <YAxis type="number" dataKey="y" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Scatter data={visualization.embedding_map || []}>
                          {(visualization.embedding_map || []).map((entry, index) => (
                            <Cell key={entry.chunk_id} fill={palette[index % palette.length]} />
                          ))}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </Panel>
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
                <Panel title="Quiz & flashcards" icon={HelpCircle}>
                  <div className="space-y-3">
                    {(assets.quiz || []).slice(0, 3).map(item => (
                      <div key={item.id} className="ui-card-muted p-3">
                        <p className="text-xs font-semibold text-[var(--text)]">{item.question}</p>
                        <p className="text-[11px] text-[var(--text-muted)] mt-1">{item.answer}</p>
                      </div>
                    ))}
                  </div>
                </Panel>

                <Panel title="Podcast script" icon={Radio}>
                  <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
                    {(assets.podcast?.turns || []).slice(0, 6).map((turn, idx) => (
                      <p key={`${turn.speaker}-${idx}`} className="text-xs text-[var(--text-secondary)]">
                        <span className="font-bold text-[var(--text)]">{turn.speaker}:</span> {turn.text}
                      </p>
                    ))}
                  </div>
                </Panel>

                <Panel title="Presentation outline" icon={Presentation}>
                  <div className="space-y-3">
                    {(assets.presentation || []).map(slide => (
                      <div key={slide.slide} className="ui-card-muted p-3">
                        <p className="text-xs font-bold text-[var(--text)]">{slide.slide}. {slide.title}</p>
                        <p className="text-[11px] text-[var(--text-muted)] mt-1">{(slide.bullets || []).join(' · ')}</p>
                      </div>
                    ))}
                  </div>
                </Panel>
              </div>
            </>
          ) : (
            <Panel title="Research workspace" icon={Brain}>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[
                  ['Upload', 'PDF/DOCX/TXT, OCR fallback, structure extraction'],
                  ['Analyze', 'Semantic chunks, embeddings, entities, timeline'],
                  ['Compare', 'Extractive vs abstractive, metrics, citations'],
                ].map(([title, text]) => (
                  <div key={title} className="ui-card-muted p-4">
                    <p className="text-sm font-bold text-[var(--text)]">{title}</p>
                    <p className="text-xs text-[var(--text-muted)] mt-1">{text}</p>
                  </div>
                ))}
              </div>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
