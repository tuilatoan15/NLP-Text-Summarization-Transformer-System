import React, { useCallback, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  UploadCloud, FileText, File, X, CheckCircle2, Loader2,
  AlertCircle, Sparkles, Brain, Layers3, Search,
} from 'lucide-react';
import { ingestDocument } from '../../services/apiService';
import { useDocumentContext } from '../../context/DocumentContext';

const ACCEPT = '.pdf,.docx,.txt,.doc';
const MAX_MB = 50;

const STEPS = [
  { icon: FileText, label: 'Đọc & parse tài liệu', sub: 'PDF / DOCX / TXT' },
  { icon: Layers3, label: 'Tạo semantic chunks', sub: 'Heading-aware, overlap' },
  { icon: Brain, label: 'Phân tích AI overview', sub: 'Key insights & keywords' },
  { icon: Search, label: 'Tạo embeddings', sub: 'Vector similarity index' },
];

function FileIcon({ ext }: { ext: string }) {
  const colors: Record<string, string> = {
    pdf: 'text-red-500',
    docx: 'text-blue-500',
    doc: 'text-blue-500',
    txt: 'text-emerald-500',
  };
  return <File className={`w-10 h-10 ${colors[ext] ?? 'text-slate-400'}`} />;
}

export default function DocumentUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(-1);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const { setDocument } = useDocumentContext();
  const navigate = useNavigate();

  const ext = file?.name.split('.').pop()?.toLowerCase() ?? '';
  const sizeMB = file ? (file.size / 1024 / 1024).toFixed(2) : '0';

  const validate = (f: File) => {
    if (f.size > MAX_MB * 1024 * 1024)
      return `Tệp quá lớn (max ${MAX_MB} MB)`;
    const ok = ACCEPT.split(',').some(a => f.name.toLowerCase().endsWith(a));
    if (!ok) return 'Định dạng không hỗ trợ (PDF, DOCX, TXT)';
    return null;
  };

  const pickFile = (f: File) => {
    const err = validate(f);
    if (err) { setError(err); return; }
    setError(null);
    setFile(f);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) pickFile(f);
  }, []);

  const simulateSteps = async () => {
    for (let i = 0; i < STEPS.length; i++) {
      setStep(i);
      await new Promise(r => setTimeout(r, 600 + Math.random() * 400));
    }
  };

  async function onUpload() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setStep(0);
    try {
      const [payload] = await Promise.all([
        ingestDocument(file, { includeEmbeddings: true, embeddingModel: 'hash' }),
        simulateSteps(),
      ]);
      setStep(STEPS.length);
      await new Promise(r => setTimeout(r, 400));
      setDocument(payload);
      navigate('../analysis');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload thất bại');
      setStep(-1);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div>
        <h2 className="ui-page-title mb-1">Upload tài liệu</h2>
        <p className="ui-page-subtitle">Hỗ trợ PDF · DOCX · TXT — tối đa {MAX_MB} MB</p>
      </div>

      {/* Drop zone */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className={`relative border-2 border-dashed rounded-2xl transition-all duration-200 cursor-pointer ${
          dragging
            ? 'border-blue-400 bg-blue-50/60 dark:bg-blue-900/20 scale-[1.01]'
            : file
            ? 'border-emerald-300 bg-emerald-50/40 dark:bg-emerald-900/10'
            : 'border-[var(--border)] hover:border-blue-300 dark:hover:border-blue-600 bg-[var(--surface-elevated)]'
        }`}
        onDrop={onDrop}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onClick={() => !loading && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          disabled={loading}
          onChange={e => { const f = e.target.files?.[0]; if (f) pickFile(f); }}
        />

        <div className="p-10 flex flex-col items-center gap-4 text-center">
          {file ? (
            <>
              <FileIcon ext={ext} />
              <div>
                <p className="font-semibold text-[var(--text)] text-base">{file.name}</p>
                <p className="text-sm text-[var(--text-muted)] mt-1">
                  {sizeMB} MB · {ext.toUpperCase()}
                </p>
              </div>
              {!loading && (
                <button
                  type="button"
                  onClick={e => { e.stopPropagation(); setFile(null); setStep(-1); }}
                  className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] hover:text-red-500 transition"
                >
                  <X size={14} /> Xóa tệp
                </button>
              )}
            </>
          ) : (
            <>
              <div className="p-4 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 shadow-lg shadow-blue-500/25">
                <UploadCloud className="w-8 h-8 text-white" />
              </div>
              <div>
                <p className="font-semibold text-[var(--text)]">
                  Kéo thả hoặc{' '}
                  <span className="text-blue-600 dark:text-blue-400">chọn tệp</span>
                </p>
                <p className="text-sm text-[var(--text-muted)] mt-1">PDF, DOCX, TXT · Tối đa {MAX_MB} MB</p>
              </div>
            </>
          )}
        </div>
      </motion.div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800/50 rounded-xl px-4 py-3"
          >
            <AlertCircle size={16} /> {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Processing steps */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}
            className="ui-card p-5 overflow-hidden"
          >
            <p className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)] mb-4">
              Đang xử lý
            </p>
            <div className="space-y-3">
              {STEPS.map((s, i) => {
                const Icon = s.icon;
                const done = step > i;
                const active = step === i;
                return (
                  <div key={i} className={`flex items-center gap-3 transition-opacity ${i > step + 1 ? 'opacity-30' : 'opacity-100'}`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 transition-all ${
                      done ? 'bg-emerald-500' : active ? 'bg-blue-600 animate-pulse' : 'bg-[var(--surface-inset)]'
                    }`}>
                      {done
                        ? <CheckCircle2 size={16} className="text-white" />
                        : active
                        ? <Loader2 size={14} className="text-white animate-spin" />
                        : <Icon size={14} className="text-[var(--text-faint)]" />
                      }
                    </div>
                    <div>
                      <p className={`text-sm font-medium ${active ? 'text-blue-600 dark:text-blue-400' : done ? 'text-emerald-600 dark:text-emerald-400' : 'text-[var(--text-muted)]'}`}>
                        {s.label}
                      </p>
                      <p className="text-xs text-[var(--text-faint)]">{s.sub}</p>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Progress bar */}
            <div className="mt-4 h-1.5 rounded-full bg-[var(--surface-inset)] overflow-hidden">
              <motion.div
                className="h-full rounded-full bg-blue-600"
                animate={{ width: `${step < 0 ? 0 : ((step + 1) / STEPS.length) * 100}%` }}
                transition={{ duration: 0.4 }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Submit button */}
      <motion.button
        type="button"
        onClick={onUpload}
        disabled={!file || loading}
        whileHover={!file || loading ? {} : { scale: 1.01 }}
        whileTap={!file || loading ? {} : { scale: 0.99 }}
        className="ui-btn-primary w-full py-3 text-base font-semibold shadow-lg shadow-blue-500/20"
      >
        {loading ? (
          <><Loader2 className="w-5 h-5 animate-spin" /> Đang phân tích...</>
        ) : (
          <><Sparkles className="w-5 h-5" /> Phân tích tài liệu AI</>
        )}
      </motion.button>

      {/* Feature badges */}
      <div className="flex flex-wrap gap-2 justify-center">
        {['Semantic chunking', 'AI overview', 'Vector search', 'Citation grounding', 'Quiz & Podcast'].map(f => (
          <span key={f} className="text-xs px-3 py-1 rounded-full bg-[var(--surface-inset)] text-[var(--text-muted)] border border-[var(--border)]">
            {f}
          </span>
        ))}
      </div>
    </div>
  );
}
