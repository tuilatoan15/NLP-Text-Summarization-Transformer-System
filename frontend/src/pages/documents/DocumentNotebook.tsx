import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  HelpCircle, Radio, Presentation, CreditCard, BookOpen,
  ChevronLeft, ChevronRight, CheckCircle2, XCircle, Loader2,
  Volume2, Play, Pause, SkipForward,
} from 'lucide-react';
import { exportPodcastTts } from '../../services/apiService';
import { useDocumentContext } from '../../context/DocumentContext';

// ─── Tab types ─────────────────────────────────────────────────────────────
const NOTEBOOK_TABS = [
  { id: 'quiz',         label: 'Quiz',         icon: HelpCircle },
  { id: 'flashcards',  label: 'Flashcards',   icon: CreditCard },
  { id: 'podcast',     label: 'Podcast',      icon: Radio },
  { id: 'presentation',label: 'Presentation', icon: Presentation },
  { id: 'report',      label: 'AI Report',    icon: BookOpen },
];

// ─── Quiz component ─────────────────────────────────────────────────────────
function QuizSection({ quiz }: { quiz: Array<Record<string, any>> }) {
  const [idx, setIdx] = useState(0);
  const [selected, setSelected] = useState<Record<number, string>>({});
  const [revealed, setRevealed] = useState<Record<number, boolean>>({});

  if (!quiz.length) return <Empty label="Không có câu hỏi quiz." />;

  const q = quiz[idx];
  const choices: string[] = q.choices ?? q.options ?? [];
  const correct: string = q.answer ?? q.correct ?? '';
  const isRevealed = !!revealed[idx];
  const userAnswer = selected[idx];

  return (
    <div className="space-y-4">
      {/* Progress */}
      <div className="flex items-center justify-between text-xs text-[var(--text-muted)]">
        <span>Câu {idx + 1} / {quiz.length}</span>
        <div className="flex gap-1">
          {quiz.map((_, i) => (
            <div
              key={i}
              className={`w-2 h-2 rounded-full transition-all cursor-pointer ${
                i === idx ? 'bg-blue-500 w-4' : selected[i] ? 'bg-emerald-400' : 'bg-[var(--border)]'
              }`}
              onClick={() => setIdx(i)}
            />
          ))}
        </div>
      </div>

      {/* Question */}
      <div className="ui-card p-5">
        <p className="text-sm font-semibold text-[var(--text)] leading-relaxed mb-4">{q.question}</p>

        {choices.length > 0 ? (
          <div className="space-y-2">
            {choices.map((choice: string) => {
              const isCorrect = choice === correct;
              const isSelected = userAnswer === choice;
              let cls = 'border-[var(--border)] bg-[var(--surface-inset)]';
              if (isRevealed) {
                if (isCorrect) cls = 'border-emerald-400 bg-emerald-50 dark:bg-emerald-900/20';
                else if (isSelected) cls = 'border-red-400 bg-red-50 dark:bg-red-900/20';
              } else if (isSelected) {
                cls = 'border-blue-400 bg-blue-50 dark:bg-blue-900/20';
              }
              return (
                <button
                  key={choice}
                  type="button"
                  onClick={() => { setSelected(s => ({ ...s, [idx]: choice })); }}
                  disabled={isRevealed}
                  className={`w-full text-left px-4 py-3 rounded-xl border text-sm transition-all ${cls}`}
                >
                  <span className="flex items-center gap-3">
                    {isRevealed && isCorrect && <CheckCircle2 size={15} className="text-emerald-500 shrink-0" />}
                    {isRevealed && isSelected && !isCorrect && <XCircle size={15} className="text-red-500 shrink-0" />}
                    {choice}
                  </span>
                </button>
              );
            })}
          </div>
        ) : (
          <div>
            <p className="text-xs text-[var(--text-muted)] mb-1">Câu trả lời:</p>
            {isRevealed && (
              <p className="text-sm text-emerald-600 dark:text-emerald-400 font-medium">{correct}</p>
            )}
          </div>
        )}

        {q.explanation && isRevealed && (
          <div className="mt-3 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 text-xs text-blue-700 dark:text-blue-300">
            💡 {q.explanation}
          </div>
        )}
      </div>

      <div className="flex gap-2">
        <button
          type="button"
          className="ui-btn-secondary flex-1"
          onClick={() => setRevealed(r => ({ ...r, [idx]: true }))}
          disabled={isRevealed}
        >
          Xem đáp án
        </button>
        <button type="button" className="ui-btn-secondary px-3" onClick={() => setIdx(i => Math.max(0, i - 1))} disabled={idx === 0}>
          <ChevronLeft size={16} />
        </button>
        <button type="button" className="ui-btn-primary px-3" onClick={() => setIdx(i => Math.min(quiz.length - 1, i + 1))} disabled={idx === quiz.length - 1}>
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
}

// ─── Flashcards ─────────────────────────────────────────────────────────────
function FlashcardSection({ quiz }: { quiz: Array<Record<string, any>> }) {
  const [idx, setIdx] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const cards = quiz.filter(q => q.question && (q.answer ?? q.correct));

  if (!cards.length) return <Empty label="Không có flashcards." />;
  const card = cards[idx];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between text-xs text-[var(--text-muted)]">
        <span>{idx + 1} / {cards.length}</span>
        <span className="text-blue-500 font-medium">{flipped ? 'Đáp án' : 'Câu hỏi'}</span>
      </div>

      {/* Card */}
      <motion.div
        className="cursor-pointer"
        onClick={() => setFlipped(f => !f)}
        whileTap={{ scale: 0.98 }}
      >
        <div className="ui-card p-8 min-h-48 flex flex-col items-center justify-center text-center relative overflow-hidden">
          <div className="absolute inset-0 opacity-5"
            style={{ background: flipped ? 'radial-gradient(circle, #10b981, transparent)' : 'radial-gradient(circle, #3b82f6, transparent)' }}
          />
          <AnimatePresence mode="wait">
            <motion.div
              key={flipped ? 'answer' : 'question'}
              initial={{ opacity: 0, rotateX: -30 }} animate={{ opacity: 1, rotateX: 0 }} exit={{ opacity: 0, rotateX: 30 }}
              transition={{ duration: 0.2 }}
            >
              {!flipped ? (
                <>
                  <HelpCircle className="w-6 h-6 text-blue-400 mx-auto mb-3" />
                  <p className="text-base font-semibold text-[var(--text)]">{card.question}</p>
                </>
              ) : (
                <>
                  <CheckCircle2 className="w-6 h-6 text-emerald-400 mx-auto mb-3" />
                  <p className="text-base text-emerald-600 dark:text-emerald-400 font-semibold">{card.answer ?? card.correct}</p>
                </>
              )}
            </motion.div>
          </AnimatePresence>
          <p className="text-xs text-[var(--text-faint)] mt-4">Nhấn để lật thẻ</p>
        </div>
      </motion.div>

      <div className="flex gap-2">
        <button type="button" className="ui-btn-secondary flex-1" onClick={() => { setIdx(i => Math.max(0, i - 1)); setFlipped(false); }} disabled={idx === 0}>
          <ChevronLeft size={15} /> Trước
        </button>
        <button type="button" className="ui-btn-primary flex-1" onClick={() => { setIdx(i => Math.min(cards.length - 1, i + 1)); setFlipped(false); }} disabled={idx === cards.length - 1}>
          Tiếp <ChevronRight size={15} />
        </button>
      </div>
    </div>
  );
}

// ─── Podcast ─────────────────────────────────────────────────────────────────
function PodcastSection({ podcast, documentId }: { podcast: Record<string, any>; documentId: string }) {
  const turns: Array<{ speaker: string; text: string }> = podcast?.turns ?? [];
  const [ttsLoading, setTtsLoading] = useState(false);
  const [ttsResult, setTtsResult] = useState<string | null>(null);
  const [playing, setPlaying] = useState<number | null>(null);

  async function exportTTS() {
    setTtsLoading(true);
    try {
      const r = await exportPodcastTts(documentId) as any;
      setTtsResult(r?.audio_uri ?? r?.message ?? r?.status ?? 'Generated');
    } finally {
      setTtsLoading(false);
    }
  }

  if (!turns.length) return <Empty label="Không có script podcast." />;

  const speakerColors: Record<string, string> = {};
  const palette = ['#3b82f6', '#10b981', '#f59e0b', '#e879f9'];
  let si = 0;
  turns.forEach(t => { if (!speakerColors[t.speaker]) { speakerColors[t.speaker] = palette[si++ % palette.length]; } });

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-3 p-3 rounded-xl bg-[var(--surface-inset)] border border-[var(--border)]">
        <Radio className="w-4 h-4 text-blue-500" />
        <div className="flex-1">
          <p className="text-xs font-bold text-[var(--text)]">AI Podcast Script</p>
          <p className="text-[10px] text-[var(--text-muted)]">{turns.length} lượt · {Object.keys(speakerColors).join(', ')}</p>
        </div>
        <button
          type="button"
          className="ui-btn-secondary text-xs px-3 py-1.5"
          onClick={exportTTS}
          disabled={ttsLoading}
        >
          {ttsLoading ? <Loader2 size={13} className="animate-spin" /> : <Volume2 size={13} />}
          Export TTS
        </button>
      </div>

      {ttsResult && (
        <div className="text-xs text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 rounded-lg px-3 py-2">
          ✓ {ttsResult}
        </div>
      )}

      {/* Transcript */}
      <div className="space-y-3 max-h-96 overflow-y-auto pr-1">
        {turns.map((turn, i) => {
          const isA = i % 2 === 0;
          const color = speakerColors[turn.speaker] ?? '#64748b';
          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: isA ? -10 : 10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.03 }}
              className={`flex gap-3 ${isA ? '' : 'flex-row-reverse'}`}
            >
              <div
                className="w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-bold shrink-0"
                style={{ background: color }}
              >
                {turn.speaker.slice(0, 1).toUpperCase()}
              </div>
              <div
                className={`flex-1 max-w-[80%] px-4 py-2.5 rounded-2xl text-sm ${
                  isA ? 'rounded-tl-sm' : 'rounded-tr-sm'
                }`}
                style={{ background: `${color}18`, border: `1px solid ${color}30` }}
              >
                <p className="text-[10px] font-bold mb-1 opacity-70" style={{ color }}>{turn.speaker}</p>
                <p className="text-[var(--text-secondary)] leading-relaxed">{turn.text}</p>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Presentation ────────────────────────────────────────────────────────────
function PresentationSection({ slides }: { slides: Array<Record<string, any>> }) {
  const [idx, setIdx] = useState(0);
  if (!slides.length) return <Empty label="Không có slide nào." />;
  const slide = slides[idx];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between text-xs text-[var(--text-muted)]">
        <span>Slide {idx + 1} / {slides.length}</span>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={idx}
          initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}
          className="ui-card p-6 min-h-40"
        >
          <div className="flex items-center gap-2 mb-3">
            <span className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 text-white text-sm font-bold flex items-center justify-center">
              {slide.slide ?? idx + 1}
            </span>
            <h3 className="font-bold text-[var(--text)]">{slide.title ?? `Slide ${idx + 1}`}</h3>
          </div>
          {slide.content && (
            <ul className="space-y-1.5 mt-3">
              {(Array.isArray(slide.content) ? slide.content : [slide.content]).map((pt: string, i: number) => (
                <li key={i} className="flex items-start gap-2 text-sm text-[var(--text-secondary)]">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-2 shrink-0" />
                  {pt}
                </li>
              ))}
            </ul>
          )}
          {slide.notes && (
            <p className="text-xs text-[var(--text-faint)] mt-4 italic border-t border-[var(--border)] pt-3">{slide.notes}</p>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Slide list */}
      <div className="flex gap-1 overflow-x-auto pb-1">
        {slides.map((s, i) => (
          <button
            key={i}
            type="button"
            onClick={() => setIdx(i)}
            className={`shrink-0 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
              i === idx
                ? 'bg-blue-600 text-white border-transparent'
                : 'border-[var(--border)] text-[var(--text-muted)] hover:border-blue-300'
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>

      <div className="flex gap-2">
        <button type="button" className="ui-btn-secondary flex-1" onClick={() => setIdx(i => Math.max(0, i - 1))} disabled={idx === 0}>
          <ChevronLeft size={15} /> Trước
        </button>
        <button type="button" className="ui-btn-primary flex-1" onClick={() => setIdx(i => Math.min(slides.length - 1, i + 1))} disabled={idx === slides.length - 1}>
          Tiếp <ChevronRight size={15} />
        </button>
      </div>
    </div>
  );
}

// ─── AI Report ───────────────────────────────────────────────────────────────
function ReportSection({ report }: { report: string | Record<string, any> }) {
  const text = typeof report === 'string' ? report : JSON.stringify(report, null, 2);
  if (!text) return <Empty label="Không có report." />;
  return (
    <div className="ui-card p-5">
      <pre className="text-sm text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap font-sans">
        {text}
      </pre>
    </div>
  );
}

// ─── Empty state ─────────────────────────────────────────────────────────────
function Empty({ label }: { label: string }) {
  return (
    <div className="py-12 text-center text-[var(--text-muted)] text-sm">{label}</div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function DocumentNotebook() {
  const { document } = useDocumentContext();
  const [activeTab, setActiveTab] = useState('quiz');

  if (!document) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <BookOpen className="w-16 h-16 text-[var(--text-faint)] mb-4" />
        <p className="text-[var(--text-muted)] font-medium">Upload tài liệu trước.</p>
      </div>
    );
  }

  const assets = (document.analysis_assets as Record<string, any>) ?? {};
  const quiz: Array<Record<string, any>> = assets.quiz ?? [];
  const podcast: Record<string, any> = assets.podcast ?? {};
  const slides: Array<Record<string, any>> = assets.presentation ?? [];
  const report = assets.report ?? assets.ai_report ?? '';

  return (
    <div className="space-y-5">
      {/* Tab bar */}
      <div className="flex flex-wrap gap-2">
        {NOTEBOOK_TABS.map(tab => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-semibold border transition-all ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white border-transparent shadow-sm shadow-blue-500/25'
                  : 'border-[var(--border)] text-[var(--text-muted)] hover:border-blue-300 dark:hover:border-blue-700'
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          {activeTab === 'quiz'          && <QuizSection quiz={quiz} />}
          {activeTab === 'flashcards'   && <FlashcardSection quiz={quiz} />}
          {activeTab === 'podcast'      && <PodcastSection podcast={podcast} documentId={document.document_id as string} />}
          {activeTab === 'presentation' && <PresentationSection slides={slides} />}
          {activeTab === 'report'       && <ReportSection report={report} />}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
