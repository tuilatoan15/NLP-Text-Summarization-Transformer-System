import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Brain, Quote, Layers3, Tag, BarChart3, FileText,
  Hash, Zap, BookOpen, TrendingUp,
} from 'lucide-react';
import {
  ResponsiveContainer, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts';
import { useDocumentContext } from '../../context/DocumentContext';

const pct = (v: number) => `${Math.round((v ?? 0) * 100)}%`;

function InfoCard({ title, icon: Icon, children, className = '' }: {
  title: string; icon: React.ElementType; children: React.ReactNode; className?: string;
}) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`ui-card p-5 ${className}`}
    >
      <div className="flex items-center gap-2 mb-4">
        <div className="p-1.5 rounded-lg bg-blue-50 dark:bg-blue-900/30">
          <Icon className="w-4 h-4 text-blue-600 dark:text-blue-400" />
        </div>
        <h3 className="text-sm font-bold text-[var(--text)]">{title}</h3>
      </div>
      {children}
    </motion.section>
  );
}

function StatPill({ label, value, color = 'blue' }: { label: string; value: string | number; color?: string }) {
  const bg: Record<string, string> = {
    blue: 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800',
    emerald: 'bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800',
    amber: 'bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800',
    violet: 'bg-violet-50 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300 border-violet-200 dark:border-violet-800',
  };
  return (
    <div className={`rounded-xl border px-4 py-3 ${bg[color] ?? bg.blue}`}>
      <p className="text-[10px] font-bold uppercase tracking-wider opacity-70">{label}</p>
      <p className="text-xl font-bold mt-0.5">{value}</p>
    </div>
  );
}

export default function DocumentAnalysis() {
  const { document } = useDocumentContext();

  if (!document) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <FileText className="w-16 h-16 text-[var(--text-faint)] mb-4" />
        <p className="text-[var(--text-muted)] font-medium">Upload tài liệu trước để xem phân tích.</p>
      </div>
    );
  }

  const assets = (document.analysis_assets as Record<string, any>) ?? {};
  const overview = assets.overview ?? {};
  const quality = (document.quality as Record<string, any>) ?? {};
  const meta = (document.meta as Record<string, any>) ?? {};
  const sections = (document.sections as Array<Record<string, any>>) ?? [];
  const chunks = (document.chunks as Array<Record<string, any>>) ?? [];

  const keywords: Array<{ term: string; score: number }> = overview.keywords ?? [];
  const insights: string[] = overview.key_insights ?? [];
  const takeaways: string[] = overview.key_takeaways ?? [];

  const chunkSizes = useMemo(() =>
    chunks.slice(0, 20).map((c, i) => ({
      name: `C${i + 1}`,
      words: (c.content ?? c.text ?? '').split(/\s+/).filter(Boolean).length,
    })), [chunks]);

  const qualityScore = (quality?.extraction?.score ?? 0) as number;
  const radarData = [
    { subject: 'Extraction', value: qualityScore * 100 },
    { subject: 'Structure', value: sections.length > 3 ? 80 : 50 },
    { subject: 'Coverage', value: Math.min(100, (quality?.chunk_count ?? 0) * 5) },
    { subject: 'Readability', value: 72 },
    { subject: 'Completeness', value: meta?.word_count ? Math.min(100, meta.word_count / 30) : 60 },
  ];

  return (
    <div className="space-y-5">
      {/* Stats row */}
      <motion.div
        initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-2 md:grid-cols-4 gap-3"
      >
        <StatPill label="Chất lượng trích xuất" value={pct(qualityScore)} color="emerald" />
        <StatPill label="Số chunks" value={quality?.chunk_count ?? chunks.length ?? '—'} color="blue" />
        <StatPill label="Số từ" value={(meta?.word_count ?? 0).toLocaleString()} color="amber" />
        <StatPill label="Số sections" value={sections.length || '—'} color="violet" />
      </motion.div>

      {/* Overview + Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <InfoCard title="AI Document Overview" icon={Brain}>
          <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
            {overview.document_overview || 'Chưa có dữ liệu overview.'}
          </p>
          {insights.length > 0 && (
            <div className="mt-4 space-y-2">
              <p className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Key Insights</p>
              {insights.map((item, i) => (
                <div key={i} className="flex gap-2 text-sm text-[var(--text-secondary)]">
                  <span className="font-bold text-blue-500 shrink-0">{i + 1}.</span>
                  <span>{item}</span>
                </div>
              ))}
            </div>
          )}
        </InfoCard>

        <InfoCard title="Key Takeaways" icon={Quote}>
          {takeaways.length > 0 ? (
            <div className="space-y-3">
              {takeaways.map((t, i) => (
                <div key={i} className="border-l-4 border-blue-400 pl-3 py-1">
                  <p className="text-sm text-[var(--text-secondary)]">{t}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-[var(--text-muted)]">Chưa có takeaways.</p>
          )}
        </InfoCard>
      </div>

      {/* Keywords */}
      {keywords.length > 0 && (
        <InfoCard title="Keywords & Trọng số" icon={Tag}>
          <div className="flex flex-wrap gap-2">
            {keywords.slice(0, 20).map((k) => {
              const intensity = Math.round((k.score ?? 0.5) * 9) + 1;
              return (
                <span
                  key={k.term}
                  className="px-3 py-1 rounded-full text-xs font-medium border transition-all hover:scale-105"
                  style={{
                    background: `rgba(37,99,235,${0.08 + (k.score ?? 0.5) * 0.25})`,
                    borderColor: `rgba(37,99,235,${0.15 + (k.score ?? 0.5) * 0.3})`,
                    color: `rgb(${37 + intensity * 5},${99 - intensity * 3},235)`,
                    fontSize: `${10 + Math.round((k.score ?? 0.5) * 4)}px`,
                  }}
                >
                  {k.term}
                </span>
              );
            })}
          </div>
        </InfoCard>
      )}

      {/* Quality radar + Chunk size distribution */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <InfoCard title="Document Quality Radar" icon={TrendingUp}>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="var(--border)" />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                <Radar
                  name="Quality"
                  dataKey="value"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.25}
                />
                <Tooltip
                  contentStyle={{ background: 'var(--surface-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
                  formatter={(v: number) => [`${v.toFixed(0)}%`]}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </InfoCard>

        <InfoCard title="Chunk Size Distribution" icon={BarChart3}>
          {chunkSizes.length > 0 ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chunkSizes} margin={{ left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                  <XAxis dataKey="name" tick={{ fontSize: 9, fill: 'var(--text-muted)' }} />
                  <YAxis tick={{ fontSize: 9, fill: 'var(--text-muted)' }} />
                  <Tooltip
                    contentStyle={{ background: 'var(--surface-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
                    formatter={(v: number) => [`${v} từ`]}
                  />
                  <Bar dataKey="words" fill="#6366f1" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-sm text-[var(--text-muted)] py-8 text-center">Không có dữ liệu chunks.</p>
          )}
        </InfoCard>
      </div>

      {/* Sections tree */}
      {sections.length > 0 && (
        <InfoCard title="Cấu trúc tài liệu" icon={BookOpen}>
          <div className="space-y-1.5 max-h-64 overflow-y-auto pr-2">
            {sections.map((s, i) => (
              <div
                key={i}
                className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-[var(--surface-inset)] transition"
                style={{ paddingLeft: `${12 + ((s.level ?? 1) - 1) * 16}px` }}
              >
                <Hash className="w-3 h-3 text-[var(--text-faint)] shrink-0" />
                <span className="text-sm text-[var(--text-secondary)] flex-1 truncate">{s.title ?? s.heading ?? `Section ${i + 1}`}</span>
                <span className="text-xs text-[var(--text-faint)]">{s.word_count ?? 0} từ</span>
              </div>
            ))}
          </div>
        </InfoCard>
      )}

      {/* Document ID */}
      <div className="flex items-center gap-2 text-xs text-[var(--text-faint)] px-1">
        <Zap className="w-3 h-3" />
        <span>Document ID:</span>
        <code className="bg-[var(--surface-inset)] px-2 py-0.5 rounded font-mono">{document.document_id as string}</code>
      </div>
    </div>
  );
}
