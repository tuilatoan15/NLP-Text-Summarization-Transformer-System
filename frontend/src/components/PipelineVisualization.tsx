import React from 'react';
import { motion } from 'framer-motion';
import MetricBadge from './animations/MetricBadge';

export const PIPELINE_STEPS = [
  { id: 'question', label: 'Question', short: 'Q' },
  { id: 'embedding', label: 'Embedding', short: 'Emb' },
  { id: 'retrieval', label: 'Hybrid Retrieval', short: 'Ret' },
  { id: 'crossencoder', label: 'CrossEncoder', short: 'CE' },
  { id: 'acb_intent', label: 'Intent Analysis', short: 'Int' },
  { id: 'acb_summary', label: 'Query Summary', short: 'Sum' },
  { id: 'acb_chunks', label: 'Dynamic Chunks', short: 'Chk' },
  { id: 'acb_compose', label: 'Prompt Compose', short: 'Pr' },
  { id: 'generation', label: 'Qwen / LLM', short: 'Gen' },
  { id: 'streaming', label: 'Streaming', short: 'Str' },
] as const;

/** Legacy stage IDs for backward compat when adaptive is off */
export const LEGACY_COMPRESSION_STAGE = 'context_compression';

export type PipelineStageId = (typeof PIPELINE_STEPS)[number]['id'] | typeof LEGACY_COMPRESSION_STAGE;

export interface AdaptiveContextMetrics {
  compressionRatio?: number | null;
  tokenReduction?: number | null;
  chunksKept?: number | null;
  summaryTokens?: number | null;
  generationTokens?: number | null;
  latencySavingS?: number | null;
  compressionTier?: string | null;
  adaptiveMode?: boolean;
}

interface PipelineVisualizationProps {
  activeStage: PipelineStageId | null;
  completedStages?: PipelineStageId[];
  metrics?: AdaptiveContextMetrics;
  className?: string;
}

export default function PipelineVisualization({
  activeStage,
  completedStages = [],
  metrics,
  className = '',
}: PipelineVisualizationProps) {
  const completed = new Set(completedStages);
  const activeIdx = activeStage
    ? PIPELINE_STEPS.findIndex(s => s.id === activeStage)
    : -1;

  const showMetrics = metrics?.adaptiveMode !== false && (
    metrics?.compressionRatio != null ||
    metrics?.tokenReduction != null ||
    metrics?.chunksKept != null
  );

  return (
    <div className={`rounded-xl border border-[var(--border)] bg-[var(--bg-muted)]/30 p-3 ${className}`}>
      <div className="flex items-center justify-between mb-2.5">
        <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-faint)]">
          Adaptive Context Builder Pipeline
        </span>
        {metrics?.compressionTier && (
          <span className="text-[9px] font-bold text-violet-600 dark:text-violet-400 bg-violet-500/10 px-2 py-0.5 rounded-full capitalize">
            {metrics.compressionTier}
          </span>
        )}
      </div>

      {showMetrics && (
        <div className="flex flex-wrap gap-1.5 mb-2.5">
          {metrics?.compressionRatio != null && (
            <MetricBadge label="Nén" value={`${(metrics.compressionRatio * 100).toFixed(0)}%`} color="sky" />
          )}
          {metrics?.tokenReduction != null && (
            <MetricBadge label="Token −" value={`${(metrics.tokenReduction * 100).toFixed(0)}%`} color="emerald" />
          )}
          {metrics?.chunksKept != null && (
            <MetricBadge label="Chunks" value={String(metrics.chunksKept)} color="amber" />
          )}
          {metrics?.summaryTokens != null && metrics.summaryTokens > 0 && (
            <MetricBadge label="Sum tok" value={String(metrics.summaryTokens)} color="violet" />
          )}
          {metrics?.latencySavingS != null && metrics.latencySavingS > 0 && (
            <MetricBadge label="Tiết kiệm" value={`${metrics.latencySavingS.toFixed(2)}s`} color="rose" />
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-1">
        {PIPELINE_STEPS.map((step, idx) => {
          const isActive = activeStage === step.id;
          const isDone = completed.has(step.id) || (activeIdx > idx);

          return (
            <React.Fragment key={step.id}>
              <motion.div
                layout
                animate={{
                  scale: isActive ? 1.05 : 1,
                  boxShadow: isActive
                    ? '0 0 0 2px rgba(139, 92, 246, 0.35)'
                    : '0 0 0 0px rgba(0,0,0,0)',
                }}
                transition={{ type: 'spring', stiffness: 400, damping: 28 }}
                className={`relative flex items-center gap-1 px-2 py-1 rounded-lg text-[9px] font-bold border transition-colors ${
                  isActive
                    ? 'bg-violet-500 text-white border-violet-600'
                    : isDone
                      ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/25'
                      : 'bg-[var(--bg)] text-[var(--text-faint)] border-[var(--border)]'
                }`}
                title={step.label}
              >
                {isActive && (
                  <motion.span
                    className="absolute inset-0 rounded-lg bg-violet-400/20"
                    animate={{ opacity: [0.3, 0.7, 0.3] }}
                    transition={{ repeat: Infinity, duration: 1.2 }}
                  />
                )}
                <span className="relative z-10">{step.short}</span>
                <span className="relative z-10 hidden sm:inline truncate max-w-[72px]">{step.label}</span>
              </motion.div>
              {idx < PIPELINE_STEPS.length - 1 && (
                <motion.span
                  animate={{ color: isDone ? 'var(--accent)' : 'var(--text-faint)' }}
                  className="text-[10px] font-bold opacity-50"
                >
                  →
                </motion.span>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
