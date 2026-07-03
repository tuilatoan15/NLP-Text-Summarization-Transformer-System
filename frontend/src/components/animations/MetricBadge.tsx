import React from 'react';
import { motion } from 'framer-motion';

type MetricColor = 'sky' | 'emerald' | 'amber' | 'violet' | 'rose';

const COLOR_MAP: Record<MetricColor, string> = {
  sky: 'text-sky-600 dark:text-sky-400 bg-sky-500/10',
  emerald: 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10',
  amber: 'text-amber-600 dark:text-amber-400 bg-amber-500/10',
  violet: 'text-violet-600 dark:text-violet-400 bg-violet-500/10',
  rose: 'text-rose-600 dark:text-rose-400 bg-rose-500/10',
};

interface MetricBadgeProps {
  label: string;
  value: string;
  color?: MetricColor;
}

export default function MetricBadge({ label, value, color = 'sky' }: MetricBadgeProps) {
  return (
    <motion.span
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className={`text-[9px] font-bold px-2 py-0.5 rounded-full ${COLOR_MAP[color]}`}
    >
      {label}: {value}
    </motion.span>
  );
}
