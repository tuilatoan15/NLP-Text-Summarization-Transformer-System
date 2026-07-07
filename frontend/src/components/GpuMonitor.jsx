import React, { memo } from 'react';
import { motion } from 'framer-motion';
import { Cpu, HardDrive, Server, Thermometer } from 'lucide-react';

const GaugeBar = memo(({ label, value, max, display, color, icon: Icon }) => {
  const pct = max > 0 ? Math.min(100, Math.round((value / max) * 100)) : 0;
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center text-xs font-semibold">
        <span className="text-[var(--text-muted)] flex items-center gap-1.5">
          {Icon && <Icon size={14} style={{ color }} />}
          {label}
        </span>
        <span className="text-[var(--text-primary)]">{display}</span>
      </div>
      <div className="h-1.5 rounded-full bg-[var(--bg-inset)] overflow-hidden">
        <motion.div
          className="h-full w-full rounded-full origin-left gpu-bar-fill"
          style={{ backgroundColor: color }}
          initial={{ scaleX: 0 }}
          animate={{ scaleX: pct / 100 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
});

const GpuMonitor = memo(({ gpu, node, models, loading, compact = false }) => {
  if (loading) {
    return (
      <div className="space-y-4">
        {Array.from({ length: compact ? 2 : 3 }).map((_, i) => (
          <div key={i} className="ui-skeleton h-8 w-full rounded-lg" />
        ))}
      </div>
    );
  }

  const unavailable = !gpu?.available;
  const host = gpu?.host || {};
  const gpuName = gpu?.gpu_name || 'Unavailable';
  const util = gpu?.gpu_utilization_percent;
  const totalVram = gpu?.total_vram_mb;
  const usedVram = gpu?.vram_used_mb ?? gpu?.allocated_vram_mb;
  const ramTotal = host.ram_total_gb;
  const ramUsed = host.ram_used_gb;

  const loadedSummarizers = (models?.summarizers || [])
    .filter((m) => m.loaded)
    .map((m) => m.key)
    .join(', ') || '—';

  const statusLabel = {
    healthy: 'Healthy',
    busy: 'Busy',
    offline: 'Offline',
  }[node?.status] || node?.status || 'Unknown';

  const inferenceLabel = {
    idle: 'Sẵn sàng (Idle)',
    busy: 'Đang suy luận',
    lazy: 'Lazy load',
  }[node?.inference_state] || node?.inference_state || '—';

  return (
    <div className="space-y-4">
      {unavailable ? (
        <p className="text-xs font-medium text-[var(--text-muted)] bg-[var(--bg-muted)]/50 rounded-lg px-3 py-2 border border-[var(--border)]">
          GPU: Unavailable — {gpu?.device === 'cpu' ? 'CPU mode' : 'No CUDA device detected'}
        </p>
      ) : (
        <>
          <GaugeBar
            label={gpuName}
            value={util ?? 0}
            max={100}
            display={util != null ? `${util}% Load` : 'N/A'}
            color="#6366f1"
            icon={Cpu}
          />
          <GaugeBar
            label="GPU VRAM"
            value={usedVram ?? 0}
            max={totalVram ?? 1}
            display={
              totalVram
                ? `${((usedVram ?? 0) / 1024).toFixed(1)} / ${(totalVram / 1024).toFixed(1)} GB`
                : 'N/A'
            }
            color="#0ea5e9"
            icon={HardDrive}
          />
        </>
      )}

      {!compact && ramTotal != null && (
        <GaugeBar
          label="System RAM"
          value={ramUsed ?? 0}
          max={ramTotal}
          display={`${ramUsed ?? '—'} / ${ramTotal} GB`}
          color="#10b981"
          icon={Server}
        />
      )}

      <div className="pt-3 border-t border-[var(--border)] space-y-2 text-xs font-medium text-[var(--text-secondary)]">
        <div className="flex justify-between">
          <span>Mô hình đã nạp</span>
          <span className="font-bold text-sky-600 dark:text-sky-400 text-right max-w-[55%] truncate">
            {loadedSummarizers}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Trạng thái suy luận</span>
          <span className={`font-bold ${node?.inference_state === 'busy' ? 'text-amber-500' : 'text-emerald-500'}`}>
            {inferenceLabel}
          </span>
        </div>
        {gpu?.temperature_c != null && (
          <div className="flex justify-between items-center">
            <span className="flex items-center gap-1">
              <Thermometer size={12} />
              GPU Temperature
            </span>
            <span>{gpu.temperature_c}°C</span>
          </div>
        )}
      </div>

      <div className="text-[10px] text-[var(--text-faint)] font-bold uppercase tracking-wider text-center">
        Node: {statusLabel}
        {node?.node_id ? ` · ${node.node_id}` : ''}
      </div>
    </div>
  );
});

export default GpuMonitor;
