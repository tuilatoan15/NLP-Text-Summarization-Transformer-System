import React, { memo } from 'react';
import { NavLink } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard, MessageSquare,
  BarChart3, Settings, Sparkles,
  GitCompareArrows, PanelLeftClose, PanelLeftOpen, Cpu, Database
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { useSidebarSystemQueries } from '../hooks/useApiQueries';
import { usePrefersReducedMotion } from '../hooks/usePrefersReducedMotion';
import BrandLogo from '../components/BrandLogo';

const navGroups = [
  {
    labelKey: 'navSectionWorkspace',
    labelVie: 'KHÔNG GIAN LÀM VIỆC',
    labelEng: 'WORKSPACE',
    items: [
      { nameKey: 'navOverview', path: '/', icon: LayoutDashboard, labelVie: 'Bảng điều khiển', labelEng: 'Dashboard' },
      { nameKey: 'navSummarize', path: '/summarize', icon: Sparkles, labelVie: 'Tóm tắt tài liệu', labelEng: 'Summarization' },
      { nameKey: 'navChat', path: '/chat', icon: MessageSquare, labelVie: 'Trò chuyện AI', labelEng: 'AI Chat' },
    ],
  },
  {
    labelKey: 'navSectionAI',
    labelVie: 'PHÂN TÍCH & SO SÁNH',
    labelEng: 'AI & ANALYTICS',
    items: [
      { nameKey: 'navCompare', path: '/compare', icon: GitCompareArrows, labelVie: 'So sánh Mô hình', labelEng: 'Model Comparison' },
      { nameKey: 'navAnalytics', path: '/analytics', icon: BarChart3, labelVie: 'Phân tích hiệu năng', labelEng: 'Analytics' },
      { nameKey: 'navDatasetAnalytics', path: '/dataset-analytics', icon: Database, labelVie: 'Dataset Analytics', labelEng: 'Dataset Analytics' },
    ],
  },
  {
    labelKey: 'navSectionSystem',
    labelVie: 'QUẢN TRỊ HỆ THỐNG',
    labelEng: 'SYSTEM & DATA',
    items: [
      { nameKey: 'navBenchmark', path: '/benchmark', icon: Cpu, labelVie: 'Kết quả Benchmark', labelEng: 'Models' },
      { nameKey: 'navSettings', path: '/settings', icon: Settings, labelVie: 'Cấu hình hệ thống', labelEng: 'Settings' },
    ],
  },
];

const NavItem = memo(({ item, collapsed }) => {
  const { locale } = useApp();
  const Icon = item.icon;
  const displayName = locale === 'vie' ? item.labelVie : item.labelEng;

  return (
    <NavLink
      to={item.path}
      className={({ isActive }) =>
        `group relative flex items-center gap-3 px-3 py-2 rounded-lg text-[13px] font-medium transition-all duration-150 ${isActive
          ? 'text-[var(--accent)] font-semibold bg-[var(--accent-muted)]'
          : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-muted)]'
        } ${collapsed ? 'justify-center px-2' : ''}`
      }
    >
      <Icon className="w-[18px] h-[18px] shrink-0" />
      {!collapsed && (
        <span className="truncate">{displayName}</span>
      )}
    </NavLink>
  );
});

const Sidebar = () => {
  const { locale, sidebarCollapsed, toggleSidebar } = useApp();
  const reducedMotion = usePrefersReducedMotion();
  const { gpu: gpuQuery, node: nodeQuery } = useSidebarSystemQueries();
  const gpu = gpuQuery.data;
  const node = nodeQuery.data;

  const gpuLabel = gpu?.available ? (gpu.gpu_name || 'GPU') : 'Unavailable';
  const gpuUtil = gpu?.gpu_utilization_percent;
  const nodeStatus = node?.status || 'offline';
  const statusColor = nodeStatus === 'healthy' ? 'bg-emerald-500' : nodeStatus === 'busy' ? 'bg-amber-500' : 'bg-red-500';
  const sidebarWidth = sidebarCollapsed ? 68 : 260;

  return (
    <aside
      className="bg-[var(--bg-elevated)] border-r border-[var(--border)] flex flex-col h-screen fixed left-0 top-0 z-30 overflow-hidden gpu-layer"
      style={{
        width: sidebarWidth,
        transition: reducedMotion ? 'none' : 'width 200ms cubic-bezier(0.16, 1, 0.3, 1)',
      }}
    >
      <div className={`flex items-center gap-3 h-15 border-b border-[var(--border)] shrink-0 ${sidebarCollapsed ? 'justify-center px-2' : 'px-4'}`}>
        <BrandLogo size="md" />
        <AnimatePresence>
          {!sidebarCollapsed && (
            <motion.div
              initial={reducedMotion ? false : { opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={reducedMotion ? undefined : { opacity: 0, x: -10 }}
              transition={{ duration: 0.15 }}
              className="overflow-hidden whitespace-nowrap"
            >
              <h1 className="text-sm font-bold text-[var(--text-primary)] leading-tight tracking-tight">AI Document Hub</h1>
              <p className="text-[10px] text-[var(--text-faint)] font-medium leading-tight">Research Workspace v2.1</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="flex-1 overflow-y-auto py-4 px-3 space-y-5 scrollbar-none">
        {navGroups.map((group) => (
          <div key={group.labelKey} className="space-y-1.5">
            {!sidebarCollapsed ? (
              <p className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-faint)] px-3">
                {locale === 'vie' ? group.labelVie : group.labelEng}
              </p>
            ) : (
              <div className="h-px bg-[var(--border)] mx-1" />
            )}
            <nav className="space-y-0.5">
              {group.items.map((item) => (
                <NavItem key={item.path} item={item} collapsed={sidebarCollapsed} />
              ))}
            </nav>
          </div>
        ))}
      </div>

      <div className="border-t border-[var(--border)] bg-[var(--bg-subtle)] p-3 shrink-0 space-y-3">
        {!sidebarCollapsed && (
          <div className="p-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg)] space-y-1.5">
            <div className="flex items-center justify-between text-[10px] font-semibold text-[var(--text-muted)]">
              <span className="flex items-center gap-1.5">
                <span className={`w-1.5 h-1.5 rounded-full ${statusColor} ${reducedMotion ? '' : 'animate-pulse'}`} />
                Live Node
              </span>
              <span className="truncate max-w-[100px]" title={gpuLabel}>{gpuLabel}</span>
            </div>
            {gpu?.available && gpuUtil != null ? (
              <div className="space-y-1">
                <div className="flex justify-between text-[9px] text-[var(--text-faint)]">
                  <span>GPU Usage</span>
                  <span>{gpuUtil}%</span>
                </div>
                <div className="h-1 rounded-full bg-[var(--bg-inset)] overflow-hidden">
                  <div
                    className="h-full bg-sky-500 rounded-full gpu-bar-fill"
                    style={{
                      width: `${Math.min(100, gpuUtil)}%`,
                      transition: reducedMotion ? 'none' : 'width 500ms ease-out',
                    }}
                  />
                </div>
              </div>
            ) : (
              <p className="text-[9px] text-[var(--text-faint)]">GPU: Unavailable</p>
            )}
          </div>
        )}

        {!sidebarCollapsed && (
          <div className="px-1 text-[10px] text-[var(--text-faint)] truncate">
            {node?.node_id || 'local-node'}
          </div>
        )}

        <button
          onClick={toggleSidebar}
          className="ui-btn-ghost w-full justify-center gap-2 py-1.5 text-xs border border-[var(--border)] bg-[var(--bg-elevated)] hover:bg-[var(--bg-muted)] text-[var(--text-secondary)] shadow-sm"
          title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {sidebarCollapsed ? (
            <PanelLeftOpen className="w-4 h-4 text-sky-500" />
          ) : (
            <>
              <PanelLeftClose className="w-4 h-4 text-sky-500" />
              <span className="text-xs font-semibold text-[var(--text-muted)]">Thu gọn</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
};

export default memo(Sidebar);
