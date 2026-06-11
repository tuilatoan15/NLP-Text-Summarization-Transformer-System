import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard, FileText, PlaySquare, MessageSquare,
  LineChart, BarChart3, Settings, Activity, Sparkles,
  Search, GitCompareArrows, PanelLeftClose, PanelLeftOpen,
} from 'lucide-react';
import { useApp } from '../context/AppContext';

const navGroups = [
  {
    labelKey: 'navSectionWorkspace',
    items: [
      { nameKey: 'navOverview', path: '/', icon: LayoutDashboard },
      { nameKey: 'navSummarize', path: '/summarize', icon: Sparkles },
      { nameKey: 'navChat', path: '/chat', icon: MessageSquare },
    ],
  },
  {
    labelKey: 'navSectionAI',
    items: [
      { nameKey: 'navCompare', path: '/compare', icon: GitCompareArrows },
      { nameKey: 'navSearch', path: '/search', icon: Search },
    ],
  },
  {
    labelKey: 'navSectionAnalytics',
    items: [
      { nameKey: 'navAnalytics', path: '/analytics', icon: BarChart3 },
    ],
  },
  {
    labelKey: 'navSectionSystem',
    items: [
      { nameKey: 'navSettings', path: '/settings', icon: Settings },
    ],
  },
];

const NavItem = ({ item, collapsed }) => {
  const { t } = useApp();
  const Icon = item.icon;

  return (
    <NavLink
      to={item.path}
      className={({ isActive }) =>
        `group relative flex items-center gap-3 px-3 py-2 rounded-lg text-[13px] font-medium transition-all duration-150 ${
          isActive
            ? 'text-[var(--accent)] font-semibold'
            : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-muted)]'
        } ${collapsed ? 'justify-center px-2' : ''}`
      }
    >
      {({ isActive }) => (
        <>
          {isActive && (
            <motion.div
              layoutId="sidebar-active"
              className="absolute inset-0 rounded-lg -z-10"
              style={{ backgroundColor: 'var(--accent-muted)' }}
              transition={{ type: 'spring', stiffness: 400, damping: 30 }}
            />
          )}
          <Icon className="w-[18px] h-[18px] shrink-0" strokeWidth={isActive ? 2.2 : 1.8} />
          {!collapsed && (
            <span className="truncate">{t(item.nameKey)}</span>
          )}
        </>
      )}
    </NavLink>
  );
};

const Sidebar = () => {
  const { t, sidebarCollapsed, toggleSidebar } = useApp();
  const location = useLocation();

  return (
    <motion.aside
      animate={{ width: sidebarCollapsed ? 64 : 256 }}
      transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
      className="bg-[var(--bg-elevated)] border-r border-[var(--border)] flex flex-col h-screen fixed left-0 top-0 z-30 overflow-hidden"
    >
      {/* Logo */}
      <div className={`flex items-center gap-3 h-14 border-b border-[var(--border)] shrink-0 ${sidebarCollapsed ? 'justify-center px-2' : 'px-4'}`}>
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shrink-0 shadow-sm">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M2 4h12M2 8h9M2 12h6" stroke="white" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </div>
        <AnimatePresence>
          {!sidebarCollapsed && (
            <motion.div
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: 'auto' }}
              exit={{ opacity: 0, width: 0 }}
              transition={{ duration: 0.15 }}
              className="overflow-hidden whitespace-nowrap"
            >
              <h1 className="text-sm font-bold text-[var(--text-primary)] leading-tight">AI Document Hub</h1>
              <p className="text-[10px] text-[var(--text-faint)] leading-tight">v2.0 — Research Platform</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto py-3 px-2 space-y-4">
        {navGroups.map((group) => (
          <div key={group.labelKey}>
            {!sidebarCollapsed && (
              <p className="ui-overline px-3 mb-1.5 text-[10px]">
                {t(group.labelKey)}
              </p>
            )}
            {sidebarCollapsed && <div className="h-px bg-[var(--border)] mx-2 mb-2" />}
            <nav className="space-y-0.5">
              {group.items.map((item) => (
                <NavItem key={item.path} item={item} collapsed={sidebarCollapsed} />
              ))}
            </nav>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className={`border-t border-[var(--border)] p-2 shrink-0 ${sidebarCollapsed ? 'flex justify-center' : ''}`}>
        <button
          onClick={toggleSidebar}
          className="ui-btn-ghost w-full justify-center gap-2 py-1.5 text-xs"
          title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {sidebarCollapsed ? (
            <PanelLeftOpen className="w-4 h-4" />
          ) : (
            <>
              <PanelLeftClose className="w-4 h-4" />
              <span className="text-[var(--text-faint)]">Thu gọn</span>
            </>
          )}
        </button>
      </div>
    </motion.aside>
  );
};

export default Sidebar;
