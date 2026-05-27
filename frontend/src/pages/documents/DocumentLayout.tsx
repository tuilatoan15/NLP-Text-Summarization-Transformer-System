import React from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain, GitCompare, ChartColumn,
  Info, BookOpen, ChevronRight, Zap
} from 'lucide-react';
import { DocumentProvider, useDocumentContext } from '../../context/DocumentContext';

const TABS = [
  { to: 'compare',        label: 'Compare',        icon: GitCompare, reqDoc: true },
  { to: 'evaluation',     label: 'Evaluation',     icon: ChartColumn,reqDoc: true },
  { to: 'explainability', label: 'Explainability', icon: Info,       reqDoc: true },
  { to: 'notebook',       label: 'NotebookLM',     icon: BookOpen,   reqDoc: true },
];

function LayoutInner() {
  const { document } = useDocumentContext();
  const location = useLocation();

  const currentTab = TABS.find(t => location.pathname.includes(t.to));

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div>
        <h1 className="ui-page-title mb-1 flex items-center gap-2">
          Document Intelligence
          {document && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-mono tracking-tight font-medium flex items-center gap-1 mt-1">
              <Zap size={10} /> {document.document_id as string}
            </span>
          )}
        </h1>
        <div className="flex items-center gap-2 text-[var(--text-muted)] text-sm">
          <span>Hệ thống phân tích tài liệu chuyên sâu</span>
          {currentTab && (
            <>
              <ChevronRight size={14} className="text-[var(--text-faint)]" />
              <span className="font-medium text-[var(--text)]">{currentTab.label}</span>
            </>
          )}
        </div>
      </div>

      {/* Tabs */}
      <nav className="flex flex-wrap gap-2">
        {TABS.map(tab => {
          const disabled = tab.reqDoc && !document;
          const Icon = tab.icon;
          return (
            <NavLink
              key={tab.to}
              to={tab.to}
              onClick={e => { if (disabled) e.preventDefault(); }}
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                  isActive
                    ? 'bg-blue-600 text-white border-transparent shadow-sm shadow-blue-500/25'
                    : 'bg-[var(--surface-elevated)] text-[var(--text-muted)] border-[var(--border)] hover:border-blue-300'
                } ${disabled ? 'opacity-40 cursor-not-allowed hover:border-[var(--border)]' : ''}`
              }
            >
              <Icon className="w-3.5 h-3.5" />
              {tab.label}
            </NavLink>
          );
        })}
      </nav>

      {/* Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={location.pathname}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
        >
          <Outlet />
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

export default function DocumentLayout() {
  return (
    <DocumentProvider>
      <LayoutInner />
    </DocumentProvider>
  );
}
