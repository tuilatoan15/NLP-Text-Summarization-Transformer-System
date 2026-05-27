import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, LineChart, PlaySquare, MessageSquare, Library,
} from 'lucide-react';
import { useApp } from '../context/AppContext';

const Sidebar = () => {
  const { t } = useApp();

  const navItems = [
    { nameKey: 'navOverview', path: '/', icon: LayoutDashboard },
    { nameKey: 'navPlayground', path: '/playground', icon: PlaySquare },
    { nameKey: 'navDocuments', path: '/documents', icon: Library },
    { nameKey: 'navAnalytics', path: '/analytics', icon: LineChart },
  ];

  return (
    <aside className="w-64 bg-[var(--surface-elevated)] border-r border-[var(--border)] flex flex-col h-screen fixed left-0 top-0 z-30 transition-colors duration-200">
      <div className="p-6">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-br from-blue-600 to-indigo-600 p-2 rounded-xl shadow-sm shadow-blue-500/20">
            <LineChart className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-[var(--text)] leading-tight">{t('appName')}</h1>
            <p className="text-xs text-[var(--text-muted)]">{t('appTagline')}</p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-2">
        <div className="mb-6">
          <p className="text-xs font-semibold text-[var(--text-faint)] uppercase tracking-wider mb-3 px-2">
            {t('navSectionOverview')}
          </p>
          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-[var(--accent-muted)] text-blue-600 dark:text-blue-400'
                        : 'text-[var(--text-muted)] hover:bg-[var(--surface-inset)] hover:text-[var(--text)]'
                    }`
                  }
                >
                  <Icon className="w-4 h-4" />
                  {t(item.nameKey)}
                </NavLink>
              );
            })}
          </nav>
        </div>

        <div>
          <p className="text-xs font-semibold text-[var(--text-faint)] uppercase tracking-wider mb-3 px-2">
            {t('navSectionCore')}
          </p>
          <nav className="space-y-1">
            <NavLink
              to="/chat"
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-[var(--accent-muted)] text-blue-600 dark:text-blue-400'
                    : 'text-[var(--text-muted)] hover:bg-[var(--surface-inset)] hover:text-[var(--text)]'
                }`
              }
            >
              <MessageSquare className="w-4 h-4" />
              {t('navChat')}
            </NavLink>
          </nav>
        </div>
      </div>

      <div className="p-4 border-t border-[var(--border)] text-xs text-center text-[var(--text-faint)]">
        v1.0.0-beta
      </div>
    </aside>
  );
};

export default Sidebar;
