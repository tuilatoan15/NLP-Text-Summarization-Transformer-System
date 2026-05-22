import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, Settings2, LineChart, PlaySquare, Library, MessageSquare,
} from 'lucide-react';
import { useApp } from '../context/AppContext';

const Sidebar = () => {
  const { t } = useApp();

  const navItems = [
    { nameKey: 'navOverview', path: '/', icon: LayoutDashboard },
    { nameKey: 'navPlayground', path: '/playground', icon: PlaySquare },
    { nameKey: 'navAnalytics', path: '/analytics', icon: LineChart },
    { nameKey: 'navSettings', path: '/settings', icon: Settings2 },
  ];

  return (
    <aside className="w-64 bg-white dark:bg-slate-900 border-r border-gray-100 dark:border-slate-800 flex flex-col h-screen fixed left-0 top-0 z-30">
      <div className="p-6">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded-lg">
            <LineChart className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-gray-900 dark:text-slate-100 leading-tight">{t('appName')}</h1>
            <p className="text-xs text-gray-500 dark:text-slate-400">{t('appTagline')}</p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-2">
        <div className="mb-6">
          <p className="text-xs font-semibold text-gray-400 dark:text-slate-500 uppercase tracking-wider mb-3 px-2">
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
                        ? 'bg-blue-50 dark:bg-blue-950/50 text-blue-600 dark:text-blue-400'
                        : 'text-gray-600 dark:text-slate-400 hover:bg-gray-50 dark:hover:bg-slate-800 hover:text-gray-900 dark:hover:text-slate-100'
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
          <p className="text-xs font-semibold text-gray-400 dark:text-slate-500 uppercase tracking-wider mb-3 px-2">
            {t('navSectionCore')}
          </p>
          <nav className="space-y-1">
            <div className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-gray-400 dark:text-slate-600 cursor-not-allowed">
              <MessageSquare className="w-4 h-4" />
              {t('navChat')}
            </div>
            <div className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-gray-400 dark:text-slate-600 cursor-not-allowed">
              <Library className="w-4 h-4" />
              {t('navRag')}
            </div>
          </nav>
        </div>
      </div>

      <div className="p-4 border-t border-gray-100 dark:border-slate-800 text-xs text-center text-gray-400 dark:text-slate-500">
        v1.0.0-beta
      </div>
    </aside>
  );
};

export default Sidebar;
