import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { Search, Moon, Sun, Bell, CheckCircle2, AlertCircle, Info, X } from 'lucide-react';
import { useApp } from '../context/AppContext';

const typeIcon = {
  success: CheckCircle2,
  error: AlertCircle,
  info: Info,
};

const typeColor = {
  success: 'text-emerald-500',
  error: 'text-red-500',
  info: 'text-blue-500',
};

const Header = () => {
  const {
    theme, isDark, locale, toggleTheme, setLanguage, t,
    notifications, notifOpen, setNotifOpen, unreadCount,
    markAsRead, markAllRead, clearNotifications,
    requestNotificationPermission, formatTimeAgo,
  } = useApp();

  const panelRef = useRef(null);
  const bellRef = useRef(null);

  useEffect(() => {
    requestNotificationPermission();
  }, [requestNotificationPermission]);

  useEffect(() => {
    function onDocClick(e) {
      if (!notifOpen) return;
      if (panelRef.current?.contains(e.target) || bellRef.current?.contains(e.target)) return;
      setNotifOpen(false);
    }
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, [notifOpen, setNotifOpen]);

  return (
    <header className="bg-white dark:bg-slate-900 border-b border-gray-100 dark:border-slate-800 h-16 flex items-center justify-between px-6 sticky top-0 z-20 w-full">
      <div className="flex-1 max-w-xl">
        <div className="relative group">
          <Search className="w-4 h-4 text-gray-400 dark:text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder={t('searchPlaceholder')}
            className="w-full bg-gray-50 dark:bg-slate-800 border border-transparent dark:border-slate-700 rounded-lg pl-10 pr-4 py-2 text-sm text-gray-900 dark:text-slate-100 placeholder:text-gray-400 dark:placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-100 dark:focus:ring-blue-900 transition-all"
          />
          <div className="absolute right-3 top-1/2 -translate-y-1/2 hidden sm:flex items-center gap-1">
            <kbd className="border border-gray-200 dark:border-slate-600 rounded px-1.5 py-0.5 text-[10px] text-gray-400 dark:text-slate-500 bg-white dark:bg-slate-800 shadow-sm">Ctrl</kbd>
            <kbd className="border border-gray-200 dark:border-slate-600 rounded px-1.5 py-0.5 text-[10px] text-gray-400 dark:text-slate-500 bg-white dark:bg-slate-800 shadow-sm">K</kbd>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3 ml-4">
        {/* VIE / ENG */}
        <div
          className="flex items-center gap-0.5 bg-gray-100 dark:bg-slate-800 rounded-lg p-1"
          role="group"
          aria-label="Language"
        >
          <button
            type="button"
            onClick={() => setLanguage('vie')}
            className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-all ${
              locale === 'vie'
                ? 'bg-white dark:bg-slate-700 text-gray-900 dark:text-white shadow-sm'
                : 'text-gray-500 dark:text-slate-400 hover:text-gray-800 dark:hover:text-slate-200'
            }`}
          >
            {t('langVie')}
          </button>
          <button
            type="button"
            onClick={() => setLanguage('eng')}
            className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-all ${
              locale === 'eng'
                ? 'bg-white dark:bg-slate-700 text-gray-900 dark:text-white shadow-sm'
                : 'text-gray-500 dark:text-slate-400 hover:text-gray-800 dark:hover:text-slate-200'
            }`}
          >
            {t('langEng')}
          </button>
        </div>

        {/* Theme */}
        <button
          type="button"
          onClick={toggleTheme}
          title={isDark ? t('themeLight') : t('themeDark')}
          className="p-2 text-gray-500 dark:text-slate-400 hover:text-gray-800 dark:hover:text-slate-200 hover:bg-gray-50 dark:hover:bg-slate-800 rounded-full transition-colors"
        >
          {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </button>

        {/* Notifications */}
        <div className="relative">
          <button
            ref={bellRef}
            type="button"
            onClick={() => setNotifOpen(v => !v)}
            title={t('notifications')}
            className="p-2 text-gray-500 dark:text-slate-400 hover:text-gray-800 dark:hover:text-slate-200 hover:bg-gray-50 dark:hover:bg-slate-800 rounded-full transition-colors relative"
          >
            <Bell className="w-5 h-5" />
            {unreadCount > 0 && (
              <span className="absolute top-1 right-1 min-w-[8px] h-2 px-0.5 bg-red-500 rounded-full border-2 border-white dark:border-slate-900" title={String(unreadCount)} />
            )}
          </button>

          {notifOpen && (
            <div
              ref={panelRef}
              className="absolute right-0 top-full mt-2 w-80 sm:w-96 rounded-xl border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl shadow-gray-200/50 dark:shadow-black/40 overflow-hidden z-50"
            >
              <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-slate-800">
                <h3 className="text-sm font-semibold text-gray-900 dark:text-slate-100">{t('notifications')}</h3>
                <div className="flex gap-2">
                  {notifications.length > 0 && (
                    <>
                      <button
                        type="button"
                        onClick={markAllRead}
                        className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
                      >
                        {t('markAllRead')}
                      </button>
                      <button
                        type="button"
                        onClick={clearNotifications}
                        className="text-xs text-gray-500 dark:text-slate-400 hover:underline"
                      >
                        {t('clearAll')}
                      </button>
                    </>
                  )}
                  <button
                    type="button"
                    onClick={() => setNotifOpen(false)}
                    className="p-0.5 text-gray-400 hover:text-gray-600 dark:hover:text-slate-300"
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>

              <div className="max-h-80 overflow-y-auto">
                {notifications.length === 0 ? (
                  <p className="px-4 py-8 text-center text-sm text-gray-500 dark:text-slate-400">
                    {t('noNotifications')}
                  </p>
                ) : (
                  notifications.map((item) => {
                    const Icon = typeIcon[item.type] || Info;
                    const content = (
                      <div
                        className={`flex gap-3 px-4 py-3 border-b border-gray-50 dark:border-slate-800/80 hover:bg-gray-50 dark:hover:bg-slate-800/50 transition ${
                          !item.read ? 'bg-blue-50/50 dark:bg-blue-950/20' : ''
                        }`}
                      >
                        <Icon className={`w-5 h-5 shrink-0 mt-0.5 ${typeColor[item.type] || typeColor.info}`} />
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-medium text-gray-900 dark:text-slate-100">{item.title}</p>
                          <p className="text-xs text-gray-600 dark:text-slate-400 mt-0.5 line-clamp-2">{item.message}</p>
                          <p className="text-[10px] text-gray-400 dark:text-slate-500 mt-1">{formatTimeAgo(item.createdAt)}</p>
                        </div>
                        {!item.read && (
                          <span className="w-2 h-2 rounded-full bg-blue-500 shrink-0 mt-2" />
                        )}
                      </div>
                    );

                    if (item.link) {
                      return (
                        <Link
                          key={item.id}
                          to={item.link}
                          onClick={() => { markAsRead(item.id); setNotifOpen(false); }}
                        >
                          {content}
                        </Link>
                      );
                    }
                    return (
                      <button
                        key={item.id}
                        type="button"
                        className="w-full text-left"
                        onClick={() => markAsRead(item.id)}
                      >
                        {content}
                      </button>
                    );
                  })
                )}
              </div>
            </div>
          )}
        </div>

        <div className="h-6 w-px bg-gray-200 dark:bg-slate-700 mx-1" />

        <div className="flex items-center gap-3 cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-800 p-1.5 rounded-lg transition-colors">
          <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/50 rounded-full flex items-center justify-center text-blue-600 dark:text-blue-400 font-semibold text-sm border border-blue-200 dark:border-blue-800">
            A
          </div>
          <div className="hidden sm:block text-sm">
            <p className="font-medium text-gray-900 dark:text-slate-100 leading-none mb-1">Admin</p>
            <p className="text-xs text-gray-500 dark:text-slate-400 leading-none">admin@agentic.ai</p>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
