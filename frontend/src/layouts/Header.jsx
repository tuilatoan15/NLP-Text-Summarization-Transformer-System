import React, { useEffect, useRef } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Search, Moon, Sun, Bell, CheckCircle2, AlertCircle, Info, X, Command,
} from 'lucide-react';
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

const routeLabels = {
  '/': 'Dashboard',
  '/summarize': 'Tóm tắt',
  '/chat': 'Chat với Tài liệu',
  '/compare': 'So sánh Mô hình',
  '/search': 'Semantic Search',
  '/analytics': 'Phân tích & Báo cáo',
  '/settings': 'Cài đặt',
  '/documents': 'Tài liệu',
};

const Header = () => {
  const {
    isDark, locale, toggleTheme, setLanguage, t,
    notifications, notifOpen, setNotifOpen, unreadCount,
    markAsRead, markAllRead, clearNotifications,
    requestNotificationPermission, formatTimeAgo,
    setCommandPaletteOpen,
  } = useApp();

  const location = useLocation();
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

  const pageLabel = routeLabels[location.pathname] || location.pathname;

  return (
    <header
      className="h-14 flex items-center justify-between px-5 sticky top-0 z-20 w-full border-b transition-colors duration-150"
      style={{
        backgroundColor: 'var(--bg-elevated)',
        borderColor: 'var(--border)',
      }}
    >
      {/* Left: Breadcrumb + Search trigger */}
      <div className="flex items-center gap-3 min-w-0">
        {/* Breadcrumb */}
        <nav className="flex items-center gap-1.5 text-sm min-w-0">
          <span className="text-[var(--text-faint)] hidden sm:inline">AI Document Hub</span>
          <span className="text-[var(--text-faint)] hidden sm:inline">/</span>
          <span className="font-semibold text-[var(--text-primary)] truncate">{pageLabel}</span>
        </nav>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-1">
        {/* Command Palette Trigger */}
        <button
          type="button"
          onClick={() => setCommandPaletteOpen(true)}
          className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs transition-colors duration-150 cursor-pointer"
          style={{
            backgroundColor: 'var(--bg-muted)',
            color: 'var(--text-faint)',
            border: '1px solid var(--border)',
          }}
        >
          <Search className="w-3.5 h-3.5" />
          <span>{t('searchPlaceholder')}</span>
          <div className="flex items-center gap-0.5 ml-2">
            <kbd className="ui-kbd">⌘</kbd>
            <kbd className="ui-kbd">K</kbd>
          </div>
        </button>

        {/* Spacer */}
        <div className="w-px h-5 mx-1 hidden sm:block" style={{ backgroundColor: 'var(--border)' }} />

        {/* Language */}
        <div
          className="flex items-center gap-0.5 p-0.5 rounded-lg"
          style={{ backgroundColor: 'var(--bg-muted)' }}
          role="group"
          aria-label="Language"
        >
          <button
            type="button"
            onClick={() => setLanguage('vie')}
            className={`px-2 py-1 text-[11px] font-semibold rounded-md transition-all cursor-pointer ${
              locale === 'vie'
                ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)] shadow-sm'
                : 'text-[var(--text-faint)] hover:text-[var(--text-secondary)]'
            }`}
          >
            VI
          </button>
          <button
            type="button"
            onClick={() => setLanguage('eng')}
            className={`px-2 py-1 text-[11px] font-semibold rounded-md transition-all cursor-pointer ${
              locale === 'eng'
                ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)] shadow-sm'
                : 'text-[var(--text-faint)] hover:text-[var(--text-secondary)]'
            }`}
          >
            EN
          </button>
        </div>

        {/* Theme */}
        <button
          type="button"
          onClick={toggleTheme}
          title={isDark ? t('themeLight') : t('themeDark')}
          className="ui-btn-icon cursor-pointer"
        >
          {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>

        {/* Notifications */}
        <div className="relative">
          <button
            ref={bellRef}
            type="button"
            onClick={() => setNotifOpen(v => !v)}
            title={t('notifications')}
            className="ui-btn-icon relative cursor-pointer"
          >
            <Bell className="w-4 h-4" />
            {unreadCount > 0 && (
              <span
                className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full border-2"
                style={{
                  backgroundColor: 'var(--error)',
                  borderColor: 'var(--bg-elevated)',
                }}
              />
            )}
          </button>

          {notifOpen && (
            <div
              ref={panelRef}
              className="absolute right-0 top-full mt-2 w-80 sm:w-96 rounded-xl border overflow-hidden z-50 animate-fade-in"
              style={{
                backgroundColor: 'var(--bg-elevated)',
                borderColor: 'var(--border)',
                boxShadow: 'var(--shadow-lg)',
              }}
            >
              <div
                className="flex items-center justify-between px-4 py-3 border-b"
                style={{ borderColor: 'var(--border)' }}
              >
                <h3 className="text-sm font-semibold text-[var(--text-primary)]">{t('notifications')}</h3>
                <div className="flex gap-2">
                  {notifications.length > 0 && (
                    <>
                      <button
                        type="button"
                        onClick={markAllRead}
                        className="text-xs font-medium cursor-pointer hover:underline"
                        style={{ color: 'var(--accent)' }}
                      >
                        {t('markAllRead')}
                      </button>
                      <button
                        type="button"
                        onClick={clearNotifications}
                        className="text-xs text-[var(--text-faint)] cursor-pointer hover:underline"
                      >
                        {t('clearAll')}
                      </button>
                    </>
                  )}
                  <button
                    type="button"
                    onClick={() => setNotifOpen(false)}
                    className="p-0.5 text-[var(--text-faint)] hover:text-[var(--text-secondary)] cursor-pointer"
                  >
                    <X size={14} />
                  </button>
                </div>
              </div>

              <div className="max-h-80 overflow-y-auto">
                {notifications.length === 0 ? (
                  <p className="px-4 py-8 text-center text-sm text-[var(--text-faint)]">
                    {t('noNotifications')}
                  </p>
                ) : (
                  notifications.map((item) => {
                    const Icon = typeIcon[item.type] || Info;
                    const content = (
                      <div
                        className="flex gap-3 px-4 py-3 border-b transition-colors duration-100 cursor-pointer"
                        style={{
                          borderColor: 'var(--border-subtle)',
                          backgroundColor: !item.read ? 'var(--accent-muted)' : 'transparent',
                        }}
                        onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'var(--bg-muted)'; }}
                        onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = !item.read ? 'var(--accent-muted)' : 'transparent'; }}
                      >
                        <Icon className={`w-4 h-4 shrink-0 mt-0.5 ${typeColor[item.type] || typeColor.info}`} />
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-medium text-[var(--text-primary)]">{item.title}</p>
                          <p className="text-xs text-[var(--text-muted)] mt-0.5 line-clamp-2">{item.message}</p>
                          <p className="text-[10px] text-[var(--text-faint)] mt-1">{formatTimeAgo(item.createdAt)}</p>
                        </div>
                        {!item.read && (
                          <span
                            className="w-2 h-2 rounded-full shrink-0 mt-2"
                            style={{ backgroundColor: 'var(--accent)' }}
                          />
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
      </div>
    </header>
  );
};

export default Header;
