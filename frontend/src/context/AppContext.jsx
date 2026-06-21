import React, {
  createContext, useCallback, useContext, useEffect, useMemo, useState,
} from 'react';
import { t as translate } from '../i18n/translations';

const STORAGE_THEME = 'aidh_theme';
const STORAGE_LOCALE = 'aidh_locale';
const STORAGE_NOTIFICATIONS = 'aidh_notifications';
const STORAGE_SIDEBAR = 'aidh_sidebar';
const MAX_NOTIFICATIONS = 50;

const AppContext = createContext(null);

function loadJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function getInitialTheme() {
  try {
    const stored = localStorage.getItem(STORAGE_THEME);
    if (stored === 'dark' || stored === 'light') return stored;
    // System preference
    if (window.matchMedia?.('(prefers-color-scheme: dark)').matches) return 'dark';
  } catch { /* ignore */ }
  return 'light';
}

function formatTimeAgo(ts, locale) {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return translate(locale, 'justNow');
  if (mins < 60) return translate(locale, 'minutesAgo', { n: mins });
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export function AppProvider({ children }) {
  const [theme, setTheme] = useState(getInitialTheme);
  const [locale, setLocale] = useState(() => localStorage.getItem(STORAGE_LOCALE) || 'vie');
  const [notifications, setNotifications] = useState(() => loadJson(STORAGE_NOTIFICATIONS, []));
  const [notifOpen, setNotifOpen] = useState(false);
  const [overviewCache, setOverviewCache] = useState(null);
  const [analyticsCache, setAnalyticsCache] = useState({});

  // Sidebar collapsed state
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_SIDEBAR) === 'collapsed';
    } catch {
      return false;
    }
  });

  // Command palette state
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);

  // Apply theme to DOM
  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle('dark', theme === 'dark');
    localStorage.setItem(STORAGE_THEME, theme);
  }, [theme]);

  // Listen for system theme changes
  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)');
    if (!mq) return;
    const handler = (e) => {
      const stored = localStorage.getItem(STORAGE_THEME);
      if (!stored || stored === 'system') {
        setTheme(e.matches ? 'dark' : 'light');
      }
    };
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_LOCALE, locale);
  }, [locale]);

  useEffect(() => {
    localStorage.setItem(STORAGE_NOTIFICATIONS, JSON.stringify(notifications.slice(0, MAX_NOTIFICATIONS)));
  }, [notifications]);

  useEffect(() => {
    localStorage.setItem(STORAGE_SIDEBAR, sidebarCollapsed ? 'collapsed' : 'expanded');
  }, [sidebarCollapsed]);

  // Ctrl+K command palette shortcut
  useEffect(() => {
    function onKeyDown(e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setCommandPaletteOpen(prev => !prev);
      }
      if (e.key === 'Escape') {
        setCommandPaletteOpen(false);
      }
    }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, []);

  const unreadCount = useMemo(
    () => notifications.filter(n => !n.read).length,
    [notifications],
  );

  const toggleTheme = useCallback((targetTheme) => {
    if (targetTheme === 'light' || targetTheme === 'dark') {
      setTheme(targetTheme);
    } else {
      setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
    }
  }, []);

  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed(prev => !prev);
  }, []);

  const setLanguage = useCallback((lang) => {
    if (lang === 'vie' || lang === 'eng') setLocale(lang);
  }, []);

  const t = useCallback((key, vars) => translate(locale, key, vars), [locale]);

  const pushBrowserNotification = useCallback((title, body) => {
    if (typeof Notification === 'undefined' || Notification.permission !== 'granted') return;
    try {
      new Notification(title, { body, icon: '/favicon.ico' });
    } catch { /* ignore */ }
  }, []);

  const addNotification = useCallback(({
    title,
    message,
    type = 'info',
    link = null,
    showBrowser = true,
  }) => {
    const item = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      title,
      message,
      type,
      link,
      read: false,
      createdAt: Date.now(),
    };
    setNotifications(prev => [item, ...prev].slice(0, MAX_NOTIFICATIONS));
    if (showBrowser) pushBrowserNotification(title, message);
    return item.id;
  }, [pushBrowserNotification]);

  const markAsRead = useCallback((id) => {
    setNotifications(prev => prev.map(n => (n.id === id ? { ...n, read: true } : n)));
  }, []);

  const markAllRead = useCallback(() => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
  }, []);

  const clearNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  const requestNotificationPermission = useCallback(async () => {
    if (typeof Notification === 'undefined') return 'unsupported';
    if (Notification.permission === 'granted') return 'granted';
    if (Notification.permission === 'denied') return 'denied';
    return Notification.requestPermission();
  }, []);

  const value = useMemo(() => ({
    theme,
    locale,
    isDark: theme === 'dark',
    notifications,
    notifOpen,
    setNotifOpen,
    unreadCount,
    toggleTheme,
    setLanguage,
    t,
    addNotification,
    markAsRead,
    markAllRead,
    clearNotifications,
    requestNotificationPermission,
    formatTimeAgo: (ts) => formatTimeAgo(ts, locale),
    overviewCache,
    setOverviewCache,
    analyticsCache,
    setAnalyticsCache,
    // New: sidebar
    sidebarCollapsed,
    setSidebarCollapsed,
    toggleSidebar,
    // New: command palette
    commandPaletteOpen,
    setCommandPaletteOpen,
  }), [
    theme, locale, notifications, notifOpen, unreadCount,
    toggleTheme, setLanguage, t, addNotification, markAsRead, markAllRead,
    clearNotifications, requestNotificationPermission,
    overviewCache, analyticsCache,
    sidebarCollapsed, toggleSidebar,
    commandPaletteOpen,
  ]);

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}
