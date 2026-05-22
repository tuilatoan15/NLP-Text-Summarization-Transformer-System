import React, {
  createContext, useCallback, useContext, useEffect, useMemo, useState,
} from 'react';
import { t as translate } from '../i18n/translations';

const STORAGE_THEME = 'nlp_theme';
const STORAGE_LOCALE = 'nlp_locale';
const STORAGE_NOTIFICATIONS = 'nlp_notifications';
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

function formatTimeAgo(ts, locale) {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return translate(locale, 'justNow');
  return translate(locale, 'minutesAgo', { n: mins });
}

export function AppProvider({ children }) {
  const [theme, setTheme] = useState(() => localStorage.getItem(STORAGE_THEME) || 'light');
  const [locale, setLocale] = useState(() => localStorage.getItem(STORAGE_LOCALE) || 'vie');
  const [notifications, setNotifications] = useState(() => loadJson(STORAGE_NOTIFICATIONS, []));
  const [notifOpen, setNotifOpen] = useState(false);

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle('dark', theme === 'dark');
    localStorage.setItem(STORAGE_THEME, theme);
    document.body.style.backgroundColor = theme === 'dark' ? '#0f172a' : '#f9fafb';
  }, [theme]);

  useEffect(() => {
    localStorage.setItem(STORAGE_LOCALE, locale);
  }, [locale]);

  useEffect(() => {
    localStorage.setItem(STORAGE_NOTIFICATIONS, JSON.stringify(notifications.slice(0, MAX_NOTIFICATIONS)));
  }, [notifications]);

  const unreadCount = useMemo(
    () => notifications.filter(n => !n.read).length,
    [notifications],
  );

  const toggleTheme = useCallback(() => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  const setLanguage = useCallback((lang) => {
    if (lang === 'vie' || lang === 'eng') setLocale(lang);
  }, []);

  const t = useCallback((key, vars) => translate(locale, key, vars), [locale]);

  const pushBrowserNotification = useCallback((title, body) => {
    if (typeof Notification === 'undefined' || Notification.permission !== 'granted') return;
    try {
      new Notification(title, { body, icon: '/favicon.ico' });
    } catch {
      /* ignore */
    }
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
  }), [
    theme, locale, notifications, notifOpen, unreadCount,
    toggleTheme, setLanguage, t, addNotification, markAsRead, markAllRead,
    clearNotifications, requestNotificationPermission,
  ]);

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}
