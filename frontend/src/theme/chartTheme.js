export function getChartTheme(isDark) {
  return {
    grid: isDark ? '#334155' : '#f3f4f6',
    axis: isDark ? '#94a3b8' : '#6b7280',
    tooltipStyle: {
      backgroundColor: isDark ? '#1e293b' : '#ffffff',
      border: `1px solid ${isDark ? '#475569' : '#e5e7eb'}`,
      borderRadius: '8px',
      fontSize: '12px',
      color: isDark ? '#e2e8f0' : '#111827',
    },
  };
}

export function dateLocale(locale) {
  return locale === 'eng' ? 'en-US' : 'vi-VN';
}
