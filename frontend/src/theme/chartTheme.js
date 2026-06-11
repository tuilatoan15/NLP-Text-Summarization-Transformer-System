export function getChartTheme(isDark) {
  return {
    grid: isDark ? '#27272a' : '#e4e4e7',
    axis: isDark ? '#71717a' : '#71717a',
    accent: isDark ? '#818cf8' : '#6366f1',
    tooltipStyle: {
      backgroundColor: isDark ? '#18181b' : '#ffffff',
      border: `1px solid ${isDark ? '#27272a' : '#e4e4e7'}`,
      borderRadius: '8px',
      fontSize: '12px',
      color: isDark ? '#d4d4d8' : '#09090b',
      boxShadow: isDark
        ? '0 4px 16px rgba(0, 0, 0, 0.5)'
        : '0 4px 16px rgba(0, 0, 0, 0.08)',
    },
  };
}

export function dateLocale(locale) {
  return locale === 'eng' ? 'en-US' : 'vi-VN';
}
