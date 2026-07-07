/**
 * Human-readable duration from seconds (API metadata: analysis_duration_sec, etc.).
 */

export function getAnalysisDurationSec(meta) {
  if (!meta || typeof meta !== 'object') return null;
  const raw = meta.analysis_duration_sec ?? meta.analysis_time_s ?? meta.duration;
  if (raw == null || raw === '') return null;
  const sec = Number(raw);
  return Number.isFinite(sec) ? sec : null;
}

export function formatDuration(totalSeconds, locale = 'vie') {
  const sec = Math.round(Number(totalSeconds));
  if (!Number.isFinite(sec) || sec < 0) return null;

  if (sec < 60) {
    return locale === 'eng'
      ? `~${sec} second${sec === 1 ? '' : 's'}`
      : `khoảng ${sec} giây`;
  }

  const minutes = Math.floor(sec / 60);
  if (minutes < 60) {
    return locale === 'eng'
      ? `~${minutes} minute${minutes === 1 ? '' : 's'}`
      : `khoảng ${minutes} phút`;
  }

  const hours = Math.floor(minutes / 60);
  const remMin = minutes % 60;

  if (locale === 'eng') {
    if (remMin === 0) return `~${hours} hour${hours === 1 ? '' : 's'}`;
    return `~${hours}h ${remMin}m`;
  }

  if (remMin === 0) return `khoảng ${hours} giờ`;
  return `khoảng ${hours} giờ ${remMin} phút`;
}

function formatCompletedTimestamp(isoString, locale) {
  const d = new Date(isoString);
  if (Number.isNaN(d.getTime())) return null;

  const loc = locale === 'eng' ? 'en-GB' : 'vi-VN';
  const time = d.toLocaleString(loc, {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
  const date = d.toLocaleString(loc, {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });

  return { time, date };
}

/**
 * Build subtitle fragment for dataset analytics metadata line.
 * @param {object} meta - API metadata (generated_at, analysis_duration_sec, source, …)
 * @param {string} locale - 'vie' | 'eng'
 * @param {(key: string, vars?: object) => string} t - i18n translate function
 */
export function buildDatasetAnalyticsMetaLine(meta, locale, t) {
  if (!meta || typeof meta !== 'object') return '';

  const parts = [];

  if (meta.generated_at) {
    const ts = formatCompletedTimestamp(meta.generated_at, locale);
    if (ts) {
      parts.push(t('datasetAnalyticsCompletedAt', ts));
    }
  }

  const durSec = getAnalysisDurationSec(meta);
  if (durSec != null) {
    const duration = formatDuration(durSec, locale);
    if (duration) {
      const suffix = meta.source === 'colab' ? t('datasetAnalyticsColabSuffix') : '';
      parts.push(`${t('datasetAnalyticsProcessingTime', { duration })}${suffix}`);
    }
  }

  return parts.join(t('datasetAnalyticsMetaSeparator'));
}
