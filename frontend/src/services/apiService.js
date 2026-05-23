const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export async function getHealth() {
  const response = await fetch(`${API}/health`);
  if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
  return response.json();
}

export async function getMetrics() {
  const response = await fetch(`${API}/metrics`);
  if (!response.ok) throw new Error(`Metrics failed: ${response.status}`);
  return response.json();
}

export async function getModels() {
  const response = await fetch(`${API}/models`);
  if (!response.ok) throw new Error(`Models failed: ${response.status}`);
  return response.json();
}

export async function summarizeCompare(
  text,
  reference,
  algorithms,
  extractiveSentences,
  maxAbstractiveLength,
  targetLengthRatio = 50,
) {
  const response = await fetch(`${API}/summarize/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      reference: reference || null,
      algorithms,
      extractive_sentences: extractiveSentences,
      max_abstractive_length: maxAbstractiveLength,
      target_length_ratio: targetLengthRatio,
      use_length_ratio: true,
      save_result: true,
    }),
  });
  if (!response.ok) throw new Error(`Compare failed: ${response.status}`);
  return response.json();
}

export async function getAnalyticsDashboard(timeRange = '30d', limit = 15) {
  const response = await fetch(
    `${API}/analytics/dashboard?time_range=${encodeURIComponent(timeRange)}&limit=${limit}`,
  );
  if (!response.ok) throw new Error(`Analytics failed: ${response.status}`);
  return response.json();
}

export async function getAnalyticsHistory(limit = 30) {
  const response = await fetch(`${API}/analytics/history?limit=${limit}`);
  if (!response.ok) throw new Error(`History failed: ${response.status}`);
  return response.json();
}
