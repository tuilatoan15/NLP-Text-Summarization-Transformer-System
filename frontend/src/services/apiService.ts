const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export type DocumentPayload = {
  document_id: string;
  metadata?: Record<string, unknown>;
  quality?: Record<string, unknown>;
  analysis_assets?: Record<string, unknown>;
  visualization?: Record<string, unknown>;
  chunks?: Array<Record<string, unknown>>;
};

async function parseJson<T>(response: Response, label: string): Promise<T> {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${label} failed: ${response.status} ${detail}`);
  }
  return response.json() as Promise<T>;
}

export async function getHealth(): Promise<any> {
  const response = await fetch(`${API}/health`);
  return parseJson(response, 'Health check');
}

export async function getMetrics(): Promise<any> {
  const response = await fetch(`${API}/metrics`);
  return parseJson(response, 'Metrics');
}

export async function getModels(): Promise<any> {
  const response = await fetch(`${API}/models`);
  return parseJson(response, 'Models');
}

export async function summarizeCompare(
  text: string,
  reference: string | null,
  algorithms: string[],
  extractiveSentences: number,
  maxAbstractiveLength: number,
  targetLengthRatio = 20,
): Promise<any> {
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
  return parseJson(response, 'Compare');
}

export async function getAnalyticsDashboard(timeRange = '30d', limit = 15): Promise<any> {
  const response = await fetch(
    `${API}/analytics/dashboard?time_range=${encodeURIComponent(timeRange)}&limit=${limit}`,
  );
  return parseJson(response, 'Analytics dashboard');
}

export async function getAnalyticsHistory(limit = 30): Promise<any> {
  const response = await fetch(`${API}/analytics/history?limit=${limit}`);
  return parseJson(response, 'Analytics history');
}

export async function ingestDocument(
  file: File,
  { includeEmbeddings = true, embeddingModel = 'hash' }: { includeEmbeddings?: boolean; embeddingModel?: string } = {},
): Promise<DocumentPayload> {
  const form = new FormData();
  form.append('file', file);
  form.append('include_embeddings', String(includeEmbeddings));
  if (embeddingModel) form.append('embedding_model', embeddingModel);
  const response = await fetch(`${API}/documents/ingest`, { method: 'POST', body: form });
  return parseJson(response, 'Document ingest');
}

export async function getDocument(documentId: string): Promise<DocumentPayload> {
  const response = await fetch(`${API}/documents/${documentId}`);
  return parseJson(response, 'Get document');
}

export async function searchDocument(documentId: string, query: string, topK = 5): Promise<any> {
  const response = await fetch(`${API}/documents/${documentId}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  });
  return parseJson(response, 'Document search');
}

export async function getDocumentAssets(documentId: string): Promise<any> {
  const response = await fetch(`${API}/documents/${documentId}/assets`);
  return parseJson(response, 'Document assets');
}

export async function getDocumentVisualization(documentId: string): Promise<any> {
  const response = await fetch(`${API}/documents/${documentId}/visualization`);
  return parseJson(response, 'Document visualization');
}

export async function compareDocumentSummaries(
  documentId: string,
  {
    reference = null,
    algorithms = ['textrank', 'lexrank', 'lsa', 'tfidf'],
    targetLengthRatio = 20,
    extractiveSentences = 4,
    maxAbstractiveLength = 160,
  }: {
    reference?: string | null;
    algorithms?: string[];
    targetLengthRatio?: number;
    extractiveSentences?: number;
    maxAbstractiveLength?: number;
  } = {},
): Promise<any> {
  const response = await fetch(`${API}/documents/${documentId}/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      reference,
      algorithms,
      target_length_ratio: targetLengthRatio,
      extractive_sentences: extractiveSentences,
      max_abstractive_length: maxAbstractiveLength,
    }),
  });
  return parseJson(response, 'Document compare');
}

export async function getExplainability(documentId: string, algorithm = 'textrank'): Promise<any> {
  const response = await fetch(`${API}/documents/${documentId}/explainability?algorithm=${encodeURIComponent(algorithm)}`);
  return parseJson(response, 'Explainability');
}

export async function hierarchicalSummarize(documentId: string, modelKey = 'vit5', useExtractiveMap = false): Promise<any> {
  const response = await fetch(`${API}/documents/${documentId}/summarize/hierarchical`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_key: modelKey, use_extractive_map: useExtractiveMap }),
  });
  return parseJson(response, 'Hierarchical summarize');
}

export async function exportPodcastTts(documentId: string): Promise<any> {
  const response = await fetch(`${API}/documents/${documentId}/podcast/tts`, { method: 'POST' });
  return parseJson(response, 'Podcast TTS');
}

export async function getResearchLeaderboard(): Promise<any> {
  const response = await fetch(`${API}/research/leaderboard`);
  return parseJson(response, 'Research leaderboard');
}

export async function getResearchBenchmarkSamples(
  page = 1,
  limit = 10,
  category = 'All',
  search = '',
): Promise<any> {
  const url = new URL(`${API}/research/benchmark/samples`);
  url.searchParams.append('page', String(page));
  url.searchParams.append('limit', String(limit));
  if (category && category !== 'All') {
    url.searchParams.append('category', category);
  }
  if (search) {
    url.searchParams.append('search', search);
  }
  const response = await fetch(url.toString());
  return parseJson(response, 'Research benchmark samples');
}

export async function getResearchHybridStudy(locale?: string): Promise<any> {
  const url = new URL(`${API}/research/hybrid-study`);
  if (locale) url.searchParams.append('locale', locale);
  const response = await fetch(url.toString());
  return parseJson(response, 'Research hybrid study');
}

export async function getResearchReport(locale?: string): Promise<any> {
  const url = new URL(`${API}/research/report`);
  if (locale) url.searchParams.append('locale', locale);
  const response = await fetch(url.toString());
  return parseJson(response, 'Research report');
}

export async function runResearchBenchmark(): Promise<any> {
  const response = await fetch(`${API}/research/benchmark/run`, { method: 'POST' });
  return parseJson(response, 'Run research benchmark');
}

export async function getLeaderboardByCategory(category: string): Promise<any> {
  const response = await fetch(`${API}/research/leaderboard/by-category?category=${encodeURIComponent(category)}`);
  return parseJson(response, 'Leaderboard by category');
}


