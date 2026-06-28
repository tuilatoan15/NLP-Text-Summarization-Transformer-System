import { cacheLog } from '../lib/cacheLogger';

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export type FileExtractResult = {
  text: string;
  documents?: unknown;
};

export async function extractFilesFromUpload(files: File[]): Promise<FileExtractResult> {
  const form = new FormData();
  files.forEach((file) => form.append('files', file));
  cacheLog('API', 'POST /summarize/files/extract');
  const response = await fetch(`${API}/summarize/files/extract`, { method: 'POST', body: form });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `HTTP ${response.status}`);
  }
  return response.json();
}

export type CompareStreamEvent = Record<string, unknown>;

export async function streamCompareSummaries(
  payload: {
    text: string;
    reference: string | null;
    algorithms: string[];
    saveResult?: boolean;
  },
  onEvent: (event: CompareStreamEvent) => void,
): Promise<void> {
  cacheLog('API', 'POST /summarize/compare/stream');
  const response = await fetch(`${API}/summarize/compare/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: payload.text,
      reference: payload.reference,
      algorithms: payload.algorithms,
      save_result: payload.saveResult ?? true,
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `HTTP ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('Streaming not supported by browser');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split('\n\n');
    buffer = chunks.pop() || '';
    for (const chunk of chunks) {
      for (const line of chunk.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        onEvent(JSON.parse(trimmed.slice(5).trim()));
      }
    }
  }
}

export async function fetchCompareHistoryList(limit = 20): Promise<any[]> {
  cacheLog('API', `GET /summarize/history?limit=${limit}`);
  const response = await fetch(`${API}/summarize/history?limit=${limit}`);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `HTTP ${response.status}`);
  }
  return response.json();
}

export async function fetchCompareHistoryDetail(resultId: string): Promise<any> {
  cacheLog('API', `GET /summarize/history/${resultId}`);
  const response = await fetch(`${API}/summarize/history/${resultId}`);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `HTTP ${response.status}`);
  }
  return response.json();
}

export async function deleteCompareHistoryRecord(resultId: string): Promise<any> {
  cacheLog('API', `DELETE /summarize/history/${resultId}`);
  const response = await fetch(`${API}/summarize/history/${resultId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `HTTP ${response.status}`);
  }
  return response.json();
}
