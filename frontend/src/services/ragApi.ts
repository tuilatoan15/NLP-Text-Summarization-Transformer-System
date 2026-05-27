import type { RAGChatRequest, RAGChatResponse, RAGDocument, RAGMessage } from '../types/rag';

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

async function parseJson<T>(response: Response, label: string): Promise<T> {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${label} failed: ${response.status} ${detail}`);
  }
  return response.json() as Promise<T>;
}

export async function uploadRagDocument(
  file: File,
  options: { chunkSize: number; chunkOverlap: number; embeddingModel: string },
): Promise<any> {
  const form = new FormData();
  form.append('file', file);
  form.append('chunk_size', String(options.chunkSize));
  form.append('chunk_overlap', String(options.chunkOverlap));
  form.append('embedding_model', options.embeddingModel);
  const response = await fetch(`${API}/rag/documents/upload`, { method: 'POST', body: form });
  return parseJson(response, 'RAG upload');
}

export async function listRagDocuments(): Promise<RAGDocument[]> {
  const response = await fetch(`${API}/rag/documents`);
  const payload = await parseJson<{ items: RAGDocument[] }>(response, 'RAG documents');
  return payload.items;
}

export async function deleteRagDocument(documentId: string): Promise<void> {
  const response = await fetch(`${API}/rag/documents/${documentId}`, { method: 'DELETE' });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Delete document failed: ${response.status} ${detail}`);
  }
}

export async function listRagConversations(): Promise<Array<{ id: string; title: string }>> {
  const response = await fetch(`${API}/rag/conversations`);
  const payload = await parseJson<{ items: Array<{ id: string; title: string }> }>(
    response,
    'RAG conversations',
  );
  return payload.items;
}

export async function listConversationMessages(conversationId: string): Promise<RAGMessage[]> {
  const response = await fetch(`${API}/rag/conversations/${conversationId}/messages`);
  const payload = await parseJson<{ items: RAGMessage[] }>(response, 'RAG messages');
  return payload.items;
}

export async function listEmbeddingModels(): Promise<string[]> {
  const response = await fetch(`${API}/rag/embedding-models`);
  const payload = await parseJson<{ models: Record<string, unknown> }>(response, 'Embedding models');
  return Object.keys(payload.models || {});
}

export async function streamRagChat(
  request: RAGChatRequest,
  onToken: (text: string, conversationId: string) => void,
): Promise<RAGChatResponse> {
  const response = await fetch(`${API}/rag/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok || !response.body) {
    const detail = await response.text();
    throw new Error(`RAG stream failed: ${response.status} ${detail}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalResponse: RAGChatResponse | null = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() || '';

    for (const event of events) {
      if (!event.startsWith('data: ')) continue;
      const payloadText = event.slice(6).trim();
      const parsedEvent = JSON.parse(payloadText) as any;
      if (parsedEvent.event === 'done') {
        finalResponse = parsedEvent.response as RAGChatResponse;
      } else if (parsedEvent.event === 'token') {
        onToken(parsedEvent.content as string, parsedEvent.conversation_id as string);
      }
    }
  }
  if (!finalResponse) throw new Error('RAG stream ended without final response payload');
  return finalResponse;
}

