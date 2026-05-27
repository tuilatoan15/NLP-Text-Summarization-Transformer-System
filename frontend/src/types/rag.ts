export interface RAGDocument {
  id: string;
  filename: string;
  source_type: string;
  status: string;
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface RAGCitation {
  chunk_id: string;
  document_id: string;
  filename: string;
  page: number | null;
  text: string;
  embedding_score: number;
  bm25_score: number;
  combined_score: number;
  rank: number;
}

export interface RAGMessage {
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  confidence?: number | null;
  citations?: RAGCitation[];
  created_at?: string;
}

export interface RAGChatRequest {
  query: string;
  conversation_id?: string | null;
  document_ids: string[];
  top_k: number;
  threshold: number;
  retrieval_mode: 'hybrid' | 'embedding' | 'bm25';
  use_reranking: boolean;
  embedding_model: string;
  temperature: number;
}

export interface RAGChatResponse {
  conversation_id: string;
  answer: string;
  confidence: number;
  grounded: boolean;
  retrieved_context: RAGCitation[];
  retrieval_threshold: number;
  prompt_template: string;
}

