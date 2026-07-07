export interface RAGDocument {
  id: string;
  filename: string;
  source_type: string;
  status: string;
  created_at: string;
  metadata: Record<string, unknown>;
  file_size?: number;
  chunks_count?: number;
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
  model_used?: string | null;
  evaluation?: {
    faithfulness?: number;
    consistency_score: number;
    grounding_coverage: number;
    semantic_coverage: number;
    hallucination_risk: string;
  } | null;
  faithfulness?: number | null;
  retrieval_confidence?: number | null;
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

export interface AdaptiveContextDetails {
  mode?: string;
  query_intent?: string;
  query_focus?: string;
  compression_tier?: string;
  input_chars?: number;
  output_chars?: number;
  input_tokens_est?: number;
  output_tokens_est?: number;
  summary_tokens?: number;
  token_reduction?: number;
  compression_ratio?: number;
  chunks_kept?: number;
  chunks_total?: number;
  facts_preserved?: number;
  citations_count?: number;
  latency_saving_estimate_s?: number;
}

export interface RAGChatResponse {
  conversation_id: string;
  answer: string;
  confidence: number;
  faithfulness?: number;
  retrieval_confidence?: number;
  grounded: boolean;
  retrieved_context: RAGCitation[];
  retrieval_threshold: number;
  prompt_template: string;
  model_used?: string | null;
  evaluation?: {
    faithfulness?: number;
    consistency_score: number;
    grounding_coverage: number;
    semantic_coverage: number;
    hallucination_risk: string;
  } | null;
  context_compression?: {
    enabled: boolean;
    skipped_reason?: string | null;
    compression_ratio?: number;
    hybrid_algo?: string | null;
    summary_model?: string | null;
    top_original_count?: number;
    mode?: string;
    compression_tier?: string | null;
    query_intent?: string | null;
    query_focus?: string | null;
    facts_preserved?: number;
    citations_count?: number;
    summary_tokens?: number;
  };
  context_details?: AdaptiveContextDetails | null;
  latency_details?: {
    compression_enabled?: boolean;
    compression_ratio?: number;
    adaptive_mode?: boolean;
    token_reduction?: number;
    chunks_kept?: number;
    summary_tokens?: number;
    latency_saving_estimate_s?: number;
    stage_breakdown?: Record<string, unknown>;
  };
}

export type PipelineStageId =
  | 'question'
  | 'embedding'
  | 'retrieval'
  | 'crossencoder'
  | 'top_k'
  | 'context_compression'
  | 'acb_intent'
  | 'acb_summary'
  | 'acb_chunks'
  | 'acb_facts'
  | 'acb_compose'
  | 'prompt'
  | 'generation'
  | 'streaming';
