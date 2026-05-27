import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  UploadCloud, Trash2, Plus, MessageSquare, Settings, Sliders,
  Loader2, Send, CheckCircle2, AlertTriangle, FileText,
  ChevronRight, Sparkles, ShieldCheck, Bookmark, Settings2
} from 'lucide-react';
import * as ragApi from '../services/ragApi';
import type { RAGDocument, RAGMessage, RAGCitation } from '../types/rag';

export default function Chat() {
  // --- Refs ---
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isStreamingRef = useRef(false);

  // --- State ---
  const [documents, setDocuments] = useState<RAGDocument[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [conversations, setConversations] = useState<Array<{ id: string; title: string }>>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<RAGMessage[]>([]);
  const [inputQuery, setInputQuery] = useState('');
  
  // Advanced Ingestion Settings
  const [chunkSize, setChunkSize] = useState(500);
  const [chunkOverlap, setChunkOverlap] = useState(80);
  const [ingestionModel, setIngestionModel] = useState('bge-m3');

  // Ingestion loading & files
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  
  // Embedding models list & active RAG configuration
  const [embeddingModels, setEmbeddingModels] = useState<string[]>([]);
  const [ragModel, setRagModel] = useState('bge-m3');
  const [topK, setTopK] = useState(5);
  const [threshold, setThreshold] = useState(0.25);
  const [retrievalMode, setRetrievalMode] = useState<'hybrid' | 'embedding' | 'bm25'>('hybrid');
  const [useReranking, setUseReranking] = useState(true);
  const [temperature, setTemperature] = useState(0.2);

  // Active generation/interaction states
  const [chatLoading, setChatLoading] = useState(false);
  const [activeCitations, setActiveCitations] = useState<RAGCitation[]>([]);
  const [highlightedCitationId, setHighlightedCitationId] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  // Right Panel tabs
  const [rightPanelTab, setRightPanelTab] = useState<'settings' | 'citations'>('settings');

  // --- Initial Data Load ---
  useEffect(() => {
    loadDocuments();
    loadConversations();
    loadEmbeddingModels();
  }, []);

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, chatLoading]);

  // Load message history when active conversation changes (excl. ongoing streams)
  useEffect(() => {
    if (activeConversationId) {
      if (!isStreamingRef.current) {
        loadMessages(activeConversationId);
      }
    } else {
      setMessages([]);
      setActiveCitations([]);
    }
  }, [activeConversationId]);

  // --- API Integrations ---
  const loadDocuments = async (selectNewId?: string) => {
    try {
      const items = await ragApi.listRagDocuments();
      setDocuments(items);
      if (selectNewId) {
        setSelectedDocIds(prev => [...new Set([...prev, selectNewId])]);
      } else if (selectedDocIds.length === 0 && items.length > 0) {
        setSelectedDocIds(items.map(d => d.id));
      }
    } catch (err: any) {
      showError('Không thể tải danh sách tài liệu: ' + err.message);
    }
  };

  const loadConversations = async () => {
    try {
      const list = await ragApi.listRagConversations();
      setConversations(list);
    } catch (err: any) {
      showError('Không thể tải lịch sử trò chuyện: ' + err.message);
    }
  };

  const loadEmbeddingModels = async () => {
    try {
      const models = await ragApi.listEmbeddingModels();
      setEmbeddingModels(models);
      if (models.length > 0) {
        setIngestionModel(models[0]);
        setRagModel(models[0]);
      }
    } catch (err: any) {
      showError('Không thể tải danh sách embedding model: ' + err.message);
    }
  };

  const loadMessages = async (convId: string) => {
    try {
      const msgs = await ragApi.listConversationMessages(convId);
      setMessages(msgs);
      
      const assistantMsgs = msgs.filter(m => m.role === 'assistant');
      if (assistantMsgs.length > 0) {
        const lastMsg = assistantMsgs[assistantMsgs.length - 1];
        if (lastMsg.citations && lastMsg.citations.length > 0) {
          setActiveCitations(lastMsg.citations);
          setRightPanelTab('citations');
        }
      }
    } catch (err: any) {
      showError('Không thể tải tin nhắn: ' + err.message);
    }
  };

  const showSuccess = (msg: string) => {
    setStatusMessage(msg);
    setTimeout(() => setStatusMessage(null), 4000);
  };

  const showError = (msg: string) => {
    setErrorMessage(msg);
    setTimeout(() => setErrorMessage(null), 5000);
  };

  // --- Document Operations ---
  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) return;

    setUploading(true);
    setErrorMessage(null);
    try {
      const res = await ragApi.uploadRagDocument(uploadFile, {
        chunkSize,
        chunkOverlap,
        embeddingModel: ingestionModel
      });
      showSuccess(`Đã tải lên và nạp thành công: ${uploadFile.name}`);
      setUploadFile(null);
      loadDocuments(res.document_id);
    } catch (err: any) {
      showError('Upload tài liệu thất bại: ' + err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDoc = async (docId: string, filename: string) => {
    if (!window.confirm(`Bạn có chắc chắn muốn xóa tài liệu "${filename}"? Dữ liệu chunks và vector index sẽ bị hủy hoàn toàn.`)) return;

    try {
      await ragApi.deleteRagDocument(docId);
      showSuccess(`Đã xóa tài liệu: ${filename}`);
      setSelectedDocIds(prev => prev.filter(id => id !== docId));
      loadDocuments();
    } catch (err: any) {
      showError('Xóa tài liệu thất bại: ' + err.message);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setUploadFile(e.dataTransfer.files[0]);
    }
  };

  const toggleSelectDoc = (docId: string) => {
    setSelectedDocIds(prev =>
      prev.includes(docId) ? prev.filter(id => id !== docId) : [...prev, docId]
    );
  };

  const toggleSelectAllDocs = () => {
    if (selectedDocIds.length === documents.length) {
      setSelectedDocIds([]);
    } else {
      setSelectedDocIds(documents.map(d => d.id));
    }
  };

  // --- Chat Operations ---
  const handleStartNewChat = () => {
    if (chatLoading) return;
    setActiveConversationId(null);
    setMessages([]);
    setActiveCitations([]);
    setInputQuery('');
    setRightPanelTab('settings');
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || chatLoading) return;

    const currentQuery = inputQuery;
    setInputQuery('');
    setChatLoading(true);
    setErrorMessage(null);
    isStreamingRef.current = true;

    // Create immediate user message
    const userMsg: RAGMessage = {
      role: 'user',
      content: currentQuery,
      created_at: new Date().toISOString()
    };
    
    // Add to messages local state
    setMessages(prev => [...prev, userMsg]);

    // Temp assistant bubble placeholder for streaming
    const assistantPlaceholder: RAGMessage = {
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString()
    };
    setMessages(prev => [...prev, assistantPlaceholder]);

    try {
      const requestPayload = {
        query: currentQuery,
        conversation_id: activeConversationId,
        document_ids: selectedDocIds,
        top_k: topK,
        threshold,
        retrieval_mode: retrievalMode,
        use_reranking: useReranking,
        embedding_model: ragModel,
        temperature
      };

      // Call streaming API
      const finalResult = await ragApi.streamRagChat(requestPayload, (streamingText, conversationId) => {
        // Update streaming text in real-time
        setMessages(prev => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          if (updated[lastIdx] && updated[lastIdx].role === 'assistant') {
            updated[lastIdx] = {
              ...updated[lastIdx],
              content: streamingText,
              id: `stream_${conversationId}`
            };
          }
          return updated;
        });

        // Set conversation ID if it was a new chat without triggering reloads
        if (!activeConversationId && conversationId) {
          setActiveConversationId(conversationId);
        }
      });

      // Stream successfully resolved, update with final RAG citations & confidence parameters
      setMessages(prev => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        if (updated[lastIdx] && updated[lastIdx].role === 'assistant') {
          updated[lastIdx] = {
            role: 'assistant',
            content: finalResult.answer,
            confidence: finalResult.confidence,
            citations: finalResult.retrieved_context,
            created_at: new Date().toISOString()
          };
        }
        return updated;
      });

      if (finalResult.retrieved_context && finalResult.retrieved_context.length > 0) {
        setActiveCitations(finalResult.retrieved_context);
        setRightPanelTab('citations');
      } else {
        setActiveCitations([]);
      }

      // Reload conversations dropdown
      loadConversations();

    } catch (err: any) {
      showError('Lỗi gửi tin nhắn: ' + err.message);
      // Remove assistant placeholder on complete failure
      setMessages(prev => prev.filter((_, i) => i !== prev.length - 1));
    } finally {
      setChatLoading(false);
      isStreamingRef.current = false;
    }
  };

  const handleCitationClick = (citation: RAGCitation) => {
    setRightPanelTab('citations');
    setHighlightedCitationId(citation.chunk_id);
    setTimeout(() => {
      const targetElement = document.getElementById(`citation-chunk-${citation.chunk_id}`);
      targetElement?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
  };

  // Quick prompt injection
  const handleQuickPrompt = (prompt: string) => {
    setInputQuery(prompt);
  };

  // Helper to render citation superscript links in assistant response with regex mapping
  const renderMessageContent = (msg: RAGMessage) => {
    if (msg.role === 'user') {
      return <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>;
    }

    const text = msg.content;
    if (!msg.citations || msg.citations.length === 0) {
      return <p className="whitespace-pre-wrap leading-relaxed">{text}</p>;
    }

    // Split text by reference blocks [1], [2] to inject clickable superscript motion buttons
    const parts = text.split(/(\[\d+\])/g);
    
    return (
      <div className="space-y-3">
        <p className="whitespace-pre-wrap leading-relaxed">
          {parts.map((part, i) => {
            const match = part.match(/^\[(\d+)\]$/);
            if (match) {
              const num = parseInt(match[1], 10);
              const cite = msg.citations?.[num - 1];
              if (cite) {
                return (
                  <motion.button
                    key={i}
                    whileHover={{ scale: 1.18, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => handleCitationClick(cite)}
                    className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300 text-[10px] font-extrabold mx-0.5 border border-emerald-300 dark:border-emerald-800 align-super cursor-pointer shadow-sm shadow-emerald-500/10"
                    title={`Nguồn: ${cite.filename}`}
                  >
                    {num}
                  </motion.button>
                );
              }
            }
            return part;
          })}
        </p>
        
        <div className="pt-2.5 border-t border-[var(--border)]/60 mt-3 opacity-90">
          <p className="text-[10px] font-bold text-[var(--text-muted)] uppercase tracking-wider mb-1.5 flex items-center gap-1">
            <ShieldCheck size={11} className="text-emerald-500" /> Nguồn tham chiếu tài liệu:
          </p>
          <div className="flex flex-wrap gap-1.5">
            {msg.citations.map((cite, i) => (
              <motion.button
                key={cite.chunk_id}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => handleCitationClick(cite)}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-xl text-xs font-semibold bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 border border-emerald-200/50 dark:border-emerald-800/40 hover:border-emerald-400 hover:bg-emerald-100 dark:hover:bg-emerald-900/30 transition-all cursor-pointer"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                <span className="font-mono">[{i + 1}]</span>
                <span className="truncate max-w-[120px]">{cite.filename}</span>
                {cite.page !== null && <span className="opacity-60 text-[10px]">(Tr. {cite.page})</span>}
              </motion.button>
            ))}
          </div>
        </div>
      </div>
    );
  };

  // --- Framer Motion variants for lists ---
  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.04 }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 8 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <div className="h-[calc(100vh-100px)] flex flex-col lg:flex-row gap-4 relative overflow-hidden -mx-4 -my-4 lg:mx-0 lg:my-0">
      
      {/* ─────────────────────────────────────────────────────────────
          LEFT PANEL: Documents Ingestion & Scope Select
          ───────────────────────────────────────────────────────────── */}
      <section className="w-full lg:w-80 bg-[var(--surface-elevated)] border border-[var(--border)] rounded-2xl flex flex-col overflow-hidden shrink-0 shadow-sm shadow-black/5">
        {/* Header */}
        <div className="p-4 border-b border-[var(--border)] bg-[var(--surface-muted)] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText size={16} className="text-blue-500" />
            <h2 className="text-sm font-bold text-[var(--text)]">Tài liệu RAG</h2>
          </div>
          <span className="text-xs px-2 py-0.5 rounded-full bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-bold">
            {documents.length} File
          </span>
        </div>

        {/* Ingestion & Selection Tabs */}
        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          {/* Upload Form */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold text-[var(--text-faint)] uppercase tracking-wider">
              Nạp tài liệu mới
            </h3>
            <form onSubmit={handleUpload} className="space-y-3">
              {/* Drag/Drop Box */}
              <motion.div
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                animate={{
                  boxShadow: dragActive ? "0 0 15px rgba(37, 99, 235, 0.4)" : "0 0 0px rgba(0,0,0,0)",
                  scale: dragActive ? 1.02 : 1
                }}
                className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all ${
                  dragActive
                    ? 'border-blue-500 bg-blue-500/10'
                    : uploadFile
                    ? 'border-emerald-400 bg-emerald-400/5'
                    : 'border-[var(--border)] hover:border-blue-400 hover:bg-[var(--surface-muted)]'
                }`}
                onClick={() => document.getElementById('file-upload-input')?.click()}
              >
                <input
                  id="file-upload-input"
                  type="file"
                  accept=".pdf,.docx,.txt,.md"
                  className="hidden"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                />
                
                {uploadFile ? (
                  <div className="space-y-1">
                    <FileText className="w-8 h-8 text-emerald-500 mx-auto" />
                    <p className="text-xs font-semibold text-[var(--text)] truncate px-2">{uploadFile.name}</p>
                    <p className="text-[10px] text-[var(--text-muted)]">{(uploadFile.size / 1024).toFixed(1)} KB</p>
                  </div>
                ) : (
                  <div className="space-y-1">
                    <UploadCloud className="w-8 h-8 text-blue-500 mx-auto" />
                    <p className="text-xs font-semibold text-[var(--text)]">Kéo thả hoặc nhấp để chọn file</p>
                    <p className="text-[10px] text-[var(--text-faint)]">PDF, DOCX, TXT, MD</p>
                  </div>
                )}
              </motion.div>

              {/* Ingestion Parameters Accordion */}
              <details className="group border border-[var(--border)] rounded-xl bg-[var(--surface-muted)] overflow-hidden transition-all">
                <summary className="flex items-center justify-between p-2.5 text-xs font-bold text-[var(--text-secondary)] cursor-pointer hover:bg-[var(--surface-inset)]">
                  <span className="flex items-center gap-1.5">
                    <Settings size={12} /> Cấu hình Chunk & Model
                  </span>
                  <ChevronRight size={12} className="group-open:rotate-90 transition-transform" />
                </summary>
                
                <div className="p-3 border-t border-[var(--border)] space-y-3 bg-[var(--surface-elevated)]">
                  <div>
                    <label className="block text-[10px] font-bold text-[var(--text-muted)] uppercase mb-1">
                      Kích thước chunk (tokens)
                    </label>
                    <div className="flex gap-2 items-center">
                      <input
                        type="range" min="100" max="1500" step="50"
                        value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))}
                        className="ui-range flex-1"
                      />
                      <span className="text-xs font-mono font-bold w-12 text-right">{chunkSize}</span>
                    </div>
                  </div>

                  <div>
                    <label className="block text-[10px] font-bold text-[var(--text-muted)] uppercase mb-1">
                      Độ trùng lặp chunk (overlap)
                    </label>
                    <div className="flex gap-2 items-center">
                      <input
                        type="range" min="0" max="400" step="10"
                        value={chunkOverlap} onChange={(e) => setChunkOverlap(Number(e.target.value))}
                        className="ui-range flex-1"
                      />
                      <span className="text-xs font-mono font-bold w-12 text-right">{chunkOverlap}</span>
                    </div>
                  </div>

                  <div>
                    <label className="block text-[10px] font-bold text-[var(--text-muted)] uppercase mb-1">
                      Embedding Model nạp
                    </label>
                    <select
                      value={ingestionModel}
                      onChange={(e) => setIngestionModel(e.target.value)}
                      className="ui-input py-1 text-xs"
                    >
                      {embeddingModels.map((m) => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </details>

              <button
                type="submit"
                disabled={!uploadFile || uploading}
                className="ui-btn-primary w-full text-xs py-2 gap-1.5"
              >
                {uploading ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    Đang nhúng & lưu...
                  </>
                ) : (
                  <>
                    <Sparkles size={14} />
                    Nạp tài liệu & Index
                  </>
                )}
              </button>
            </form>
          </div>

          <div className="h-px bg-[var(--border)]" />

          {/* Documents Selection Checklist */}
          <div className="space-y-2 flex-1">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold text-[var(--text-faint)] uppercase tracking-wider">
                Chọn tài liệu trò chuyện
              </h3>
              <button
                onClick={toggleSelectAllDocs}
                className="text-[10px] text-blue-500 font-bold hover:underline cursor-pointer"
              >
                {selectedDocIds.length === documents.length ? 'Bỏ chọn hết' : 'Chọn tất cả'}
              </button>
            </div>

            {documents.length === 0 ? (
              <div className="py-6 text-center text-xs text-[var(--text-faint)]">
                Chưa có tài liệu nào được nạp. Hãy tải file lên trước.
              </div>
            ) : (
              <motion.div
                variants={containerVariants}
                initial="hidden"
                animate="show"
                className="space-y-1.5 max-h-[260px] overflow-y-auto pr-1"
              >
                {documents.map((doc) => {
                  const isChecked = selectedDocIds.includes(doc.id);
                  return (
                    <motion.div
                      variants={itemVariants}
                      key={doc.id}
                      className={`flex items-center justify-between p-2.5 rounded-xl border transition-colors ${
                        isChecked
                          ? 'border-blue-500/30 bg-blue-500/5'
                          : 'border-[var(--border)] hover:bg-[var(--surface-muted)]'
                      }`}
                    >
                      <label className="flex items-center gap-2.5 flex-1 cursor-pointer min-w-0">
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => toggleSelectDoc(doc.id)}
                          className="rounded border-[var(--border)] text-blue-500 focus:ring-blue-500 cursor-pointer"
                        />
                        <div className="min-w-0">
                          <p className="text-xs font-semibold text-[var(--text)] truncate">{doc.filename}</p>
                          <p className="text-[9px] text-[var(--text-faint)] flex items-center gap-1 mt-0.5">
                            <span className="uppercase font-mono">{doc.source_type}</span> ·
                            <span>{new Date(doc.created_at).toLocaleDateString()}</span>
                          </p>
                        </div>
                      </label>
                      <button
                        onClick={() => handleDeleteDoc(doc.id, doc.filename)}
                        className="text-[var(--text-faint)] hover:text-red-500 p-1 rounded-lg hover:bg-red-50 dark:hover:bg-red-950/20 transition-colors ml-1 cursor-pointer"
                        title="Xóa tài liệu"
                      >
                        <Trash2 size={13} />
                      </button>
                    </motion.div>
                  );
                })}
              </motion.div>
            )}
          </div>
        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────
          MIDDLE PANEL: Streaming Chat & Session Manager
          ───────────────────────────────────────────────────────────── */}
      <section className="flex-1 bg-[var(--surface-elevated)] border border-[var(--border)] rounded-2xl flex flex-col overflow-hidden shadow-sm shadow-black/5 relative">
        {/* Chat Header */}
        <div className="p-4 border-b border-[var(--border)] bg-[var(--surface-muted)] flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Conversation Selector / History Toggle */}
            <select
              value={activeConversationId || ''}
              onChange={(e) => {
                if (chatLoading) return;
                const val = e.target.value;
                setActiveConversationId(val ? val : null);
              }}
              className="ui-input py-1 text-xs max-w-[200px] h-8 font-semibold shadow-none border-[var(--border)] cursor-pointer"
            >
              <option value="">+ Cuộc trò chuyện mới</option>
              {conversations.map((c) => (
                <option key={c.id} value={c.id}>{c.title}</option>
              ))}
            </select>

            <span className="text-xs text-[var(--text-muted)] hidden md:inline-flex items-center gap-1.5 bg-[var(--surface-inset)] px-2.5 py-1 rounded-lg">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              Đang nhắm {selectedDocIds.length} tài liệu
            </span>
          </div>

          <button
            onClick={handleStartNewChat}
            disabled={chatLoading}
            className="ui-btn-secondary py-1.5 px-3 text-xs gap-1 h-8 bg-blue-50 dark:bg-blue-900/10 text-blue-600 dark:text-blue-400 border border-blue-200/50 dark:border-blue-800/40 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Plus size={13} /> Chat mới
          </button>
        </div>

        {/* Global Notifications inside Panel */}
        <AnimatePresence>
          {statusMessage && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="bg-emerald-50 dark:bg-emerald-950/20 text-emerald-700 dark:text-emerald-300 text-xs px-4 py-2 border-b border-emerald-200 dark:border-emerald-800 flex items-center gap-2 font-medium"
            >
              <CheckCircle2 size={13} className="text-emerald-500" />
              <span>{statusMessage}</span>
            </motion.div>
          )}

          {errorMessage && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="bg-red-50 dark:bg-red-950/20 text-red-700 dark:text-red-300 text-xs px-4 py-2 border-b border-red-200 dark:border-red-900 flex items-center gap-2 font-medium"
            >
              <AlertTriangle size={13} className="text-red-500" />
              <span>{errorMessage}</span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Message Thread */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-4">
              <motion.div
                initial={{ scale: 0.5, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: "spring", stiffness: 260, damping: 20 }}
                className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-500 text-white flex items-center justify-center shadow-lg shadow-blue-500/20 scale-110"
              >
                <MessageSquare className="w-8 h-8" />
              </motion.div>
              <div className="space-y-1 max-w-sm">
                <h3 className="font-bold text-base text-[var(--text)]">Hỏi đáp tài liệu RAG</h3>
                <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                  Nhập câu hỏi của bạn. AI sẽ tìm kiếm thông tin liên quan trong các tài liệu đã chọn và phản hồi có nguồn dẫn.
                </p>
              </div>

              {/* Quick Prompts suggestions */}
              {documents.length > 0 && (
                <div className="pt-4 space-y-2 w-full max-w-md">
                  <p className="text-[10px] font-bold text-[var(--text-faint)] uppercase tracking-wider">Gợi ý truy vấn nhanh</p>
                  <div className="grid grid-cols-1 gap-2">
                    {[
                      "Tóm tắt ngắn gọn các nội dung cốt lõi của tài liệu.",
                      "Tài liệu này đề cập đến các vấn đề chính nào?",
                      "Tìm các thông tin quan trọng nhất trong văn bản."
                    ].map((promptText) => (
                      <motion.button
                        key={promptText}
                        whileHover={{ scale: 1.01, x: 2 }}
                        whileTap={{ scale: 0.99 }}
                        onClick={() => handleQuickPrompt(promptText)}
                        className="text-left text-xs p-3 rounded-xl border border-[var(--border)] hover:border-blue-400 bg-[var(--surface-muted)] hover:bg-[var(--surface-inset)] transition-all text-[var(--text-secondary)] font-medium cursor-pointer"
                      >
                        {promptText}
                      </motion.button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((msg, index) => {
                const isUser = msg.role === 'user';
                return (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ type: "spring", stiffness: 350, damping: 28 }}
                    className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}
                  >
                    {/* Bot Avatar */}
                    {!isUser && (
                      <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 text-white text-xs font-bold flex items-center justify-center shrink-0 shadow-md shadow-blue-500/10">
                        <Sparkles size={14} />
                      </div>
                    )}

                    <div className="space-y-1 max-w-[85%]">
                      {/* Sub-label showing confidence */}
                      {!isUser && (msg.confidence !== undefined && msg.confidence !== null) && (
                        <div className="flex items-center gap-1.5 pl-1">
                          <span className={`text-[9px] font-extrabold uppercase px-1.5 py-0.5 rounded tracking-wider ${
                            msg.confidence > 0.7 
                              ? 'bg-emerald-50 dark:bg-emerald-950/20 text-emerald-700 dark:text-emerald-400 border border-emerald-200/40' 
                              : 'bg-amber-50 dark:bg-amber-950/20 text-amber-700 dark:text-amber-400 border border-amber-200/40'
                          }`}>
                            Tin cậy: {Math.round(msg.confidence * 100)}%
                          </span>
                          <span className="text-[10px] text-[var(--text-faint)]">RAG Grounded</span>
                        </div>
                      )}

                      <div className={`p-4 rounded-2xl text-sm leading-relaxed border shadow-sm ${
                        isUser
                          ? 'bg-blue-600 border-blue-500 text-white rounded-tr-sm shadow-blue-600/10'
                          : 'bg-[var(--surface-elevated)] border-[var(--border)] text-[var(--text-secondary)] rounded-tl-sm'
                      }`}>
                        {renderMessageContent(msg)}
                      </div>
                    </div>
                  </motion.div>
                );
              })}

              {/* Shimmer Skeleton Loader when generating */}
              {chatLoading && messages[messages.length - 1]?.role !== 'assistant' && (
                <motion.div
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-3 justify-start"
                >
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 text-white text-xs font-bold flex items-center justify-center shrink-0 shadow-md shadow-blue-500/10">
                    <Loader2 size={13} className="animate-spin" />
                  </div>
                  <div className="p-4 rounded-2xl text-sm bg-[var(--surface-muted)] border border-[var(--border)] text-[var(--text-secondary)] rounded-tl-sm space-y-2 w-full max-w-lg shadow-sm">
                    <div className="flex items-center gap-2 font-bold text-xs text-blue-500">
                      <Loader2 size={13} className="animate-spin" />
                      <span>AI đang tìm kiếm & tổng hợp...</span>
                    </div>
                    <div className="space-y-1.5 pt-1">
                      <div className="h-3 bg-gradient-to-r from-[var(--border)] via-[var(--surface-inset)] to-[var(--border)] bg-[length:200%_100%] animate-[algo-shimmer_1.5s_linear_infinite] rounded w-full" />
                      <div className="h-3 bg-gradient-to-r from-[var(--border)] via-[var(--surface-inset)] to-[var(--border)] bg-[length:200%_100%] animate-[algo-shimmer_1.5s_linear_infinite] rounded w-[90%]" />
                      <div className="h-3 bg-gradient-to-r from-[var(--border)] via-[var(--surface-inset)] to-[var(--border)] bg-[length:200%_100%] animate-[algo-shimmer_1.5s_linear_infinite] rounded w-[60%]" />
                    </div>
                  </div>
                </motion.div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Bar */}
        <div className="p-4 border-t border-[var(--border)] bg-[var(--surface-muted)]">
          <form onSubmit={handleSendMessage} className="flex gap-2">
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder={
                selectedDocIds.length === 0
                  ? "Vui lòng chọn ít nhất 1 tài liệu ở thanh bên..."
                  : "Hỏi AI bất cứ điều gì về tài liệu..."
              }
              disabled={chatLoading || selectedDocIds.length === 0}
              className="ui-input flex-1 bg-[var(--surface-elevated)] border-[var(--border)] text-sm rounded-xl py-3 px-4 shadow-sm"
            />
            <button
              type="submit"
              disabled={!inputQuery.trim() || chatLoading || selectedDocIds.length === 0}
              className="ui-btn-primary rounded-xl px-5 py-3 shadow-md shadow-blue-600/10 hover:shadow-blue-600/20 shrink-0 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {chatLoading ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Send size={16} />
              )}
            </button>
          </form>
        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────
          RIGHT PANEL: Citations Grounds & Parameters Settings
          ───────────────────────────────────────────────────────────── */}
      <section className="w-full lg:w-80 bg-[var(--surface-elevated)] border border-[var(--border)] rounded-2xl flex flex-col overflow-hidden shrink-0 shadow-sm shadow-black/5">
        {/* Toggle tabs with sliding background pill indicator */}
        <div className="border-b border-[var(--border)] bg-[var(--surface-muted)] p-1">
          <div className="flex border border-[var(--border)] bg-[var(--surface-muted)] p-1 rounded-2xl relative overflow-hidden">
            <button
              onClick={() => setRightPanelTab('settings')}
              className={`relative z-10 flex-grow flex-shrink flex-basis-0 flex items-center justify-center gap-1.5 py-2 rounded-xl text-xs font-bold transition-colors cursor-pointer ${
                rightPanelTab === 'settings' ? 'text-blue-600 dark:text-blue-400' : 'text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
            >
              {rightPanelTab === 'settings' && (
                <motion.div
                  layoutId="activeTabPill"
                  className="absolute inset-0 bg-[var(--surface-elevated)] border border-[var(--border)]/50 shadow-sm rounded-xl"
                  transition={{ type: "spring", stiffness: 380, damping: 30 }}
                />
              )}
              <span className="relative z-20 flex items-center gap-1.5">
                <Sliders size={13} /> Tham số RAG
              </span>
            </button>
            
            <button
              onClick={() => setRightPanelTab('citations')}
              className={`relative z-10 flex-grow flex-shrink flex-basis-0 flex items-center justify-center gap-1.5 py-2 rounded-xl text-xs font-bold transition-colors cursor-pointer ${
                rightPanelTab === 'citations' ? 'text-blue-600 dark:text-blue-400' : 'text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
            >
              {rightPanelTab === 'citations' && (
                <motion.div
                  layoutId="activeTabPill"
                  className="absolute inset-0 bg-[var(--surface-elevated)] border border-[var(--border)]/50 shadow-sm rounded-xl"
                  transition={{ type: "spring", stiffness: 380, damping: 30 }}
                />
              )}
              <span className="relative z-20 flex items-center gap-1.5">
                <ShieldCheck size={13} /> Dẫn nguồn ({activeCitations.length})
              </span>
            </button>
          </div>
        </div>

        {/* Tab contents */}
        <div className="flex-1 overflow-y-auto p-4">
          
          {/* TAB 1: Advanced settings */}
          {rightPanelTab === 'settings' && (
            <div className="space-y-4">
              <div className="space-y-1">
                <p className="text-[10px] font-bold text-[var(--text-faint)] uppercase tracking-wider">Mô hình truy vấn</p>
                <select
                  value={ragModel}
                  onChange={(e) => setRagModel(e.target.value)}
                  className="ui-input py-1.5 text-xs rounded-xl cursor-pointer"
                >
                  {embeddingModels.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>

              <div className="space-y-1">
                <p className="text-[10px] font-bold text-[var(--text-faint)] uppercase tracking-wider">Phương thức Retrieval</p>
                <div className="grid grid-cols-3 gap-1 bg-[var(--surface-muted)] p-1 rounded-xl border border-[var(--border)]">
                  {(['hybrid', 'embedding', 'bm25'] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setRetrievalMode(mode)}
                      className={`text-[10px] font-bold py-1.5 rounded-lg capitalize transition-all cursor-pointer ${
                        retrievalMode === mode
                          ? 'bg-blue-600 text-white shadow-sm'
                          : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                      }`}
                    >
                      {mode === 'hybrid' ? 'Hybrid' : mode === 'embedding' ? 'Vector' : 'BM25'}
                    </button>
                  ))}
                </div>
              </div>

              <div className="h-px bg-[var(--border)]" />

              <div>
                <div className="flex justify-between text-xs font-bold mb-1">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Số nguồn lấy (Top K)</span>
                  <span className="font-mono text-blue-500">{topK} chunks</span>
                </div>
                <input
                  type="range" min="1" max="15" step="1"
                  value={topK} onChange={(e) => setTopK(Number(e.target.value))}
                  className="ui-range"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs font-bold mb-1">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Ngưỡng tương đồng</span>
                  <span className="font-mono text-blue-500">{threshold.toFixed(2)}</span>
                </div>
                <input
                  type="range" min="0.0" max="0.9" step="0.05"
                  value={threshold} onChange={(e) => setThreshold(Number(e.target.value))}
                  className="ui-range"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs font-bold mb-1">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Nhiệt độ (Temperature)</span>
                  <span className="font-mono text-blue-500">{temperature.toFixed(1)}</span>
                </div>
                <input
                  type="range" min="0.0" max="1.0" step="0.1"
                  value={temperature} onChange={(e) => setTemperature(Number(e.target.value))}
                  className="ui-range"
                />
              </div>

              <div className="flex items-center justify-between p-3 rounded-xl bg-[var(--surface-muted)] border border-[var(--border)]">
                <div className="space-y-0.5">
                  <p className="text-xs font-bold text-[var(--text)]">Sử dụng Reranking</p>
                  <p className="text-[9px] text-[var(--text-muted)]">Tái định thứ hạng bằng Cross-Encoder</p>
                </div>
                <input
                  type="checkbox"
                  checked={useReranking}
                  onChange={(e) => setUseReranking(e.target.checked)}
                  className="rounded border-[var(--border)] text-blue-500 focus:ring-blue-500 cursor-pointer"
                />
              </div>
            </div>
          )}

          {/* TAB 2: Grounding citations list */}
          {rightPanelTab === 'citations' && (
            <div className="space-y-3">
              {activeCitations.length === 0 ? (
                <div className="py-12 text-center text-xs text-[var(--text-faint)]">
                  Chưa có trích dẫn từ tin nhắn phản hồi gần nhất. Gửi một tin nhắn để xem dẫn nguồn chi tiết.
                </div>
              ) : (
                <div className="space-y-3">
                  {activeCitations.map((cite, index) => {
                    const isHighlighted = highlightedCitationId === cite.chunk_id;
                    return (
                      <motion.div
                        key={cite.chunk_id}
                        id={`citation-chunk-${cite.chunk_id}`}
                        animate={{
                          borderColor: isHighlighted ? 'var(--accent)' : 'var(--border)',
                          backgroundColor: isHighlighted ? 'rgba(37,99,235,0.06)' : 'rgba(255,255,255,0)',
                        }}
                        transition={{ duration: 0.3 }}
                        className={`p-3.5 rounded-xl border ${
                          isHighlighted
                            ? 'shadow-md shadow-blue-500/5 ring-1 ring-blue-500/20'
                            : 'bg-[var(--surface-muted)]/50'
                        }`}
                      >
                        {/* Source Details Header */}
                        <div className="flex items-center justify-between gap-2 mb-2">
                          <span className="text-[10px] font-extrabold px-2 py-0.5 rounded bg-emerald-500 text-white font-mono uppercase">
                            Nguồn #{index + 1}
                          </span>
                          <span className="text-[9px] font-bold text-[var(--text-faint)]">
                            Hạng {cite.rank ?? (index + 1)}
                          </span>
                        </div>

                        {/* File Name info */}
                        <p className="text-[11px] font-bold text-[var(--text)] truncate flex items-center gap-1.5">
                          <FileText size={12} className="text-blue-500 shrink-0" />
                          {cite.filename}
                          {cite.page !== null && (
                            <span className="text-[9px] px-1 py-0.5 rounded bg-[var(--surface-inset)] text-[var(--text-muted)] font-mono">
                              Tr. {cite.page}
                            </span>
                          )}
                        </p>

                        {/* Chunk Content Text */}
                        <p className="text-xs text-[var(--text-secondary)] leading-relaxed mt-2 p-2 bg-[var(--surface-elevated)] rounded-lg border border-[var(--border)]/60 max-h-[140px] overflow-y-auto whitespace-pre-wrap font-sans">
                          {cite.text}
                        </p>

                        {/* Detail Scores */}
                        <div className="mt-2.5 pt-2 border-t border-[var(--border)]/40 grid grid-cols-3 gap-1 text-center">
                          <div className="p-1 rounded bg-[var(--surface-inset)]">
                            <p className="text-[8px] text-[var(--text-faint)] uppercase font-semibold">Embed F1</p>
                            <p className="text-xs font-mono font-bold text-blue-500">{(cite.embedding_score || 0).toFixed(3)}</p>
                          </div>
                          <div className="p-1 rounded bg-[var(--surface-inset)]">
                            <p className="text-[8px] text-[var(--text-faint)] uppercase font-semibold">BM25</p>
                            <p className="text-xs font-mono font-bold text-blue-500">{(cite.bm25_score || 0).toFixed(1)}</p>
                          </div>
                          <div className="p-1 rounded bg-[var(--surface-inset)]">
                            <p className="text-[8px] text-[var(--text-faint)] uppercase font-semibold">Combined</p>
                            <p className="text-xs font-mono font-bold text-emerald-500">{(cite.combined_score || 0).toFixed(3)}</p>
                          </div>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

        </div>
      </section>

    </div>
  );
}
