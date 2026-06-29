import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  UploadCloud, Trash2, Plus, MessageSquare,
  Loader2, Send, CheckCircle2, AlertTriangle, FileText,
  ChevronRight, Sparkles, ShieldCheck, Bookmark,
  PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen,
  Search, Edit2, Check, Bot
} from 'lucide-react';
import * as ragApi from '../services/ragApi';
import { useApp } from '../context/AppContext';
import { RAGDocument, RAGMessage, RAGCitation, RAGChatRequest } from '../types/rag';

const formatModelName = (name: string | null | undefined) => {
  if (!name) return "";
  const n = name.toLowerCase();
  if (n.includes("gemini")) return "Google Gemini API";
  if (n.includes("openai")) return "OpenAI GPT API";
  if (n.includes("ollama")) return "Ollama Local LLM";
  if (n.includes("vit5")) return "ViT5 Local";
  if (n.includes("bartpho")) return "BARTPho Local";
  if (n.includes("mt5")) return "mT5 Local";
  if (n.includes("extractive")) return "Trích xuất Fallback";
  return name;
};

const SUGGESTED_QUESTIONS = [
  "Tài liệu này đề cập đến các vấn đề chính nào?",
  "Tóm tắt các phát hiện quan trọng nhất trong tài liệu này.",
  "Phương pháp nghiên cứu nào được tác giả sử dụng?",
  "Có những kết luận thực tiễn nào rút ra được?"
];

export default function Chat() {
  const { t } = useApp() as any;
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isStreamingRef = useRef(false);
  const queryClient = useQueryClient();

  // Tab & search states
  const [leftTab, setLeftTab] = useState<'history' | 'documents'>('history');
  const [searchQuery, setSearchQuery] = useState('');
  const [editingConvId, setEditingConvId] = useState<string | null>(null);
  const [editTitleVal, setEditTitleVal] = useState('');

  // RAG datasets & active conversation state
  const [documents, setDocuments] = useState<RAGDocument[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<RAGMessage[]>([]);
  const [inputQuery, setInputQuery] = useState('');
  
  // Advanced configuration — hardcoded best defaults
  const [chunkSize] = useState(512);
  const [chunkOverlap] = useState(80);
  const [ingestionModel] = useState('intfloat/multilingual-e5-large');

  // RAG defaults (không hiển thị cho người dùng sửa)
  const ragModel = 'intfloat/multilingual-e5-large';
  const topK = 3;
  const threshold = 0.35;
  const retrievalMode = 'hybrid';
  const useReranking = true;
  const temperature = 0.15;

  // Ingestion loading & files
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  // Layout panels & citations
  const [chatLoading, setChatLoading] = useState(false);
  const [activeCitations, setActiveCitations] = useState<RAGCitation[]>([]);
  const [highlightedCitationId, setHighlightedCitationId] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  const [showLeftPanel, setShowLeftPanel] = useState(true);
  const [showRightPanel, setShowRightPanel] = useState(true);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 1024);
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // --- React Query Integrations ---
  const { data: conversationsData = [], isLoading: isConvsLoading } = useQuery({
    queryKey: ['conversations', searchQuery],
    queryFn: async () => {
      if (searchQuery.trim()) return await ragApi.searchConversations(searchQuery.trim());
      return await ragApi.listRagConversations();
    },
    placeholderData: (previousData) => previousData,
  });

  const { data: fetchedMessages } = useQuery({
    queryKey: ['messages', activeConversationId],
    queryFn: async () => {
      if (!activeConversationId) return [];
      return await ragApi.listConversationMessages(activeConversationId);
    },
    enabled: !!activeConversationId,
  });

  useEffect(() => {
    if (activeConversationId && fetchedMessages) {
      if (!isStreamingRef.current) {
        setMessages(fetchedMessages);
        const assistantMsgs = fetchedMessages.filter(m => m.role === 'assistant');
        if (assistantMsgs.length > 0) {
          const lastMsg = assistantMsgs[assistantMsgs.length - 1];
          setActiveCitations(lastMsg.citations || []);
        } else {
          setActiveCitations([]);
        }
      }
    } else if (!activeConversationId) {
      setMessages([]);
      setActiveCitations([]);
    }
  }, [fetchedMessages, activeConversationId]);

  const createConversationMutation = useMutation({
    mutationFn: async () => await ragApi.createConversation("Cuộc trò chuyện mới"),
    onSuccess: (newConv) => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      setActiveConversationId(newConv.id);
      setLeftTab('history');
    },
    onError: (err: any) => showError("Không thể tạo cuộc trò chuyện: " + err.message)
  });

  const renameConversationMutation = useMutation({
    mutationFn: async ({ id, title }: { id: string; title: string }) => await ragApi.renameConversation(id, title),
    onMutate: async ({ id, title }) => {
      await queryClient.cancelQueries({ queryKey: ['conversations'] });
      const previousConvs = queryClient.getQueryData(['conversations']);
      queryClient.setQueryData(['conversations'], (old: any) =>
        old ? old.map((c: any) => c.id === id ? { ...c, title } : c) : []
      );
      return { previousConvs };
    },
    onError: (err: any, _, context) => {
      if (context?.previousConvs) queryClient.setQueryData(['conversations'], context.previousConvs);
      showError("Không thể đổi tên cuộc trò chuyện: " + err.message);
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['conversations'] })
  });

  const deleteConversationMutation = useMutation({
    mutationFn: async (id: string) => await ragApi.deleteConversation(id),
    onSuccess: (_, deletedId) => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      if (activeConversationId === deletedId) {
        setActiveConversationId(null);
        setMessages([]);
        setActiveCitations([]);
      }
      showSuccess("Đã xóa cuộc trò chuyện");
    },
    onError: (err: any) => showError("Không thể xóa cuộc trò chuyện: " + err.message)
  });

  useEffect(() => {
    loadDocuments();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, chatLoading]);

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

  const handleDeleteAll = async () => {
    if (!window.confirm('Bạn có chắc chắn muốn XÓA TẤT CẢ cuộc trò chuyện và tài liệu RAG? Hành động này không thể hoàn tác.')) return;
    try {
      await ragApi.deleteAllConversations();
      await ragApi.deleteAllDocuments();
      setActiveConversationId(null);
      setMessages([]);
      setActiveCitations([]);
      setDocuments([]);
      setSelectedDocIds([]);
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      showSuccess('Đã xóa toàn bộ lịch sử trò chuyện và tài liệu RAG');
    } catch (err: any) {
      showError('Không thể xóa tất cả: ' + err.message);
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

  const handleStartRename = (id: string, currentTitle: string) => {
    setEditingConvId(id);
    setEditTitleVal(currentTitle);
  };

  const handleRenameSubmit = (id: string) => {
    if (editTitleVal.trim()) {
      renameConversationMutation.mutate({ id, title: editTitleVal.trim() });
    }
    setEditingConvId(null);
  };

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
      showSuccess(`Đã nạp thành công: ${uploadFile.name}`);
      setUploadFile(null);
      loadDocuments(res.document_id);
    } catch (err: any) {
      showError('Upload tài liệu thất bại: ' + err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDoc = async (docId: string, filename: string) => {
    if (!window.confirm(`Bạn có chắc chắn muốn xóa tài liệu "${filename}"? Chunks và vector index sẽ bị xóa hoàn toàn.`)) return;
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
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
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

  const handleSendMessage = async (e?: React.FormEvent, customQuery?: string) => {
    if (e) e.preventDefault();
    const queryToSend = customQuery || inputQuery;
    if (!queryToSend.trim() || chatLoading || selectedDocIds.length === 0) return;

    setInputQuery('');
    setChatLoading(true);
    setErrorMessage(null);
    isStreamingRef.current = true;

    const userMsg: RAGMessage = {
      role: 'user',
      content: queryToSend,
      created_at: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMsg]);

    const assistantPlaceholder: RAGMessage = {
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString()
    };
    setMessages(prev => [...prev, assistantPlaceholder]);

    try {
      const requestPayload: RAGChatRequest = {
        query: queryToSend,
        conversation_id: activeConversationId,
        document_ids: selectedDocIds,
        top_k: topK,
        threshold,
        retrieval_mode: retrievalMode as 'hybrid' | 'embedding' | 'bm25',
        use_reranking: useReranking,
        embedding_model: ragModel,
        temperature
      };

      const finalResult = await ragApi.streamRagChat(requestPayload, (streamingText, conversationId) => {
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

        if (!activeConversationId && conversationId) {
          setActiveConversationId(conversationId);
        }
      });

      setMessages(prev => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        if (updated[lastIdx] && updated[lastIdx].role === 'assistant') {
          updated[lastIdx] = {
            role: 'assistant',
            content: finalResult.answer,
            confidence: finalResult.confidence,
            citations: finalResult.retrieved_context,
            model_used: finalResult.model_used,
            evaluation: finalResult.evaluation,
            created_at: new Date().toISOString()
          };
        }
        return updated;
      });

      setActiveCitations(finalResult.retrieved_context || []);
      const activeId = activeConversationId || finalResult.conversation_id;
      if (!activeConversationId && finalResult.conversation_id) {
        setActiveConversationId(finalResult.conversation_id);
      }
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      queryClient.invalidateQueries({ queryKey: ['messages', activeId] });

    } catch (err: any) {
      showError('Lỗi gửi tin nhắn: ' + err.message);
      setMessages(prev => prev.filter((_, i) => i !== prev.length - 1));
    } finally {
      setChatLoading(false);
      isStreamingRef.current = false;
    }
  };

  const handleCitationClick = (citation: RAGCitation) => {
    setHighlightedCitationId(citation.chunk_id);
    setTimeout(() => {
      const targetElement = document.getElementById(`citation-chunk-${citation.chunk_id}`);
      targetElement?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
  };

  const renderMessageContent = (msg: RAGMessage, isStreaming: boolean = false) => {
    if (msg.role === 'user') {
      return <p className="whitespace-pre-wrap leading-relaxed text-sm font-semibold">{msg.content}</p>;
    }

    const text = msg.content;
    if (!msg.citations || msg.citations.length === 0) {
      return (
        <p className="whitespace-pre-wrap leading-relaxed text-sm">
          {text}
          {isStreaming && (
            <motion.span
              animate={{ opacity: [1, 0, 1] }}
              transition={{ repeat: Infinity, duration: 0.8 }}
              className="inline-block w-1.5 h-3.5 ml-1 bg-sky-500 align-middle"
            />
          )}
        </p>
      );
    }

    const parts = text.split(/(\[\d+\])/g);
    return (
      <div className="space-y-4">
        <p className="whitespace-pre-wrap leading-relaxed text-sm text-[var(--text-secondary)]">
          {parts.map((part: string, i: number) => {
            const match = part.match(/^\[(\d+)\]$/);
            if (match) {
              const num = parseInt(match[1], 10);
              const cite = msg.citations?.[num - 1];
              if (cite) {
                return (
                  <motion.button
                    key={i}
                    whileHover={{ scale: 1.15, y: -1 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => handleCitationClick(cite)}
                    className="inline-flex items-center justify-center w-4.5 h-4.5 rounded-full bg-sky-105 dark:bg-sky-900 text-sky-700 dark:text-sky-300 text-[9px] font-bold mx-0.5 border border-sky-300 dark:border-sky-800 align-super cursor-pointer shadow-sm"
                    title={`Nguồn: ${cite.filename}`}
                  >
                    {num}
                  </motion.button>
                );
              }
            }
            return part;
          })}
          {isStreaming && (
            <motion.span
              animate={{ opacity: [1, 0, 1] }}
              transition={{ repeat: Infinity, duration: 0.8 }}
              className="inline-block w-1.5 h-3.5 ml-1 bg-sky-500 align-middle"
            />
          )}
        </p>
        
        <div className="pt-2 border-t border-[var(--border)] opacity-95">
          <p className="text-[10px] font-bold text-[var(--text-faint)] uppercase tracking-wider mb-2 flex items-center gap-1">
            <ShieldCheck size={12} className="text-emerald-500" /> Nguồn tham chiếu từ file:
          </p>
          <div className="flex flex-wrap gap-1.5">
            {msg.citations.map((cite: RAGCitation, i: number) => (
              <motion.button
                key={cite.chunk_id}
                whileHover={{ scale: 1.02, borderColor: 'var(--accent)' }}
                whileTap={{ scale: 0.98 }}
                onClick={() => handleCitationClick(cite)}
                className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-50 dark:bg-emerald-950/20 text-emerald-700 dark:text-emerald-400 border border-emerald-250/30 dark:border-emerald-800/40 hover:bg-emerald-100/50 dark:hover:bg-emerald-900/10 transition-all cursor-pointer"
              >
                <span className="font-mono">[{i + 1}]</span>
                <span className="truncate max-w-[100px]">{cite.filename}</span>
                {cite.page !== null && <span className="opacity-60">(Tr. {cite.page})</span>}
              </motion.button>
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="h-[calc(100vh-100px)] flex gap-4 relative overflow-hidden -mx-6 -my-6 px-6 py-6 bg-[var(--bg)]">
      
      {/* ─────────────────────────────────────────────────────────────
          LEFT PANEL: Documents Ingestion & Chat History
          ───────────────────────────────────────────────────────────── */}
      <AnimatePresence initial={false}>
        {showLeftPanel && (
          <motion.section
            key="left-panel"
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: isMobile ? '100%' : 300, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded-2xl flex flex-col overflow-hidden shrink-0 shadow-sm"
          >
            {/* Tabs */}
            <div className="flex border-b border-[var(--border)] p-1 bg-[var(--bg-muted)]/50">
              <button
                onClick={() => setLeftTab('history')}
                className={`flex-1 py-1.5 text-xs font-bold rounded-lg flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
                  leftTab === 'history'
                    ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)] shadow-sm'
                    : 'text-[var(--text-muted)] hover:bg-[var(--bg-muted)]'
                }`}
              >
                <MessageSquare size={13} className="text-sky-500" />
                Hội thoại
              </button>
              <button
                onClick={() => setLeftTab('documents')}
                className={`flex-1 py-1.5 text-xs font-bold rounded-lg flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
                  leftTab === 'documents'
                    ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)] shadow-sm'
                    : 'text-[var(--text-muted)] hover:bg-[var(--bg-muted)]'
                }`}
              >
                <FileText size={13} className="text-sky-500" />
                Tài liệu ({documents.length})
              </button>
            </div>

            {/* TAB CONTENT: HISTORY */}
            {leftTab === 'history' && (
              <div className="flex-1 flex flex-col overflow-hidden p-3 space-y-3">
                <div className="flex gap-2">
                  <button
                    onClick={() => createConversationMutation.mutate()}
                    disabled={createConversationMutation.isPending}
                    className="flex-1 py-2 px-3 rounded-xl text-xs font-bold bg-sky-600 hover:bg-sky-700 text-white flex items-center justify-center gap-1.5 shadow-sm transition-all disabled:opacity-50 cursor-pointer"
                  >
                    {createConversationMutation.isPending ? <Loader2 size={13} className="animate-spin" /> : <Plus size={13} />}
                    Tạo mới
                  </button>
                  <button
                    onClick={handleDeleteAll}
                    className="py-2 px-3 rounded-xl text-xs font-bold bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20 flex items-center justify-center gap-1 shadow-sm transition-all cursor-pointer"
                    title="Xóa tất cả lịch sử trò chuyện và tài liệu RAG"
                  >
                    <Trash2 size={13} />
                    Xóa hết
                  </button>
                </div>

                <div className="relative">
                  <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-faint)]" />
                  <input
                    type="text"
                    placeholder="Tìm cuộc trò chuyện..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-8 pr-3 py-1.5 text-xs rounded-xl border border-[var(--border)] bg-[var(--bg)] focus:border-sky-500 focus:ring-1 focus:ring-sky-500 outline-none transition-all"
                  />
                </div>

                <div className="flex-1 overflow-y-auto space-y-1 scrollbar-none pr-0.5">
                  {isConvsLoading && conversationsData.length === 0 ? (
                    <div className="flex items-center justify-center py-6">
                      <Loader2 size={16} className="animate-spin text-sky-500" />
                    </div>
                  ) : conversationsData.length === 0 ? (
                    <p className="text-center text-xs text-[var(--text-faint)] py-6 font-medium">Không tìm thấy hội thoại</p>
                  ) : (
                    conversationsData.map((conv: any) => {
                      const isSelected = activeConversationId === conv.id;
                      const isEditing = editingConvId === conv.id;

                      if (isEditing) {
                        return (
                          <div key={conv.id} className="flex items-center gap-1.5 p-2 bg-[var(--bg-muted)] border border-sky-500 rounded-xl">
                            <input
                              type="text"
                              value={editTitleVal}
                              onChange={(e) => setEditTitleVal(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') handleRenameSubmit(conv.id);
                                if (e.key === 'Escape') setEditingConvId(null);
                              }}
                              className="flex-1 text-xs bg-transparent border-none outline-none p-0"
                              autoFocus
                            />
                            <button onClick={() => handleRenameSubmit(conv.id)} className="text-emerald-500 cursor-pointer"><CheckCircle2 size={14} /></button>
                          </div>
                        );
                      }

                      return (
                        <div
                          key={conv.id}
                          onClick={() => !chatLoading && setActiveConversationId(conv.id)}
                          className={`group relative flex items-center justify-between p-2.5 rounded-xl border transition-all cursor-pointer ${
                            isSelected
                              ? 'border-sky-500/20 bg-sky-500/5 text-sky-600 dark:text-sky-400'
                              : 'border-transparent hover:bg-[var(--bg-muted)] text-[var(--text-secondary)]'
                          }`}
                        >
                          <div className="min-w-0 pr-8">
                            <p className="text-xs font-bold truncate leading-tight">{conv.title}</p>
                            <p className="text-[10px] text-[var(--text-faint)] mt-0.5 font-medium">
                              {new Date(conv.updated_at).toLocaleDateString()}
                            </p>
                          </div>
                          <div className="hidden group-hover:flex items-center gap-1 absolute right-2">
                            <button
                              onClick={(e) => { e.stopPropagation(); handleStartRename(conv.id, conv.title); }}
                              className="p-1 rounded hover:bg-[var(--bg-inset)] text-[var(--text-muted)] hover:text-sky-500 cursor-pointer"
                            >
                              <Edit2 size={12} />
                            </button>
                            <button
                              onClick={(e) => { e.stopPropagation(); deleteConversationMutation.mutate(conv.id); }}
                              className="p-1 rounded hover:bg-[var(--bg-inset)] text-[var(--text-muted)] hover:text-red-500 cursor-pointer"
                            >
                              <Trash2 size={12} />
                            </button>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            )}

            {/* TAB CONTENT: DOCUMENTS */}
            {leftTab === 'documents' && (
              <div className="flex-1 flex flex-col overflow-hidden p-3 space-y-3">
                {/* Upload Section */}
                <form onSubmit={handleUpload} className="space-y-2">
                  <div
                    onDragEnter={handleDrag}
                    onDragOver={handleDrag}
                    onDragLeave={handleDrag}
                    onDrop={handleDrop}
                    className={`border border-dashed rounded-xl p-4 text-center cursor-pointer transition-colors relative flex flex-col items-center justify-center ${
                      dragActive ? 'border-sky-500 bg-sky-500/5' : 'border-[var(--border)] hover:border-sky-400'
                    }`}
                  >
                    <input
                      type="file"
                      accept=".pdf,.docx,.txt"
                      onChange={(e) => e.target.files && setUploadFile(e.target.files[0])}
                      className="absolute inset-0 opacity-0 cursor-pointer"
                    />
                    <UploadCloud size={24} className="text-sky-500 mb-1" />
                    <span className="text-[10px] font-bold text-[var(--text-muted)]">Kéo thả file PDF, DOCX, TXT vào đây</span>
                  </div>
                  {uploadFile && (
                    <div className="flex items-center justify-between bg-[var(--bg-muted)] px-3 py-1.5 rounded-lg border border-[var(--border)]">
                      <span className="text-[10px] font-bold text-sky-600 truncate max-w-[180px]">{uploadFile.name}</span>
                      <button
                        type="submit"
                        disabled={uploading}
                        className="text-[10px] font-bold bg-sky-600 text-white rounded px-2 py-0.5 hover:bg-sky-700 disabled:opacity-50 cursor-pointer"
                      >
                        {uploading ? <Loader2 size={10} className="animate-spin" /> : 'Nạp'}
                      </button>
                    </div>
                  )}
                </form>

                {/* Doc List */}
                <div className="flex-1 overflow-y-auto space-y-1.5 scrollbar-none pr-0.5">
                  <div className="flex items-center justify-between text-[10px] font-bold text-[var(--text-faint)] uppercase tracking-wider px-1 mb-1">
                    <span>Chọn tài liệu nạp RAG</span>
                    <div className="flex gap-2 items-center">
                      <button onClick={() => setSelectedDocIds(documents.map(d => d.id))} className="hover:underline hover:text-sky-500 cursor-pointer">Chọn tất cả</button>
                      <span className="text-[var(--text-faint)]">|</span>
                      <button onClick={() => setSelectedDocIds([])} className="hover:underline hover:text-red-500 cursor-pointer">Bỏ chọn</button>
                    </div>
                  </div>
                  {documents.length === 0 ? (
                    <p className="text-center text-xs text-[var(--text-faint)] py-6 font-medium">Chưa có tài liệu nào được nạp</p>
                  ) : (
                    documents.map((doc) => {
                      const isSelected = selectedDocIds.includes(doc.id);
                      return (
                        <div
                          key={doc.id}
                          onClick={() => toggleSelectDoc(doc.id)}
                          className={`flex items-center justify-between p-2 rounded-xl border transition-all cursor-pointer ${
                            isSelected
                              ? 'border-sky-500/20 bg-sky-500/5'
                              : 'border-[var(--border)] hover:bg-[var(--bg-muted)]'
                          }`}
                        >
                          <div className="min-w-0 flex-1 flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => {}}
                              className="rounded border-[var(--border)] text-sky-500 focus:ring-sky-500 shrink-0"
                            />
                            <div className="min-w-0">
                              <p className="text-xs font-bold text-[var(--text-primary)] truncate leading-tight">{doc.filename}</p>
                              <p className="text-[9px] text-[var(--text-faint)] mt-0.5 font-medium">
                                {(((doc.file_size || 0) / 1024)).toFixed(1)} KB • {doc.chunks_count ?? 0} chunks
                              </p>
                            </div>
                          </div>
                          <button
                            onClick={(e) => { e.stopPropagation(); handleDeleteDoc(doc.id, doc.filename); }}
                            className="p-1 text-[var(--text-faint)] hover:text-red-500 cursor-pointer"
                            title="Xóa tài liệu"
                          >
                            <Trash2 size={12} />
                          </button>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            )}
          </motion.section>
        )}
      </AnimatePresence>

      {/* ─────────────────────────────────────────────────────────────
          CENTER WORKSPACE: Main Chat Area & Input
          ───────────────────────────────────────────────────────────── */}
      <section className="flex-1 flex flex-col bg-[var(--bg-elevated)] border border-[var(--border)] rounded-2xl overflow-hidden shadow-sm">
        {/* Workspace Header */}
        <header className="px-5 h-13 border-b border-[var(--border)] flex items-center justify-between bg-[var(--bg-subtle)] shrink-0">
          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setShowLeftPanel(v => !v)}
              className="ui-btn-icon hover:bg-[var(--bg-muted)] rounded-lg cursor-pointer"
              title="Toggle sidebar"
            >
              {showLeftPanel ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
            </button>
            <div className="h-4 w-px bg-[var(--border)]" />
            <div className="min-w-0">
              <span className="text-xs font-bold text-[var(--text-primary)]">AI Assistant Workspace</span>
              {selectedDocIds.length > 0 && (
                <span className="text-[10px] text-sky-600 dark:text-sky-400 font-bold bg-sky-50 dark:bg-sky-950/30 px-2 py-0.5 rounded-full ml-2 border border-sky-100 dark:border-sky-900/50">
                  {selectedDocIds.length} tài liệu RAG active
                </span>
              )}
            </div>
          </div>
          <button
            onClick={() => setShowRightPanel(v => !v)}
            className="ui-btn-icon hover:bg-[var(--bg-muted)] rounded-lg cursor-pointer"
            title="Toggle parameter panel"
          >
            {showRightPanel ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
          </button>
        </header>

        {/* Message List */}
        <div className="flex-1 overflow-y-auto p-5 space-y-6 scroll-smooth">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center max-w-lg mx-auto text-center space-y-4">
              <div className="w-12 h-12 rounded-2xl bg-sky-100 dark:bg-sky-950/40 text-sky-600 flex items-center justify-center shadow-sm">
                <Bot size={24} />
              </div>
              <div className="space-y-1.5">
                <h2 className="text-base font-extrabold tracking-tight text-[var(--text-primary)]">Hệ thống Trò chuyện Ngữ nghĩa (RAG)</h2>
                <p className="text-xs text-[var(--text-muted)] font-medium leading-relaxed">
                  Chọn hoặc nạp tài liệu khoa học ở cột bên trái. Hệ thống sẽ tự động phân tách tài liệu, tìm kiếm chunks có độ tương đồng ngữ nghĩa cao nhất để làm ngữ cảnh trả lời câu hỏi của bạn.
                </p>
              </div>
              <div className="w-full grid grid-cols-1 gap-2 pt-2 text-left">
                <span className="text-[10px] font-bold text-[var(--text-faint)] uppercase tracking-wider px-1">Câu hỏi gợi ý:</span>
                {SUGGESTED_QUESTIONS.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSendMessage(undefined, q)}
                    disabled={selectedDocIds.length === 0}
                    className="w-full text-left p-3 rounded-xl border border-[var(--border)] hover:border-sky-500 hover:bg-sky-500/5 text-xs text-[var(--text-secondary)] font-bold transition-all flex items-center justify-between group cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <span>{q}</span>
                    <ChevronRight size={13} className="text-[var(--text-faint)] group-hover:text-sky-500 group-hover:translate-x-0.5 transition-all" />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => {
              const isUser = msg.role === 'user';
              const isStreaming = isStreamingRef.current && idx === messages.length - 1 && msg.role === 'assistant';
              return (
                <div key={idx} className={`flex gap-3 max-w-3xl ${isUser ? 'ml-auto flex-row-reverse' : 'mr-auto'}`}>
                  {/* Avatar */}
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs shrink-0 shadow-sm border ${
                    isUser
                      ? 'bg-sky-600 text-white border-sky-700'
                      : 'bg-white dark:bg-slate-900 text-sky-600 border-[var(--border)]'
                  }`}>
                    {isUser ? 'ME' : 'AI'}
                  </div>
                  {/* Content Bubble */}
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold text-[var(--text-muted)]">
                        {isUser ? 'Bạn' : 'AI Assistant'}
                      </span>
                      {!isUser && msg.model_used && (
                        <span className="text-[9px] font-bold bg-[var(--bg-muted)] text-[var(--text-muted)] px-1.5 py-0.5 rounded border border-[var(--border)] uppercase">
                          {formatModelName(msg.model_used)}
                        </span>
                      )}
                      {!isUser && msg.confidence !== undefined && msg.confidence !== null && (
                        <span className="text-[9px] font-bold bg-emerald-50 dark:bg-emerald-950/20 text-emerald-600 dark:text-emerald-400 px-1.5 py-0.5 rounded border border-emerald-200/20 dark:border-emerald-800/40">
                          Độ tin cậy: {Math.round(msg.confidence * 100)}%
                        </span>
                      )}
                    </div>
                    <div className={`p-4 rounded-2xl border ${
                      isUser
                        ? 'bg-sky-500/5 border-sky-500/20 text-[var(--text-primary)] rounded-tr-none'
                        : 'bg-[var(--bg-muted)]/20 border-[var(--border)] rounded-tl-none'
                    }`}>
                      {renderMessageContent(msg, isStreaming)}
                    </div>
                  </div>
                </div>
              );
            })
          )}
          {chatLoading && messages.length > 0 && messages[messages.length - 1].role === 'user' && (
            <div className="flex gap-3 mr-auto max-w-3xl animate-pulse">
              <div className="w-8 h-8 rounded-full bg-[var(--bg-muted)] border border-[var(--border)] flex items-center justify-center text-xs font-bold text-[var(--text-faint)]">
                AI
              </div>
              <div className="space-y-1">
                <span className="text-[10px] font-bold text-[var(--text-faint)]">AI Assistant</span>
                <div className="p-4 rounded-2xl bg-[var(--bg-muted)]/20 border border-[var(--border)] rounded-tl-none flex items-center gap-2 text-xs text-[var(--text-muted)] font-semibold">
                  <Loader2 size={13} className="animate-spin text-sky-500" />
                  Đang truy xuất tài liệu và suy luận câu trả lời...
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input Panel */}
        <div className="p-4 border-t border-[var(--border)] bg-[var(--bg-subtle)] shrink-0">
          {selectedDocIds.length === 0 ? (
            <div className="p-3 bg-amber-50 dark:bg-amber-950/20 border border-amber-200/50 dark:border-amber-900/50 rounded-xl text-amber-700 dark:text-amber-400 text-xs font-bold flex items-center gap-2">
              <AlertTriangle size={15} />
              Vui lòng chọn hoặc nạp ít nhất một tài liệu ở tab "Tài liệu" cột bên trái để bắt đầu trò chuyện.
            </div>
          ) : (
            <form onSubmit={handleSendMessage} className="relative flex items-center">
              <textarea
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                placeholder="Hỏi AI về nội dung tài liệu của bạn (Enter để gửi)..."
                rows={1}
                className="w-full pl-4 pr-14 py-3 rounded-2xl border border-[var(--border)] bg-[var(--bg)] focus:border-sky-500 focus:ring-1 focus:ring-sky-500 outline-none text-xs font-semibold leading-relaxed shadow-sm resize-none scrollbar-none"
              />
              <button
                type="submit"
                disabled={chatLoading || !inputQuery.trim()}
                className="absolute right-2.5 p-2 rounded-xl bg-sky-600 hover:bg-sky-700 text-white disabled:opacity-40 disabled:cursor-not-allowed shadow-sm transition-all cursor-pointer"
              >
                {chatLoading ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
              </button>
            </form>
          )}
        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────
          RIGHT PANEL: Citations Chunks & RAG Hyperparameters
          ───────────────────────────────────────────────────────────── */}
      <AnimatePresence>
        {showRightPanel && (
          <motion.section
            key="right-panel"
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: isMobile ? '100%' : 320, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded-2xl flex flex-col overflow-hidden shrink-0 shadow-sm"
          >
            {/* Header citations tab */}
            <div className="flex border-b border-[var(--border)] p-1 bg-[var(--bg-muted)]/50 shrink-0">
              <div className="flex-1 py-1.5 text-xs font-bold text-center text-sky-600 dark:text-sky-400 flex items-center justify-center gap-1">
                <Bookmark size={13} />
                Nguồn dẫn Chunks
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-5 scrollbar-none">

              {/* Citations List */}
              <div className="space-y-3">
                <h3 className="text-[10px] font-bold text-[var(--text-faint)] uppercase tracking-wider flex items-center gap-1.5">
                  <ShieldCheck size={13} className="text-emerald-500" />
                  Nội dung Chunks trích xuất ({activeCitations.length})
                </h3>

                {activeCitations.length === 0 ? (
                  <p className="text-xs text-[var(--text-faint)] italic py-4 text-center font-medium bg-[var(--bg-muted)]/10 rounded-xl border border-[var(--border)]/30">
                    Chưa có trích xuất ngữ cảnh. Hãy gửi tin nhắn cho AI.
                  </p>
                ) : (
                  <div className="space-y-3.5">
                    {activeCitations.map((cite, i) => {
                      const isHighlighted = highlightedCitationId === cite.chunk_id;
                      return (
                        <div
                          key={cite.chunk_id}
                          id={`citation-chunk-${cite.chunk_id}`}
                          className={`p-3.5 rounded-xl border transition-all duration-200 ${
                            isHighlighted
                              ? 'border-sky-500 bg-sky-500/5 shadow-md shadow-sky-500/5'
                              : 'border-[var(--border)] bg-[var(--bg-muted)]/10'
                          }`}
                        >
                          <div className="flex items-center justify-between text-[10px] font-bold mb-2">
                            <span className="text-sky-600 dark:text-sky-400 font-mono">[{i + 1}] {cite.filename}</span>
                            <span className="bg-[var(--bg-inset)] text-[var(--text-muted)] px-1.5 py-0.5 rounded text-[8px]">
                              Score: {(cite.combined_score ?? (cite as any).score ?? 0).toFixed(3)}
                            </span>
                          </div>
                          <p className="text-[11px] text-[var(--text-secondary)] font-medium leading-relaxed whitespace-pre-wrap font-sans italic">
                            "{cite.text}"
                          </p>
                          {cite.page !== null && (
                            <div className="mt-2 text-[9px] text-[var(--text-faint)] font-bold text-right">
                              Trang {cite.page}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </motion.section>
        )}
      </AnimatePresence>
    </div>
  );
}
