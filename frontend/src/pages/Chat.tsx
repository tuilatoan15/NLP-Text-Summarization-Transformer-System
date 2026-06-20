import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  UploadCloud, Trash2, Plus, MessageSquare, Settings, Sliders,
  Loader2, Send, CheckCircle2, AlertTriangle, FileText,
  ChevronRight, Sparkles, ShieldCheck, Bookmark, Settings2,
  PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen,
  Search, Edit2
} from 'lucide-react';
import * as ragApi from '../services/ragApi';
import { useApp } from '../context/AppContext';
import { RAGDocument, RAGMessage, RAGCitation } from '../types/rag';
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

export default function Chat() {
  // --- Context & Hooks ---
  const { t } = useApp() as any;

  // --- Refs ---
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isStreamingRef = useRef(false);

  // --- State ---
  const queryClient = useQueryClient();
  const [leftTab, setLeftTab] = useState<'history' | 'documents'>('history');
  const [searchQuery, setSearchQuery] = useState('');
  const [editingConvId, setEditingConvId] = useState<string | null>(null);
  const [editTitleVal, setEditTitleVal] = useState('');
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null);
  const [deleteTargetTitle, setDeleteTargetTitle] = useState('');

  const [documents, setDocuments] = useState<RAGDocument[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<RAGMessage[]>([]);
  const [inputQuery, setInputQuery] = useState('');
  
  // Advanced Ingestion Settings (Cố định ở tầng State, không hiển thị trên UI)
  const [chunkSize, setChunkSize] = useState(512);
  const [chunkOverlap, setChunkOverlap] = useState(80);
  const [ingestionModel, setIngestionModel] = useState('intfloat/multilingual-e5-large');

  // Ingestion loading & files
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  
  // Embedding models list & active RAG configuration
  const [embeddingModels, setEmbeddingModels] = useState<string[]>([]);
  const [ragModel, setRagModel] = useState('intfloat/multilingual-e5-large');
  const [topK, setTopK] = useState(4);
  const [threshold, setThreshold] = useState(0.35);
  const [retrievalMode, setRetrievalMode] = useState<'hybrid' | 'embedding' | 'bm25'>('hybrid');
  const [useReranking, setUseReranking] = useState(true);
  const [temperature, setTemperature] = useState(0.15);

  // Active generation/interaction states
  const [chatLoading, setChatLoading] = useState(false);
  const [activeCitations, setActiveCitations] = useState<RAGCitation[]>([]);
  const [highlightedCitationId, setHighlightedCitationId] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  // Right Panel: Chỉ hiển thị tab dẫn nguồn ('citations') theo thiết kế tối giản mới
  const [rightPanelTab, setRightPanelTab] = useState<'settings' | 'citations'>('citations');

  // Sidebars toggle states for modern layout
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
  
  // Conversations list query
  const { data: conversationsData = [], isLoading: isConvsLoading } = useQuery({
    queryKey: ['conversations', searchQuery],
    queryFn: async () => {
      if (searchQuery.trim()) {
        return await ragApi.searchConversations(searchQuery.trim());
      }
      return await ragApi.listRagConversations();
    },
    placeholderData: (previousData) => previousData,
  });

  // Messages of active conversation query
  const { data: fetchedMessages } = useQuery({
    queryKey: ['messages', activeConversationId],
    queryFn: async () => {
      if (!activeConversationId) return [];
      return await ragApi.listConversationMessages(activeConversationId);
    },
    enabled: !!activeConversationId,
  });

  // Sync React Query messages to local state (supporting streaming)
  useEffect(() => {
    if (activeConversationId && fetchedMessages) {
      if (!isStreamingRef.current) {
        setMessages(fetchedMessages);
        
        const assistantMsgs = fetchedMessages.filter(m => m.role === 'assistant');
        if (assistantMsgs.length > 0) {
          const lastMsg = assistantMsgs[assistantMsgs.length - 1];
          if (lastMsg.citations && lastMsg.citations.length > 0) {
            setActiveCitations(lastMsg.citations);
          } else {
            setActiveCitations([]);
          }
        } else {
          setActiveCitations([]);
        }
      }
    } else if (!activeConversationId) {
      setMessages(prev => prev.length === 0 ? prev : []);
      setActiveCitations(prev => prev.length === 0 ? prev : []);
    }
  }, [fetchedMessages, activeConversationId]);

  // Mutations
  const createConversationMutation = useMutation({
    mutationFn: async () => {
      return await ragApi.createConversation("New chat");
    },
    onSuccess: (newConv) => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      setActiveConversationId(newConv.id);
      setLeftTab('history');
    },
    onError: (err: any) => {
      showError("Không thể tạo cuộc trò chuyện: " + err.message);
    }
  });

  const renameConversationMutation = useMutation({
    mutationFn: async ({ id, title }: { id: string; title: string }) => {
      return await ragApi.renameConversation(id, title);
    },
    onMutate: async ({ id, title }) => {
      await queryClient.cancelQueries({ queryKey: ['conversations'] });
      const previousConvs = queryClient.getQueryData(['conversations']);
      queryClient.setQueryData(['conversations'], (old: any) =>
        old ? old.map((c: any) => c.id === id ? { ...c, title } : c) : []
      );
      return { previousConvs };
    },
    onError: (err: any, _, context) => {
      if (context?.previousConvs) {
        queryClient.setQueryData(['conversations'], context.previousConvs);
      }
      showError("Không thể đổi tên cuộc trò chuyện: " + err.message);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
    }
  });

  const deleteConversationMutation = useMutation({
    mutationFn: async (id: string) => {
      return await ragApi.deleteConversation(id);
    },
    onMutate: async (id) => {
      await queryClient.cancelQueries({ queryKey: ['conversations'] });
      const previousConvs = queryClient.getQueryData(['conversations']);
      queryClient.setQueryData(['conversations'], (old: any) =>
        old ? old.filter((c: any) => c.id !== id) : []
      );
      return { previousConvs };
    },
    onSuccess: (_, deletedId) => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      if (activeConversationId === deletedId) {
        setActiveConversationId(null);
        setMessages([]);
        setActiveCitations([]);
      }
      showSuccess("Đã xóa cuộc trò chuyện");
    },
    onError: (err: any, _, context) => {
      if (context?.previousConvs) {
        queryClient.setQueryData(['conversations'], context.previousConvs);
      }
      showError("Không thể xóa cuộc trò chuyện: " + err.message);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
    }
  });

  // --- Initial Data Load ---
  useEffect(() => {
    loadDocuments();
    loadEmbeddingModels();
  }, []);

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, chatLoading]);

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

  // --- Date Grouping Helper ---
  const groupConversationsByDate = (convList: any[]) => {
    const groups: Record<string, any[]> = {
      today: [],
      yesterday: [],
      last7Days: [],
      last30Days: [],
      older: []
    };
    
    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const startOfYesterday = new Date(startOfToday.getTime() - 24 * 60 * 60 * 1000);
    const startOf7DaysAgo = new Date(startOfToday.getTime() - 7 * 24 * 60 * 60 * 1000);
    const startOf30DaysAgo = new Date(startOfToday.getTime() - 30 * 24 * 60 * 60 * 1000);

    convList.forEach(conv => {
      const updatedDate = new Date(conv.updated_at);
      if (updatedDate >= startOfToday) {
        groups.today.push(conv);
      } else if (updatedDate >= startOfYesterday) {
        groups.yesterday.push(conv);
      } else if (updatedDate >= startOf7DaysAgo) {
        groups.last7Days.push(conv);
      } else if (updatedDate >= startOf30DaysAgo) {
        groups.last30Days.push(conv);
      } else {
        groups.older.push(conv);
      }
    });

    return groups;
  };

  // --- Inline Rename Handlers ---
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
    if (!inputQuery.trim() || chatLoading || selectedDocIds.length === 0) return;

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
            model_used: finalResult.model_used,
            evaluation: finalResult.evaluation,
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

      const activeId = activeConversationId || finalResult.conversation_id;
      if (!activeConversationId && finalResult.conversation_id) {
        setActiveConversationId(finalResult.conversation_id);
      }
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      queryClient.invalidateQueries({ queryKey: ['messages', activeId] });

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
  const renderMessageContent = (msg: RAGMessage, isStreaming: boolean = false) => {
    if (msg.role === 'user') {
      return <p className="whitespace-pre-wrap leading-relaxed font-medium">{msg.content}</p>;
    }

    const text = msg.content;
    if (!msg.citations || msg.citations.length === 0) {
      return (
        <p className="whitespace-pre-wrap leading-relaxed">
          {text}
          {isStreaming && (
            <motion.span
              animate={{ opacity: [1, 0, 1] }}
              transition={{ repeat: Infinity, duration: 0.8 }}
              className="inline-block w-1.5 h-3.5 ml-1 bg-blue-500 align-middle"
            />
          )}
        </p>
      );
    }

    // Split text by reference blocks [1], [2] to inject clickable superscript motion buttons
    const parts = text.split(/(\[\d+\])/g);
    
    return (
      <div className="space-y-3">
        <p className="whitespace-pre-wrap leading-relaxed">
          {parts.map((part: string, i: number) => {
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
          {isStreaming && (
            <motion.span
              animate={{ opacity: [1, 0, 1] }}
              transition={{ repeat: Infinity, duration: 0.8 }}
              className="inline-block w-1.5 h-3.5 ml-1 bg-blue-500 align-middle"
            />
          )}
        </p>
        
        <div className="pt-2.5 border-t border-[var(--border)]/60 mt-3 opacity-90">
          <p className="text-[10px] font-bold text-[var(--text-muted)] uppercase tracking-wider mb-1.5 flex items-center gap-1">
            <ShieldCheck size={11} className="text-emerald-500" /> Nguồn tham chiếu tài liệu:
          </p>
          <div className="flex flex-wrap gap-1.5">
            {msg.citations.map((cite: RAGCitation, i: number) => (
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
          LEFT PANEL: Documents Ingestion & Scope Select (Collapsible)
          ───────────────────────────────────────────────────────────── */}
      <AnimatePresence initial={false}>
        {showLeftPanel && (
          <motion.section
            key="left-panel"
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: isMobile ? '100%' : 320, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="bg-[var(--surface-elevated)] border border-[var(--border)] rounded-2xl flex flex-col overflow-hidden shrink-0 shadow-sm shadow-black/5"
          >
            {/* Tab Switcher Header */}
            <div className="flex border-b border-[var(--border)] bg-[var(--surface-muted)] p-1 gap-1">
              <button
                onClick={() => setLeftTab('history')}
                className={`flex-1 py-2 px-3 text-xs font-bold rounded-xl flex items-center justify-center gap-1.5 transition-all ${
                  leftTab === 'history'
                    ? 'bg-[var(--surface-elevated)] text-[var(--text)] shadow-sm'
                    : 'text-[var(--text-muted)] hover:bg-[var(--surface-inset)]'
                }`}
              >
                <MessageSquare size={14} />
                Hội thoại
              </button>
              <button
                onClick={() => setLeftTab('documents')}
                className={`flex-1 py-2 px-3 text-xs font-bold rounded-xl flex items-center justify-center gap-1.5 transition-all ${
                  leftTab === 'documents'
                    ? 'bg-[var(--surface-elevated)] text-[var(--text)] shadow-sm'
                    : 'text-[var(--text-muted)] hover:bg-[var(--surface-inset)]'
                }`}
              >
                <FileText size={14} />
                Tài liệu ({documents.length})
              </button>
            </div>

            {leftTab === 'history' ? (
              /* HISTORY TAB CONTENT */
              <div className="flex-1 flex flex-col overflow-hidden p-4 space-y-4">
                {/* New Chat Button */}
                <button
                  onClick={() => createConversationMutation.mutate()}
                  disabled={createConversationMutation.isPending}
                  className="w-full py-2.5 px-4 rounded-xl text-xs font-bold bg-blue-600 hover:bg-blue-700 text-white flex items-center justify-center gap-2 shadow-sm transition-all disabled:opacity-50 cursor-pointer"
                >
                  {createConversationMutation.isPending ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Plus size={14} />
                  )}
                  Cuộc trò chuyện mới
                </button>

                {/* Real-time Search Input */}
                <div className="relative">
                  <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-faint)]" />
                  <input
                    type="text"
                    placeholder="Tìm cuộc trò chuyện..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-9 pr-8 py-2 text-xs rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all text-[var(--text)] outline-none"
                  />
                  {searchQuery && (
                    <button
                      onClick={() => setSearchQuery('')}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] text-[var(--text-faint)] hover:text-[var(--text)] font-semibold"
                    >
                      Xóa
                    </button>
                  )}
                </div>

                {/* Conversations List grouped by time */}
                <div className="flex-1 overflow-y-auto pr-1 space-y-4">
                  {isConvsLoading && conversationsData.length === 0 ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 size={16} className="animate-spin text-blue-500" />
                    </div>
                  ) : conversationsData.length === 0 ? (
                    <div className="text-center text-xs text-[var(--text-faint)] py-8">
                      Không tìm thấy cuộc trò chuyện nào.
                    </div>
                  ) : (
                    (() => {
                      const groups = groupConversationsByDate(conversationsData);
                      const groupKeys: Array<{ key: keyof typeof groups; label: string }> = [
                        { key: 'today', label: 'Hôm nay' },
                        { key: 'yesterday', label: 'Hôm qua' },
                        { key: 'last7Days', label: '7 ngày gần nhất' },
                        { key: 'last30Days', label: '30 ngày gần nhất' },
                        { key: 'older', label: 'Cũ hơn' }
                      ];

                      const formatUpdateTime = (dateStr: string) => {
                        try {
                          const d = new Date(dateStr);
                          const now = new Date();
                          if (d.toDateString() === now.toDateString()) {
                            return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                          }
                          if (d.getFullYear() === now.getFullYear()) {
                            return `${d.getDate()}/${d.getMonth() + 1}`;
                          }
                          return `${d.getDate()}/${d.getMonth() + 1}/${d.getFullYear()}`;
                        } catch {
                          return "";
                        }
                      };

                      return groupKeys.map(({ key, label }) => {
                        const items = groups[key];
                        if (!items || items.length === 0) return null;

                        return (
                          <div key={key} className="space-y-1.5">
                            <h3 className="text-[10px] font-bold text-[var(--text-faint)] uppercase tracking-wider px-2">
                              {label}
                            </h3>
                            <div className="space-y-1">
                              {items.map((conv: any) => {
                                const isSelected = activeConversationId === conv.id;
                                const isEditing = editingConvId === conv.id;

                                if (isEditing) {
                                  return (
                                    <div
                                      key={conv.id}
                                      className="flex items-center gap-1.5 w-full p-2 bg-[var(--surface-inset)] rounded-xl border border-blue-500"
                                    >
                                      <input
                                        type="text"
                                        value={editTitleVal}
                                        onChange={(e) => setEditTitleVal(e.target.value)}
                                        onKeyDown={(e) => {
                                          if (e.key === 'Enter') handleRenameSubmit(conv.id);
                                          if (e.key === 'Escape') setEditingConvId(null);
                                        }}
                                        className="flex-1 text-xs bg-transparent border-none focus:outline-none text-[var(--text)] p-0"
                                        autoFocus
                                      />
                                      <button
                                        onClick={() => handleRenameSubmit(conv.id)}
                                        className="text-emerald-500 hover:text-emerald-600 transition-colors cursor-pointer"
                                        title="Lưu"
                                      >
                                        <CheckCircle2 size={14} />
                                      </button>
                                      <button
                                        onClick={() => setEditingConvId(null)}
                                        className="text-red-500 hover:text-red-600 transition-colors cursor-pointer"
                                        title="Hủy"
                                      >
                                        <Trash2 size={14} className="rotate-45" />
                                      </button>
                                    </div>
                                  );
                                }

                                return (
                                  <div
                                    key={conv.id}
                                    className={`group relative flex items-center justify-between p-2.5 rounded-xl border transition-all cursor-pointer ${
                                      isSelected
                                        ? 'border-blue-500/30 bg-blue-500/5 text-blue-600 dark:text-blue-400'
                                        : 'border-[var(--border)] hover:bg-[var(--surface-muted)] text-[var(--text-secondary)]'
                                    }`}
                                    onClick={() => {
                                      if (!chatLoading) {
                                        setActiveConversationId(conv.id);
                                      }
                                    }}
                                  >
                                    <div className="flex flex-col min-w-0 pr-12">
                                      <p className={`text-xs font-semibold truncate ${isSelected ? 'text-blue-600 dark:text-blue-400' : 'text-[var(--text)]'}`}>
                                        {conv.title}
                                      </p>
                                      <p className="text-[9px] text-[var(--text-faint)] mt-0.5">
                                        Cập nhật: {formatUpdateTime(conv.updated_at)}
                                      </p>
                                    </div>
                                    
                                    {/* Hover Action Buttons */}
                                    <div className="absolute right-2 top-1/2 -translate-y-1/2 hidden group-hover:flex items-center gap-1.5 bg-gradient-to-l from-[var(--surface-elevated)] via-[var(--surface-elevated)] to-transparent pl-4 py-1.5">
                                      <button
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          handleStartRename(conv.id, conv.title);
                                        }}
                                        className="p-1 rounded hover:bg-[var(--surface-inset)] text-[var(--text-muted)] hover:text-blue-500 transition-colors cursor-pointer"
                                        title="Đổi tên"
                                      >
                                        <Edit2 size={13} />
                                      </button>
                                      <button
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          setDeleteTargetId(conv.id);
                                          setDeleteTargetTitle(conv.title);
                                        }}
                                        className="p-1 rounded hover:bg-[var(--surface-inset)] text-[var(--text-muted)] hover:text-red-500 transition-colors cursor-pointer"
                                        title="Xóa"
                                      >
                                        <Trash2 size={13} />
                                      </button>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        );
                      });
                    })()
                  )}
                </div>
              </div>
            ) : (
              /* DOCUMENTS TAB CONTENT */
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

                    <button
                      type="submit"
                      disabled={!uploadFile || uploading}
                      className="ui-btn-primary w-full text-xs py-2 gap-1.5 cursor-pointer"
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
            )}
          </motion.section>
        )}
      </AnimatePresence>

      {/* ─────────────────────────────────────────────────────────────
          MIDDLE PANEL: Streaming Chat & Session Manager
          ───────────────────────────────────────────────────────────── */}
      <section className="flex-1 bg-[var(--surface-elevated)] border border-[var(--border)] rounded-2xl flex flex-col overflow-hidden shadow-sm shadow-black/5 relative">
        {/* Chat Header (Glassmorphism & Collapsible buttons) */}
        <div className="p-4 border-b border-[var(--border)] bg-[var(--surface-muted)]/70 backdrop-blur-md flex items-center justify-between z-10">
          <div className="flex items-center gap-3">
            {/* Toggle Left Panel */}
            <button
              onClick={() => setShowLeftPanel(!showLeftPanel)}
              className="p-1.5 rounded-lg hover:bg-[var(--surface-inset)] border border-[var(--border)]/60 text-[var(--text-secondary)] hover:text-[var(--text)] transition-colors cursor-pointer"
              title={showLeftPanel ? "Ẩn danh sách tài liệu" : "Hiện danh sách tài liệu"}
            >
              {showLeftPanel ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
            </button>

            {/* Active Conversation Title */}
            <h1 className="text-sm font-bold text-[var(--text)] truncate max-w-[150px] md:max-w-[280px]">
              {activeConversationId
                ? conversationsData.find((c: any) => c.id === activeConversationId)?.title || "Cuộc trò chuyện"
                : "Cuộc trò chuyện mới"}
            </h1>

            <span className="text-xs text-[var(--text-muted)] hidden md:inline-flex items-center gap-1.5 bg-[var(--surface-inset)] px-2.5 py-1 rounded-lg">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              Đang nhắm {selectedDocIds.length} tài liệu
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleStartNewChat}
              disabled={chatLoading}
              className="ui-btn-secondary py-1.5 px-3 text-xs gap-1 h-8 bg-blue-50 dark:bg-blue-900/10 text-blue-600 dark:text-blue-400 border border-blue-200/50 dark:border-blue-800/40 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Plus size={13} /> Chat mới
            </button>

            {/* Toggle Right Panel */}
            <button
              onClick={() => setShowRightPanel(!showRightPanel)}
              className="p-1.5 rounded-lg hover:bg-[var(--surface-inset)] border border-[var(--border)]/60 text-[var(--text-secondary)] hover:text-[var(--text)] transition-colors cursor-pointer"
              title={showRightPanel ? "Ẩn dẫn nguồn tham chiếu" : "Hiện dẫn nguồn tham chiếu"}
            >
              {showRightPanel ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
            </button>
          </div>
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
                      t('mainIssues'),
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
                      {/* Sub-label showing confidence as a premium progress bar */}
                      {!isUser && (msg.confidence !== undefined && msg.confidence !== null) && (
                        <div className="flex flex-col gap-1 w-full max-w-[200px] mt-1 mb-1.5 pl-1">
                          <div className="flex items-center justify-between text-[10px] font-semibold text-[var(--text-secondary)]">
                            <span className="flex items-center gap-1">
                              <ShieldCheck size={12} className={msg.confidence > 0.7 ? 'text-emerald-500' : 'text-amber-500'} />
                              Độ tin cậy RAG
                            </span>
                            <span className={msg.confidence > 0.7 ? 'text-emerald-600 dark:text-emerald-400 font-bold' : 'text-amber-600 dark:text-amber-400 font-bold'}>
                              {Math.round(msg.confidence * 100)}%
                            </span>
                          </div>
                          <div className="w-full h-1.5 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${msg.confidence * 100}%` }}
                              transition={{ duration: 0.8, ease: "easeOut" }}
                              className={`h-full rounded-full ${
                                msg.confidence > 0.7
                                  ? 'bg-gradient-to-r from-emerald-400 to-teal-500'
                                  : 'bg-gradient-to-r from-amber-400 to-orange-500'
                              }`}
                            />
                          </div>
                           {msg.model_used && (
                            <div className="text-[9px] text-[var(--text-faint)] font-semibold flex items-center gap-1 mt-0.5">
                              <span>Mô hình:</span>
                              <span className="text-blue-600 dark:text-blue-400 font-mono">
                                {formatModelName(msg.model_used)}
                              </span>
                            </div>
                          )}
                          {msg.evaluation && (
                            <div className="mt-2 pt-2 border-t border-[var(--border)]/40 flex flex-col gap-1.5 text-[10px] w-full max-w-[240px]">
                              <div className="flex items-center justify-between font-semibold">
                                <span className="text-[var(--text-muted)]">Rác / Hoang đường:</span>
                                <span className={`px-1.5 py-0.5 rounded text-[9px] font-extrabold uppercase ${
                                  msg.evaluation.hallucination_risk === 'low'
                                    ? 'bg-emerald-500/10 text-emerald-500'
                                    : msg.evaluation.hallucination_risk === 'medium'
                                    ? 'bg-amber-500/10 text-amber-500'
                                    : 'bg-red-500/10 text-red-500'
                                }`}>
                                  {msg.evaluation.hallucination_risk === 'low' ? 'Thấp' : msg.evaluation.hallucination_risk === 'medium' ? 'Trung bình' : 'Cao'}
                                </span>
                              </div>
                              <div className="space-y-1">
                                <div className="flex items-center justify-between text-[9px]">
                                  <span className="text-[var(--text-faint)]">Nhất quán (Consistency):</span>
                                  <span className="font-bold text-[var(--text-secondary)]">{Math.round(msg.evaluation.consistency_score * 100)}%</span>
                                </div>
                                <div className="flex items-center justify-between text-[9px]">
                                  <span className="text-[var(--text-faint)]">Đo bao phủ nguồn (Grounding):</span>
                                  <span className="font-bold text-[var(--text-secondary)]">{Math.round(msg.evaluation.grounding_coverage * 100)}%</span>
                                </div>
                                <div className="flex items-center justify-between text-[9px]">
                                  <span className="text-[var(--text-faint)]">Độ bao phủ ngữ nghĩa:</span>
                                  <span className="font-bold text-[var(--text-secondary)]">{Math.round(msg.evaluation.semantic_coverage * 100)}%</span>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      <div className={`p-4 rounded-2xl text-sm leading-relaxed border shadow-sm ${
                        isUser
                          ? 'bg-gradient-to-r from-indigo-600 to-blue-600 border-indigo-500 text-white rounded-tr-sm shadow-indigo-600/10'
                          : 'bg-[var(--surface-elevated)] border-[var(--border)] text-[var(--text-secondary)] rounded-tl-sm'
                      }`}>
                        {renderMessageContent(msg, !isUser && index === messages.length - 1 && chatLoading)}
                      </div>
                    </div>
                  </motion.div>
                );
              })}

              {/* Bouncing Dots Typing Indicator when generating */}
              {chatLoading && messages[messages.length - 1]?.role !== 'assistant' && (
                <motion.div
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-3 justify-start"
                >
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 text-white text-xs font-bold flex items-center justify-center shrink-0 shadow-md shadow-blue-500/10">
                    <Sparkles size={14} />
                  </div>
                  <div className="p-4 rounded-2xl bg-[var(--surface-elevated)] border border-[var(--border)] text-[var(--text-secondary)] rounded-tl-sm flex flex-col gap-2 shadow-sm max-w-xs">
                    <div className="text-xs font-bold text-blue-500 flex items-center gap-1.5">
                      <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
                      </span>
                      AI đang tìm kiếm & suy nghĩ
                    </div>
                    <div className="flex items-center gap-1.5 py-1">
                      {[0, 1, 2].map((dotIndex) => (
                        <motion.span
                          key={dotIndex}
                          animate={{ y: [0, -6, 0] }}
                          transition={{
                            repeat: Infinity,
                            duration: 0.6,
                            delay: dotIndex * 0.15,
                            ease: "easeInOut"
                          }}
                          className="w-2.5 h-2.5 rounded-full bg-blue-500/80 inline-block"
                        />
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Bar (Glassmorphism & Responsive layout) */}
        <div className="p-4 border-t border-[var(--border)] bg-[var(--surface-muted)]/70 backdrop-blur-md">
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
              className="ui-input flex-1 bg-[var(--surface-elevated)]/60 backdrop-blur-sm border-[var(--border)] text-sm rounded-xl py-3 px-4 shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all"
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
          RIGHT PANEL: Citations Grounds & Parameters Settings (Collapsible)
          ───────────────────────────────────────────────────────────── */}
      <AnimatePresence initial={false}>
        {showRightPanel && (
          <motion.section
            key="right-panel"
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: isMobile ? '100%' : 320, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="bg-[var(--surface-elevated)] border border-[var(--border)] rounded-2xl flex flex-col overflow-hidden shrink-0 shadow-sm shadow-black/5"
          >
            {/* Right Panel Header */}
            <div className="p-4 border-b border-[var(--border)] bg-[var(--surface-muted)] flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ShieldCheck size={16} className="text-emerald-500 animate-pulse" />
                <h2 className="text-sm font-bold text-[var(--text)]">Dẫn nguồn tham chiếu</h2>
              </div>
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 font-bold font-mono">
                {activeCitations.length} đoạn
              </span>
            </div>

            {/* Citations Content list */}
            <div className="flex-1 overflow-y-auto p-4">
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
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      {/* Floating Panel Reopen Buttons */}
      <AnimatePresence>
        {!showLeftPanel && (
          <motion.button
            key="restore-left-btn"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowLeftPanel(true)}
            className="absolute left-2 top-1/2 -translate-y-1/2 z-40 w-10 h-10 rounded-full bg-blue-600 text-white flex items-center justify-center shadow-lg hover:bg-blue-700 transition-colors border border-blue-500 cursor-pointer"
            title="Hiện danh sách tài liệu"
          >
            <PanelLeftOpen size={18} />
          </motion.button>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {!showRightPanel && (
          <motion.button
            key="restore-right-btn"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowRightPanel(true)}
            className="absolute right-2 top-1/2 -translate-y-1/2 z-40 w-10 h-10 rounded-full bg-emerald-600 text-white flex items-center justify-center shadow-lg hover:bg-emerald-700 transition-colors border border-emerald-500 cursor-pointer"
            title="Hiện dẫn nguồn tham chiếu"
          >
            <PanelRightOpen size={18} />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Delete Confirmation Modal */}
      <AnimatePresence>
        {deleteTargetId && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-[var(--surface-elevated)] border border-[var(--border)] rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-4"
            >
              <div className="flex items-center gap-3 text-red-500">
                <AlertTriangle size={24} />
                <h3 className="text-base font-bold text-[var(--text)]">Xóa cuộc trò chuyện?</h3>
              </div>
              <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
                Bạn có chắc muốn xóa cuộc trò chuyện <span className="font-semibold text-[var(--text)]">"{deleteTargetTitle}"</span>? Toàn bộ lịch sử tin nhắn liên quan sẽ bị xóa vĩnh viễn và không thể khôi phục.
              </p>
              <div className="flex items-center justify-end gap-3 pt-2">
                <button
                  onClick={() => {
                    setDeleteTargetId(null);
                    setDeleteTargetTitle('');
                  }}
                  className="px-4 py-2 rounded-xl text-xs font-bold bg-[var(--surface-inset)] hover:bg-[var(--surface-muted)] text-[var(--text)] border border-[var(--border)] transition-colors cursor-pointer"
                >
                  Hủy
                </button>
                <button
                  onClick={() => {
                    if (deleteTargetId) {
                      deleteConversationMutation.mutate(deleteTargetId);
                      setDeleteTargetId(null);
                      setDeleteTargetTitle('');
                    }
                  }}
                  disabled={deleteConversationMutation.isPending}
                  className="px-4 py-2 rounded-xl text-xs font-bold bg-red-600 hover:bg-red-700 text-white shadow-md shadow-red-600/10 transition-colors flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                >
                  {deleteConversationMutation.isPending && <Loader2 size={13} className="animate-spin" />}
                  Xóa
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

    </div>
  );
}
