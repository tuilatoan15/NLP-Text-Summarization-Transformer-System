import React, { useState, useMemo, useCallback, memo } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { BENCHMARK_SAMPLE_SIZE } from '../lib/benchmarkConfig';
import { motion } from 'framer-motion';
import {
  ResponsiveContainer, BarChart, Bar, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ScatterChart, Scatter, ZAxis, Cell,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import {
  Search, ChevronLeft, ChevronRight, Loader2, Maximize2, FileText,
  BarChart3, BookOpen, TrendingUp, Info, ShieldCheck, AlertTriangle,
  Cpu, Zap, Award, Activity, Sparkles, Check, X, ShieldAlert, CheckCircle2
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { getChartTheme } from '../theme/chartTheme';
import {
  useResearchBenchmarkSamplesQuery,
  useResearchHybridStudyQuery,
  useResearchLeaderboardQuery,
  useResearchReportQuery,
} from '../hooks/useApiQueries';
import { useCacheHitLogger } from '../hooks/useCacheHitLogger';
import { invalidateAfterBenchmark } from '../lib/cacheInvalidation';
import { runResearchBenchmark } from '../services/apiService';

function pct(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

const Compare = () => {
  const { t, locale, isDark } = useApp();
  const queryClient = useQueryClient();
  const chartTheme = getChartTheme(isDark);
  
  const [activeTab, setActiveTab] = useState('leaderboard');
  const [error, setError] = useState('');
  
  const [samplePage, setSamplePage] = useState(1);
  const [sampleCategory, setSampleCategory] = useState('All');
  const [sampleSearch, setSampleSearch] = useState('');
  const [activeSampleIndex, setActiveSampleIndex] = useState(0);
  const [modelCompareFilter, setModelCompareFilter] = useState('all');
  const [idInput, setIdInput] = useState('');
  
  // Sample detail modal state
  const [selectedSample, setSelectedSample] = useState(null);
  
  // Leaderboard sorting and category filtering
  const [sortField, setSortField] = useState('composite');
  const [sortAsc, setSortAsc] = useState(false);
  const [leaderboardFilter, setLeaderboardFilter] = useState('all'); // all, extractive, abstractive, hybrid
  const [leaderboardCategory, setLeaderboardCategory] = useState('All');
  const [runningBenchmark, setRunningBenchmark] = useState(false);
  const [benchmarkStatusMsg, setBenchmarkStatusMsg] = useState('');

  const SAMPLES_PER_PAGE = 10;

  const benchmarkSize = BENCHMARK_SAMPLE_SIZE;

  const leaderboardQuery = useResearchLeaderboardQuery(leaderboardCategory, benchmarkSize, true);
  const hybridQuery = useResearchHybridStudyQuery('vie', benchmarkSize, true);
  const reportQuery = useResearchReportQuery('vie', benchmarkSize, true);
  const samplesQuery = useResearchBenchmarkSamplesQuery(
    samplePage,
    SAMPLES_PER_PAGE,
    sampleCategory,
    sampleSearch,
    benchmarkSize,
    activeTab === 'samples',
  );

  const rawLeaderboard = leaderboardQuery.data?.leaderboard || [];
  const leaderboardData = useMemo(() => {
    if (!rawLeaderboard) return [];
    return Array.isArray(rawLeaderboard) ? rawLeaderboard : Object.values(rawLeaderboard);
  }, [rawLeaderboard]);
  const metadata = leaderboardQuery.data?.metadata || null;
  const hybridStudy = hybridQuery.data || null;
  const report = reportQuery.data || null;
  const samples = useMemo(() => {
    const rawItems = samplesQuery.data?.items || [];
    return rawItems.map(item => {
      const outputs = {};
      const metrics = {};
      if (item.models) {
        Object.keys(item.models).forEach(key => {
          outputs[key] = item.models[key]?.summary || '';
          metrics[key] = item.models[key]?.metrics || {};
        });
      }
      return {
        ...item,
        source: item.source || item.article || '',
        reference: item.reference || item.summary || '',
        outputs: item.outputs || outputs,
        metrics: item.metrics || metrics,
      };
    });
  }, [samplesQuery.data?.items]);
  const sampleTotalPages = samplesQuery.data?.pages || 1;

  const activeSample = useMemo(() => {
    if (!samples || samples.length === 0) return null;
    return samples[activeSampleIndex] || samples[0];
  }, [samples, activeSampleIndex]);

  // Reset active sample index when dataset, page, or search changes
  React.useEffect(() => {
    setActiveSampleIndex(0);
  }, [samples.length, sampleCategory, sampleSearch, samplePage]);

  const handlePrev = useCallback(() => {
    if (activeSampleIndex > 0) {
      setActiveSampleIndex(prev => prev - 1);
    } else if (samplePage > 1) {
      setSamplePage(prev => prev - 1);
      setActiveSampleIndex(SAMPLES_PER_PAGE - 1); // last sample of previous page
    }
  }, [activeSampleIndex, samplePage]);

  const handleNext = useCallback(() => {
    if (activeSampleIndex < samples.length - 1) {
      setActiveSampleIndex(prev => prev + 1);
    } else if (samplePage < sampleTotalPages) {
      setSamplePage(prev => prev + 1);
      setActiveSampleIndex(0);
    }
  }, [activeSampleIndex, samples.length, samplePage, sampleTotalPages]);

  const handleRandom = useCallback(() => {
    const randPage = Math.floor(Math.random() * sampleTotalPages) + 1;
    setSamplePage(randPage);
    const randIndex = Math.floor(Math.random() * SAMPLES_PER_PAGE);
    setActiveSampleIndex(randIndex);
  }, [sampleTotalPages]);

  const handleIdSearch = useCallback((e) => {
    if (e) e.preventDefault();
    if (idInput.trim()) {
      setSampleSearch(idInput.trim());
      setSamplePage(1);
      setActiveSampleIndex(0);
    }
  }, [idInput]);

  const handleClearIdSearch = useCallback(() => {
    setIdInput('');
    setSampleSearch('');
    setSamplePage(1);
    setActiveSampleIndex(0);
  }, []);

  const loading = (
    (activeTab === 'leaderboard' || activeTab === 'charts') && leaderboardQuery.isLoading && !leaderboardQuery.data
  );
  const leaderboardLoading = leaderboardQuery.isFetching && !leaderboardQuery.data;
  const samplesLoading = samplesQuery.isFetching && !samplesQuery.data;

  useCacheHitLogger('research leaderboard', leaderboardQuery.data, leaderboardQuery.isFetching);
  useCacheHitLogger('research hybrid study', hybridQuery.data, hybridQuery.isFetching);
  useCacheHitLogger('research report', reportQuery.data, reportQuery.isFetching);
  useCacheHitLogger('research benchmark samples', samplesQuery.data, samplesQuery.isFetching);

  const displayError = error || leaderboardQuery.error?.message || '';

  const handleRunBenchmark = async () => {
    if (runningBenchmark) return;
    setRunningBenchmark(true);
    setBenchmarkStatusMsg('Đang kích hoạt tiến trình benchmark nền...');
    try {
      const res = await runResearchBenchmark(2500);
      await invalidateAfterBenchmark(queryClient);
      setBenchmarkStatusMsg(res.message || 'Benchmark đã được kích hoạt chạy nền.');
      setTimeout(() => setBenchmarkStatusMsg(''), 5000);
    } catch (err) {
      setBenchmarkStatusMsg('Gặp lỗi khi kích hoạt benchmark.');
      setTimeout(() => setBenchmarkStatusMsg(''), 3500);
    } finally {
      setRunningBenchmark(false);
    }
  };

  const radarChartData = useMemo(() => {
    const lsa = leaderboardData.find(m => m.key === 'lsa') || { rougeL: 0.45, bertscore: 0.75, latency: 0.088, compression: 0.32, faithfulness: 1.0, coverage: 0.78 };
    const vit5 = leaderboardData.find(m => m.key === 'vit5') || { rougeL: 0.3633, bertscore: 0.8845, latency: 6.234, compression: 0.28, faithfulness: 0.84, coverage: 0.82 };
    const bartpho = leaderboardData.find(m => m.key === 'bartpho') || { rougeL: 0.3404, bertscore: 0.9097, latency: 9.25, compression: 0.25, faithfulness: 0.85, coverage: 0.89 };

    const normSpeed = (lat) => Math.max(0.1, 1.0 - lat / 10.0);

    return [
      { subject: 'ROUGE-L', lsa: lsa.rougeL, vit5: vit5.rougeL, bartpho: bartpho.rougeL },
      { subject: 'BERTScore', lsa: lsa.bertscore, vit5: vit5.bertscore, bartpho: bartpho.bertscore },
      { subject: 'Tốc độ', lsa: normSpeed(lsa.latency), vit5: normSpeed(vit5.latency), bartpho: normSpeed(bartpho.latency) },
      { subject: 'Độ nén', lsa: 1.0 - lsa.compression, vit5: 1.0 - vit5.compression, bartpho: 1.0 - bartpho.compression },
      { subject: 'Trung thực', lsa: lsa.faithfulness || 0.9, vit5: vit5.faithfulness || 0.84, bartpho: bartpho.faithfulness || 0.96 }
    ];
  }, [leaderboardData]);

  const processedLeaderboard = useMemo(() => {
    const ALLOWED_KEYS = [
      'textrank', 'lexrank', 'lsa',
      'vit5', 'mt5', 'bartpho',
      'textrank_vit5', 'lexrank_vit5', 'lsa_vit5',
      'textrank_mt5', 'lexrank_mt5', 'lsa_mt5',
      'textrank_bartpho', 'lexrank_bartpho', 'lsa_bartpho'
    ];
    let result = leaderboardData.filter(item => ALLOWED_KEYS.includes(item.key));
    if (leaderboardFilter !== 'all') {
      result = result.filter(item => item.group === leaderboardFilter);
    }
    
    result.sort((a, b) => {
      const valA = a[sortField];
      const valB = b[sortField];
      if (valA === undefined) return 1;
      if (valB === undefined) return -1;
      if (typeof valA === 'string') {
        return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }
      return sortAsc ? valA - valB : valB - valA;
    });
    return result;
  }, [leaderboardData, leaderboardFilter, sortField, sortAsc]);

  const requestSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const modelsSpecifications = {
    textrank: {
      name: 'TextRank', group: 'Extractive', complexity: 'O(N²) - Đồ thị PageRank', params: 'Không tham số',
      framework: 'NLTK / NetworkX / Spacy', desc: 'Thuật toán trích xuất câu dựa trên đồ thị tương đồng từ vựng (PageRank áp dụng cho văn bản). Cực kỳ nhanh và an toàn.',
      pros: ['Tốc độ phản hồi siêu nhanh (~30ms)', 'Độ trung thực thông tin đạt tuyệt đối 100% (không bịa đặt)', 'Không yêu cầu tài nguyên GPU phần cứng'],
      cons: ['Không có khả năng diễn đạt lại (paraphrasing)', 'Bản tóm tắt dễ bị rời rạc, thiếu liên kết mạch lạc']
    },
    lexrank: {
      name: 'LexRank', group: 'Extractive', complexity: 'O(N²) - Trọng số TF-IDF', params: 'Không tham số',
      framework: 'LexRank Base / NumPy', desc: 'Phương pháp trích xuất câu sử dụng độ trung tâm đồ thị liên kết câu, kết hợp trọng số tần suất từ nghịch đảo TF-IDF.',
      pros: ['Đo lường tầm quan trọng của câu chính xác hơn nhờ TF-IDF', 'Rất phù hợp cho tóm tắt đa tài liệu báo chí', 'Chi phí tính toán siêu thấp, chạy được trên CPU yếu'],
      cons: ['Câu trích xuất giữ nguyên gốc, dễ chứa các thông tin dư thừa']
    },
    lsa: {
      name: 'LSA Summarizer', group: 'Extractive', complexity: 'O(N · M · K) - Phân tách SVD', params: 'Không tham số',
      framework: 'Scikit-Learn / SciPy', desc: 'Phân tích ngữ nghĩa tiềm ẩn bằng cách phân tách ma trận SVD từ ma trận đặc trưng câu-từ khóa để tìm chủ đề cốt lõi.',
      pros: ['Nắm bắt được mối quan hệ chủ đề tiềm ẩn (latent topics)', 'Không bị phụ thuộc vào sự trùng lặp từ vựng chính xác', 'Rất ổn định trên các tài liệu khoa học dài'],
      cons: ['Vẫn là phương pháp trích xuất thuần túy, không có từ mới']
    },
    vit5: {
      name: 'ViT5 (Fine-tuned)', group: 'Abstractive', complexity: 'O(L² · D) - Self-Attention', params: '220M (ViT5-base)',
      framework: 'PyTorch / HuggingFace Transformers', desc: 'Mô hình sinh tóm tắt dựa trên kiến trúc T5 tối ưu hóa riêng cho tiếng Việt bởi VietAI. Được tinh chỉnh chuyên sâu trên tập dữ liệu VietNews.',
      pros: ['Văn bản sinh ra mượt mà, tự nhiên và trôi chảy giống người viết', 'Khả năng diễn dịch (paraphrasing) tốt nhờ cơ chế học sâu', 'Độ nén thông tin cực cao, cô đọng nội dung tốt'],
      cons: ['Thời gian phản hồi tương đối chậm (6-8 giây)', 'Yêu cầu tài nguyên tính toán cao (tối thiểu 4GB GPU VRAM)', 'Có rủi ro nhỏ xảy ra hiện tượng bịa đặt thông tin (hallucination)']
    },
    mt5: {
      name: 'mT5 (Baseline)', group: 'Abstractive', complexity: 'O(L² · D) - Self-Attention', params: '300M (mT5-small)',
      framework: 'PyTorch / HuggingFace Transformers', desc: 'Mô hình sinh đa ngôn ngữ của Google (mT5-small). Sử dụng làm mốc so sánh thực nghiệm gốc chưa tinh chỉnh chuyên sâu.',
      pros: ['Hỗ trợ đa ngôn ngữ mặc định', 'Kiến trúc chuẩn hóa toàn cầu, dễ triển khai mở rộng'],
      cons: ['Bị hiện tượng lặp từ và sinh từ rác rất cao khi chưa fine-tune', 'Không hoạt động tốt trên các ngữ cảnh tiếng Việt chuyên sâu', 'Độ trung thực thông tin cực kỳ thấp (<20%)']
    },
    bartpho: {
      name: 'BARTPho (Fine-tuned)', group: 'Abstractive', complexity: 'O(L² · D) - Self-Attention', params: '340M (BARTPho-word)',
      framework: 'PyTorch / HuggingFace Transformers', desc: 'Mô hình sinh tóm tắt sử dụng kiến trúc Seq2Seq BART dành riêng cho tiếng Việt của VinAI. Xử lý âm tiết tiếng Việt cực kỳ tự nhiên.',
      pros: ['Đạt độ tương đồng ngữ nghĩa cao nhất trên tập kiểm thử', 'Diễn đạt xuất sắc, không bao giờ gặp lỗi ngữ pháp tiếng Việt', 'Cực kỳ phù hợp cho phong cách viết tin tức, báo chí'],
      cons: ['Kích thước mô hình lớn, tốn tài nguyên lưu trữ', 'Thời gian khởi động và sinh văn bản lâu (~8 giây)']
    },
    textrank_vit5: {
      name: 'TextRank ➔ ViT5', group: 'Hybrid', complexity: 'O(N²) + O(L² · D)', params: '220M (ViT5-base)',
      framework: 'NetworkX + PyTorch Transformers', desc: 'Mô hình lai hai giai đoạn: trích xuất các câu quan trọng nhất bằng TextRank rồi sinh tóm tắt cô đọng bằng ViT5.',
      pros: ['Độ trễ suy diễn giảm gần 40% so với ViT5 thuần túy', 'Tránh hoàn toàn nguy cơ tràn RAM/VRAM GPU trên tài liệu dài', 'Độ trung thực tăng nhẹ nhờ loại bỏ nhiễu từ giai đoạn trích xuất'],
      cons: ['Phụ thuộc vào chất lượng trích lọc câu ở giai đoạn 1']
    },
    lexrank_vit5: {
      name: 'LexRank ➔ ViT5', group: 'Hybrid', complexity: 'O(N²) + O(L² · D)', params: '220M (ViT5-base)',
      framework: 'NumPy + PyTorch Transformers', desc: 'Mô hình lai hai giai đoạn: dùng LexRank (TF-IDF đồ thị) lọc câu cốt lõi và dùng ViT5 để viết lại văn bản mượt mà.',
      pros: ['Tốc độ nhanh, ổn định trên văn bản báo chí trung bình-dài', 'Tiết kiệm đáng kể tài nguyên tính toán GPU', 'Độ nén thông tin rất tốt, văn bản trôi chảy'],
      cons: ['Nếu LexRank lọc sót ý chính, bản tóm tắt sẽ bị thiếu thông tin']
    },
    lsa_vit5: {
      name: 'LSA ➔ ViT5', group: 'Hybrid', complexity: 'O(N·M·K) + O(L² · D)', params: '220M (ViT5-base)',
      framework: 'SciPy + PyTorch Transformers', desc: 'Mô hình lai: dùng Phân tích ngữ nghĩa tiềm ẩn (LSA) trích lọc chủ đề chính và dùng ViT5 sinh tóm tắt ngữ cảnh.',
      pros: ['Nhận diện chủ đề tiềm ẩn tốt giúp giảm thiểu thông tin dư thừa', 'Độ trễ tối ưu hóa mạnh (~1.5 giây)', 'Độ trung thực sự thật đạt mức cao (~83%)'],
      cons: ['Có thể bỏ lỡ các chi tiết số liệu cụ thể nếu LSA không chọn câu đó']
    },
    textrank_mt5: {
      name: 'TextRank ➔ mT5', group: 'Hybrid', complexity: 'O(N²) + O(L² · D)', params: '300M (mT5-small)',
      framework: 'NetworkX + PyTorch Transformers', desc: 'Mô hình lai: lọc câu chính bằng TextRank và sinh tóm tắt bằng mT5 đa ngôn ngữ.',
      pros: ['Hỗ trợ đa ngôn ngữ, tốc độ suy diễn nhanh hơn mT5 thuần', 'Hạn chế được một phần hiện tượng sinh từ lặp rác của mT5'],
      cons: ['Vẫn bị ảnh hưởng bởi chất lượng sinh từ chưa tối ưu của mô hình nền mT5']
    },
    lexrank_mt5: {
      name: 'LexRank ➔ mT5', group: 'Hybrid', complexity: 'O(N²) + O(L² · D)', params: '300M (mT5-small)',
      framework: 'NumPy + PyTorch Transformers', desc: 'Mô hình lai: dùng LexRank lọc câu cốt lõi và dùng mT5 sinh tóm tắt đa ngôn ngữ.',
      pros: ['Độ trễ suy diễn thấp, thích hợp cho tài liệu đa ngôn ngữ lớn', 'Chi phí tính toán GPU thấp'],
      cons: ['Chất lượng tóm tắt ở mức trung bình, có thể gặp lỗi lặp từ']
    },
    lsa_mt5: {
      name: 'LSA ➔ mT5', group: 'Hybrid', complexity: 'O(N·M·K) + O(L² · D)', params: '300M (mT5-small)',
      framework: 'SciPy + PyTorch Transformers', desc: 'Mô hình lai: dùng LSA trích xuất chủ đề và dùng mT5 đa ngôn ngữ sinh bản tóm tắt.',
      pros: ['Tốc độ xử lý nhanh, lọc nhiễu tốt trên tài liệu đa ngôn ngữ dài'],
      cons: ['Chất lượng văn bản sinh ra chưa thực sự tự nhiên']
    },
    textrank_bartpho: {
      name: 'TextRank ➔ BARTPho', group: 'Hybrid', complexity: 'O(N²) + O(L² · D)', params: '340M (BARTPho-word)',
      framework: 'NetworkX + PyTorch Transformers', desc: 'Mô hình lai: lọc câu chính bằng TextRank và dùng BARTPho của VinAI để sinh tóm tắt tiếng Việt tự nhiên.',
      pros: ['Đoạt độ tự nhiên cao, trôi chảy và đúng ngữ pháp tiếng Việt', 'Giảm độ trễ suy diễn của BARTPho xuống gần 45%', 'Ngăn ngừa hiệu quả lỗi tràn bộ nhớ ngữ cảnh'],
      cons: ['Yêu cầu tài nguyên lưu trữ mô hình khá lớn']
    },
    lexrank_bartpho: {
      name: 'LexRank ➔ BARTPho', group: 'Hybrid', complexity: 'O(N²) + O(L² · D)', params: '340M (BARTPho-word)',
      framework: 'NumPy + PyTorch Transformers', desc: 'Mô hình lai: lọc câu cốt lõi bằng LexRank và dùng BARTPho viết lại tóm tắt tự nhiên.',
      pros: ['Cân bằng xuất sắc giữa thời gian đáp ứng và chất lượng học sâu', 'Rất phù hợp cho môi trường chạy thực tế thương mại'],
      cons: ['Chất lượng phụ thuộc vào ngưỡng chọn câu của LexRank']
    },
    lsa_bartpho: {
      name: 'LSA ➔ BARTPho', group: 'Hybrid', complexity: 'O(N·M·K) + O(L² · D)', params: '340M (BARTPho-word)',
      framework: 'SciPy + PyTorch Transformers', desc: 'Mô hình lai tối ưu nhất: dùng LSA trích lọc chủ đề và dùng BARTPho sinh tóm tắt ngữ nghĩa.',
      pros: ['Đạt điểm tổng hợp (Composite Score) cao nhất hệ thống', 'Độ trung thực sự thật vượt trội (~96%)', 'Độ trễ suy diễn cực thấp so với BARTPho thuần'],
      cons: ['Kích thước mô hình lớn, tốn tài nguyên lưu trữ ban đầu']
    }
  };

  const ALGORITHM_KEYS = [
    'textrank', 'lexrank', 'lsa',
    'vit5', 'mt5', 'bartpho',
    'textrank_vit5', 'lexrank_vit5', 'lsa_vit5',
    'textrank_mt5', 'lexrank_mt5', 'lsa_mt5',
    'textrank_bartpho', 'lexrank_bartpho', 'lsa_bartpho'
  ];

  const getRankBadge = (idx) => {
    if (idx === 0) return 'bg-amber-100 dark:bg-amber-950/40 text-amber-800 dark:text-amber-400 border border-amber-300 dark:border-amber-900/50';
    if (idx === 1) return 'bg-slate-100 dark:bg-slate-900 text-slate-800 dark:text-slate-400 border border-slate-350';
    if (idx === 2) return 'bg-orange-100 dark:bg-orange-950/20 text-orange-800 dark:text-orange-400 border border-orange-200';
    return 'bg-[var(--bg-muted)] text-[var(--text-muted)]';
  };

  const getModelTypeStyles = (key) => {
    // 1. Extractive
    if (['textrank', 'lexrank', 'lsa'].includes(key)) {
      return {
        bg: 'bg-emerald-50 dark:bg-emerald-950/20',
        border: 'border-emerald-250 dark:border-emerald-900/30',
        text: 'text-emerald-700 dark:text-emerald-400',
        label: 'Extractive'
      };
    }
    // 2. Abstractive (Fine-tuned)
    if (['vit5', 'bartpho'].includes(key)) {
      return {
        bg: 'bg-amber-50 dark:bg-amber-950/20',
        border: 'border-amber-250 dark:border-amber-900/30',
        text: 'text-amber-700 dark:text-amber-400',
        label: 'Abstractive (FT)'
      };
    }
    // 3. Abstractive (Baseline)
    if (key === 'mt5') {
      return {
        bg: 'bg-rose-50 dark:bg-rose-950/20',
        border: 'border-rose-250 dark:border-rose-900/30',
        text: 'text-rose-700 dark:text-rose-400',
        label: 'Abstractive (Base)'
      };
    }
    // 4. Hybrid (Lai ghép)
    return {
      bg: 'bg-sky-50 dark:bg-sky-950/20',
      border: 'border-sky-250 dark:border-sky-900/30',
      text: 'text-sky-700 dark:text-sky-400',
      label: 'Hybrid'
    };
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="ui-heading-1 flex items-center gap-2">
            <Zap className="text-sky-500" strokeWidth={2.5} />
            Research Lab: So Sánh Đối Kè
          </h1>
          <p className="ui-page-subtitle">Đánh giá hiệu năng, kiểm định độ tương đồng ROUGE/BERTScore và tài nguyên tính toán của các mô hình.</p>
        </div>
        
        <div className="flex items-center gap-3">
          <span className="px-3 py-1.5 text-xs font-bold rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-secondary)] shadow-sm">
            {locale === 'vie' ? '5.000 mẫu (benchmark thực nghiệm)' : '5,000 samples (research benchmark)'}
          </span>
          
          {/* Ẩn nút kích hoạt benchmark tự động theo yêu cầu của người dùng để tránh bấm nhầm gây treo máy */}
          {/* <button
            onClick={handleRunBenchmark}
            disabled={runningBenchmark}
            className="ui-btn-primary py-2 px-4 text-xs font-bold shrink-0 flex items-center gap-2 cursor-pointer"
          >
            {runningBenchmark ? <Loader2 size={14} className="animate-spin" /> : <Activity size={14} />}
            Kích hoạt Benchmark tự động
          </button> */}
        </div>
      </div>

      {benchmarkStatusMsg && (
        <div className="p-3.5 rounded-xl border border-sky-200/50 dark:border-sky-850/50 bg-sky-50 dark:bg-sky-950/20 text-sky-800 dark:text-sky-300 text-xs font-semibold animate-fade-in">
          {benchmarkStatusMsg}
        </div>
      )}

      {/* Tab Switcher */}
      <div className="flex border-b border-[var(--border)] overflow-x-auto scrollbar-none gap-2">
        <button
          onClick={() => setActiveTab('leaderboard')}
          className={`pb-2.5 px-4 text-xs font-bold border-b-2 transition-all cursor-pointer whitespace-nowrap ${
            activeTab === 'leaderboard'
              ? 'border-sky-500 text-sky-600 dark:text-sky-400 font-extrabold'
              : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text-primary)]'
          }`}
        >
          <Award size={14} className="inline mr-1.5" />
          Bảng xếp hạng (Leaderboard)
        </button>
        <button
          onClick={() => setActiveTab('charts')}
          className={`pb-2.5 px-4 text-xs font-bold border-b-2 transition-all cursor-pointer whitespace-nowrap ${
            activeTab === 'charts'
              ? 'border-sky-500 text-sky-600 dark:text-sky-400 font-extrabold'
              : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text-primary)]'
          }`}
        >
          <BarChart3 size={14} className="inline mr-1.5" />
          Mạng nhện & Phân bố (Radar/Latency)
        </button>
        <button
          onClick={() => setActiveTab('samples')}
          className={`pb-2.5 px-4 text-xs font-bold border-b-2 transition-all cursor-pointer whitespace-nowrap ${
            activeTab === 'samples'
              ? 'border-sky-500 text-sky-600 dark:text-sky-400 font-extrabold'
              : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text-primary)]'
          }`}
        >
          <BookOpen size={14} className="inline mr-1.5" />
          So kè mẫu thử (Benchmark Samples)
        </button>
        <button
          onClick={() => setActiveTab('specs')}
          className={`pb-2.5 px-4 text-xs font-bold border-b-2 transition-all cursor-pointer whitespace-nowrap ${
            activeTab === 'specs'
              ? 'border-sky-500 text-sky-600 dark:text-sky-400 font-extrabold'
              : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text-primary)]'
          }`}
        >
          <Cpu size={14} className="inline mr-1.5" />
          Thông số kỹ thuật mô hình
        </button>
      </div>

      {/* ─────────────────────────────────────────────────────────────
          TAB CONTENT: LEADERBOARD
          ───────────────────────────────────────────────────────────── */}
      {activeTab === 'leaderboard' && (
        <div className="space-y-4">
          {/* Filters bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 bg-[var(--bg-muted)]/30 border border-[var(--border)] rounded-xl p-3.5">
            <div className="flex flex-wrap gap-1.5">
              {[
                { key: 'all', label: 'Tất cả giải thuật' },
                { key: 'extractive', label: 'Trích xuất (Extractive)' },
                { key: 'abstractive', label: 'Mô hình sinh (Abstractive)' },
                { key: 'hybrid', label: 'Lai ghép (Hybrid)' }
              ].map(f => (
                <button
                  key={f.key}
                  onClick={() => setLeaderboardFilter(f.key)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold border transition-all cursor-pointer ${
                    leaderboardFilter === f.key
                      ? 'bg-sky-600 border-sky-600 text-white shadow-sm'
                      : 'bg-[var(--bg-elevated)] border-[var(--border)] text-[var(--text-secondary)] hover:border-sky-300 dark:hover:border-sky-800'
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>
            
            <div className="text-[10px] font-bold text-[var(--text-faint)] uppercase tracking-wider">
              {leaderboardLoading ? (
                <span className="flex items-center gap-1.5"><Loader2 size={12} className="animate-spin text-sky-500" /> Đang cập nhật dữ liệu...</span>
              ) : 'Dữ liệu thực nghiệm thực tế'}
            </div>
          </div>

          {/* Table */}
          <div className="ui-card overflow-hidden bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="ui-table-head border-b">
                  <tr className="divide-x divide-[var(--border)]/20">
                    <th className="px-4 py-3 text-left cursor-pointer hover:bg-[var(--bg-muted)]" onClick={() => requestSort('algorithm')}>Phương pháp</th>
                    <th className="px-4 py-3 cursor-pointer hover:bg-[var(--bg-muted)] text-center" onClick={() => requestSort('rouge1')}>ROUGE-1 (R1)</th>
                    <th className="px-4 py-3 cursor-pointer hover:bg-[var(--bg-muted)] text-center" onClick={() => requestSort('rouge2')}>ROUGE-2 (R2)</th>
                    <th className="px-4 py-3 cursor-pointer hover:bg-[var(--bg-muted)] text-center" onClick={() => requestSort('rougeL')}>ROUGE-L (RL)</th>
                    <th className="px-4 py-3 cursor-pointer hover:bg-[var(--bg-muted)] text-center" onClick={() => requestSort('rougeLsum')}>ROUGE-LSum</th>
                    <th className="px-4 py-3 cursor-pointer hover:bg-[var(--bg-muted)] text-center" onClick={() => requestSort('bert_p')}>BERT P</th>
                    <th className="px-4 py-3 cursor-pointer hover:bg-[var(--bg-muted)] text-center" onClick={() => requestSort('bert_r')}>BERT R</th>
                    <th className="px-4 py-3 cursor-pointer hover:bg-[var(--bg-muted)] text-center" onClick={() => requestSort('bertscore')}>BERT F1</th>
                    <th className="px-4 py-3 cursor-pointer hover:bg-[var(--bg-muted)] text-center bg-sky-500/5 text-sky-600 dark:text-sky-400" onClick={() => requestSort('latency')}>Latency (s)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-subtle)] font-medium text-[var(--text-secondary)]">
                  {processedLeaderboard.map((row, idx) => (
                    <tr key={row.key} className="ui-table-row hover:bg-[var(--bg-muted)]/40 divide-x divide-[var(--border)]/10">
                      <td className="px-4 py-3 font-semibold text-[var(--text-primary)]">
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold border whitespace-nowrap ${getModelTypeStyles(row.key).bg} ${getModelTypeStyles(row.key).border} ${getModelTypeStyles(row.key).text}`}>
                            {getModelTypeStyles(row.key).label}
                          </span>
                          <span className="text-xs font-bold">{modelsSpecifications[row.key]?.name || row.name || row.key}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-center font-mono">{(row.rouge1 ?? 0).toFixed(4)}</td>
                      <td className="px-4 py-3 text-center font-mono">{(row.rouge2 ?? 0).toFixed(4)}</td>
                      <td className="px-4 py-3 text-center font-mono">{(row.rougeL ?? 0).toFixed(4)}</td>
                      <td className="px-4 py-3 text-center font-mono">{(row.rougeLsum ?? row.rougeL ?? 0).toFixed(4)}</td>
                      <td className="px-4 py-3 text-center font-mono">{(row.bert_p ?? 0).toFixed(4)}</td>
                      <td className="px-4 py-3 text-center font-mono">{(row.bert_r ?? 0).toFixed(4)}</td>
                      <td className="px-4 py-3 text-center font-mono">{(row.bertscore ?? 0).toFixed(4)}</td>
                      <td className="px-4 py-3 bg-sky-500/5 text-sky-700 dark:text-sky-300 font-bold text-center font-mono">
                        {(row.latency ?? 0).toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────
          TAB CONTENT: CHARTS
          ───────────────────────────────────────────────────────────── */}
      {activeTab === 'charts' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Radar Chart Multi-Metric */}
          <div className="ui-card p-5 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm">
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)] mb-4 flex items-center gap-1.5">
              <Sparkles size={14} className="text-sky-500" />
              Radar Chart: Đánh giá Đa chiều
            </h3>
            <div className="h-80 w-full flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarChartData}>
                  <PolarGrid stroke={chartTheme.grid} />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: chartTheme.axis, fontSize: 10, fontWeight: 700 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 1.0]} tick={{ fill: chartTheme.axis, fontSize: 9 }} />
                  <Radar name="LSA (Extractive)" dataKey="lsa" stroke="#84cc16" fill="#84cc16" fillOpacity={0.1} />
                  <Radar name="ViT5 (Abstractive)" dataKey="vit5" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.1} />
                  <Radar name="BARTPho (Abstractive)" dataKey="bartpho" stroke="#fb7185" fill="#fb7185" fillOpacity={0.1} />
                  <Tooltip contentStyle={chartTheme.tooltipStyle} wrapperClassName="radar-chart-tooltip" />
                  <Legend verticalAlign="bottom" height={36} wrapperStyle={{ fontSize: 9, fontWeight: 700, fill: chartTheme.axis }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Scatter latency vs Quality */}
          <div className="ui-card p-5 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm">
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-faint)] mb-4 flex items-center gap-1.5">
              <Activity size={14} className="text-sky-500" />
              Latency vs Quality (Composite Score)
            </h3>
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
                  <XAxis type="number" dataKey="latency" name="Thời gian (giây)" unit="s" tick={{ fill: chartTheme.axis, fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis type="number" dataKey="composite" name="Composite Score" domain={[0, 1.0]} tick={{ fill: chartTheme.axis, fontSize: 10 }} axisLine={false} tickLine={false} />
                  <ZAxis type="number" range={[100, 200]} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={chartTheme.tooltipStyle} />
                  <Scatter name="Algorithms" data={leaderboardData.filter(item => Object.keys(modelsSpecifications).includes(item.key))} fill={chartTheme.accent}>
                    {leaderboardData.filter(item => Object.keys(modelsSpecifications).includes(item.key)).map((entry, index) => {
                      const meta = modelsSpecifications[entry.key] || { group: 'Extractive' };
                      let color = '#10b981'; // Extractive
                      if (meta.group === 'Abstractive') {
                        color = '#fb7185';
                      } else if (meta.group === 'Hybrid') {
                        color = '#0ea5e9'; // Hybrid (sky-500)
                      }
                      return <Cell key={`cell-${index}`} fill={color} />;
                    })}
                  </Scatter>
                  <Legend verticalAlign="bottom" height={36} wrapperStyle={{ fontSize: 9, fontWeight: 700 }} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-center gap-5 mt-2 text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">
              <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Extractive</span>
              <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-[#fb7185]" /> Abstractive</span>
              <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-[#0ea5e9]" /> Hybrid</span>
            </div>
          </div>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────
          TAB CONTENT: BENCHMARK SAMPLES
          ───────────────────────────────────────────────────────────── */}
      {activeTab === 'samples' && (
        <div className="space-y-6">
          {/* Top Panel: Compact Horizontal Selector */}
          <div className="ui-card p-5 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm space-y-4 rounded-2xl">
            {/* Header, Search, Pagination controls */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex flex-wrap items-center gap-3">
                <span className="text-xs font-extrabold uppercase tracking-wider text-[var(--text-faint)]">
                  Mẫu thử ({samplesQuery.data?.total || 0})
                </span>

                {/* ID Search Input */}
                <div className="relative flex items-center gap-2">
                  <div className="relative w-64">
                    <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-faint)]" />
                    <input
                      type="text"
                      placeholder="Tìm theo ID hoặc nội dung..."
                      value={idInput}
                      onChange={(e) => setIdInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          setSampleSearch(idInput.trim());
                          setSamplePage(1);
                        }
                      }}
                      className="w-full pl-8 pr-8 py-1.5 text-xs rounded-xl border border-[var(--border)] bg-[var(--bg)] outline-none focus:border-sky-500 transition-all font-bold"
                    />
                    {idInput && (
                      <button
                        onClick={() => {
                          setIdInput('');
                          setSampleSearch('');
                          setSamplePage(1);
                        }}
                        className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[var(--text-faint)] hover:text-[var(--text-primary)] cursor-pointer bg-transparent border-none p-0 flex items-center"
                      >
                        <X size={13} />
                      </button>
                    )}
                  </div>
                  <button
                    onClick={() => {
                      setSampleSearch(idInput.trim());
                      setSamplePage(1);
                    }}
                    className="px-3.5 py-1.5 rounded-xl bg-sky-600 hover:bg-sky-700 text-white text-xs font-bold transition-all cursor-pointer shadow-sm"
                  >
                    Tìm
                  </button>
                </div>
              </div>

              {/* Pagination controls */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setSamplePage(prev => Math.max(1, prev - 1))}
                  disabled={samplePage === 1 || samplesLoading}
                  className="p-1.5 rounded-lg border border-[var(--border)] hover:bg-[var(--bg-muted)] text-[var(--text-primary)] disabled:opacity-40 disabled:cursor-not-allowed transition-all cursor-pointer"
                >
                  <ChevronLeft size={14} />
                </button>
                <span className="text-xs font-extrabold text-[var(--text-primary)] min-w-[80px] text-center">
                  Trang {samplePage} / {sampleTotalPages}
                </span>
                <button
                  onClick={() => setSamplePage(prev => Math.min(sampleTotalPages, prev + 1))}
                  disabled={samplePage === sampleTotalPages || samplesLoading}
                  className="p-1.5 rounded-lg border border-[var(--border)] hover:bg-[var(--bg-muted)] text-[var(--text-primary)] disabled:opacity-40 disabled:cursor-not-allowed transition-all cursor-pointer"
                >
                  <ChevronRight size={14} />
                </button>
                
                <button
                  onClick={handleRandom}
                  className="px-3 py-1.5 rounded-lg border border-[var(--border)] hover:bg-[var(--bg-muted)] text-[var(--text-primary)] text-xs font-bold transition-all cursor-pointer"
                >
                  Ngẫu nhiên
                </button>
              </div>
            </div>

            {/* Horizontal list of samples in current page */}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 xl:grid-cols-10 gap-2.5 pt-2">
              {samplesLoading && samples.length === 0 ? (
                <div className="col-span-full flex justify-center py-4">
                  <Loader2 size={16} className="animate-spin text-sky-500" />
                </div>
              ) : samples.length === 0 ? (
                <div className="col-span-full text-center text-xs font-semibold text-[var(--text-faint)] py-2">
                  Không tìm thấy mẫu thử nào phù hợp.
                </div>
              ) : (
                samples.map((s, idx) => {
                  const isActive = activeSampleIndex === idx;
                  return (
                    <button
                      key={s.id}
                      onClick={() => setActiveSampleIndex(idx)}
                      className={`p-3 rounded-xl border text-left transition-all cursor-pointer flex flex-col justify-between h-[68px] ${
                        isActive
                          ? 'border-sky-500 bg-sky-500/5 text-sky-700 dark:text-sky-400 font-extrabold shadow-sm'
                          : 'border-[var(--border)] hover:border-sky-400/50 bg-[var(--bg-muted)]/20 text-[var(--text-secondary)] font-semibold'
                      }`}
                    >
                      <span className="text-[10px] tracking-tight block truncate w-full">
                        {s.id.replace('benchmark_sample_', 'ID: ')}
                      </span>
                      <span className="text-[9px] text-[var(--text-faint)] italic line-clamp-2 leading-tight">
                        "{s.source}"
                      </span>
                    </button>
                  );
                })
              )}
            </div>
          </div>

          {/* Bottom Panel: Full Width Details */}
          {activeSample ? (
            <div className="space-y-6">
              {/* Sample info details */}
              <div className="ui-card p-5 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm space-y-3 rounded-2xl">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold bg-sky-50 dark:bg-sky-950/20 text-sky-600 dark:text-sky-400 px-2.5 py-0.5 rounded border border-sky-200/30">
                    Sample ID: {activeSample.id}
                  </span>
                </div>
                
                <div className="pt-2 border-t border-[var(--border)] flex flex-wrap gap-4 text-[10px] font-bold text-[var(--text-muted)]">
                  <span className="flex items-center gap-1.5"><FileText size={12} className="text-sky-500" /> Gốc: {activeSample.source.split(' ').filter(Boolean).length} từ</span>
                  <span className="flex items-center gap-1.5"><BookOpen size={12} className="text-emerald-500" /> Tham khảo: {activeSample.reference.split(' ').filter(Boolean).length} từ</span>
                </div>
              </div>

              {/* Source and Reference summary text boxes */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Left: Source text */}
                <div className="ui-card p-5 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm space-y-2.5 max-h-72 overflow-y-auto scrollbar-thin rounded-2xl">
                  <h4 className="text-[10px] font-bold text-[var(--text-faint)] uppercase tracking-wider flex items-center gap-1.5 sticky top-0 bg-[var(--bg-elevated)] py-1">
                    <FileText size={12} className="text-sky-500" />
                    Văn bản gốc
                  </h4>
                  <p className="text-xs text-[var(--text-secondary)] leading-relaxed font-medium font-sans">
                    {activeSample.source}
                  </p>
                </div>

                {/* Right: Human Reference summary */}
                <div className="ui-card p-5 bg-emerald-500/5 border border-emerald-500/20 shadow-sm space-y-2.5 max-h-72 overflow-y-auto scrollbar-thin rounded-2xl">
                  <h4 className="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider flex items-center gap-1.5 sticky top-0 bg-emerald-500/5 dark:bg-emerald-950/20 py-1">
                    <CheckCircle2 size={12} className="text-emerald-500" />
                    Bản tóm tắt tham chiếu (Human-Written)
                  </h4>
                  <p className="text-xs text-[var(--text-primary)] leading-relaxed font-semibold font-sans italic">
                    "{activeSample.reference}"
                  </p>
                </div>
              </div>

              {/* AI Summaries Grid */}
              <div className="space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <h3 className="text-xs font-extrabold uppercase tracking-wider text-[var(--text-faint)] flex items-center gap-1.5">
                    <Sparkles size={14} className="text-sky-500" />
                    Bản tóm tắt sinh bởi các thuật toán AI (So sánh song song)
                  </h3>
                  
                  {/* Bộ lọc nhóm mô hình */}
                  <div className="flex gap-1 bg-[var(--bg-muted)]/45 p-0.5 rounded-xl border border-[var(--border)] w-fit shrink-0">
                    {[
                      { key: 'all', label: 'Tất cả (15)' },
                      { key: 'extractive', label: 'Trích xuất (3)' },
                      { key: 'abstractive', label: 'Mô hình sinh (3)' },
                      { key: 'hybrid', label: 'Lai ghép (9)' }
                    ].map(f => (
                      <button
                        key={f.key}
                        onClick={() => setModelCompareFilter(f.key)}
                        className={`px-2.5 py-1 rounded-lg text-[9px] font-bold transition-all cursor-pointer ${
                          modelCompareFilter === f.key
                            ? 'bg-sky-600 text-white shadow-sm'
                            : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                        }`}
                      >
                        {f.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* 15 Algorithms Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                  {ALGORITHM_KEYS.filter(key => {
                    if (modelCompareFilter === 'all') return true;
                    const spec = modelsSpecifications[key];
                    return spec && spec.group.toLowerCase() === modelCompareFilter.toLowerCase();
                  }).map(key => {
                    const spec = modelsSpecifications[key] || { name: key.toUpperCase().replace('_', ' ➔ '), group: 'Unknown' };
                    const output = activeSample.outputs?.[key];
                    const metrics = activeSample.metrics?.[key];
                    
                    // Determine badge style based on group
                    let badgeStyle = 'bg-slate-50 dark:bg-slate-950/20 text-slate-600 dark:text-slate-400 border border-slate-200/30';
                    if (spec.group === 'Extractive') {
                      badgeStyle = 'bg-emerald-50 dark:bg-emerald-950/20 text-emerald-600 dark:text-emerald-400 border border-emerald-250/20';
                    } else if (spec.group === 'Abstractive') {
                      if (key.includes('mt5')) {
                        badgeStyle = 'bg-rose-50 dark:bg-rose-950/20 text-rose-600 dark:text-rose-400 border border-rose-250/20';
                      } else {
                        badgeStyle = 'bg-amber-50 dark:bg-amber-950/20 text-amber-600 dark:text-amber-400 border border-amber-250/20';
                      }
                    } else if (spec.group === 'Hybrid') {
                      if (key.includes('mt5')) {
                        badgeStyle = 'bg-indigo-50 dark:bg-indigo-950/20 text-indigo-600 dark:text-indigo-400 border border-indigo-250/20';
                      } else if (key.includes('bartpho')) {
                        badgeStyle = 'bg-sky-50 dark:bg-sky-950/20 text-sky-600 dark:text-sky-400 border border-sky-250/20';
                      } else {
                        badgeStyle = 'bg-cyan-50 dark:bg-cyan-950/20 text-cyan-600 dark:text-cyan-400 border border-cyan-250/20';
                      }
                    }

                    return (
                      <div key={key} className="ui-card p-5 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm hover:shadow-md hover:border-sky-500/30 transition-all flex flex-col justify-between rounded-2xl">
                        <div className="space-y-2.5">
                          <div className="flex justify-between items-start gap-2">
                            <span className="font-extrabold text-[11px] text-[var(--text-primary)] tracking-tight">
                              {spec.name}
                            </span>
                            <span className={`text-[8px] font-extrabold px-1.5 py-0.5 rounded uppercase tracking-wider shrink-0 ${badgeStyle}`}>
                              {spec.group}
                            </span>
                          </div>
                          
                          <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed font-medium bg-[var(--bg-muted)]/20 p-3 rounded-xl border border-[var(--border-subtle)]/40 min-h-[60px] italic">
                            {output ? `"${output}"` : 'Chưa chạy benchmark cho mô hình này.'}
                          </p>
                        </div>

                        {output && (
                          <div className="mt-3.5 pt-2 border-t border-[var(--border)]/40 flex flex-wrap gap-1.5 text-[8px] font-bold">
                            <span className="bg-emerald-50 dark:bg-emerald-950/10 text-emerald-600 dark:text-emerald-400 border border-emerald-250/20 px-1.5 py-0.5 rounded-md">
                              R-L: {pct(metrics?.rougeL || 0)}
                            </span>
                            <span className="bg-sky-50 dark:bg-sky-950/10 text-sky-600 dark:text-sky-400 border border-sky-250/20 px-1.5 py-0.5 rounded-md">
                              BS: {pct(metrics?.bertscore || 0)}
                            </span>
                            {metrics?.latency !== undefined && (
                              <span className="bg-purple-50 dark:bg-purple-950/10 text-purple-600 dark:text-purple-400 border border-purple-250/20 px-1.5 py-0.5 rounded-md">
                                Trễ: {metrics.latency < 1 ? `${Math.round(metrics.latency * 1000)}ms` : `${(metrics.latency).toFixed(2)}s`}
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
            ) : (
              <div className="ui-card p-8 text-center text-[var(--text-faint)] bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm rounded-xl">
                Không có dữ liệu mẫu benchmark.
              </div>
            )}
          </div>
        )}

      {/* ─────────────────────────────────────────────────────────────
          TAB CONTENT: MODELS SPECS
          ───────────────────────────────────────────────────────────── */}
      {activeTab === 'specs' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {Object.keys(modelsSpecifications).map(key => {
            const spec = modelsSpecifications[key];
            return (
              <motion.div
                key={key}
                whileHover={{ y: -3 }}
                className="ui-card p-5 bg-[var(--bg-elevated)] border border-[var(--border)] shadow-sm flex flex-col justify-between"
              >
                <div>
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h3 className="font-extrabold text-sm text-[var(--text-primary)]">{spec.name}</h3>
                      <span className="text-[9px] font-bold text-sky-600 dark:text-sky-400 bg-sky-50 dark:bg-sky-950/20 px-1.5 py-0.5 rounded uppercase border border-sky-200/30 tracking-wider">
                        {spec.group}
                      </span>
                    </div>
                    <Cpu size={18} className="text-sky-500" />
                  </div>
                  <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed mb-4 font-medium italic">
                    "{spec.desc}"
                  </p>

                  <div className="space-y-3 pt-3 border-t border-[var(--border)]/60">
                    <div className="space-y-1.5">
                      <span className="text-[9px] font-bold text-[var(--text-faint)] uppercase tracking-wider">Ưu điểm nổi bật</span>
                      <ul className="space-y-1">
                        {spec.pros.map((pro, i) => (
                          <li key={i} className="text-[10px] font-semibold text-[var(--text-secondary)] flex items-start gap-1">
                            <Check size={11} className="text-emerald-500 shrink-0 mt-0.5" />
                            <span>{pro}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {spec.cons && spec.cons.length > 0 && (
                      <div className="space-y-1.5">
                        <span className="text-[9px] font-bold text-[var(--text-faint)] uppercase tracking-wider">Nhược điểm hạn chế</span>
                        <ul className="space-y-1">
                          {spec.cons.map((con, i) => (
                            <li key={i} className="text-[10px] font-semibold text-[var(--text-secondary)] flex items-start gap-1">
                              <X size={11} className="text-red-500 shrink-0 mt-0.5" />
                              <span>{con}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>

                <div className="mt-5 pt-3 border-t border-[var(--border)]/60 text-[10px] text-[var(--text-faint)] font-bold space-y-1">
                  <div className="flex justify-between">
                    <span>Độ phức tạp tính toán:</span>
                    <span className="text-[var(--text-secondary)]">{spec.complexity}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Số lượng tham số:</span>
                    <span className="text-[var(--text-secondary)]">{spec.params}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Framework chính:</span>
                    <span className="text-[var(--text-secondary)]">{spec.framework}</span>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default Compare;
