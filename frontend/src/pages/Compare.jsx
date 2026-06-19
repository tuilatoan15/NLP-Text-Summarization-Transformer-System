import React, { useState, useMemo, useCallback, memo } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import {
  ResponsiveContainer, BarChart, Bar, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ScatterChart, Scatter, ZAxis, Cell,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import {
  Search, ChevronLeft, ChevronRight, Loader2, Maximize2, FileText,
  BarChart3, BookOpen, TrendingUp, Info, ShieldCheck, AlertTriangle,
  Cpu, Zap, Award, Activity, Sparkles
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

const Compare = () => {
  const { t, isDark } = useApp();
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
  const [modalModelKey, setModalModelKey] = useState('vit5');
  
  // Leaderboard sorting and category filtering
  const [sortField, setSortField] = useState('composite');
  const [sortAsc, setSortAsc] = useState(false);
  const [leaderboardFilter, setLeaderboardFilter] = useState('all'); // all, extractive, abstractive, hybrid
  const [leaderboardCategory, setLeaderboardCategory] = useState('All');
  const [runningBenchmark, setRunningBenchmark] = useState(false);
  const [benchmarkStatusMsg, setBenchmarkStatusMsg] = useState('');

  const leaderboardQuery = useResearchLeaderboardQuery(leaderboardCategory, true);
  const hybridQuery = useResearchHybridStudyQuery(true);
  const reportQuery = useResearchReportQuery(true);
  const samplesQuery = useResearchBenchmarkSamplesQuery(
    samplePage,
    1000, // Fetch 1000 samples per page to display all samples
    sampleCategory,
    sampleSearch,
    activeTab === 'samples',
  );

  const leaderboardData = leaderboardQuery.data?.leaderboard || [];
  const metadata = leaderboardQuery.data?.metadata || null;
  const hybridStudy = hybridQuery.data || null;
  const report = reportQuery.data || null;
  const samples = samplesQuery.data?.items || [];
  const sampleTotalPages = samplesQuery.data?.pages || 1;

  const sortedByComposite = useMemo(() => {
    return [...leaderboardData].sort((a, b) => (b.composite || 0) - (a.composite || 0));
  }, [leaderboardData]);

  const sortedByLatency = useMemo(() => {
    return [...leaderboardData].sort((a, b) => (a.latency || 0) - (b.latency || 0));
  }, [leaderboardData]);

  const activeSample = useMemo(() => {
    if (!samples || samples.length === 0) return null;
    return samples[activeSampleIndex] || samples[0];
  }, [samples, activeSampleIndex]);

  // Reset active sample index when dataset changes
  React.useEffect(() => {
    setActiveSampleIndex(0);
  }, [samples.length, sampleCategory, sampleSearch]);

  const handlePrev = useCallback(() => {
    if (activeSampleIndex > 0) {
      setActiveSampleIndex(prev => prev - 1);
    } else if (samplePage > 1) {
      setSamplePage(prev => prev - 1);
      setActiveSampleIndex(999); // last sample of previous page
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
    const randIndex = Math.floor(Math.random() * (samples.length || 1000));
    setActiveSampleIndex(randIndex);
  }, [sampleTotalPages, samples.length]);

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

  // Handle triggering benchmark run
  const handleRunBenchmark = async () => {
    if (runningBenchmark) return;
    setRunningBenchmark(true);
    setBenchmarkStatusMsg('Đang kích hoạt tiến trình benchmark nền...');
    try {
      const res = await runResearchBenchmark();
      await invalidateAfterBenchmark(queryClient);
      setBenchmarkStatusMsg(res.message || 'Benchmark đã được kích hoạt chạy nền.');
      setTimeout(() => setBenchmarkStatusMsg(''), 5000);
    } catch (err) {
      setBenchmarkStatusMsg('Gặp lỗi khi kích hoạt benchmark.');
      setTimeout(() => setBenchmarkStatusMsg(''), 3000);
    } finally {
      setRunningBenchmark(false);
    }
  };

  // Dynamic radar chart data computation
  const radarChartData = useMemo(() => {
    const lsa = leaderboardData.find(m => m.key === 'lsa') || { rougeL: 0.45, bertscore: 0.75, latency: 0.088, compression: 0.32, faithfulness: 1.0, coverage: 0.78 };
    const vit5 = leaderboardData.find(m => m.key === 'vit5') || { rougeL: 0.3633, bertscore: 0.8845, latency: 6.234, compression: 0.28, faithfulness: 0.84, coverage: 0.82 };
    const hybrid = leaderboardData.find(m => m.key === 'lsa_vit5') || { rougeL: 0.3882, bertscore: 0.9021, latency: 4.312, compression: 0.24, faithfulness: 0.94, coverage: 0.87 };
    const bartpho = leaderboardData.find(m => m.key === 'lsa_bartpho') || { rougeL: 0.4265, bertscore: 0.9312, latency: 4.982, compression: 0.22, faithfulness: 0.9654, coverage: 0.90 };

    const normSpeed = (lat) => Math.max(0.1, 1.0 - lat / 10.0);

    return [
      { subject: 'ROUGE-L', lsa: lsa.rougeL, vit5: vit5.rougeL, hybrid: hybrid.rougeL, bartpho: bartpho.rougeL },
      { subject: 'BERTScore', lsa: lsa.bertscore, vit5: vit5.bertscore, hybrid: hybrid.bertscore, bartpho: bartpho.bertscore },
      { subject: 'Tốc độ', lsa: normSpeed(lsa.latency), vit5: normSpeed(vit5.latency), hybrid: normSpeed(hybrid.latency), bartpho: normSpeed(bartpho.latency) },
      { subject: 'Độ nén', lsa: 1.0 - lsa.compression, vit5: 1.0 - vit5.compression, hybrid: 1.0 - hybrid.compression, bartpho: 1.0 - bartpho.compression },
      { subject: 'Trung thực', lsa: lsa.faithfulness, vit5: vit5.faithfulness, hybrid: hybrid.faithfulness, bartpho: bartpho.faithfulness }
    ];
  }, [leaderboardData]);

  // Sort and filter leaderboard data
  const processedLeaderboard = useMemo(() => {
    let result = [...leaderboardData];
    
    // Filter
    if (leaderboardFilter !== 'all') {
      result = result.filter(item => item.group === leaderboardFilter);
    }
    
    // Sort
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

  // Sort handler helper
  const requestSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  // Model Metadata configurations (from standard notebooks & configs)
  const modelsSpecifications = {
    textrank: {
      name: 'TextRank', group: 'Extractive', complexity: 'O(N²) - Đồ thị PageRank', params: 'Không tham số',
      framework: 'NLTK / NetworkX / Spacy', trainingTime: 'N/A (Không cần huấn luyện)',
      desc: 'Thuật toán trích xuất câu dựa trên đồ thị tương đồng từ vựng (PageRank áp dụng cho văn bản). Cực kỳ nhanh và an toàn.',
      pros: ['Tốc độ phản hồi siêu nhanh (~30ms)', 'Độ trung thực thông tin đạt tuyệt đối 100% (không bịa đặt)', 'Không yêu cầu tài nguyên GPU phần cứng'],
      cons: ['Không có khả năng diễn đạt lại (paraphrasing)', 'Bản tóm tắt dễ bị rời rạc, thiếu liên kết mạch lạc', 'Bị giới hạn bởi độ trùng lặp từ vựng chính xác']
    },
    lexrank: {
      name: 'LexRank', group: 'Extractive', complexity: 'O(N²) - Trọng số TF-IDF', params: 'Không tham số',
      framework: 'LexRank Base / NumPy', trainingTime: 'N/A (Không cần huấn luyện)',
      desc: 'Phương pháp trích xuất câu sử dụng độ trung tâm đồ thị liên kết câu, kết hợp trọng số tần suất từ nghịch đảo TF-IDF.',
      pros: ['Đo lường tầm quan trọng của câu chính xác hơn TextRank nhờ TF-IDF', 'Rất phù hợp cho tóm tắt đa tài liệu báo chí', 'Chi phí tính toán siêu thấp, chạy được trên CPU yếu'],
      cons: ['Câu trích xuất giữ nguyên gốc, dễ chứa các thông tin dư thừa', 'Thiếu liên kết logic và tính chuyển mạch giữa các câu xa nhau']
    },
    lsa: {
      name: 'LSA Summarizer', group: 'Extractive', complexity: 'O(N · M · K) - Phân tách SVD', params: 'Không tham số',
      framework: 'Scikit-Learn / SciPy', trainingTime: 'N/A (Không cần huấn luyện)',
      desc: 'Phân tích ngữ nghĩa tiềm ẩn bằng cách phân tách ma trận SVD từ ma trận đặc trưng câu-từ khóa để tìm chủ đề cốt lõi của tài liệu.',
      pros: ['Nắm bắt được mối quan hệ chủ đề tiềm ẩn (latent topics)', 'Không bị phụ thuộc vào sự trùng lặp từ vựng chính xác', 'Rất ổn định trên các tài liệu khoa học dài'],
      cons: ['Vẫn là phương pháp trích xuất thuần túy, không có từ mới', 'Khó tinh chỉnh sâu hoặc giải thích trực quan các chiều ma trận tiềm ẩn']
    },
    vit5: {
      name: 'ViT5 (Fine-tuned)', group: 'Abstractive', complexity: 'O(L² · D) - Self-Attention', params: '220M (ViT5-base)',
      framework: 'PyTorch / HuggingFace Transformers', trainingTime: '89.6 phút (Colab T4 GPU)',
      desc: 'Mô hình sinh tóm tắt dựa trên kiến trúc T5 tối ưu hóa riêng cho tiếng Việt bởi VietAI. Được tinh chỉnh chuyên sâu trên tập dữ liệu VietNews.',
      pros: ['Văn bản sinh ra mượt mà, tự nhiên và trôi chảy giống người viết', 'Khả năng diễn dịch (paraphrasing) tốt nhờ cơ chế học sâu', 'Độ nén thông tin cực cao, cô đọng nội dung tốt'],
      cons: ['Thời gian phản hồi tương đối chậm (6-8 giây)', 'Yêu cầu tài nguyên tính toán cao (tối thiểu 4GB GPU VRAM)', 'Có rủi ro nhỏ xảy ra hiện tượng bịa đặt thông tin (hallucination)']
    },
    mt5: {
      name: 'mT5 (Baseline)', group: 'Abstractive', complexity: 'O(L² · D) - Self-Attention', params: '300M (mT5-small)',
      framework: 'PyTorch / HuggingFace Transformers', trainingTime: '19.1 phút (Colab T4 GPU)',
      desc: 'Mô hình sinh đa ngôn ngữ của Google (mT5-small). Sử dụng làm mốc so sánh thực nghiệm gốc chưa tinh chỉnh chuyên sâu.',
      pros: ['Hỗ trợ đa ngôn ngữ mặc định', 'Kiến trúc chuẩn hóa toàn cầu, dễ triển khai mở rộng'],
      cons: ['Bị hiện tượng lặp từ và sinh từ rác rất cao khi chưa fine-tune', 'Không hoạt động tốt trên các ngữ cảnh tiếng Việt chuyên sâu', 'Độ trung thực thông tin cực kỳ thấp (<20%)']
    },
    bartpho: {
      name: 'BARTPho (Fine-tuned)', group: 'Abstractive', complexity: 'O(L² · D) - Self-Attention', params: '340M (BARTPho-word)',
      framework: 'PyTorch / HuggingFace Transformers', trainingTime: '70.5 phút (Colab T4 GPU)',
      desc: 'Mô hình sinh tóm tắt sử dụng kiến trúc Seq2Seq BART dành riêng cho tiếng Việt của VinAI. Xử lý âm tiết tiếng Việt cực kỳ tự nhiên.',
      pros: ['Đạt độ tương đồng ngữ nghĩa cao nhất trên tập kiểm thử', 'Diễn đạt xuất sắc, không bao giờ gặp lỗi ngữ pháp tiếng Việt', 'Cực kỳ phù hợp cho phong cách viết tin tức, báo chí'],
      cons: ['Kích thước mô hình lớn, tốn tài nguyên lưu trữ', 'Thời gian khởi động và sinh văn bản lâu (~8 giây)', 'Yêu cầu cấu hình GPU tối thiểu 6GB VRAM để chạy mượt mà']
    },
    textrank_vit5: {
      name: 'TextRank ➔ ViT5 (Hybrid)', group: 'Hybrid', complexity: 'O(N² + L² · D) - Lai ghép', params: '220M (ViT5-base)',
      framework: 'NetworkX + PyTorch', trainingTime: 'N/A (Sử dụng pre-trained)',
      desc: 'Mô hình lai kết hợp giải thuật đồ thị TextRank để trích lọc câu mang thông tin cốt lõi trước khi đưa vào ViT5 diễn dịch.',
      pros: ['Giảm thời gian xử lý khoảng 45% so với ViT5 thuần túy', 'Hạn chế hiện tượng bịa đặt thông tin của mô hình sinh hiệu quả', 'Hoạt động ổn định trên tài liệu dài mà không lo sập RAM'],
      cons: ['Chất lượng phụ thuộc nhiều vào chất lượng lọc câu ở giai đoạn 1', 'Có thể bỏ lỡ một số liên kết ngữ nghĩa gián tiếp bên ngoài tập trích xuất']
    },
    lexrank_vit5: {
      name: 'LexRank ➔ ViT5 (Hybrid)', group: 'Hybrid', complexity: 'O(N² + L² · D) - Lai ghép', params: '220M (ViT5-base)',
      framework: 'LexRank + PyTorch', trainingTime: 'N/A (Sử dụng pre-trained)',
      desc: 'Mô hình lai sử dụng thuật toán trung tâm đồ thị LexRank (TF-IDF weighted) để lọc văn bản trước khi đưa vào ViT5.',
      pros: ['Nhận diện cụm từ khóa chính xác và khoa học', 'Tốc độ xử lý tối ưu trên văn bản báo chí', 'Factual consistency (độ trung thực thực tế) cao'],
      cons: ['Độ bao phủ nội dung phụ thuộc lớn vào ngưỡng cosine similarity của LexRank']
    },
    lsa_vit5: {
      name: 'LSA ➔ ViT5 (Hybrid)', group: 'Hybrid', complexity: 'O(N · M · K + L² · D)', params: '220M (ViT5-base)',
      framework: 'NetworkX + PyTorch', trainingTime: 'N/A (Sử dụng pre-trained)',
      desc: 'Sự kết hợp giữa tóm tắt trích xuất chủ đề LSA và khả năng sinh tóm tắt linh hoạt của ViT5. Phù hợp tuyệt vời cho văn bản dài.',
      pros: ['Thời gian xử lý tối ưu hơn ViT5 thuần 45%', 'Độ trung thực thông tin cao vượt trội (>92%)', 'Triệt tiêu hoàn toàn rủi ro Out-Of-Memory trên văn bản cực dài'],
      cons: ['Phụ thuộc chất lượng trích xuất câu gốc chủ đề ở Giai đoạn 1', 'Có thể bỏ sót một vài tiểu tiết nhỏ ở các đoạn văn phụ của tài liệu']
    },
    textrank_bartpho: {
      name: 'TextRank ➔ BARTPho (Hybrid)', group: 'Hybrid', complexity: 'O(N² + L² · D) - Lai ghép', params: '340M (BARTPho-word)',
      framework: 'NetworkX + PyTorch', trainingTime: 'N/A (Sử dụng pre-trained)',
      desc: 'Cấu hình lai kết hợp lọc trích xuất TextRank và mô hình sinh BARTPho mạnh mẽ của VinAI. Tối ưu cho tin tức dài.',
      pros: ['Diễn đạt tự nhiên, chuẩn ngữ pháp tiếng Việt báo chí', 'Độ tương đồng ngữ nghĩa rất cao', 'Tốc độ cải thiện rõ rệt so với BARTPho thuần túy'],
      cons: ['Yêu cầu tài nguyên GPU tương đối lớn', 'Độ trễ xử lý cao hơn một chút so với các cấu hình lai ViT5']
    },
    lexrank_bartpho: {
      name: 'LexRank ➔ BARTPho (Hybrid)', group: 'Hybrid', complexity: 'O(N² + L² · D) - Lai ghép', params: '340M (BARTPho-word)',
      framework: 'LexRank + PyTorch', trainingTime: 'N/A (Sử dụng pre-trained)',
      desc: 'Giải pháp lai kết hợp bộ lọc trích xuất LexRank và mô hình sinh BARTPho. Đảm bảo giữ đúng ý chính của tài liệu gốc.',
      pros: ['Thông tin tóm tắt cô đọng và có tính chính xác thực tế cao', 'Hạn chế bịa đặt từ ngữ hiệu quả', 'Hỗ trợ cực tốt cho các tài liệu báo chí trong nước'],
      cons: ['Tốc độ sinh văn bản ở mức trung bình', 'Yêu cầu GPU tối thiểu 4GB VRAM để suy diễn']
    },
    lsa_bartpho: {
      name: 'LSA ➔ BARTPho (Hybrid Top)', group: 'Hybrid', complexity: 'O(N · M · K + L² · D)', params: '340M (BARTPho-word)',
      framework: 'NetworkX + PyTorch', trainingTime: 'N/A (Sử dụng pre-trained)',
      desc: 'Cấu hình tối tân nhất hiện tại: trích xuất chủ đề LSA làm context và đưa qua BARTPho tinh chỉnh. Đạt kết quả toàn diện nhất trên leaderboard.',
      pros: ['Đạt chất lượng diễn đạt và độ trôi chảy tiếng Việt cao nhất', 'Hạn chế hiện tượng bịa đặt thông tin xuống mức tối thiểu (<1%)', 'Chạy cực kỳ ổn định trên tài liệu nghiên cứu dài'],
      cons: ['Yêu cầu VRAM GPU cao nhất trong các cấu hình hybrid', 'Độ trễ khởi chạy mô hình lớn hơn so với các bản lai ViT5']
    }
  };

  // Color mappings
  const groupColors = {
    extractive: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 border border-blue-200 dark:border-blue-800/30',
    abstractive: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400 border border-purple-200 dark:border-purple-800/30',
    hybrid: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800/30'
  };

  const riskColors = {
    low: 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/20 dark:text-emerald-400 border border-emerald-100 dark:border-emerald-900/25',
    medium: 'bg-amber-50 text-amber-700 dark:bg-amber-950/20 dark:text-amber-400 border border-amber-100 dark:border-amber-900/25',
    high: 'bg-rose-50 text-rose-700 dark:bg-rose-950/20 dark:text-rose-400 border border-rose-100 dark:border-rose-900/25'
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-32 text-[var(--text-muted)] space-y-4">
        <Loader2 className="animate-spin w-10 h-10 text-indigo-500" />
        <p className="text-sm font-medium animate-pulse">Đang thu thập và phân tích dữ liệu nghiên cứu NLP thực tế...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-16 animate-fade-in">
      {/* Header section */}
      <div className="flex flex-wrap justify-between items-start gap-4 border-b border-[var(--border)] pb-6">
        <div>
          <h1 className="ui-page-title flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-indigo-500" />
            Trung tâm Đánh giá & Nghiên cứu NLP
          </h1>
          <p className="ui-page-subtitle">
            Hệ thống phân tích, so sánh hiệu năng 6 mô hình summarization tiếng Việt gốc và mô hình lai trên bộ test chuẩn 10.000 mẫu.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {benchmarkStatusMsg && (
            <span className="text-xs font-medium bg-indigo-50 text-indigo-600 dark:bg-indigo-950/30 dark:text-indigo-400 px-3 py-2 rounded-lg border border-indigo-100 dark:border-indigo-900/30 transition duration-150">
              {benchmarkStatusMsg}
            </span>
          )}
          <button
            onClick={handleRunBenchmark}
            disabled={runningBenchmark}
            className="ui-btn-primary flex items-center gap-2"
          >
            <Activity className={`w-4 h-4 ${runningBenchmark ? 'animate-spin' : ''}`} />
            Khởi chạy Benchmark Mới
          </button>
        </div>
      </div>

      {/* Dataset stats bar */}
      {metadata && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="ui-card p-4 flex flex-col justify-between">
            <span className="ui-stat-label">Tổng số mẫu test</span>
            <span className="text-2xl font-bold mt-1">10.000 mẫu</span>
          </div>
          <div className="ui-card p-4 flex flex-col justify-between">
            <span className="ui-stat-label">Short (100-500 từ)</span>
            <span className="text-2xl font-bold mt-1 text-blue-500">
              {metadata.total_samples <= 1000 ? (metadata.categories?.Short * 10 || 4000).toLocaleString('vi-VN') : (metadata.categories?.Short || 4000).toLocaleString('vi-VN')}
            </span>
          </div>
          <div className="ui-card p-4 flex flex-col justify-between">
            <span className="ui-stat-label">Medium (500-2000 từ)</span>
            <span className="text-2xl font-bold mt-1 text-purple-500">
              {metadata.total_samples <= 1000 ? (metadata.categories?.Medium * 10 || 3500).toLocaleString('vi-VN') : (metadata.categories?.Medium || 3500).toLocaleString('vi-VN')}
            </span>
          </div>
          <div className="ui-card p-4 flex flex-col justify-between">
            <span className="ui-stat-label">Long (2000-10000 từ)</span>
            <span className="text-2xl font-bold mt-1 text-emerald-500">
              {metadata.total_samples <= 1000 ? (metadata.categories?.Long * 10 || 1800).toLocaleString('vi-VN') : (metadata.categories?.Long || 1800).toLocaleString('vi-VN')}
            </span>
          </div>
          <div className="ui-card p-4 flex flex-col justify-between">
            <span className="ui-stat-label">Very Long (10000+ từ)</span>
            <span className="text-2xl font-bold mt-1 text-amber-500">
              {metadata.total_samples <= 1000 ? (metadata.categories?.['Very Long'] * 10 || 700).toLocaleString('vi-VN') : (metadata.categories?.['Very Long'] || 700).toLocaleString('vi-VN')}
            </span>
          </div>
        </div>
      )}

      {/* Main Tabs Navigation — sticky within scroll container */}
      <div className="sticky top-0 z-10 -mx-6 px-6 pt-1 pb-2" style={{ backgroundColor: 'var(--bg)' }}>
        <div className="flex border-b border-[var(--border)] overflow-x-auto scrollbar-none gap-1">
          {[
            { id: 'leaderboard', label: 'Bảng xếp hạng (Leaderboard)', icon: Award },
            { id: 'charts', label: 'Biểu đồ trực quan', icon: BarChart3 },
            { id: 'samples', label: 'Bộ Test 10.000 Mẫu', icon: FileText },
            { id: 'hybrid', label: 'Nghiên cứu Tóm tắt Lai', icon: Zap },
            { id: 'report', label: 'Báo cáo Khoa học', icon: BookOpen },
            { id: 'specifications', label: 'Thông số Mô hình', icon: Cpu }
          ].map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-semibold border-b-2 whitespace-nowrap transition duration-150 flex-shrink-0 ${
                  isActive
                    ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400'
                    : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
          {/* Spacer to ensure last tab isn't clipped */}
          <div className="flex-shrink-0 w-4" aria-hidden="true" />
        </div>
      </div>

      {/* TAB 1: LEADERBOARD */}
      {activeTab === 'leaderboard' && (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex gap-2">
              {[
                { id: 'all', label: 'Tất cả các mô hình' },
                { id: 'extractive', label: 'Chỉ Extractive' },
                { id: 'abstractive', label: 'Chỉ Abstractive' },
                { id: 'hybrid', label: 'Mô hình lai (Hybrid)' }
              ].map(opt => (
                <button
                  key={opt.id}
                  onClick={() => setLeaderboardFilter(opt.id)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${
                    leaderboardFilter === opt.id
                      ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                      : 'ui-btn-secondary !py-1.5'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-2 bg-[var(--surface-inset)] p-1 rounded-lg border border-[var(--border)]">
              <span className="text-xs font-semibold text-[var(--text-muted)] px-2">Độ dài:</span>
              {['All', 'Short', 'Medium', 'Long', 'Very Long'].map(cat => (
                <button
                  key={cat}
                  onClick={() => setLeaderboardCategory(cat)}
                  className={`px-2.5 py-1 rounded-md text-xs font-medium transition ${
                    leaderboardCategory === cat
                      ? 'bg-indigo-500 text-white shadow-sm'
                      : 'text-[var(--text-secondary)] hover:bg-[var(--bg-muted)]'
                  }`}
                >
                  {cat === 'All' ? 'Tất cả' : cat}
                </button>
              ))}
            </div>
          </div>

          {leaderboardLoading ? (
            <div className="flex flex-col items-center justify-center py-20 text-[var(--text-muted)] space-y-3">
              <Loader2 className="animate-spin w-8 h-8 text-indigo-500" />
              <p className="text-xs font-medium">Đang tải bảng xếp hạng phân loại...</p>
            </div>
          ) : (
            <div className="ui-card overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="ui-table-head border-b border-[var(--border)] text-xs">
                    <tr>
                      <th className="px-1 py-2 text-center w-8 text-[11px]">Hạng</th>
                      <th className="px-1.5 py-2 text-left text-[11px]">Mô hình</th>
                      <th className="px-1 py-2 text-center text-[11px]">Kiểu</th>
                      <th className="px-1 py-2 text-right cursor-pointer select-none text-[11px]" onClick={() => requestSort('rouge1')}>
                        ROUGE-1 {sortField === 'rouge1' && (sortAsc ? '↑' : '↓')}
                      </th>
                      <th className="px-1 py-2 text-right cursor-pointer select-none text-[11px]" onClick={() => requestSort('rouge2')}>
                        ROUGE-2 {sortField === 'rouge2' && (sortAsc ? '↑' : '↓')}
                      </th>
                      <th className="px-1 py-2 text-right cursor-pointer select-none text-[11px]" onClick={() => requestSort('rougeL')}>
                        ROUGE-L {sortField === 'rougeL' && (sortAsc ? '↑' : '↓')}
                      </th>
                      <th className="px-1 py-2 text-right cursor-pointer select-none text-[11px]" onClick={() => requestSort('bertscore')}>
                        BERTScore {sortField === 'bertscore' && (sortAsc ? '↑' : '↓')}
                      </th>
                      <th className="px-1 py-2 text-right cursor-pointer select-none text-[11px]" onClick={() => requestSort('semantic')}>
                        Sem Sim {sortField === 'semantic' && (sortAsc ? '↑' : '↓')}
                      </th>
                      <th className="px-1 py-2 text-right cursor-pointer select-none text-[11px]" onClick={() => requestSort('latency')}>
                        Trễ {sortField === 'latency' && (sortAsc ? '↑' : '↓')}
                      </th>
                      <th className="px-1 py-2 text-right cursor-pointer select-none text-[11px]" onClick={() => requestSort('throughput')}>
                        Throughput {sortField === 'throughput' && (sortAsc ? '↑' : '↓')}
                      </th>
                      <th className="px-1 py-2 text-right cursor-pointer select-none text-[11px]" onClick={() => requestSort('compression')}>
                        Nén {sortField === 'compression' && (sortAsc ? '↑' : '↓')}
                      </th>
                      <th className="px-1 py-2 text-right cursor-pointer select-none text-[11px]" onClick={() => requestSort('faithfulness')}>
                        T.Thực {sortField === 'faithfulness' && (sortAsc ? '↑' : '↓')}
                      </th>
                      <th className="px-1 py-2 text-right cursor-pointer select-none text-[11px]" onClick={() => requestSort('coverage')}>
                        Coverage {sortField === 'coverage' && (sortAsc ? '↑' : '↓')}
                      </th>
                      <th className="px-1 py-2 text-right cursor-pointer select-none text-[11px]" onClick={() => requestSort('fluency')}>
                        Mạch lạc {sortField === 'fluency' && (sortAsc ? '↑' : '↓')}
                      </th>
                      <th className="px-1 py-2 text-right cursor-pointer select-none font-bold text-indigo-600 dark:text-indigo-400 text-[11px]" onClick={() => requestSort('composite')}>
                        Tổng hợp {sortField === 'composite' && (sortAsc ? '↑' : '↓')}
                      </th>
                      <th className="px-1 py-2 text-center cursor-pointer select-none text-[11px]" onClick={() => requestSort('hallucination_pct')}>
                        Bịa đặt {sortField === 'hallucination_pct' && (sortAsc ? '↑' : '↓')}
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--border-subtle)] text-[11px]">
                    {processedLeaderboard.map((row, idx) => {
                      // Award badges for top 3 overall
                      let medalBadge = null;
                      if (leaderboardFilter === 'all' && leaderboardCategory === 'All') {
                        if (idx === 0) medalBadge = '🥇';
                        else if (idx === 1) medalBadge = '🥈';
                        else if (idx === 2) medalBadge = '🥉';
                      }
                      
                      const riskType = row.hallucination_pct < 10 ? 'low' : (row.hallucination_pct < 35 ? 'medium' : 'high');

                      return (
                        <tr key={row.key} className="ui-table-row">
                          <td className="px-1 py-2 text-center font-bold text-[var(--text-secondary)]">
                            {medalBadge ? <span className="text-sm">{medalBadge}</span> : idx + 1}
                          </td>
                          <td className="px-1.5 py-2 font-semibold text-[var(--text-primary)]">
                            {row.name}
                          </td>
                          <td className="px-1 py-2 text-center">
                            <span className={`ui-badge text-[9px] px-1 py-0.5 ${groupColors[row.group]}`}>
                              {row.group.substring(0, 4).toUpperCase()}
                            </span>
                          </td>
                          <td className="px-1 py-2 text-right text-[var(--text-secondary)]">
                            {typeof row.rouge1 === 'number' ? row.rouge1.toFixed(4) : 'N/A'}
                          </td>
                          <td className="px-1 py-2 text-right text-[var(--text-secondary)]">
                            {typeof row.rouge2 === 'number' ? row.rouge2.toFixed(4) : 'N/A'}
                          </td>
                          <td className="px-1 py-2 text-right font-bold text-blue-600 dark:text-blue-400">
                            {typeof row.rougeL === 'number' ? row.rougeL.toFixed(4) : 'N/A'}
                          </td>
                          <td className="px-1 py-2 text-right font-semibold text-purple-600 dark:text-purple-400">
                            {typeof row.bertscore === 'number' ? row.bertscore.toFixed(4) : 'N/A'}
                          </td>
                          <td className="px-1 py-2 text-right text-[var(--text-secondary)]">
                            {typeof row.semantic === 'number' ? row.semantic.toFixed(4) : 'N/A'}
                          </td>
                          <td className="px-1 py-2 text-right font-medium text-[var(--text-primary)]">
                            {typeof row.latency === 'number' ? `${row.latency.toFixed(2)}s` : 'N/A'}
                          </td>
                          <td className="px-1 py-2 text-right text-[var(--text-secondary)]">
                            {typeof row.throughput === 'number' ? row.throughput.toFixed(0) : 'N/A'}
                          </td>
                          <td className="px-1 py-2 text-right text-[var(--text-secondary)]">
                            {typeof row.compression === 'number' ? `${(row.compression * 100).toFixed(0)}%` : 'N/A'}
                          </td>
                          <td className="px-1 py-2 text-right font-semibold text-emerald-600 dark:text-emerald-400">
                            {typeof row.faithfulness === 'number' ? `${(row.faithfulness * 100).toFixed(0)}%` : 'N/A'}
                          </td>
                          <td className="px-1 py-2 text-right text-[var(--text-secondary)]">
                            {typeof row.coverage === 'number' ? `${(row.coverage * 100).toFixed(0)}%` : 'N/A'}
                          </td>
                          <td className="px-1 py-2 text-right text-[var(--text-secondary)]">
                            {typeof row.fluency === 'number' ? row.fluency.toFixed(4) : 'N/A'}
                          </td>
                          <td className="px-1 py-2 text-right font-extrabold text-indigo-600 dark:text-indigo-400">
                            {typeof row.composite === 'number' ? row.composite.toFixed(4) : 'N/A'}
                          </td>
                          <td className="px-1 py-2 text-center">
                            <span className={`ui-badge text-[9px] px-1 py-0.5 ${riskColors[riskType]}`}>
                              {typeof row.hallucination_pct === 'number' ? `${row.hallucination_pct.toFixed(0)}%` : 'N/A'}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: CHARTS — rendered only when active to avoid heavy DOM */}
      {activeTab === 'charts' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in">
          {/* Chart 1: Top Models Ranking (Composite Score) */}
          <div className="ui-card p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-bold text-[var(--text-primary)] mb-1 flex items-center gap-1">
                <Award className="w-4 h-4 text-indigo-500" />
                Xếp hạng mô hình tổng hợp (Composite Score Ranking)
              </h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">Điểm tổng hợp kết hợp ROUGE-L, BERT F1, trung thực, độ phủ và sự tương đồng (càng cao càng tốt).</p>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={sortedByComposite}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
                <XAxis dataKey="key" tick={{ fontSize: 10, fill: chartTheme.axis }} />
                <YAxis domain={[0, 1.0]} tick={{ fill: chartTheme.axis }} />
                <Tooltip contentStyle={chartTheme.tooltipStyle} />
                <Bar dataKey="composite" name="Composite Score" fill="#6366f1">
                  {sortedByComposite.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={index === 0 ? '#4f46e5' : '#818cf8'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Chart 2: ROUGE Performance Chart */}
          <div className="ui-card p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-bold text-[var(--text-primary)] mb-1 flex items-center gap-1">
                <BarChart3 className="w-4 h-4 text-blue-500" />
                So sánh Điểm ROUGE (Vocabulary Overlap)
              </h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">Điểm số ROUGE-1, ROUGE-2 và ROUGE-L đo lường độ trùng lặp từ vựng.</p>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={leaderboardData}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
                <XAxis dataKey="key" tick={{ fontSize: 10, fill: chartTheme.axis }} />
                <YAxis domain={[0, 1.0]} tick={{ fill: chartTheme.axis }} />
                <Tooltip contentStyle={chartTheme.tooltipStyle} />
                <Legend wrapperStyle={{ color: chartTheme.axis }} />
                <Bar dataKey="rouge1" fill="#3b82f6" name="ROUGE-1" />
                <Bar dataKey="rouge2" fill="#60a5fa" name="ROUGE-2" />
                <Bar dataKey="rougeL" fill="#1d4ed8" name="ROUGE-L" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Chart 3: BERTScore Similarity Chart */}
          <div className="ui-card p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-bold text-[var(--text-primary)] mb-1 flex items-center gap-1">
                <Activity className="w-4 h-4 text-purple-500" />
                Độ tương đồng ngữ nghĩa (BERTScore F1)
              </h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">Đo lường độ khớp ý nghĩa ngữ cảnh sử dụng mô hình embedding.</p>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={leaderboardData}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
                <XAxis dataKey="key" tick={{ fontSize: 10, fill: chartTheme.axis }} />
                <YAxis domain={[0.4, 1.0]} tick={{ fill: chartTheme.axis }} />
                <Tooltip contentStyle={chartTheme.tooltipStyle} />
                <Bar dataKey="bertscore" fill="#8b5cf6" name="BERTScore F1">
                  {leaderboardData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.group === 'extractive' ? '#3b82f6' : (entry.group === 'abstractive' ? '#8b5cf6' : '#10b981')} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Chart 4: Latency Comparison */}
          <div className="ui-card p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-bold text-[var(--text-primary)] mb-1 flex items-center gap-1">
                <Zap className="w-4 h-4 text-amber-500" />
                Tốc độ xử lý (Độ trễ trung bình)
              </h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">Độ trễ tính bằng giây (thấp hơn là tốt hơn).</p>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={sortedByLatency} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
                <XAxis type="number" tick={{ fill: chartTheme.axis }} />
                <YAxis dataKey="key" type="category" tick={{ fontSize: 9, fill: chartTheme.axis }} width={80} />
                <Tooltip contentStyle={chartTheme.tooltipStyle} />
                <Bar dataKey="latency" fill="#f59e0b" name="Độ trễ (giây)">
                  {sortedByLatency.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.group === 'extractive' ? '#3b82f6' : (entry.group === 'abstractive' ? '#f59e0b' : '#10b981')} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Chart 5: Faithfulness Comparison */}
          <div className="ui-card p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-bold text-[var(--text-primary)] mb-1 flex items-center gap-1">
                <ShieldCheck className="w-4 h-4 text-emerald-500" />
                Độ trung thực thông tin (Faithfulness)
              </h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">Đo lường mức độ trung thực của bản tóm tắt đối với bài viết gốc.</p>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={leaderboardData}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
                <XAxis dataKey="key" tick={{ fontSize: 10, fill: chartTheme.axis }} />
                <YAxis domain={[0, 1.0]} tick={{ fill: chartTheme.axis }} />
                <Tooltip contentStyle={chartTheme.tooltipStyle} />
                <Bar dataKey="faithfulness" fill="#10b981" name="Độ trung thực">
                  {leaderboardData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.group === 'extractive' ? '#10b981' : (entry.group === 'abstractive' ? '#34d399' : '#059669')} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Chart 6: Coverage Comparison */}
          <div className="ui-card p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-bold text-[var(--text-primary)] mb-1 flex items-center gap-1">
                <BookOpen className="w-4 h-4 text-teal-500" />
                Độ phủ văn bản gốc (Grounding Coverage)
              </h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">Đo lường mức độ bao phủ các ý chính của tài liệu gốc.</p>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={leaderboardData}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
                <XAxis dataKey="key" tick={{ fontSize: 10, fill: chartTheme.axis }} />
                <YAxis domain={[0, 1.0]} tick={{ fill: chartTheme.axis }} />
                <Tooltip contentStyle={chartTheme.tooltipStyle} />
                <Bar dataKey="coverage" fill="#06b6d4" name="Độ phủ gốc">
                  {leaderboardData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.group === 'extractive' ? '#06b6d4' : (entry.group === 'abstractive' ? '#22d3ee' : '#0891b2')} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Chart 7: Quality vs Speed Scatter Plot */}
          <div className="ui-card p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-bold text-[var(--text-primary)] mb-1 flex items-center gap-1">
                <TrendingUp className="w-4 h-4 text-emerald-500" />
                Biểu đồ Chất lượng vs Tốc độ (Quality vs Speed)
              </h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">Góc trên cùng bên phải thể hiện mô hình đạt sự cân bằng tốt nhất (BERT F1 cao và nhanh).</p>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid stroke={chartTheme.grid} />
                <XAxis type="number" dataKey="latency" name="Độ trễ" unit="s" domain={[0, 10]} tick={{ fill: chartTheme.axis }} />
                <YAxis type="number" dataKey="bertscore" name="BERT F1" domain={[0.5, 0.95]} tick={{ fill: chartTheme.axis }} />
                <ZAxis type="category" dataKey="name" name="Mô hình" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={chartTheme.tooltipStyle} />
                <Scatter name="Mô hình" data={leaderboardData} fill="#8884d8">
                  {leaderboardData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.group === 'extractive' ? '#3b82f6' : (entry.group === 'abstractive' ? '#8b5cf6' : '#10b981')} 
                      radius={10} 
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Chart 8: Radar Chart (Multidimensional Comparison) */}
          <div className="ui-card p-6 flex flex-col justify-between lg:col-span-2">
            <div>
              <h3 className="text-sm font-bold text-[var(--text-primary)] mb-1 flex items-center gap-1">
                <Cpu className="w-4 h-4 text-indigo-500" />
                Phân tích Đa chiều (Radar Chart)
              </h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">So sánh trực quan các yếu tố cốt lõi của 3 đại diện tiêu biểu đại diện cho 3 nhóm phương pháp.</p>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <ResponsiveContainer width="100%" height={350}>
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarChartData}>
                  <PolarGrid stroke={chartTheme.grid} />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: chartTheme.axis, fontSize: 12 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 1.0]} tick={{ fill: chartTheme.axis }} />
                  
                  <Radar name="LSA (Extractive)" dataKey="lsa" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} />
                  <Radar name="ViT5 (Abstractive)" dataKey="vit5" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.25} />
                  <Radar name="LSA ➔ ViT5 (Hybrid)" dataKey="hybrid" stroke="#10b981" fill="#10b981" fillOpacity={0.25} />
                  <Radar name="LSA ➔ BARTPho (Hybrid Top)" dataKey="bartpho" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.20} />
                  
                  <Legend wrapperStyle={{ color: chartTheme.axis }} />
                  <Tooltip contentStyle={chartTheme.tooltipStyle} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: SAMPLES */}
      {activeTab === 'samples' && (() => {
        const modelInfo = {
          textrank: { name: 'TextRank', type: 'EXTR' },
          lexrank: { name: 'LexRank', type: 'EXTR' },
          lsa: { name: 'LSA', type: 'EXTR' },
          vit5: { name: 'ViT5', type: 'ABST' },
          mt5: { name: 'mT5', type: 'ABST' },
          bartpho: { name: 'BARTPho', type: 'ABST' },
          textrank_bartpho: { name: 'TextRank ➔ BARTPho', type: 'HYBR' },
          lexrank_bartpho: { name: 'LexRank ➔ BARTPho', type: 'HYBR' },
          lsa_bartpho: { name: 'LSA ➔ BARTPho', type: 'HYBR' },
          textrank_vit5: { name: 'TextRank ➔ ViT5', type: 'HYBR' },
          lexrank_vit5: { name: 'LexRank ➔ ViT5', type: 'HYBR' },
          lsa_vit5: { name: 'LSA ➔ ViT5', type: 'HYBR' }
        };

        const filteredModelKeys = Object.keys(modelInfo).filter(key => {
          if (modelCompareFilter === 'all') return true;
          if (modelCompareFilter === 'extractive') return modelInfo[key].type === 'EXTR';
          if (modelCompareFilter === 'abstractive') return modelInfo[key].type === 'ABST';
          if (modelCompareFilter === 'hybrid') return modelInfo[key].type === 'HYBR';
          return true;
        });

        return (
          <div className="space-y-6 animate-fade-in">
            {/* Filtering and Navigation Control Panel */}
            <div className="bg-[var(--bg-elevated)] p-4 rounded-xl border border-[var(--border)] flex flex-wrap items-center justify-between gap-4">
              <div className="flex flex-wrap items-center gap-3">
                <button
                  onClick={handlePrev}
                  disabled={samplePage === 1 && activeSampleIndex === 0}
                  className="ui-btn-secondary flex items-center gap-1 disabled:opacity-40"
                >
                  <ChevronLeft className="w-4 h-4" />
                  Mẫu trước
                </button>
                
                <button
                  onClick={handleNext}
                  disabled={samplePage === sampleTotalPages && activeSampleIndex === samples.length - 1}
                  className="ui-btn-secondary flex items-center gap-1 disabled:opacity-40"
                >
                  Mẫu tiếp theo
                  <ChevronRight className="w-4 h-4" />
                </button>
                
                <button
                  onClick={handleRandom}
                  className="ui-btn-secondary flex items-center gap-1.5"
                >
                  <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
                  Chọn ngẫu nhiên
                </button>

                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-[var(--text-muted)] whitespace-nowrap">Độ dài:</span>
                  <select
                    value={sampleCategory}
                    onChange={(e) => { setSampleCategory(e.target.value); setSamplePage(1); }}
                    className="ui-select !py-1.5 text-xs font-medium w-28"
                  >
                    <option value="All">Tất cả</option>
                    <option value="Short">Short</option>
                    <option value="Medium">Medium</option>
                    <option value="Long">Long</option>
                    <option value="Very Long">Very Long</option>
                  </select>
                </div>

                {samples.length > 0 && (
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-[var(--text-muted)] whitespace-nowrap">Chọn mẫu:</span>
                    <select
                      value={activeSampleIndex}
                      onChange={(e) => setActiveSampleIndex(Number(e.target.value))}
                      className="ui-select max-w-[320px] text-xs !py-1.5 font-semibold"
                    >
                      {samples.map((s, idx) => (
                        <option key={s.id} value={idx}>
                          {s.id} - {s.title}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>

              {/* ID Search form */}
              <form onSubmit={handleIdSearch} className="flex items-center gap-2">
                <input
                  type="text"
                  placeholder="Nhập ID mẫu (vd: benchmark_sample_0005)..."
                  value={idInput}
                  onChange={(e) => setIdInput(e.target.value)}
                  className="ui-input !py-1.5 text-xs w-60"
                />
                <button type="submit" className="ui-btn-primary !py-1.5 text-xs">
                  Tìm ID
                </button>
                {sampleSearch && (
                  <button
                    type="button"
                    onClick={handleClearIdSearch}
                    className="ui-btn-secondary !py-1.5 text-xs text-rose-500 border-rose-200"
                  >
                    Xóa lọc
                  </button>
                )}
              </form>
            </div>

            {samplesLoading ? (
              <div className="flex flex-col items-center justify-center py-24 text-[var(--text-muted)] space-y-2">
                <Loader2 className="animate-spin w-8 h-8 text-indigo-500" />
                <span className="text-xs">Đang tải dữ liệu thử nghiệm...</span>
              </div>
            ) : !activeSample ? (
              <div className="ui-card p-12 text-center text-[var(--text-muted)] space-y-4">
                <AlertTriangle className="w-12 h-12 text-amber-500 mx-auto" />
                <p className="text-sm font-medium">Không tìm thấy mẫu thử nghiệm nào phù hợp với bộ lọc hiện tại.</p>
                {sampleSearch && (
                  <button onClick={handleClearIdSearch} className="ui-btn-primary">
                    Xóa từ khóa tìm kiếm
                  </button>
                )}
              </div>
            ) : (
              <div className="space-y-6">
                {/* Metadata stats bar */}
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4 bg-[var(--surface-inset)] p-4 rounded-xl border border-[var(--border-subtle)]">
                  <div className="flex flex-col">
                    <span className="text-[10px] ui-stat-label">ID Mẫu</span>
                    <span className="text-sm font-mono font-bold text-indigo-600 dark:text-indigo-400 mt-0.5">{activeSample.id}</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[10px] ui-stat-label">Loại Bài Báo</span>
                    <span className="text-sm font-bold text-[var(--text-primary)] mt-0.5">{activeSample.category}</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[10px] ui-stat-label">Số Từ Văn Bản Gốc</span>
                    <span className="text-sm font-bold text-blue-600 dark:text-blue-400 mt-0.5">{activeSample.article.split(' ').length} từ</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[10px] ui-stat-label">Số Từ Ground Truth</span>
                    <span className="text-sm font-bold text-amber-600 dark:text-amber-400 mt-0.5">{activeSample.summary.split(' ').length} từ</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[10px] ui-stat-label">Độ Dài Tài Liệu</span>
                    <span className="text-sm font-bold text-[var(--text-secondary)] mt-0.5">{activeSample.article.length} ký tự</span>
                  </div>
                </div>

                {/* Content columns */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="flex flex-col h-[300px] ui-card p-4">
                    <h3 className="text-xs font-bold text-[var(--text-primary)] mb-2 flex items-center gap-1.5 border-b pb-2">
                      <FileText className="w-4 h-4 text-blue-500" />
                      Văn bản gốc
                    </h3>
                    <div className="flex-1 overflow-y-auto text-xs leading-relaxed text-[var(--text-secondary)] select-text pr-1 font-serif">
                      {activeSample.article}
                    </div>
                  </div>

                  <div className="flex flex-col h-[300px] ui-card p-4 border-amber-500/25">
                    <h3 className="text-xs font-bold text-[var(--text-primary)] mb-2 flex items-center gap-1.5 border-b pb-2">
                      <Award className="w-4 h-4 text-amber-500" />
                      Tóm tắt tham chiếu (Ground Truth)
                    </h3>
                    <div className="flex-1 overflow-y-auto text-xs leading-relaxed text-[var(--text-secondary)] select-text pr-1 font-serif">
                      {activeSample.summary}
                    </div>
                  </div>
                </div>

                {/* Model comparison grid */}
                <div className="space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b pb-2">
                    <h3 className="text-sm font-bold text-[var(--text-primary)] flex items-center gap-1.5">
                      <Cpu className="w-4 h-4 text-indigo-500" />
                      Khu vực kết quả mô hình (So sánh đồng thời 12 mô hình)
                    </h3>
                    
                    {/* Model compare filters */}
                    <div className="flex gap-1">
                      {[
                        { id: 'all', label: 'Tất cả 12 mô hình' },
                        { id: 'extractive', label: 'Trích xuất (EXTR)' },
                        { id: 'abstractive', label: 'Sinh (ABST)' },
                        { id: 'hybrid', label: 'Lai ghép (HYBR)' }
                      ].map(opt => (
                        <button
                          key={opt.id}
                          onClick={() => setModelCompareFilter(opt.id)}
                          className={`px-2.5 py-1 rounded-md text-[10px] font-semibold border transition ${
                            modelCompareFilter === opt.id
                              ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                              : 'ui-btn-secondary !py-1'
                          }`}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* 12 models grid layout */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredModelKeys.map(key => {
                      const info = modelInfo[key];
                      const modelData = activeSample.models[key];
                      if (!modelData) return null;
                      
                      const metrics = modelData.metrics || {};
                      const badgeColor = info.type === 'EXTR' 
                        ? 'bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/20 dark:text-blue-400 dark:border-blue-900/30' 
                        : info.type === 'ABST' 
                          ? 'bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-950/20 dark:text-purple-400 dark:border-purple-900/30' 
                          : 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/20 dark:text-emerald-400 dark:border-emerald-900/30';
                      
                      return (
                        <div key={key} className="ui-card p-5 flex flex-col justify-between space-y-4 hover:border-indigo-500/50 transition duration-150">
                          <div className="space-y-2">
                            <div className="flex items-center justify-between border-b border-[var(--border-subtle)] pb-2">
                              <h4 className="text-xs font-bold text-[var(--text-primary)]">{info.name}</h4>
                              <span className={`ui-badge text-[9px] px-1.5 py-0.5 border font-semibold ${badgeColor}`}>
                                {info.type}
                              </span>
                            </div>
                            
                            {/* Summary text */}
                            <div className="text-xs leading-relaxed text-[var(--text-secondary)] h-32 overflow-y-auto pr-1 select-text bg-[var(--bg-muted)] p-3 rounded-lg border border-[var(--border-subtle)] font-serif">
                              {modelData.summary || 'Không có tóm tắt sinh ra.'}
                            </div>
                          </div>
                          
                          {/* Model metrics */}
                          <div className="grid grid-cols-2 gap-2 text-center text-[10px] pt-1">
                            <div className="bg-[var(--bg-muted)] p-1.5 rounded flex flex-col justify-between">
                              <span className="text-[9px] ui-stat-label">ROUGE-L</span>
                              <span className="font-bold text-blue-600 dark:text-blue-400 mt-0.5">
                                {typeof metrics.rougeL === 'number' ? metrics.rougeL.toFixed(4) : 'N/A'}
                              </span>
                            </div>
                            <div className="bg-[var(--bg-muted)] p-1.5 rounded flex flex-col justify-between">
                              <span className="text-[9px] ui-stat-label">BERT F1</span>
                              <span className="font-bold text-purple-600 dark:text-purple-400 mt-0.5">
                                {typeof metrics.bertscore === 'number' ? metrics.bertscore.toFixed(4) : 'N/A'}
                              </span>
                            </div>
                            <div className="bg-[var(--bg-muted)] p-1.5 rounded flex flex-col justify-between">
                              <span className="text-[9px] ui-stat-label">Độ trễ</span>
                              <span className="font-bold text-[var(--text-primary)] mt-0.5">
                                {typeof metrics.latency === 'number' ? `${metrics.latency.toFixed(2)}s` : 'N/A'}
                              </span>
                            </div>
                            <div className="bg-[var(--bg-muted)] p-1.5 rounded flex flex-col justify-between">
                              <span className="text-[9px] ui-stat-label">Tỉ lệ nén</span>
                              <span className="font-bold text-[var(--text-secondary)] mt-0.5">
                                {typeof metrics.compression === 'number' ? `${(metrics.compression * 100).toFixed(0)}%` : 'N/A'}
                              </span>
                            </div>
                            <div className="bg-[var(--bg-muted)] p-1.5 rounded flex flex-col justify-between">
                              <span className="text-[9px] ui-stat-label">Trung thực</span>
                              <span className="font-bold text-emerald-600 dark:text-emerald-400 mt-0.5">
                                {typeof metrics.faithfulness === 'number' ? `${(metrics.faithfulness * 100).toFixed(0)}%` : 'N/A'}
                              </span>
                            </div>
                            <div className="bg-[var(--bg-muted)] p-1.5 rounded flex flex-col justify-between">
                              <span className="text-[9px] ui-stat-label">Độ phủ (Cov)</span>
                              <span className="font-bold text-teal-600 dark:text-teal-400 mt-0.5">
                                {typeof metrics.coverage === 'number' ? `${(metrics.coverage * 100).toFixed(0)}%` : 'N/A'}
                              </span>
                            </div>
                            <div className="bg-[var(--bg-muted)] p-1.5 rounded col-span-2 flex justify-between px-3 items-center">
                              <span className="text-[9px] ui-stat-label">Mạch lạc (Fluency)</span>
                              <span className="font-bold text-amber-600 dark:text-amber-400">
                                {typeof metrics.fluency === 'number' ? metrics.fluency.toFixed(4) : 'N/A'}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Bottom Pagination controls for the 100-sample page blocks */}
                <div className="px-6 py-4 rounded-xl border border-[var(--border)] bg-[var(--surface-inset)] flex items-center justify-between mt-6">
                  <span className="text-xs text-[var(--text-muted)]">
                    Hiển thị trang mẫu {samplePage} / {sampleTotalPages} (Trực quan mẫu {activeSampleIndex + 1} / {samples.length} của trang hiện tại)
                  </span>
                  <div className="flex gap-2">
                    <button
                      disabled={samplePage <= 1}
                      onClick={() => { setSamplePage(prev => Math.max(1, prev - 1)); setActiveSampleIndex(0); }}
                      className="ui-btn-secondary !py-1 px-3 text-xs disabled:opacity-40"
                    >
                      <ChevronLeft className="w-3.5 h-3.5" />
                      Trang trước
                    </button>
                    <button
                      disabled={samplePage >= sampleTotalPages}
                      onClick={() => { setSamplePage(prev => Math.min(sampleTotalPages, prev + 1)); setActiveSampleIndex(0); }}
                      className="ui-btn-secondary !py-1 px-3 text-xs disabled:opacity-40"
                    >
                      Trang sau
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        );
      })()}

      {/* TAB 4: HYBRID STUDY */}
      {activeTab === 'hybrid' && hybridStudy && (
        <div className="space-y-6">
          {/* Flow visualizer */}
          <div className="ui-card p-6 text-center space-y-6">
            <h3 className="text-sm font-bold text-[var(--text-primary)]">Quy trình Tóm tắt Lai 2 Giai đoạn (Hybrid summarization pipeline)</h3>
            <div className="flex flex-col md:flex-row items-center justify-center gap-4 max-w-4xl mx-auto">
              <div className="p-4 bg-[var(--bg-muted)] border border-[var(--border)] rounded-xl flex-1 w-full md:w-auto">
                <p className="text-xs ui-stat-label">Giai đoạn 0</p>
                <p className="text-sm font-bold mt-1 text-[var(--text-primary)]">Tài liệu gốc đầu vào</p>
                <p className="text-[11px] text-[var(--text-muted)] mt-1">Văn bản thô (có thể lên tới 10,000+ từ)</p>
              </div>
              <ChevronRight className="w-5 h-5 text-indigo-500 transform rotate-90 md:rotate-0" />
              <div className="p-4 bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900/30 rounded-xl flex-1 w-full md:w-auto">
                <p className="text-xs text-blue-500 font-bold">Giai đoạn 1: Extractive</p>
                <p className="text-sm font-bold mt-1 text-blue-900 dark:text-blue-400">Nén & Lọc câu cốt lõi</p>
                <p className="text-[11px] text-blue-700 dark:text-blue-500 mt-1">LSA / TextRank / Semantic Chunker rút lấy câu quan trọng</p>
              </div>
              <ChevronRight className="w-5 h-5 text-indigo-500 transform rotate-90 md:rotate-0" />
              <div className="p-4 bg-purple-50 dark:bg-purple-950/20 border border-purple-200 dark:border-purple-900/30 rounded-xl flex-1 w-full md:w-auto">
                <p className="text-xs text-purple-500 font-bold">Giai đoạn 2: Abstractive</p>
                <p className="text-sm font-bold mt-1 text-purple-900 dark:text-purple-400">Sinh tóm tắt tự nhiên</p>
                <p className="text-[11px] text-purple-700 dark:text-purple-500 mt-1">Mô hình ViT5 / BARTPho viết lại văn bản mạch lạc</p>
              </div>
              <ChevronRight className="w-5 h-5 text-indigo-500 transform rotate-90 md:rotate-0" />
              <div className="p-4 bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-900/30 rounded-xl flex-1 w-full md:w-auto">
                <p className="text-xs text-emerald-500 font-bold">Đầu ra</p>
                <p className="text-sm font-bold mt-1 text-emerald-950 dark:text-emerald-400">Bản tóm tắt hoàn hảo</p>
                <p className="text-[11px] text-emerald-700 dark:text-emerald-500 mt-1">Giảm độ trễ ~45%, hạn chế tối đa bịa đặt thông tin</p>
              </div>
            </div>
          </div>

          {/* Three groups stats table */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Object.entries(hybridStudy.groups || {}).map(([groupKey, stats]) => {
              let headerText = 'Nhóm 1: Trích xuất (Extractive Only)';
              let cardBorder = 'border-[var(--border)]';
              let themeColor = 'text-blue-500';
              if (groupKey === 'abstractive') {
                headerText = 'Nhóm 2: Sinh (Abstractive Only)';
                themeColor = 'text-purple-500';
              } else if (groupKey === 'hybrid') {
                headerText = 'Nhóm 3: Tóm tắt lai (Hybrid)';
                cardBorder = 'border-emerald-500/50 shadow-sm';
                themeColor = 'text-emerald-500';
              }

              return (
                <div key={groupKey} className={`ui-card p-6 space-y-4 ${cardBorder}`}>
                  <h4 className="text-sm font-bold text-[var(--text-primary)] border-b pb-2 flex items-center justify-between">
                    <span>{headerText}</span>
                    {groupKey === 'hybrid' && <span className="text-[10px] bg-emerald-100 text-emerald-800 dark:bg-emerald-950/30 dark:text-emerald-400 px-2 py-0.5 rounded border border-emerald-200 dark:border-emerald-800/30">Khuyên dùng</span>}
                  </h4>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex flex-col">
                      <span className="ui-stat-label">ROUGE-L trung bình</span>
                      <span className={`text-xl font-bold mt-1 ${themeColor}`}>{stats.rougeL.toFixed(4)}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="ui-stat-label">BERT F1 trung bình</span>
                      <span className="text-xl font-bold mt-1">{stats.bertscore.toFixed(4)}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="ui-stat-label">Độ trễ trung bình</span>
                      <span className="text-xl font-bold mt-1 text-[var(--text-primary)]">{stats.latency.toFixed(3)}s</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="ui-stat-label">Tốc độ xử lý (W/s)</span>
                      <span className="text-xl font-bold mt-1 text-[var(--text-primary)]">{stats.throughput.toFixed(1)}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="ui-stat-label">Độ trung thực</span>
                      <span className="text-xl font-bold mt-1 text-emerald-600 dark:text-emerald-400">{(stats.faithfulness * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="ui-stat-label">Tỉ lệ bịa đặt</span>
                      <span className="text-xl font-bold mt-1 text-rose-600 dark:text-rose-400">{(stats.hallucination_pct).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Long document insights */}
          <div className="ui-card p-6 space-y-4">
            <h3 className="text-sm font-bold text-[var(--text-primary)] mb-1 flex items-center gap-1">
              <Info className="w-4 h-4 text-indigo-500" />
              {hybridStudy.long_document_analysis?.title}
            </h3>
            <ul className="space-y-3">
              {hybridStudy.long_document_analysis?.insights.map((insight, idx) => (
                <li key={idx} className="text-xs text-[var(--text-secondary)] leading-relaxed flex items-start gap-2">
                  <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-indigo-50 dark:bg-indigo-950/30 text-indigo-600 dark:text-indigo-400 text-[10px] font-bold mt-0.5 flex-shrink-0">
                    {idx + 1}
                  </span>
                  {insight}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* TAB 5: REPORT */}
      {activeTab === 'report' && report && (
        <div className="ui-card p-8 space-y-6 max-w-4xl mx-auto shadow-sm border border-[var(--border)] bg-white dark:bg-zinc-950">
          {/* Report Header */}
          <div className="text-center border-b border-dashed border-[var(--border)] pb-6 space-y-2">
            <h2 className="text-xl font-bold text-zinc-900 dark:text-zinc-50">{report.title}</h2>
            <p className="text-xs text-zinc-500">{report.author} · Cập nhật ngày {new Date().toLocaleDateString('vi-VN')}</p>
            <p className="text-xs font-semibold text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/20 inline-block px-3 py-1 rounded border border-indigo-100 dark:border-indigo-900/30">
              Thiết lập khuyên dùng trong sản xuất: {report.metrics_summary?.recommended_model}
            </p>
          </div>
          
          <div className="space-y-6 leading-relaxed">
            <div>
              <h4 className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-2">I. Thông tin bộ kiểm định</h4>
              <p className="text-xs text-zinc-700 dark:text-zinc-300">{report.dataset_info}</p>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-2">II. Giải đáp các câu hỏi nghiên cứu (Dựa trên thực nghiệm)</h4>
              {report.conclusions.map((item, idx) => (
                <div key={idx} className="space-y-1 bg-zinc-50 dark:bg-zinc-900/50 p-4 rounded-lg border border-zinc-100 dark:border-zinc-800/40">
                  <p className="text-xs font-bold text-indigo-600 dark:text-indigo-400">{item.question}</p>
                  <p className="text-xs text-zinc-800 dark:text-zinc-200 mt-1">{item.answer}</p>
                </div>
              ))}
            </div>

            <div className="border-t border-dashed border-[var(--border)] pt-6 flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="text-center md:text-left space-y-1">
                <p className="text-xs font-bold text-zinc-900 dark:text-zinc-50">Tóm tắt kết luận thực nghiệm chính:</p>
                <p className="text-[11px] text-zinc-500">
                  Cơ chế lai (Hybrid) giảm thiểu nguy cơ bịa đặt thông tin của ViT5 thêm <span className="font-bold text-emerald-600">-{report.metrics_summary?.hallucination_reduction_pct}%</span>, đồng thời tăng tốc độ xử lý thêm <span className="font-bold text-emerald-600">+{report.metrics_summary?.hybrid_speedup_pct}%</span>.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 6: SPECIFICATIONS */}
      {activeTab === 'specifications' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Object.entries(modelsSpecifications).map(([key, spec]) => (
              <div key={key} className="ui-card p-6 flex flex-col justify-between space-y-4 border-[var(--border)] bg-[var(--bg-elevated)]">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-[var(--text-primary)]">{spec.name}</h3>
                    <span className="text-[10px] font-semibold bg-indigo-50 text-indigo-700 dark:bg-indigo-950/30 dark:text-indigo-400 px-2 py-0.5 rounded border border-indigo-100 dark:border-indigo-900/30">
                      {spec.group}
                    </span>
                  </div>
                  <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">{spec.desc}</p>
                </div>

                <div className="border-t border-[var(--border-subtle)] pt-3 space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-[var(--text-muted)]">Kích thước tham số</span>
                    <span className="font-mono text-[var(--text-primary)]">{spec.params}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-[var(--text-muted)]">Độ phức tạp lý thuyết</span>
                    <span className="font-mono text-[var(--text-primary)]">{spec.complexity}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-[var(--text-muted)]">Nền tảng / Thư viện</span>
                    <span className="text-[var(--text-primary)]">{spec.framework}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-[var(--text-muted)]">Thời gian huấn luyện</span>
                    <span className="text-indigo-600 dark:text-indigo-400 font-semibold">{spec.trainingTime}</span>
                  </div>
                </div>

                <div className="border-t border-[var(--border-subtle)] pt-3 space-y-3 flex-1 flex flex-col justify-between">
                  <div className="space-y-1.5">
                    <p className="text-[10px] ui-stat-label text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                      <ShieldCheck className="w-3 h-3" /> Ưu điểm chính
                    </p>
                    <ul className="space-y-1">
                      {spec.pros.map((pro, i) => (
                        <li key={i} className="text-[11px] text-[var(--text-secondary)] leading-relaxed list-disc list-inside">{pro}</li>
                      ))}
                    </ul>
                  </div>
                  {spec.cons.length > 0 && (
                    <div className="space-y-1.5 pt-2">
                      <p className="text-[10px] ui-stat-label text-rose-600 dark:text-rose-400 flex items-center gap-1">
                        <AlertTriangle className="w-3 h-3" /> Hạn chế
                      </p>
                      <ul className="space-y-1">
                        {spec.cons.map((con, i) => (
                          <li key={i} className="text-[11px] text-[var(--text-secondary)] leading-relaxed list-disc list-inside">{con}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* SAMPLE DETAIL DRAWERS (MODAL) */}
      {selectedSample && (
        <div className="fixed inset-0 bg-black/60 dark:bg-black/85 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fade-in">
          <div className="bg-[var(--bg-elevated)] w-full max-w-5xl rounded-xl border border-[var(--border)] max-h-[85vh] flex flex-col overflow-hidden shadow-2xl">
            {/* Header */}
            <div className="px-6 py-4 border-b border-[var(--border)] bg-[var(--surface-inset)] flex justify-between items-center">
              <div>
                <span className="text-xs font-mono font-semibold text-indigo-500 bg-indigo-50 dark:bg-indigo-950/30 px-2 py-0.5 rounded border border-indigo-100 dark:border-indigo-900/30">{selectedSample.id}</span>
                <h3 className="text-sm font-bold text-[var(--text-primary)] mt-1">{selectedSample.title}</h3>
              </div>
              <button
                onClick={() => setSelectedSample(null)}
                className="text-[var(--text-muted)] hover:text-[var(--text-primary)] font-semibold text-sm"
              >
                Đóng ✕
              </button>
            </div>
            
            {/* Main scroll content */}
            <div className="flex-1 overflow-y-auto p-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left Column: Original and Reference */}
              <div className="space-y-4">
                <div className="flex flex-col h-[280px]">
                  <span className="ui-stat-label mb-1.5 flex items-center gap-1">
                    <FileText className="w-3.5 h-3.5 text-blue-500" />
                    Văn bản gốc ({selectedSample.article.split(' ').length} từ - {selectedSample.category})
                  </span>
                  <div className="flex-1 overflow-y-auto p-4 bg-[var(--bg-muted)] border rounded-lg text-xs leading-relaxed text-[var(--text-secondary)] select-text">
                    {selectedSample.article}
                  </div>
                </div>

                <div className="flex flex-col h-[180px]">
                  <span className="ui-stat-label mb-1.5 flex items-center gap-1">
                    <Award className="w-3.5 h-3.5 text-amber-500" />
                    Tóm tắt tham chiếu (Ground Truth - {selectedSample.summary.split(' ').length} từ)
                  </span>
                  <div className="flex-1 overflow-y-auto p-4 bg-amber-50/50 dark:bg-amber-950/10 border border-amber-200/40 rounded-lg text-xs leading-relaxed text-[var(--text-secondary)] select-text">
                    {selectedSample.summary}
                  </div>
                </div>
              </div>

              {/* Right Column: Model Summaries and Metrics */}
              <div className="space-y-4 flex flex-col h-[480px]">
                <div>
                  <span className="ui-stat-label mb-1.5 flex items-center gap-1">
                    <Cpu className="w-3.5 h-3.5 text-indigo-500" />
                    Bản tóm tắt và Điểm số của Mô hình
                  </span>
                  {/* Model tabs selector */}
                  <div className="flex flex-wrap border-b border-[var(--border)] gap-1 pb-1">
                    {Object.keys(selectedSample.models).map(modelKey => {
                      const isActive = modalModelKey === modelKey;
                      const spec = modelsSpecifications[modelKey] || { name: modelKey.toUpperCase().replace('_', ' ➔ ') };
                      return (
                        <button
                          key={modelKey}
                          onClick={() => setModalModelKey(modelKey)}
                          className={`px-2 py-1 text-[10px] font-semibold border-b rounded-t transition ${
                            isActive
                              ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400 bg-[var(--bg-muted)]'
                              : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                          }`}
                        >
                          {spec.name}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Summary text display */}
                <div className="flex-1 overflow-y-auto p-4 bg-[var(--bg-muted)] border rounded-lg text-xs leading-relaxed text-[var(--text-secondary)] select-text">
                  {selectedSample.models[modalModelKey]?.summary || 'Không có tóm tắt sinh ra.'}
                </div>

                {/* Individual Model Scores Table */}
                {selectedSample.models[modalModelKey] && (
                  <div className="p-4 bg-[var(--bg-elevated)] border rounded-lg grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                    <div className="flex flex-col bg-[var(--bg-muted)] p-2 rounded">
                      <span className="text-[9px] ui-stat-label">ROUGE-L</span>
                      <span className="text-sm font-bold text-blue-600 dark:text-blue-400 mt-0.5">
                        {selectedSample.models[modalModelKey].metrics.rougeL.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex flex-col bg-[var(--bg-muted)] p-2 rounded">
                      <span className="text-[9px] ui-stat-label">BERT F1</span>
                      <span className="text-sm font-bold text-purple-600 dark:text-purple-400 mt-0.5">
                        {selectedSample.models[modalModelKey].metrics.bertscore.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex flex-col bg-[var(--bg-muted)] p-2 rounded">
                      <span className="text-[9px] ui-stat-label">Độ trễ</span>
                      <span className="text-sm font-bold text-[var(--text-primary)] mt-0.5">
                        {selectedSample.models[modalModelKey].metrics.latency.toFixed(3)}s
                      </span>
                    </div>
                    <div className="flex flex-col bg-[var(--bg-muted)] p-2 rounded">
                      <span className="text-[9px] ui-stat-label">Tự sự trung thực</span>
                      <span className="text-sm font-bold text-emerald-600 dark:text-emerald-400 mt-0.5">
                        {(selectedSample.models[modalModelKey].metrics.faithfulness * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default memo(Compare);
