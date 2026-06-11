import React, { useState, useEffect, useMemo } from 'react';
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
  getResearchLeaderboard,
  getResearchBenchmarkSamples,
  getResearchHybridStudy,
  getResearchReport,
  runResearchBenchmark
} from '../services/apiService';

const Compare = () => {
  const { t, isDark } = useApp();
  const chartTheme = getChartTheme(isDark);
  
  // Navigation & Tabs
  const [activeTab, setActiveTab] = useState('leaderboard');
  
  // States
  const [leaderboardData, setLeaderboardData] = useState([]);
  const [metadata, setMetadata] = useState(null);
  const [hybridStudy, setHybridStudy] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  // Benchmark samples pagination & search
  const [samples, setSamples] = useState([]);
  const [samplePage, setSamplePage] = useState(1);
  const [sampleTotalPages, setSampleTotalPages] = useState(1);
  const [sampleCategory, setSampleCategory] = useState('All');
  const [sampleSearch, setSampleSearch] = useState('');
  const [samplesLoading, setSamplesLoading] = useState(false);
  
  // Sample detail modal state
  const [selectedSample, setSelectedSample] = useState(null);
  const [modalModelKey, setModalModelKey] = useState('vit5');
  
  // Leaderboard sorting
  const [sortField, setSortField] = useState('rougeL');
  const [sortAsc, setSortAsc] = useState(false);
  const [leaderboardFilter, setLeaderboardFilter] = useState('all'); // all, extractive, abstractive, hybrid
  
  // Running benchmark trigger state
  const [runningBenchmark, setRunningBenchmark] = useState(false);
  const [benchmarkStatusMsg, setBenchmarkStatusMsg] = useState('');

  // Fetch initial research data
  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError('');
      try {
        const lbRes = await getResearchLeaderboard();
        setLeaderboardData(lbRes.leaderboard || []);
        setMetadata(lbRes.metadata || null);
        
        const hbRes = await getResearchHybridStudy();
        setHybridStudy(hbRes || null);
        
        const repRes = await getResearchReport();
        setReport(repRes || null);
      } catch (err) {
        setError(err.message || 'Lỗi khi tải dữ liệu nghiên cứu NLP.');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // Fetch benchmark samples when filters change
  useEffect(() => {
    async function loadSamples() {
      setSamplesLoading(true);
      try {
        const res = await getResearchBenchmarkSamples(samplePage, 10, sampleCategory, sampleSearch);
        setSamples(res.items || []);
        setSampleTotalPages(res.pages || 1);
      } catch (err) {
        console.error('Failed to load benchmark samples:', err);
      } finally {
        setSamplesLoading(false);
      }
    }
    loadSamples();
  }, [samplePage, sampleCategory, sampleSearch]);

  // Handle triggering benchmark run
  const handleRunBenchmark = async () => {
    if (runningBenchmark) return;
    setRunningBenchmark(true);
    setBenchmarkStatusMsg('Đang kích hoạt tiến trình benchmark nền...');
    try {
      const res = await runResearchBenchmark();
      setBenchmarkStatusMsg(res.message || 'Benchmark đã được kích hoạt chạy nền.');
      setTimeout(() => setBenchmarkStatusMsg(''), 5000);
    } catch (err) {
      setBenchmarkStatusMsg('Gặp lỗi khi kích hoạt benchmark.');
      setTimeout(() => setBenchmarkStatusMsg(''), 3000);
    } finally {
      setRunningBenchmark(false);
    }
  };

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
      name: 'TextRank', group: 'Extractive', complexity: 'O(N^2)', params: 'N/A',
      framework: 'NLTK / NetworkX', trainingTime: '0 (Không cần train)',
      desc: 'Thuật toán trích xuất câu dựa trên đồ thị tương đồng từ vựng (PageRank áp dụng cho văn bản). Cực kỳ nhanh và an toàn.',
      pros: ['Tốc độ xử lý siêu nhanh (~30ms)', 'Không có hiện tượng bịa đặt câu chữ (hallucination)', 'Không tốn tài nguyên phần cứng (CPU/GPU)'],
      cons: ['Không thể diễn đạt lại ý kiến (paraphrasing)', 'Dễ bị đứt gãy mạch liên kết câu', 'Điểm chồng lấp từ vựng bị giới hạn']
    },
    lexrank: {
      name: 'LexRank', group: 'Extractive', complexity: 'O(N^2)', params: 'N/A',
      framework: 'LexRank Base', trainingTime: '0 (Không cần train)',
      desc: 'Phương pháp trích xuất câu sử dụng độ trung tâm đồ thị liên kết câu, kết hợp trọng số tần suất từ nghịch đảo TF-IDF.',
      pros: ['Đo lường tầm quan trọng của câu chính xác hơn TextRank', 'Rất phù hợp cho tóm tắt đa tài liệu', 'Chi phí tính toán siêu thấp'],
      cons: ['Câu trích xuất giữ nguyên gốc, có thể mang thông tin thừa', 'Thiếu liên kết logic giữa các câu cách biệt']
    },
    lsa: {
      name: 'LSA Summarizer', group: 'Extractive', complexity: 'O(N*M*K)', params: 'N/A',
      framework: 'Latent Semantic Analysis', trainingTime: '0 (Không cần train)',
      desc: 'Phân tích ngữ nghĩa tiềm ẩn bằng cách phân tách ma trận SVD từ ma trận đặc trưng câu-từ khóa để tìm chủ đề lõi.',
      pros: ['Nắm bắt được mối quan hệ chủ đề tiềm ẩn', 'Không bị phụ thuộc vào trùng lặp từ vựng chính xác', 'Rất ổn định trên tài liệu khoa học'],
      cons: ['Vẫn là phương pháp trích xuất thuần túy', 'Khó tinh chỉnh hoặc giải thích ma trận toán học trực quan']
    },
    vit5: {
      name: 'ViT5 (Fine-tuned)', group: 'Abstractive', complexity: 'Transformer Attention', params: '220 Million',
      framework: 'PyTorch / HuggingFace', trainingTime: '89.6 phút (Colab T4)',
      desc: 'Mô hình sinh tóm tắt dựa trên kiến trúc T5 tối ưu hóa riêng cho tiếng Việt bởi VietAI. Được tinh chỉnh trên tập dữ liệu VietNews.',
      pros: ['Câu văn mượt mà, tự nhiên giống người viết', 'Khả năng diễn dịch (paraphrasing) tốt', 'Độ nén thông tin cực cao'],
      cons: ['Thời gian phản hồi chậm (6-8 giây)', 'Tốn GPU VRAM (4GB+)', 'Có rủi ro nhỏ xảy ra hiện tượng hallucination']
    },
    mt5: {
      name: 'mT5 (Baseline)', group: 'Abstractive', complexity: 'Transformer Attention', params: '300 Million',
      framework: 'PyTorch / HuggingFace', trainingTime: '19.1 phút (Colab T4)',
      desc: 'Mô hình sinh đa ngôn ngữ của Google (mT5-small). Sử dụng làm mốc so sánh thực nghiệm gốc chưa tinh chỉnh chuyên sâu.',
      pros: ['Hỗ trợ đa ngôn ngữ', 'Kiến trúc chuẩn hóa toàn cầu'],
      cons: ['Bị hiện tượng lặp từ và sinh từ rác rất cao', 'Không hoạt động tốt trên ngữ cảnh tiếng Việt hẹp', 'Factual consistency cực thấp']
    },
    bartpho: {
      name: 'BARTPho (Fine-tuned)', group: 'Abstractive', complexity: 'Transformer Attention', params: '340 Million',
      framework: 'PyTorch / HuggingFace', trainingTime: '70.5 phút (Colab T4)',
      desc: 'Mô hình sinh tóm tắt sử dụng kiến trúc Seq2Seq BART dành riêng cho tiếng Việt của VinAI. Xử lý âm tiết và âm tự nhiên rất tốt.',
      pros: ['Độ tương đồng ngữ nghĩa cao nhất', 'Diễn đạt xuất sắc, không lỗi ngữ pháp tiếng Việt', 'Cực kỳ phù hợp cho tin tức VNExpress'],
      cons: ['Kích thước mô hình rất lớn', 'Thời gian khởi động và sinh tóm tắt lâu nhất (~8 giây)', 'Yêu cầu GPU tối thiểu 6GB VRAM']
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
            Hệ thống phân tích, so sánh hiệu năng 6 mô hình summarization tiếng Việt gốc và mô hình lai trên bộ test chuẩn 1000 mẫu.
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
            <span className="text-2xl font-bold mt-1">{metadata.total_samples} mẫu</span>
          </div>
          <div className="ui-card p-4 flex flex-col justify-between">
            <span className="ui-stat-label">Short (100-500 từ)</span>
            <span className="text-2xl font-bold mt-1 text-blue-500">{metadata.categories?.Short || 0}</span>
          </div>
          <div className="ui-card p-4 flex flex-col justify-between">
            <span className="ui-stat-label">Medium (500-2000 từ)</span>
            <span className="text-2xl font-bold mt-1 text-purple-500">{metadata.categories?.Medium || 0}</span>
          </div>
          <div className="ui-card p-4 flex flex-col justify-between">
            <span className="ui-stat-label">Long (2000-10000 từ)</span>
            <span className="text-2xl font-bold mt-1 text-emerald-500">{metadata.categories?.Long || 0}</span>
          </div>
          <div className="ui-card p-4 flex flex-col justify-between">
            <span className="ui-stat-label">Very Long (10000+ từ)</span>
            <span className="text-2xl font-bold mt-1 text-amber-500">{metadata.categories?.['Very Long'] || 0}</span>
          </div>
        </div>
      )}

      {/* Main Tabs Navigation */}
      <div className="flex border-b border-[var(--border)] overflow-x-auto scrollbar-none gap-2">
        {[
          { id: 'leaderboard', label: 'Bảng xếp hạng (Leaderboard)', icon: Award },
          { id: 'charts', label: 'Biểu đồ trực quan', icon: BarChart3 },
          { id: 'samples', label: 'Bộ Test 1000 Mẫu', icon: FileText },
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
              className={`flex items-center gap-2 px-4 py-3 text-sm font-semibold border-b-2 whitespace-nowrap transition duration-150 ${
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
          </div>

          <div className="ui-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="ui-table-head border-b border-[var(--border)]">
                  <tr>
                    <th className="px-4 py-3 text-center w-12">Hạng</th>
                    <th className="px-6 py-3 text-left">Cấu hình mô hình</th>
                    <th className="px-4 py-3 text-center">Kiểu</th>
                    <th className="px-4 py-3 text-right cursor-pointer select-none" onClick={() => requestSort('rougeL')}>
                      ROUGE-L {sortField === 'rougeL' && (sortAsc ? '↑' : '↓')}
                    </th>
                    <th className="px-4 py-3 text-right cursor-pointer select-none" onClick={() => requestSort('bertscore')}>
                      BERT F1 {sortField === 'bertscore' && (sortAsc ? '↑' : '↓')}
                    </th>
                    <th className="px-4 py-3 text-right cursor-pointer select-none" onClick={() => requestSort('bleu')}>
                      BLEU {sortField === 'bleu' && (sortAsc ? '↑' : '↓')}
                    </th>
                    <th className="px-4 py-3 text-right cursor-pointer select-none" onClick={() => requestSort('latency')}>
                      Độ trễ (s) {sortField === 'latency' && (sortAsc ? '↑' : '↓')}
                    </th>
                    <th className="px-4 py-3 text-right cursor-pointer select-none" onClick={() => requestSort('throughput')}>
                      W/s {sortField === 'throughput' && (sortAsc ? '↑' : '↓')}
                    </th>
                    <th className="px-4 py-3 text-right cursor-pointer select-none" onClick={() => requestSort('compression')}>
                      Tỉ lệ nén {sortField === 'compression' && (sortAsc ? '↑' : '↓')}
                    </th>
                    <th className="px-4 py-3 text-right cursor-pointer select-none" onClick={() => requestSort('faithfulness')}>
                      Trung thực {sortField === 'faithfulness' && (sortAsc ? '↑' : '↓')}
                    </th>
                    <th className="px-4 py-3 text-center cursor-pointer select-none" onClick={() => requestSort('hallucination_pct')}>
                      Rủi ro Bịa đặt {sortField === 'hallucination_pct' && (sortAsc ? '↑' : '↓')}
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-subtle)]">
                  {processedLeaderboard.map((row, idx) => {
                    // Award badges for top 3 overall
                    let medalBadge = null;
                    if (leaderboardFilter === 'all') {
                      if (idx === 0) medalBadge = '🥇';
                      else if (idx === 1) medalBadge = '🥈';
                      else if (idx === 2) medalBadge = '🥉';
                    }
                    
                    const riskType = row.hallucination_pct < 10 ? 'low' : (row.hallucination_pct < 35 ? 'medium' : 'high');
                    const riskText = row.hallucination_pct < 10 ? 'Thấp' : (row.hallucination_pct < 35 ? 'Trung bình' : 'Cao');

                    return (
                      <tr key={row.key} className="ui-table-row">
                        <td className="px-4 py-4 text-center font-bold text-[var(--text-secondary)]">
                          {medalBadge ? <span className="text-lg">{medalBadge}</span> : idx + 1}
                        </td>
                        <td className="px-6 py-4 font-semibold text-[var(--text-primary)]">
                          {row.name}
                        </td>
                        <td className="px-4 py-4 text-center">
                          <span className={`ui-badge text-[10px] ${groupColors[row.group]}`}>
                            {row.group.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-4 text-right font-bold text-blue-600 dark:text-blue-400">
                          {row.rougeL.toFixed(4)}
                        </td>
                        <td className="px-4 py-4 text-right font-semibold text-purple-600 dark:text-purple-400">
                          {row.bertscore.toFixed(4)}
                        </td>
                        <td className="px-4 py-4 text-right text-[var(--text-secondary)]">
                          {row.bleu.toFixed(4)}
                        </td>
                        <td className="px-4 py-4 text-right font-medium text-[var(--text-primary)]">
                          {row.latency.toFixed(3)}s
                        </td>
                        <td className="px-4 py-4 text-right text-[var(--text-secondary)]">
                          {row.throughput.toFixed(1)}
                        </td>
                        <td className="px-4 py-4 text-right text-[var(--text-secondary)]">
                          {(row.compression * 100).toFixed(1)}%
                        </td>
                        <td className="px-4 py-4 text-right font-semibold text-emerald-600 dark:text-emerald-400">
                          {(row.faithfulness * 100).toFixed(1)}%
                        </td>
                        <td className="px-4 py-4 text-center">
                          <span className={`ui-badge ${riskColors[riskType]}`}>
                            {riskText} ({(row.hallucination_pct).toFixed(1)}%)
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: CHARTS */}
      {activeTab === 'charts' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* ROUGE Performance Chart */}
          <div className="ui-card p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-bold text-[var(--text-primary)] mb-1 flex items-center gap-1">
                <BarChart3 className="w-4 h-4 text-blue-500" />
                So sánh Điểm ROUGE (Độ chồng lấp từ vựng)
              </h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">Điểm số ROUGE-1, ROUGE-2 và ROUGE-L càng cao càng tốt.</p>
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

          {/* BERTScore Similarity Chart */}
          <div className="ui-card p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-bold text-[var(--text-primary)] mb-1 flex items-center gap-1">
                <Activity className="w-4 h-4 text-purple-500" />
                Độ tương đồng ngữ nghĩa (BERTScore F1)
              </h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">Đo lường độ khớp ý tưởng ngữ nghĩa thay vì chỉ so sánh mặt chữ.</p>
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

          {/* Latency Comparison (Time) */}
          <div className="ui-card p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-bold text-[var(--text-primary)] mb-1 flex items-center gap-1">
                <Zap className="w-4 h-4 text-amber-500" />
                Tốc độ xử lý (Độ trễ trung bình)
              </h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">Độ trễ tính bằng giây. Giá trị thấp hơn thể hiện tốc độ nhanh hơn.</p>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={leaderboardData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
                <XAxis type="number" tick={{ fill: chartTheme.axis }} />
                <YAxis dataKey="key" type="category" tick={{ fontSize: 10, fill: chartTheme.axis }} width={80} />
                <Tooltip contentStyle={chartTheme.tooltipStyle} />
                <Bar dataKey="latency" fill="#f59e0b" name="Độ trễ (giây)">
                  {leaderboardData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.group === 'extractive' ? '#3b82f6' : (entry.group === 'abstractive' ? '#f59e0b' : '#10b981')} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Quality vs Speed Scatter Plot */}
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
                {/* Latency reversed on axis so left is slow, right is fast */}
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

          {/* Radar Chart (Multidimensional Comparison) */}
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
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={[
                  { subject: 'ROUGE-L', lsa: 0.45, vit5: 0.36, hybrid: 0.38 },
                  { subject: 'BERTScore', lsa: 0.75, vit5: 0.88, hybrid: 0.90 },
                  { subject: 'Tốc độ', lsa: 0.98, vit5: 0.25, hybrid: 0.58 }, // normalized
                  { subject: 'Độ nén', lsa: 0.68, vit5: 0.72, hybrid: 0.76 },
                  { subject: 'Độ trung thực', lsa: 1.0, vit5: 0.84, hybrid: 0.94 }
                ]}>
                  <PolarGrid stroke={chartTheme.grid} />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: chartTheme.axis, fontSize: 12 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 1.0]} tick={{ fill: chartTheme.axis }} />
                  
                  <Radar name="LSA (Extractive)" dataKey="lsa" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} />
                  <Radar name="ViT5 (Abstractive)" dataKey="vit5" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.25} />
                  <Radar name="LSA ➔ ViT5 (Hybrid)" dataKey="hybrid" stroke="#10b981" fill="#10b981" fillOpacity={0.25} />
                  
                  <Legend wrapperStyle={{ color: chartTheme.axis }} />
                  <Tooltip contentStyle={chartTheme.tooltipStyle} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: SAMPLES */}
      {activeTab === 'samples' && (
        <div className="space-y-6">
          {/* Filtering bar */}
          <div className="flex flex-wrap items-center justify-between gap-4 bg-[var(--bg-elevated)] p-4 rounded-xl border border-[var(--border)]">
            <div className="flex items-center gap-3 flex-1 min-w-[280px]">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-2.5 w-4 h-4 text-[var(--text-muted)]" />
                <input
                  type="text"
                  placeholder="Tìm kiếm tài liệu thử nghiệm theo tiêu đề, nội dung..."
                  value={sampleSearch}
                  onChange={(e) => { setSampleSearch(e.target.value); setSamplePage(1); }}
                  className="ui-input pl-9"
                />
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-[var(--text-muted)]">Độ dài:</span>
              <select
                value={sampleCategory}
                onChange={(e) => { setSampleCategory(e.target.value); setSamplePage(1); }}
                className="ui-select !py-1.5 w-40"
              >
                <option value="All">Tất cả</option>
                <option value="Short">Short (100-500 từ)</option>
                <option value="Medium">Medium (500-2000 từ)</option>
                <option value="Long">Long (2000-10000 từ)</option>
                <option value="Very Long">Very Long (10000+ từ)</option>
              </select>
            </div>
          </div>

          {/* Table list */}
          <div className="ui-card overflow-hidden">
            {samplesLoading ? (
              <div className="flex flex-col items-center justify-center py-24 text-[var(--text-muted)] space-y-2">
                <Loader2 className="animate-spin w-8 h-8 text-indigo-500" />
                <span className="text-xs">Đang tải danh sách mẫu...</span>
              </div>
            ) : samples.length === 0 ? (
              <div className="p-12 text-center text-[var(--text-muted)]">
                Không tìm thấy mẫu thử nghiệm nào phù hợp với bộ lọc hiện tại.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="ui-table-head border-b border-[var(--border)]">
                    <tr>
                      <th className="px-6 py-3 text-left">Mã mẫu</th>
                      <th className="px-6 py-3 text-left">Tiêu đề / Nguồn tài liệu</th>
                      <th className="px-4 py-3 text-center">Phân loại</th>
                      <th className="px-4 py-3 text-right">Độ dài văn bản</th>
                      <th className="px-4 py-3 text-right">Độ dài tóm tắt gốc</th>
                      <th className="px-6 py-3 text-center w-24">Chi tiết</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--border-subtle)]">
                    {samples.map(sample => {
                      const words = sample.article.split(' ').length;
                      const refWords = sample.summary.split(' ').length;
                      
                      let badgeColor = 'bg-gray-100 text-gray-800';
                      if (sample.category === 'Short') badgeColor = 'bg-blue-50 text-blue-700 border border-blue-100 dark:bg-blue-950/20 dark:text-blue-400 dark:border-blue-900/30';
                      else if (sample.category === 'Medium') badgeColor = 'bg-purple-50 text-purple-700 border border-purple-100 dark:bg-purple-950/20 dark:text-purple-400 dark:border-purple-900/30';
                      else if (sample.category === 'Long') badgeColor = 'bg-emerald-50 text-emerald-700 border border-emerald-100 dark:bg-emerald-950/20 dark:text-emerald-400 dark:border-emerald-900/30';
                      else if (sample.category === 'Very Long') badgeColor = 'bg-amber-50 text-amber-700 border border-amber-100 dark:bg-amber-950/20 dark:text-amber-400 dark:border-amber-900/30';

                      return (
                        <tr key={sample.id} className="ui-table-row">
                          <td className="px-6 py-3.5 font-mono text-xs font-semibold text-indigo-500">{sample.id}</td>
                          <td className="px-6 py-3.5 font-medium text-[var(--text-primary)] max-w-xs truncate">{sample.title}</td>
                          <td className="px-4 py-3.5 text-center">
                            <span className={`ui-badge text-[10px] ${badgeColor}`}>
                              {sample.category}
                            </span>
                          </td>
                          <td className="px-4 py-3.5 text-right text-[var(--text-secondary)]">{words} từ</td>
                          <td className="px-4 py-3.5 text-right text-[var(--text-secondary)]">{refWords} từ</td>
                          <td className="px-6 py-3.5 text-center">
                            <button
                              onClick={() => { setSelectedSample(sample); setModalModelKey('vit5'); }}
                              className="ui-btn-ghost !p-1.5 hover:text-indigo-500"
                            >
                              <Maximize2 className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            
            {/* Pagination footer */}
            <div className="px-6 py-4 border-t border-[var(--border)] bg-[var(--surface-inset)] flex items-center justify-between">
              <span className="text-xs text-[var(--text-muted)]">
                Hiển thị trang {samplePage} / {sampleTotalPages}
              </span>
              <div className="flex gap-2">
                <button
                  disabled={samplePage <= 1}
                  onClick={() => setSamplePage(prev => Math.max(1, prev - 1))}
                  className="ui-btn-secondary !py-1 px-3 text-xs disabled:opacity-40"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                  Trước
                </button>
                <button
                  disabled={samplePage >= sampleTotalPages}
                  onClick={() => setSamplePage(prev => Math.min(sampleTotalPages, prev + 1))}
                  className="ui-btn-secondary !py-1 px-3 text-xs disabled:opacity-40"
                >
                  Sau
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

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
                <p className="text-[11px] text-blue-700 dark:text-blue-500 mt-1">Thuật toán LSA/TextRank rút lấy top 25 câu quan trọng</p>
              </div>
              <ChevronRight className="w-5 h-5 text-indigo-500 transform rotate-90 md:rotate-0" />
              <div className="p-4 bg-purple-50 dark:bg-purple-950/20 border border-purple-200 dark:border-purple-900/30 rounded-xl flex-1 w-full md:w-auto">
                <p className="text-xs text-purple-500 font-bold">Giai đoạn 2: Abstractive</p>
                <p className="text-sm font-bold mt-1 text-purple-900 dark:text-purple-400">Sinh tóm tắt tự nhiên</p>
                <p className="text-[11px] text-purple-700 dark:text-purple-500 mt-1">Mô hình ViT5 viết lại văn bản gọn gàng, mạch lạc</p>
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
                      const spec = modelsSpecifications[modelKey.replace('_vit5', '')] || { name: modelKey.toUpperCase().replace('_', ' ➔ ') };
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

export default Compare;
