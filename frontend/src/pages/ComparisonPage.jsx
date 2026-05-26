/**
 * ComparisonPage.jsx - Research Comparison Dashboard
 * So sánh chi tiết Extractive vs Abstractive Summarization
 */

import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Download, Share2, Play, Loader } from 'lucide-react';
import ModelCard from '../components/ModelCard';
import ResearchCharts from '../components/ResearchCharts';
import ModelExplanation from '../components/ModelExplanation';
import ReportGenerator from '../components/ReportGenerator';

const ComparisonPage = () => {
  const [inputText, setInputText] = useState('');
  const [reference, setReference] = useState('');
  const [loading, setLoading] = useState(false);
  const [comparisonData, setComparisonData] = useState(null);
  const [activeTab, setActiveTab] = useState('comparison'); // comparison, charts, explanation
  const [showExplanation, setShowExplanation] = useState({});
  const fileInputRef = useRef(null);

  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const handleRunComparison = async () => {
    if (!inputText.trim()) {
      alert('Vui lòng nhập văn bản');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/research/compare/detailed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          reference: reference || null,
          extractive_sentences: 5,
          max_abstractive_length: 150,
          target_length_ratio: 50,
          use_length_ratio: true,
          include_visualization: true,
          save_result: false,
        }),
      });

      if (!response.ok) throw new Error('Comparison failed');
      const data = await response.json();
      setComparisonData(data);
      setActiveTab('comparison');
    } catch (error) {
      console.error('Error:', error);
      alert('Lỗi trong quá trình so sánh');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setInputText(e.target?.result || '');
      };
      reader.readAsText(file);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-slate-900/95 backdrop-blur border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                🔬 Research Comparison
              </h1>
              <p className="text-sm text-slate-400 mt-1">
                Extractive vs Abstractive Summarization Analysis
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors text-sm font-medium"
              >
                📄 Upload
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt"
                onChange={handleFileUpload}
                className="hidden"
              />
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Input Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8"
        >
          {/* Input Document */}
          <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-2xl p-6">
            <label className="block text-sm font-semibold text-slate-300 mb-3">
              📝 Input Document
            </label>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Nhập hoặc dán văn bản cần so sánh..."
              className="w-full h-32 bg-slate-700 border border-slate-600 rounded-lg p-4 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
            />
            <div className="mt-3 text-xs text-slate-400">
              {inputText.length} ký tự | {inputText.split(/\s+/).filter(w => w).length} từ
            </div>
          </div>

          {/* Reference Summary (Optional) */}
          <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-2xl p-6">
            <label className="block text-sm font-semibold text-slate-300 mb-3">
              📋 Reference Summary (Tùy chọn)
            </label>
            <textarea
              value={reference}
              onChange={(e) => setReference(e.target.value)}
              placeholder="Nhập tóm tắt tham chiếu để tính ROUGE..."
              className="w-full h-32 bg-slate-700 border border-slate-600 rounded-lg p-4 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
            />
            <div className="mt-3 text-xs text-slate-400">
              {reference.length} ký tự | {reference.split(/\s+/).filter(w => w).length} từ
            </div>
          </div>
        </motion.div>

        {/* Run Button */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex justify-center mb-8"
        >
          <button
            onClick={handleRunComparison}
            disabled={loading || !inputText.trim()}
            className={`px-8 py-3 rounded-lg font-semibold flex items-center gap-2 transition-all ${
              loading || !inputText.trim()
                ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white shadow-lg hover:shadow-cyan-500/50'
            }`}
          >
            {loading ? (
              <>
                <Loader size={20} className="animate-spin" />
                Đang so sánh...
              </>
            ) : (
              <>
                <Play size={20} />
                Chạy So Sánh
              </>
            )}
          </button>
        </motion.div>

        {/* Results Section */}
        <AnimatePresence mode="wait">
          {comparisonData && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-8"
            >
              {/* Tabs */}
              <div className="flex gap-2 border-b border-slate-700">
                {[
                  { id: 'comparison', label: '⚖️ Comparison' },
                  { id: 'charts', label: '📊 Charts' },
                  { id: 'explanation', label: '📚 Explanation' },
                  { id: 'report', label: '📄 Report' },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`px-4 py-3 font-medium transition-colors ${
                      activeTab === tab.id
                        ? 'text-blue-400 border-b-2 border-blue-400'
                        : 'text-slate-400 hover:text-slate-300'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Comparison Tab */}
              {activeTab === 'comparison' && (
                <ComparisonView data={comparisonData} onToggleExplanation={setShowExplanation} />
              )}

              {/* Charts Tab */}
              {activeTab === 'charts' && (
                <ResearchCharts data={comparisonData.chart_data} reference={comparisonData.reference_summary} />
              )}

              {/* Explanation Tab */}
              {activeTab === 'explanation' && (
                <ModelExplanation />
              )}

              {/* Report Tab */}
              {activeTab === 'report' && (
                <ReportGenerator data={null} comparisonData={comparisonData} />
              )}

              {/* Export Buttons */}
              <div className="flex gap-4 justify-center pt-8 border-t border-slate-700">
                <button className="px-6 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors flex items-center gap-2">
                  <Download size={18} />
                  Export CSV
                </button>
                <button className="px-6 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors flex items-center gap-2">
                  <Share2 size={18} />
                  Share Report
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
};

/**
 * ComparisonView - LEFT (Extractive) vs RIGHT (Abstractive)
 */
const ComparisonView = ({ data, onToggleExplanation }) => {
  const extractiveModels = Object.entries(data.extractive_results || {});
  const abstractiveModels = Object.entries(data.abstractive_results || {});

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* LEFT: EXTRACTIVE */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="space-y-6"
      >
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-emerald-400">
            🎯 Extractive (Trích rút)
          </h2>
          <p className="text-sm text-slate-400 mt-2">
            Lấy câu quan trọng từ văn bản gốc
          </p>
        </div>

        {extractiveModels.length > 0 ? (
          extractiveModels.map(([modelName, result]) => (
            <ModelCard
              key={modelName}
              type="extractive"
              modelName={modelName}
              result={result}
              metrics={data.all_metrics[modelName]}
              onToggleExplanation={onToggleExplanation}
            />
          ))
        ) : (
          <div className="text-center text-slate-400 py-8">
            Không có kết quả extractive
          </div>
        )}
      </motion.div>

      {/* RIGHT: ABSTRACTIVE */}
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="space-y-6"
      >
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-blue-400">
            🤖 Abstractive (Diễn giải)
          </h2>
          <p className="text-sm text-slate-400 mt-2">
            AI tự viết lại văn bản bằng cách khác
          </p>
        </div>

        {abstractiveModels.length > 0 ? (
          abstractiveModels.map(([modelName, result]) => (
            <ModelCard
              key={modelName}
              type="abstractive"
              modelName={modelName}
              result={result}
              metrics={data.all_metrics[modelName]}
              onToggleExplanation={onToggleExplanation}
            />
          ))
        ) : (
          <div className="text-center text-slate-400 py-8">
            Không có kết quả abstractive
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default ComparisonPage;
