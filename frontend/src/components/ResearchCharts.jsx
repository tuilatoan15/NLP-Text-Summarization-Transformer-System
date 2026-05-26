/**
 * ResearchCharts.jsx - Research visualization charts
 * ROUGE, Semantic, Inference Time, Compression Ratio, Radar Chart
 */

import React, { useState } from 'react';
import {
  BarChart, Bar, LineChart, Line, RadarChart, Radar, ScatterChart, Scatter,
  CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, Cell, PolarAngleAxis,
  PolarGrid, PolarRadiusAxis,
} from 'recharts';
import { motion } from 'framer-motion';

const ResearchCharts = ({ data, reference }) => {
  const [selectedChart, setSelectedChart] = useState('rouge');

  if (!data) {
    return (
      <div className="text-center text-slate-400 py-12">
        No chart data available
      </div>
    );
  }

  const chartOptions = [
    { id: 'rouge', label: '📊 ROUGE Comparison', icon: '📊' },
    { id: 'semantic', label: '🧠 Semantic Similarity', icon: '🧠' },
    { id: 'time', label: '⚡ Inference Time', icon: '⚡' },
    { id: 'compression', label: '📉 Compression Ratio', icon: '📉' },
    { id: 'radar', label: '🎯 Radar Chart', icon: '🎯' },
  ];

  return (
    <div className="space-y-6">
      {/* Chart Selector */}
      <div className="flex gap-2 overflow-x-auto pb-4">
        {chartOptions.map((option) => (
          <button
            key={option.id}
            onClick={() => setSelectedChart(option.id)}
            className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-all ${
              selectedChart === option.id
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>

      {/* Chart Container */}
      <motion.div
        key={selectedChart}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-slate-800/50 border border-slate-700 rounded-2xl p-6"
      >
        {selectedChart === 'rouge' && <ROUGEChart data={data.rouge_comparison} />}
        {selectedChart === 'semantic' && <SemanticChart data={data.semantic_comparison} />}
        {selectedChart === 'time' && <TimeChart data={data.inference_time} />}
        {selectedChart === 'compression' && <CompressionChart data={data.compression_ratio} />}
        {selectedChart === 'radar' && <RadarChartComp data={data.radar_chart} />}
      </motion.div>
    </div>
  );
};

/**
 * ROUGE Comparison Chart
 */
const ROUGEChart = ({ data }) => {
  if (!data || data.length === 0) {
    return <div className="text-slate-400 text-center py-8">No ROUGE data</div>;
  }

  const COLORS = {
    extractive: '#10b981',
    abstractive: '#3b82f6',
  };

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-bold text-white mb-4">ROUGE Score Comparison</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
            <XAxis dataKey="model" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid #475569',
                borderRadius: '8px',
              }}
            />
            <Legend />
            <Bar dataKey="rouge1" name="ROUGE-1" fill="#f59e0b" />
            <Bar dataKey="rouge2" name="ROUGE-2" fill="#06b6d4" />
            <Bar dataKey="rougeL" name="ROUGE-L" fill="#8b5cf6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Analysis */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
        <InsightBox
          title="📈 Highest ROUGE-1"
          models={data
            .sort((a, b) => b.rouge1 - a.rouge1)
            .slice(0, 2)
            .map((m) => `${m.model} (${(m.rouge1 * 100).toFixed(1)}%)`)}
        />
        <InsightBox
          title="🎯 Extractive vs Abstractive"
          content={`Abstractive typically scores higher due to semantic paraphrasing accuracy`}
        />
      </div>
    </div>
  );
};

/**
 * Semantic Similarity Chart
 */
const SemanticChart = ({ data }) => {
  if (!data || data.length === 0) {
    return <div className="text-slate-400 text-center py-8">No semantic data</div>;
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-bold text-white mb-4">Semantic Similarity Scores</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
          <XAxis dataKey="model" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" domain={[0, 1]} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid #475569',
              borderRadius: '8px',
            }}
            formatter={(value) => value.toFixed(4)}
          />
          <Legend />
          <Bar dataKey="bertscore" name="BERTScore" fill="#06b6d4" />
          <Bar dataKey="semantic" name="Semantic (SBERT)" fill="#10b981" />
        </BarChart>
      </ResponsiveContainer>

      {/* Insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
        <InsightBox
          title="🧠 Best Semantic Quality"
          models={data
            .sort((a, b) => b.semantic - a.semantic)
            .slice(0, 2)
            .map((m) => `${m.model} (${(m.semantic * 100).toFixed(1)}%)`)}
        />
        <InsightBox
          title="📌 Key Finding"
          content="Abstractive models achieve higher semantic similarity due to understanding context"
        />
      </div>
    </div>
  );
};

/**
 * Inference Time Chart
 */
const TimeChart = ({ data }) => {
  if (!data || data.length === 0) {
    return <div className="text-slate-400 text-center py-8">No time data</div>;
  }

  // Separate extractive and abstractive
  const extractiveData = data.filter((d) => d.type === 'extractive');
  const abstractiveData = data.filter((d) => d.type === 'abstractive');

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-bold text-white mb-4">Inference Time Comparison (seconds)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
          <XAxis dataKey="model" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" label={{ value: 'Time (s)', angle: -90, position: 'insideLeft' }} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid #475569',
              borderRadius: '8px',
            }}
            formatter={(value) => `${value.toFixed(3)}s`}
          />
          <Bar dataKey="time" fill="#f59e0b" name="Inference Time" />
        </BarChart>
      </ResponsiveContainer>

      {/* Insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
        <InsightBox
          title="⚡ Fastest Extractive"
          models={extractiveData
            .sort((a, b) => a.time - b.time)
            .slice(0, 1)
            .map((m) => `${m.model} (${m.time.toFixed(3)}s)`)}
        />
        <InsightBox
          title="🤖 Abstractive Speed"
          content={`Abstractive ~${(abstractiveData[0]?.time / (extractiveData[0]?.time || 1)).toFixed(0)}x slower but with better quality`}
        />
      </div>
    </div>
  );
};

/**
 * Compression Ratio Chart
 */
const CompressionChart = ({ data }) => {
  if (!data || data.length === 0) {
    return <div className="text-slate-400 text-center py-8">No compression data</div>;
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-bold text-white mb-4">Compression Ratio (lower = more compressed)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
          <XAxis dataKey="model" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" domain={[0, 1]} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid #475569',
              borderRadius: '8px',
            }}
            formatter={(value) => `${(value * 100).toFixed(1)}%`}
          />
          <Bar dataKey="compression" fill="#10b981" name="Compression Ratio" />
        </BarChart>
      </ResponsiveContainer>

      {/* Insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
        <InsightBox
          title="📉 Most Compressed"
          models={data
            .sort((a, b) => a.compression - b.compression)
            .slice(0, 2)
            .map((m) => `${m.model} (${(m.compression * 100).toFixed(1)}%)`)}
        />
        <InsightBox
          title="📊 Trade-off"
          content="Aggressive compression may lose important information. Optimal: 30-50%"
        />
      </div>
    </div>
  );
};

/**
 * Radar Chart for multi-dimensional comparison
 */
const RadarChartComp = ({ data }) => {
  if (!data || data.length === 0) {
    return <div className="text-slate-400 text-center py-8">No radar data</div>;
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-bold text-white mb-4">Multi-Dimensional Comparison (Radar Chart)</h3>
      <ResponsiveContainer width="100%" height={350}>
        <RadarChart data={data}>
          <PolarGrid stroke="#475569" />
          <PolarAngleAxis dataKey="model" stroke="#94a3b8" />
          <PolarRadiusAxis angle={90} domain={[0, 1]} stroke="#94a3b8" />
          <Radar name="ROUGE" dataKey="rouge" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.25} />
          <Radar name="Semantic" dataKey="semantic" stroke="#10b981" fill="#10b981" fillOpacity={0.25} />
          <Radar name="BERTScore" dataKey="bertscore" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} />
          <Radar name="Speed" dataKey="speed" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.25} />
          <Legend />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid #475569',
              borderRadius: '8px',
            }}
            formatter={(value) => value.toFixed(3)}
          />
        </RadarChart>
      </ResponsiveContainer>

      <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4 text-sm">
        <p className="text-blue-400 font-bold mb-2">📌 How to Read</p>
        <ul className="text-slate-300 space-y-1 text-xs">
          <li>• Larger area = better overall performance</li>
          <li>• Each dimension represents different metrics (ROUGE, Semantic, Speed)</li>
          <li>• Compare the shapes to see trade-offs between models</li>
        </ul>
      </div>
    </div>
  );
};

/**
 * InsightBox - Quick insight display
 */
const InsightBox = ({ title, models = [], content = '' }) => (
  <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-4">
    <p className="text-sm font-bold text-slate-300 mb-3">{title}</p>
    {models.length > 0 ? (
      <ul className="space-y-2">
        {models.map((model, idx) => (
          <li key={idx} className="text-sm text-cyan-400">
            • {model}
          </li>
        ))}
      </ul>
    ) : (
      <p className="text-sm text-slate-400">{content}</p>
    )}
  </div>
);

export default ResearchCharts;
