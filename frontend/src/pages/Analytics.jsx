import React, { useState } from 'react';
import {
  BarChart, Bar, LineChart, Line, RadarChart, Radar, PieChart, Pie, Cell,
  CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  PolarAngleAxis, PolarGrid, PolarRadiusAxis
} from 'recharts';
import { TrendingUp, Download, Filter } from 'lucide-react';

const mockMetricsData = [
  { model: 'ViT5', rouge1: 0.68, rouge2: 0.52, rougeL: 0.65, bertScore: 0.72 },
  { model: 'mT5', rouge1: 0.65, rouge2: 0.49, rougeL: 0.62, bertScore: 0.70 },
  { model: 'BARTPho', rouge1: 0.70, rouge2: 0.54, rougeL: 0.67, bertScore: 0.73 },
  { model: 'TextRank', rouge1: 0.58, rouge2: 0.42, rougeL: 0.55, bertScore: 0.61 },
  { model: 'LexRank', rouge1: 0.60, rouge2: 0.44, rougeL: 0.57, bertScore: 0.63 },
];

const mockTimeSeriesData = [
  { date: '2025-04-01', avgRougeL: 0.58, avgBertScore: 0.62, count: 24 },
  { date: '2025-04-05', avgRougeL: 0.62, avgBertScore: 0.65, count: 32 },
  { date: '2025-04-10', avgRougeL: 0.65, avgBertScore: 0.68, count: 45 },
  { date: '2025-04-15', avgRougeL: 0.67, avgBertScore: 0.70, count: 58 },
  { date: '2025-04-20', avgRougeL: 0.70, avgBertScore: 0.72, count: 72 },
];

const mockRadarData = [
  { metric: 'ROUGE-1', ViT5: 0.68, BARTPho: 0.70, mT5: 0.65 },
  { metric: 'ROUGE-2', ViT5: 0.52, BARTPho: 0.54, mT5: 0.49 },
  { metric: 'ROUGE-L', ViT5: 0.65, BARTPho: 0.67, mT5: 0.62 },
  { metric: 'BERTScore', ViT5: 0.72, BARTPho: 0.73, mT5: 0.70 },
  { metric: 'Semantic', ViT5: 0.71, BARTPho: 0.74, mT5: 0.69 },
];

const StatBox = ({ title, value, trend, unit = '' }) => (
  <div className="bg-white rounded-xl border border-gray-200 p-6">
    <p className="text-sm text-gray-600 mb-2">{title}</p>
    <div className="flex items-baseline gap-2">
      <span className="text-3xl font-bold text-gray-900">{value}</span>
      {unit && <span className="text-sm text-gray-500">{unit}</span>}
    </div>
    {trend && (
      <p className="text-xs text-green-600 mt-2 flex items-center gap-1">
        <TrendingUp size={14} /> {trend}
      </p>
    )}
  </div>
);

const Analytics = () => {
  const [timeRange, setTimeRange] = useState('30d');
  const [selectedModel, setSelectedModel] = useState('all');

  return (
    <div className="space-y-6 pb-12">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 mb-1">Phân tích & Báo cáo</h1>
        <p className="text-sm text-gray-500">Đánh giá sâu hơn về hiệu suất các mô hình</p>
      </div>

      {/* Header with Filters */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex gap-2">
          {['7d', '30d', '90d', 'all'].map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                timeRange === range
                  ? 'bg-blue-600 text-white'
                  : 'bg-white border border-gray-200 text-gray-700 hover:border-gray-300'
              }`}
            >
              {range === '7d' ? '7 ngày' : range === '30d' ? '30 ngày' : range === '90d' ? '90 ngày' : 'Tất cả'}
            </button>
          ))}
        </div>
        <button className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition">
          <Download size={16} /> Export
        </button>
      </div>

      {/* Key Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatBox title="Tổng số chạy" value="742" trend="+15% so với tuần trước" />
        <StatBox title="ROUGE-L TB" value="0.67" unit="score" trend="+0.05" />
        <StatBox title="BERTScore TB" value="0.70" unit="score" trend="+0.03" />
        <StatBox title="Thời gian TB" value="2.3" unit="s" trend="-0.2s" />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Metrics Comparison */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-gray-900 mb-4">Metrics Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={mockMetricsData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="model" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="rouge1" fill="#3b82f6" />
              <Bar dataKey="rouge2" fill="#10b981" />
              <Bar dataKey="rougeL" fill="#f59e0b" />
              <Bar dataKey="bertScore" fill="#8b5cf6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Time Series */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-gray-900 mb-4">Performance Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockTimeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="avgRougeL" stroke="#f59e0b" strokeWidth={2} name="Avg ROUGE-L" />
              <Line type="monotone" dataKey="avgBertScore" stroke="#8b5cf6" strokeWidth={2} name="Avg BERTScore" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Radar Chart */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-gray-900 mb-4">Model Comparison Radar</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={mockRadarData}>
              <PolarGrid stroke="#e5e7eb" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
              <PolarRadiusAxis angle={90} domain={[0, 1]} />
              <Radar name="ViT5" dataKey="ViT5" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} />
              <Radar name="BARTPho" dataKey="BARTPho" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
              <Radar name="mT5" dataKey="mT5" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.2} />
              <Legend />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Model Distribution */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-gray-900 mb-4">Usage Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={[
                  { name: 'ViT5', value: 35 },
                  { name: 'BARTPho', value: 30 },
                  { name: 'mT5', value: 20 },
                  { name: 'TextRank', value: 10 },
                  { name: 'Others', value: 5 },
                ]}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name} (${value}%)`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                <Cell fill="#3b82f6" />
                <Cell fill="#10b981" />
                <Cell fill="#f59e0b" />
                <Cell fill="#8b5cf6" />
                <Cell fill="#ef4444" />
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Summary Table */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
          <h3 className="text-sm font-semibold text-gray-900">Chi tiết theo mô hình</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-xs uppercase text-gray-600 border-b border-gray-200">
              <tr>
                <th className="px-6 py-3 text-left">Model</th>
                <th className="px-6 py-3 text-center">Lần chạy</th>
                <th className="px-6 py-3 text-center">ROUGE-L TB</th>
                <th className="px-6 py-3 text-center">BERTScore TB</th>
                <th className="px-6 py-3 text-center">Thời gian TB</th>
                <th className="px-6 py-3 text-center">Xếp hạng</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {mockMetricsData.map((row, idx) => (
                <tr key={row.model} className="hover:bg-gray-50 transition">
                  <td className="px-6 py-3 font-medium text-gray-900">{row.model}</td>
                  <td className="px-6 py-3 text-center text-gray-600">{Math.floor(Math.random() * 200) + 50}</td>
                  <td className="px-6 py-3 text-center font-semibold text-blue-600">{row.rougeL.toFixed(3)}</td>
                  <td className="px-6 py-3 text-center font-semibold text-green-600">{row.bertScore.toFixed(3)}</td>
                  <td className="px-6 py-3 text-center text-gray-600">{(Math.random() * 3 + 1).toFixed(2)}s</td>
                  <td className="px-6 py-3 text-center">
                    <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-blue-100 text-blue-700 font-bold text-xs">
                      {idx + 1}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
