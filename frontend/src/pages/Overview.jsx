import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts';
import {
  Bot, Cpu, Activity, Clock,
  BrainCircuit, BookOpen, Database, PenTool,
  CheckCircle2, XCircle, Loader2
} from 'lucide-react';
import { getHealth, getMetrics } from '../services/apiService';

const mockDailyData = Array.from({ length: 14 }).map((_, i) => ({
  name: `4/${11 + i}`,
  success: Math.floor(Math.random() * 50) + 20,
  failed: Math.floor(Math.random() * 10),
  critical: Math.floor(Math.random() * 15),
  high: Math.floor(Math.random() * 25),
  medium: Math.floor(Math.random() * 30),
  low: Math.floor(Math.random() * 20),
  todo: Math.floor(Math.random() * 20),
  progress: Math.floor(Math.random() * 30),
  done: Math.floor(Math.random() * 40),
  blocked: Math.floor(Math.random() * 5),
}));

const StatCard = ({ title, value, subtext, icon: Icon, color, trend }) => (
  <div className="bg-white p-5 rounded-2xl shadow-sm border border-gray-100 flex flex-col justify-between">
    <div className="flex justify-between items-start mb-4">
      <div>
        <p className="text-[11px] font-bold text-gray-400 uppercase tracking-wider mb-1">{title}</p>
        <h3 className="text-3xl font-bold text-gray-900">{value}</h3>
      </div>
      <div className={`p-2.5 rounded-xl text-white ${color}`}>
        <Icon className="w-5 h-5" />
      </div>
    </div>
    <p className="text-xs text-gray-500 font-medium">
      <span className={trend >= 0 ? 'text-emerald-500 font-semibold' : 'text-amber-500 font-semibold'}>
        {trend > 0 ? '+' : ''}{trend}{trend.toString().includes('%') || trend.toString().includes('s') ? '' : ' '}
      </span>{' '}
      so với tuần trước
    </p>
  </div>
);

const Overview = () => {
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [healthData, metricsData] = await Promise.all([getHealth(), getMetrics()]);
        setHealth(healthData);
        setMetrics(metricsData);
      } catch (err) {
        console.error('Failed to fetch overview data:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const agentCount = 6;
  const modelCount = metrics?.models_preloaded ? Object.keys(metrics.model_load_times || {}).length : 3;
  const flowsRun = 847;
  const avgResponseTime = metrics?.model_load_times
    ? (Object.values(metrics.model_load_times).reduce((a, b) => a + b, 0) / Object.keys(metrics.model_load_times).length).toFixed(1)
    : '2.4';

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="animate-spin mr-2" /> Đang tải dữ liệu...
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-12">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 mb-1">Bảng điều khiển</h1>
        <p className="text-sm text-gray-500">
          {health?.status === 'ok'
            ? '✓ Hệ thống hoạt động bình thường'
            : '✗ Có vấn đề với hệ thống'}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard title="Agents Hoạt động" value={agentCount} trend="+3" icon={Bot} color="bg-blue-500" />
        <StatCard title="Models Kết nối" value={modelCount} trend="+1" icon={Cpu} color="bg-cyan-500" />
        <StatCard title="Flows Đã chạy" value={flowsRun} trend="+12%" icon={Activity} color="bg-emerald-500" />
        <StatCard title="Phản hồi TB" value={`${avgResponseTime}s`} trend="-0.3s" icon={Clock} color="bg-amber-500" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-5">
        <div className="bg-white p-5 rounded-2xl shadow-sm border border-gray-100">
          <div className="mb-4">
            <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider">Agent Runs</h3>
            <p className="text-[11px] text-gray-400">14 ngày gần nhất</p>
          </div>
          <div className="h-40 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockDailyData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
                <XAxis dataKey="name" tick={{fontSize: 10}} tickLine={false} axisLine={false} />
                <Tooltip cursor={{fill: '#f9fafb'}} />
                <Bar dataKey="success" stackId="a" fill="#10b981" radius={[0, 0, 4, 4]} />
                <Bar dataKey="failed" stackId="a" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex gap-4 mt-3 text-[10px] font-medium text-gray-500">
            <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-emerald-500"></div> Succeeded</div>
            <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-red-500"></div> Failed</div>
          </div>
        </div>

        <div className="bg-white p-5 rounded-2xl shadow-sm border border-gray-100">
          <div className="mb-4">
            <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider">Theo ưu tiên</h3>
            <p className="text-[11px] text-gray-400">14 ngày gần nhất</p>
          </div>
          <div className="h-40 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockDailyData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
                <XAxis dataKey="name" tick={{fontSize: 10}} tickLine={false} axisLine={false} />
                <Tooltip cursor={{fill: '#f9fafb'}} />
                <Bar dataKey="critical" stackId="a" fill="#475569" radius={[0, 0, 4, 4]} />
                <Bar dataKey="high" stackId="a" fill="#ef4444" />
                <Bar dataKey="medium" stackId="a" fill="#f59e0b" />
                <Bar dataKey="low" stackId="a" fill="#eab308" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex gap-4 mt-3 text-[10px] font-medium text-gray-500">
            <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-slate-600"></div> Critical</div>
            <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-red-500"></div> High</div>
          </div>
        </div>

        <div className="bg-white p-5 rounded-2xl shadow-sm border border-gray-100">
          <div className="mb-4">
            <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider">Theo trạng thái</h3>
            <p className="text-[11px] text-gray-400">14 ngày gần nhất</p>
          </div>
          <div className="h-40 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockDailyData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
                <XAxis dataKey="name" tick={{fontSize: 10}} tickLine={false} axisLine={false} />
                <Tooltip cursor={{fill: '#f9fafb'}} />
                <Bar dataKey="todo" stackId="a" fill="#3b82f6" radius={[0, 0, 4, 4]} />
                <Bar dataKey="progress" stackId="a" fill="#8b5cf6" />
                <Bar dataKey="done" stackId="a" fill="#10b981" />
                <Bar dataKey="blocked" stackId="a" fill="#ec4899" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex gap-3 mt-3 text-[10px] font-medium text-gray-500">
            <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-blue-500"></div> To Do</div>
            <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-purple-500"></div> Progress</div>
            <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-emerald-500"></div> Done</div>
          </div>
        </div>

        <div className="bg-white p-5 rounded-2xl shadow-sm border border-gray-100">
          <div className="mb-4">
            <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider">Tỷ lệ thành công</h3>
            <p className="text-[11px] text-gray-400">14 ngày gần nhất</p>
          </div>
          <div className="h-40 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockDailyData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
                <XAxis dataKey="name" tick={{fontSize: 10}} tickLine={false} axisLine={false} />
                <Tooltip cursor={{fill: '#f9fafb'}} />
                <Bar dataKey="success" fill="#14b8a6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="failed" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
          <h3 className="text-sm font-bold text-gray-900 mb-4">Hành động nhanh</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-4 border border-gray-100 rounded-xl hover:shadow-md transition-all cursor-pointer flex flex-col items-center justify-center text-center gap-3 hover:border-blue-100 group">
              <div className="p-3 bg-blue-50 rounded-lg group-hover:bg-blue-600 transition-colors">
                <BrainCircuit className="w-5 h-5 text-blue-600 group-hover:text-white" />
              </div>
              <div>
                <p className="font-semibold text-gray-900 text-sm">Mô hình</p>
                <p className="text-[10px] text-gray-400 mt-0.5">Cấu hình LLMs</p>
              </div>
            </div>

            <div className="p-4 border border-gray-100 rounded-xl hover:shadow-md transition-all cursor-pointer flex flex-col items-center justify-center text-center gap-3 hover:border-orange-100 group">
              <div className="p-3 bg-orange-50 rounded-lg group-hover:bg-orange-500 transition-colors">
                <BookOpen className="w-5 h-5 text-orange-500 group-hover:text-white" />
              </div>
              <div>
                <p className="font-semibold text-gray-900 text-sm">Hướng dẫn</p>
                <p className="text-[10px] text-gray-400 mt-0.5">Prompt cho agents</p>
              </div>
            </div>

            <div className="p-4 border border-gray-100 rounded-xl hover:shadow-md transition-all cursor-pointer flex flex-col items-center justify-center text-center gap-3 hover:border-emerald-100 group">
              <div className="p-3 bg-emerald-50 rounded-lg group-hover:bg-emerald-500 transition-colors">
                <Database className="w-5 h-5 text-emerald-500 group-hover:text-white" />
              </div>
              <div>
                <p className="font-semibold text-gray-900 text-sm">Kiến thức</p>
                <p className="text-[10px] text-gray-400 mt-0.5">RAG knowledge base</p>
              </div>
            </div>

            <div className="p-4 border border-gray-100 rounded-xl hover:shadow-md transition-all cursor-pointer flex flex-col items-center justify-center text-center gap-3 hover:border-amber-100 group">
              <div className="p-3 bg-amber-50 rounded-lg group-hover:bg-amber-500 transition-colors">
                <PenTool className="w-5 h-5 text-amber-500 group-hover:text-white" />
              </div>
              <div>
                <p className="font-semibold text-gray-900 text-sm">Công cụ</p>
                <p className="text-[10px] text-gray-400 mt-0.5">Kết nối & APIs</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-sm font-bold text-gray-900">Hoạt động gần đây</h3>
            <button className="text-xs font-semibold text-blue-600 hover:text-blue-700">Xem tất cả ↗</button>
          </div>
          <div className="flex-1 space-y-4">
            <div className="flex items-start gap-3">
              <div className="mt-0.5"><CheckCircle2 className="w-4 h-4 text-emerald-500" /></div>
              <div>
                <p className="text-sm text-gray-800"><span className="font-semibold">API sẵn sàng</span> — Tất cả các mô hình đã được tải vào bộ nhớ</p>
                <p className="text-xs text-gray-400 mt-1">Vừa xong</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="mt-0.5"><CheckCircle2 className="w-4 h-4 text-emerald-500" /></div>
              <div>
                <p className="text-sm text-gray-800"><span className="font-semibold">Hệ thống khoẻ</span> — Tất cả các dịch vụ hoạt động bình thường</p>
                <p className="text-xs text-gray-400 mt-1">Vừa xong</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="mt-0.5"><CheckCircle2 className="w-4 h-4 text-emerald-500" /></div>
              <div>
                <p className="text-sm text-gray-800"><span className="font-semibold">GPU kết nối</span> — {metrics?.gpu_name || 'CPU Mode'}</p>
                <p className="text-xs text-gray-400 mt-1">2 phút trước</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Overview;
