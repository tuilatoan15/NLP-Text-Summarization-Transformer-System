import React, { useState } from 'react';
import { Save, RotateCcw, AlertCircle } from 'lucide-react';

const ModelSettings = () => {
  const [settings, setSettings] = useState({
    temperature: 0.7,
    maxLength: 150,
    topK: 50,
    topP: 0.95,
    algorithm: 'vit5',
    extractiveSentences: 5,
  });

  const [saved, setSaved] = useState(false);

  const handleChange = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
    setSaved(false);
  };

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const handleReset = () => {
    setSettings({
      temperature: 0.7,
      maxLength: 150,
      topK: 50,
      topP: 0.95,
      algorithm: 'vit5',
      extractiveSentences: 5,
    });
  };

  return (
    <div className="space-y-6 pb-12 max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 mb-1">Thiết lập Model</h1>
        <p className="text-sm text-gray-500">Điều chỉnh các siêu tham số để tối ưu hóa kết quả</p>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 flex gap-3">
        <AlertCircle className="text-blue-600 flex-shrink-0" size={20} />
        <div className="text-sm text-blue-800">
          <strong>Ghi chú:</strong> Thay đổi sẽ áp dụng cho lần chạy tiếp theo
        </div>
      </div>

      <div className="space-y-6">
        {/* Abstractive Model Settings */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-6">Abstractive Models</h2>

          <div className="space-y-6">
            <div>
              <label className="block text-sm font-semibold text-gray-900 mb-2">
                Chọn mô hình
              </label>
              <select
                value={settings.algorithm}
                onChange={(e) => handleChange('algorithm', e.target.value)}
                className="w-full rounded-lg border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="vit5">ViT5 (Fine-tuned)</option>
                <option value="mt5">mT5 (Multilingual)</option>
                <option value="bartpho">BARTPho (Vietnamese)</option>
              </select>
              <p className="text-xs text-gray-500 mt-2">
                Các mô hình đã được fine-tune trên dữ liệu tiếng Việt
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-900 mb-3">
                  Temperature
                  <span className="float-right text-lg font-bold text-blue-600">{settings.temperature.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={settings.temperature}
                  onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <p className="text-xs text-gray-500 mt-2">Kiểm soát độ ngẫu nhiên (0=xác định, 2=rất ngẫu nhiên)</p>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-900 mb-3">
                  Max Length
                  <span className="float-right text-lg font-bold text-blue-600">{settings.maxLength}</span>
                </label>
                <input
                  type="range"
                  min="50"
                  max="512"
                  step="10"
                  value={settings.maxLength}
                  onChange={(e) => handleChange('maxLength', parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <p className="text-xs text-gray-500 mt-2">Độ dài tối đa của tóm tắt (tokens)</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-900 mb-3">
                  Top-K
                  <span className="float-right text-lg font-bold text-blue-600">{settings.topK}</span>
                </label>
                <input
                  type="range"
                  min="1"
                  max="100"
                  step="5"
                  value={settings.topK}
                  onChange={(e) => handleChange('topK', parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <p className="text-xs text-gray-500 mt-2">Xem xét top-K tokens có xác suất cao nhất</p>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-900 mb-3">
                  Top-P
                  <span className="float-right text-lg font-bold text-blue-600">{settings.topP.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={settings.topP}
                  onChange={(e) => handleChange('topP', parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <p className="text-xs text-gray-500 mt-2">Nucleus sampling - chọn tokens với tích lũy xác suất</p>
              </div>
            </div>
          </div>
        </div>

        {/* Extractive Model Settings */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-6">Extractive Models</h2>

          <div className="space-y-6">
            <div>
              <label className="block text-sm font-semibold text-gray-900 mb-3">
                Số câu cần trích
                <span className="float-right text-lg font-bold text-blue-600">{settings.extractiveSentences}</span>
              </label>
              <input
                type="range"
                min="1"
                max="20"
                step="1"
                value={settings.extractiveSentences}
                onChange={(e) => handleChange('extractiveSentences', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
              />
              <p className="text-xs text-gray-500 mt-2">Số lượng câu được trích xuất từ văn bản gốc</p>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {['TextRank', 'LexRank', 'LSA'].map(algo => (
                <button
                  key={algo}
                  className="px-4 py-3 border-2 rounded-lg text-sm font-medium transition hover:border-blue-500"
                  style={{
                    borderColor: '#e5e7eb',
                    backgroundColor: '#f9fafb',
                  }}
                >
                  {algo}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3">
          <button
            onClick={handleSave}
            className="flex-1 flex items-center justify-center gap-2 bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 transition"
          >
            <Save size={18} /> Lưu thiết lập
          </button>
          <button
            onClick={handleReset}
            className="flex-1 flex items-center justify-center gap-2 border border-gray-300 bg-white text-gray-700 py-3 rounded-lg font-medium hover:bg-gray-50 transition"
          >
            <RotateCcw size={18} /> Reset
          </button>
        </div>

        {saved && (
          <div className="bg-green-50 border border-green-200 text-green-800 text-sm p-4 rounded-lg">
            ✓ Thiết lập đã được lưu thành công
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelSettings;
