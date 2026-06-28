import React from 'react';
import { Check } from 'lucide-react';

export interface AlgorithmItem {
  key: string;
  name: string;
  group: 'extractive' | 'abstractive' | 'hybrid';
  color: string;
  description: string;
}

export const ALGORITHMS: AlgorithmItem[] = [
  { key: 'textrank', name: 'TextRank', group: 'extractive', color: '#14b8a6', description: 'Xếp hạng câu dựa trên đồ thị từ vựng, tối ưu cho tin tức ngắn.' },
  { key: 'lexrank', name: 'LexRank', group: 'extractive', color: '#38bdf8', description: 'Đo độ trung tâm đồ thị tương đồng câu, xử lý cực nhanh.' },
  { key: 'lsa', name: 'LSA', group: 'extractive', color: '#84cc16', description: 'Phân tích ngữ nghĩa tiềm ẩn bằng phân rã ma trận từ-câu.' },
  { key: 'vit5', name: 'ViT5', group: 'abstractive', color: '#f59e0b', description: 'Mô hình sinh tiếng Việt tốt nhất khi fine-tuned trên VNExpress.' },
  { key: 'mt5', name: 'mT5', group: 'abstractive', color: '#e879f9', description: 'Mô hình sinh đa ngữ của Google, dịch nghĩa linh hoạt.' },
  { key: 'bartpho', name: 'BARTPho', group: 'abstractive', color: '#fb7185', description: 'Mô hình seq2seq tối ưu hóa âm tiết tiếng Việt từ VinAI.' },
  { key: 'textrank-vit5', name: 'TextRank ➔ ViT5', group: 'hybrid', color: '#d97706', description: 'Nén văn bản bằng TextRank trước rồi sinh tóm tắt bằng ViT5.' },
  { key: 'lexrank-vit5', name: 'LexRank ➔ ViT5', group: 'hybrid', color: '#7c3aed', description: 'Nén văn bản bằng LexRank trước rồi sinh tóm tắt bằng ViT5.' },
  { key: 'lsa-vit5', name: 'LSA ➔ ViT5', group: 'hybrid', color: '#dc2626', description: 'Nén văn bản bằng LSA trước rồi sinh tóm tắt bằng ViT5.' },
  { key: 'textrank-mt5', name: 'TextRank ➔ mT5', group: 'hybrid', color: '#2563eb', description: 'Nén văn bản bằng TextRank trước rồi sinh tóm tắt bằng mT5.' },
  { key: 'lexrank-mt5', name: 'LexRank ➔ mT5', group: 'hybrid', color: '#059669', description: 'Nén văn bản bằng LexRank trước rồi sinh tóm tắt bằng mT5.' },
  { key: 'lsa-mt5', name: 'LSA ➔ mT5', group: 'hybrid', color: '#0891b2', description: 'Nén văn bản bằng LSA trước rồi sinh tóm tắt bằng mT5.' },
  { key: 'textrank-bartpho', name: 'TextRank ➔ BARTPho', group: 'hybrid', color: '#c026d3', description: 'Nén văn bản bằng TextRank trước rồi sinh tóm tắt bằng BARTPho.' },
  { key: 'lexrank-bartpho', name: 'LexRank ➔ BARTPho', group: 'hybrid', color: '#0284c7', description: 'Nén văn bản bằng LexRank trước rồi sinh tóm tắt bằng BARTPho.' },
  { key: 'lsa-bartpho', name: 'LSA ➔ BARTPho', group: 'hybrid', color: '#4f46e5', description: 'Nén văn bản bằng LSA trước rồi sinh tóm tắt bằng BARTPho.' },
];

export interface AlgorithmSelectorProps {
  selected: string[];
  setSelected: (selected: string[] | ((prev: string[]) => string[])) => void;
  disabled?: boolean;
}

export const AlgorithmSelector: React.FC<AlgorithmSelectorProps> = ({ selected, setSelected, disabled }) => {
  const toggle = (key: string) => {
    if (disabled) return;
    setSelected((current) =>
      current.includes(key) ? current.filter((item) => item !== key) : [...current, key]
    );
  };

  const selectGroup = (group: 'extractive' | 'abstractive' | 'hybrid', select: boolean) => {
    if (disabled) return;
    const groupKeys = ALGORITHMS.filter((item) => item.group === group).map((item) => item.key);
    setSelected((current) => {
      if (select) {
        const newSelection = [...current];
        groupKeys.forEach((key) => {
          if (!newSelection.includes(key)) newSelection.push(key);
        });
        return newSelection;
      } else {
        return current.filter((key) => !groupKeys.includes(key));
      }
    });
  };

  const groupLabel = (group: string) => {
    switch (group) {
      case 'extractive':
        return 'Extractive (Rút trích)';
      case 'abstractive':
        return 'Abstractive (Tóm tắt sinh)';
      case 'hybrid':
        return 'Hybrid (Lai ghép 2 tầng)';
      default:
        return group;
    }
  };

  const groups: Array<'extractive' | 'abstractive' | 'hybrid'> = ['extractive', 'abstractive', 'hybrid'];

  const byKey = (key: string) => {
    return ALGORITHMS.find(item => item.key === key) || { key, name: key, group: 'extractive', color: '#64748b' };
  };

  return (
    <div className="space-y-6">
      {/* Legend / Chú giải nhóm thuật toán */}
      <div className="flex flex-wrap items-center gap-4 p-3 bg-slate-50 dark:bg-slate-900/60 rounded-xl border border-slate-200/50 dark:border-slate-800 text-[10px] font-bold text-slate-500 dark:text-slate-400">
        <span className="uppercase tracking-wider">Chú giải nhóm:</span>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded bg-[#14b8a6]" />
          <span>Extractive (Rút trích câu gốc)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded bg-[#f59e0b]" />
          <span>Abstractive (Transformer sinh câu)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded bg-[#d97706]" />
          <span>Hybrid (Lai ghép hai tầng)</span>
        </div>
      </div>

      {groups.map((group) => {
        const groupAlgos = ALGORITHMS.filter((item) => item.group === group);
        const groupSelected = selected.filter((key) => byKey(key).group === group);
        const isAllSelected = groupSelected.length === groupAlgos.length;

        return (
          <section key={group} className="space-y-3">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800/80 pb-2">
              <div className="flex items-center gap-2">
                <span className={`w-1.5 h-3.5 rounded-full ${
                  group === 'extractive' ? 'bg-[#14b8a6]' : group === 'abstractive' ? 'bg-[#f59e0b]' : 'bg-[#d97706]'
                }`} />
                <h3 className="text-xs font-bold text-slate-700 dark:text-slate-300">
                  {groupLabel(group)}
                </h3>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => selectGroup(group, !isAllSelected)}
                  className="text-[10px] font-extrabold text-sky-600 dark:text-sky-400 hover:text-sky-700 disabled:opacity-40 cursor-pointer"
                >
                  {isAllSelected ? 'Bỏ chọn nhóm' : 'Chọn cả nhóm'}
                </button>
                <span className="text-[10px] font-extrabold text-slate-400 dark:text-slate-650">|</span>
                <span className="text-[10px] font-extrabold text-sky-600 dark:text-sky-400 bg-sky-50 dark:bg-sky-950/20 px-2.5 py-0.5 rounded-full border border-sky-100 dark:border-sky-900/30">
                  {groupSelected.length}/{groupAlgos.length}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
              {groupAlgos.map((item) => {
                const isSelected = selected.includes(item.key);
                return (
                  <div key={item.key} className="relative group">
                    <button
                      type="button"
                      disabled={disabled}
                      onClick={() => toggle(item.key)}
                      className={`w-full flex flex-col items-start p-3.5 rounded-xl border text-left transition-all duration-200 cursor-pointer ${
                        isSelected
                          ? 'bg-sky-500/5 border-sky-500 dark:border-sky-600 text-sky-700 dark:text-sky-400 font-semibold shadow-sm ring-1 ring-sky-500/20'
                          : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:border-slate-300 dark:hover:border-slate-700 opacity-70 hover:opacity-100'
                      } ${disabled ? 'opacity-40 cursor-not-allowed' : ''}`}
                    >
                      <div className="flex items-center justify-between w-full mb-1.5">
                        <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: item.color }} />
                        {isSelected && (
                          <div className="p-0.5 rounded-full bg-sky-500 text-white">
                            <Check size={10} strokeWidth={3} />
                          </div>
                        )}
                      </div>
                      <span className="text-xs font-bold text-slate-800 dark:text-slate-200 break-words w-full leading-tight">
                        {item.name}
                      </span>
                    </button>
                    
                    {/* Tooltip mô tả thuật toán */}
                    <div className="absolute z-10 bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 p-2.5 bg-slate-950/95 dark:bg-slate-900/95 text-white text-[10px] rounded-lg shadow-xl opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity duration-200 leading-normal text-center border border-slate-800">
                      {item.description}
                      <div className="absolute top-full left-1/2 -translate-x-1/2 border-[5px] border-transparent border-t-slate-950/95 dark:border-t-slate-900/95" />
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        );
      })}
    </div>
  );
};
