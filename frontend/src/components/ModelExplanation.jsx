/**
 * ModelExplanation.jsx - Academic explanations for models
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, BookOpen, Zap, Brain } from 'lucide-react';

const ModelExplanation = () => {
  const [expandedModel, setExpandedModel] = useState(null);

  const models = {
    extractive: [
      {
        id: 'textrank',
        name: 'TextRank',
        emoji: '🔗',
        type: 'Graph-based Extractive',
        principle:
          'TextRank sử dụng PageRank algorithm để xác định độ quan trọng của các câu dựa trên sự xuất hiện cùng nhau của từ.',
        principle_en: 'Uses word co-occurrence graph to rank sentence importance.',
        formula: 'Importance(S) = ∑ w(S) * Importance(w)',
        advantages: [
          '✅ Rất nhanh (~30ms)',
          '✅ Không yêu cầu training data',
          '✅ Lightweight, dễ triển khai',
          '✅ Deterministic output',
        ],
        disadvantages: [
          '❌ Không hiểu semantic context',
          '❌ Chỉ trích rút câu từ gốc',
          '❌ Kém với văn bản ngắn hoặc chuyên biệt',
          '❌ Không capture long-range dependencies',
        ],
        useCases: [
          '• Real-time summarization',
          '• Tài liệu lớn cần xử lý nhanh',
          '• Mobile/embedded systems',
          '• Quick keyword extraction',
        ],
        complexity: 'O(n²) - Graph construction & PageRank iterations',
        references: 'Mihalcea & Tarau (2004)',
      },
      {
        id: 'lexrank',
        name: 'LexRank',
        emoji: '📊',
        type: 'IDF-weighted Graph-based',
        principle:
          'LexRank kết hợp TF-IDF weighting với similarity graph của các câu. Câu có nhiều kết nối đến câu khác sẽ được xếp cao hơn.',
        principle_en: 'PageRank on sentence similarity graph with IDF weighting.',
        formula: 'LexRank(S_i) = (1-d)/N + d * ∑ sim(S_i, S_j)',
        advantages: [
          '✅ Tốt hơn TextRank (~2-5% tăng ROUGE)',
          '✅ Xét TF-IDF significance',
          '✅ Vẫn tương đối nhanh',
          '✅ Phù hợp với multi-document',
        ],
        disadvantages: [
          '❌ Vẫn là extractive method',
          '❌ Phụ thuộc vào similarity metric',
          '❌ Không capture topic relationships',
        ],
        useCases: [
          '• News article summarization',
          '• Multi-document summarization',
          '• Document clustering',
          '• Topic discovery',
        ],
        complexity: 'O(n² * m) - IDF computation + PageRank',
        references: 'Erkan & Radev (2004)',
      },
      {
        id: 'lsa',
        name: 'LSA (Latent Semantic Analysis)',
        emoji: '🧮',
        type: 'Matrix Factorization',
        principle:
          'LSA áp dụng SVD trên term-sentence matrix để tìm latent semantic structure. Các câu có cao giá trị trên SVD vectors được chọn.',
        principle_en: 'Singular Value Decomposition on term-frequency matrix.',
        formula: 'A = U * Σ * V^T, Select top sentences by ∑ |v_i|',
        advantages: [
          '✅ Nắm bắt topic semantics',
          '✅ Hiệu quả với tài liệu lớn',
          '✅ Mở rộng được cho multiple dimensions',
          '✅ Robust to synonymy',
        ],
        disadvantages: [
          '❌ SVD computational overhead',
          '❌ Interpretation khó khăn (latent dimensions)',
          '❌ Hiệu suất phụ thuộc vào số dimensions',
          '❌ Không capture word order',
        ],
        useCases: [
          '• Multi-document summarization',
          '• Latent topic modeling',
          '• Semantic similarity calculation',
          '• Document clustering',
        ],
        complexity: 'O(n*m*k) - SVD decomposition O(nm*min(n,m))',
        references: 'Gong & Liu (2001)',
      },
    ],
    abstractive: [
      {
        id: 'vit5',
        name: 'ViT5 (Fine-tuned Vietnamese T5)',
        emoji: '🇻🇳',
        type: 'Transformer Encoder-Decoder',
        principle:
          'ViT5 là T5 model được fine-tune trên dữ liệu tiếng Việt. Nó sử dụng attention mechanism để hiểu context và generate tóm tắt paraphrased tự nhiên.',
        principle_en: 'Transformer encoder-decoder with attention, fine-tuned on Vietnamese data.',
        formula: 'Output = Decoder(Encoder(input) + Reference)',
        advantages: [
          '✅ Semantic understanding tốt nhất (~0.58 ROUGE-1)',
          '✅ Generate paraphrased summaries tự nhiên',
          '✅ Capture long-range dependencies',
          '✅ Contextualized word representations',
          '✅ Fine-tuned cụ thể cho Tiếng Việt',
        ],
        disadvantages: [
          '❌ Chậm (6-8s trên GPU)',
          '❌ Tốn GPU VRAM (~4GB)',
          '❌ Cần reference summaries để fine-tune',
          '❌ Cold-start latency cao',
          '❌ Output có thể hallucinates',
        ],
        useCases: [
          '• High-quality abstractive summarization',
          '• Publication/research papers',
          '• Content creation & paraphrasing',
          '• Semantic text generation',
        ],
        complexity: 'O(n²) - Quadratic attention complexity',
        references: 'Raffel et al. (2020) - Exploring the Limits of Transfer Learning',
      },
      {
        id: 'bartpho',
        name: 'BARTPho',
        emoji: '🌍',
        type: 'Multilingual Transformer',
        principle:
          'BARTPho là BART model được pre-train trên dữ liệu multilingual bao gồm Tiếng Việt. Mạnh trong denoising và generation tasks.',
        principle_en: 'Multilingual BART optimized for Vietnamese and other languages.',
        formula: 'Output = Decoder(Encoder(corrupted_input))',
        advantages: [
          '✅ Multilingual support',
          '✅ Good semantic understanding',
          '✅ Denoising pre-training',
          '✅ Tương tự ViT5 về chất lượng',
          '✅ Phù hợp cross-lingual tasks',
        ],
        disadvantages: [
          '❌ Chậm tương tự ViT5',
          '❌ Model size lớn (~1.2GB)',
          '❌ Cần GPU để inference nhanh',
          '❌ Khó khăn với domain-specific content',
        ],
        useCases: [
          '• Multilingual summarization',
          '• Cross-lingual semantic tasks',
          '• Publication-quality summaries',
          '• Translation-aware summarization',
        ],
        complexity: 'O(n²) - Transformer attention',
        references: 'Nguyen et al. (2020) - BARTPho',
      },
      {
        id: 'mt5',
        name: 'mT5 (Multilingual T5)',
        emoji: '🌐',
        type: 'Multilingual Baseline',
        principle:
          'mT5 là T5 pre-trained trên 101 ngôn ngữ. Phục vụ như baseline để so sánh với fine-tuned models.',
        principle_en: 'Pretrained T5 for 101 languages, used as comparison baseline.',
        formula: 'Output = Decoder(Encoder(input))',
        advantages: [
          '✅ Baseline comparison',
          '✅ Lighterweight than specialized models',
          '✅ Multi-language support',
          '✅ Public pre-trained weights',
        ],
        disadvantages: [
          '❌ Chưa fine-tune cho Tiếng Việt',
          '❌ Kém hơn ViT5/BARTPho (~10-15% ROUGE drop)',
          '❌ Vẫn tốn GPU',
          '❌ Generic representations',
        ],
        useCases: [
          '• Baseline comparisons',
          '• Cross-language experiments',
          '• When fine-tuned models unavailable',
          '• Research & experimentation',
        ],
        complexity: 'O(n²) - Transformer attention',
        references: 'Xue et al. (2021) - mT5',
      },
    ],
  };

  return (
    <div className="space-y-8">
      {/* Overview Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="text-left p-3 text-slate-300 font-bold">Model</th>
              <th className="text-left p-3 text-slate-300 font-bold">Type</th>
              <th className="text-left p-3 text-slate-300 font-bold">ROUGE (~)</th>
              <th className="text-left p-3 text-slate-300 font-bold">Speed</th>
              <th className="text-left p-3 text-slate-300 font-bold">Best For</th>
            </tr>
          </thead>
          <tbody>
            {[
              { name: 'TextRank', type: 'Extractive', rouge: '0.43', speed: '~30ms', use: 'Quick Summarization' },
              { name: 'LexRank', type: 'Extractive', rouge: '0.45', speed: '~50ms', use: 'News Articles' },
              { name: 'LSA', type: 'Extractive', rouge: '0.47', speed: '~90ms', use: 'Multi-Document' },
              { name: 'ViT5', type: 'Abstractive', rouge: '0.58', speed: '~6.5s', use: 'High Quality' },
              { name: 'BARTPho', type: 'Abstractive', rouge: '0.61', speed: '~8.2s', use: 'Multilingual' },
              { name: 'mT5', type: 'Abstractive (Base)', rouge: '0.48', speed: '~7s', use: 'Baseline' },
            ].map((model, idx) => (
              <tr key={idx} className="border-b border-slate-700 hover:bg-slate-700/30">
                <td className="p-3 font-bold text-cyan-400">{model.name}</td>
                <td className="p-3 text-slate-300">{model.type}</td>
                <td className="p-3 text-amber-400">{model.rouge}</td>
                <td className="p-3">{model.speed}</td>
                <td className="p-3 text-slate-300">{model.use}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Detailed Explanations */}
      <div className="space-y-4">
        {/* Extractive Models */}
        <div>
          <h3 className="text-xl font-bold text-emerald-400 mb-4 flex items-center gap-2">
            🎯 Extractive Models
          </h3>
          <div className="space-y-3">
            {models.extractive.map((model) => (
              <ModelExplanationCard
                key={model.id}
                model={model}
                expanded={expandedModel === model.id}
                onToggle={() => setExpandedModel(expandedModel === model.id ? null : model.id)}
              />
            ))}
          </div>
        </div>

        {/* Abstractive Models */}
        <div>
          <h3 className="text-xl font-bold text-blue-400 mb-4 flex items-center gap-2">
            🤖 Abstractive Models
          </h3>
          <div className="space-y-3">
            {models.abstractive.map((model) => (
              <ModelExplanationCard
                key={model.id}
                model={model}
                expanded={expandedModel === model.id}
                onToggle={() => setExpandedModel(expandedModel === model.id ? null : model.id)}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Key Insights */}
      <div className="bg-blue-900/20 border border-blue-600 rounded-xl p-6">
        <h3 className="text-lg font-bold text-blue-400 mb-4">📌 Key Insights for Thesis Defense</h3>
        <div className="space-y-4 text-sm text-slate-300">
          <div>
            <p className="font-bold text-blue-300 mb-2">Extractive vs Abstractive Trade-off</p>
            <p>
              Extractive methods (TextRank, LexRank, LSA) được xếp hạng theo graph-based hoặc matrix factorization,
              hoạt động rất nhanh (~30-90ms) nhưng chỉ có thể trích rút câu từ văn bản gốc. Abstractive methods sử dụng
              Transformer architecture để hiểu context sâu và generate paraphrased summaries, đạt ROUGE cao hơn nhưng
              chậm hơn 100x (6-8 giây).
            </p>
          </div>
          <div>
            <p className="font-bold text-green-300 mb-2">🎯 Recommendation</p>
            <p>
              • Sử dụng <strong>Extractive</strong> khi cần speed hoặc resource-constraint
              <br />
              • Sử dụng <strong>Abstractive</strong> khi cần high-quality, publication-ready summaries
              <br />
              • Hybrid approach: Extract → Reorder → Abstractive refinement
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * ModelExplanationCard - Expandable model detail card
 */
const ModelExplanationCard = ({ model, expanded, onToggle }) => {
  const isExtraction = model.type.includes('Extractive') || model.type.includes('Graph') || model.type.includes('Matrix');
  const bgColor = isExtraction ? 'bg-emerald-900/20 border-emerald-600' : 'bg-blue-900/20 border-blue-600';
  const accentColor = isExtraction ? 'text-emerald-400' : 'text-blue-400';

  return (
    <motion.div
      className={`border ${bgColor} rounded-xl overflow-hidden`}
    >
      {/* Header */}
      <div
        onClick={onToggle}
        className="p-4 cursor-pointer hover:bg-slate-700/20 transition-colors"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-3xl">{model.emoji}</span>
            <div>
              <h4 className={`text-lg font-bold ${accentColor}`}>{model.name}</h4>
              <p className="text-xs text-slate-400">{model.type}</p>
            </div>
          </div>
          <ChevronDown
            size={20}
            className={`transition-transform ${expanded ? 'rotate-180' : ''}`}
          />
        </div>
      </div>

      {/* Expanded Content */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-t border-slate-700 p-4 space-y-4"
          >
            {/* Principle */}
            <div>
              <p className="text-xs font-bold text-slate-300 mb-2">📖 Principle (Tiếng Việt)</p>
              <p className="text-sm text-slate-300 bg-slate-700/30 p-3 rounded-lg leading-relaxed">
                {model.principle}
              </p>
              <p className="text-xs font-bold text-slate-300 mt-3 mb-2">📖 Principle (English)</p>
              <p className="text-sm text-slate-400 bg-slate-700/30 p-3 rounded-lg italic">
                {model.principle_en}
              </p>
            </div>

            {/* Formula */}
            <div>
              <p className="text-xs font-bold text-slate-300 mb-2">∑ Mathematical Formula</p>
              <div className="bg-slate-900/50 border border-slate-600 rounded-lg p-3 font-mono text-sm text-cyan-400 overflow-x-auto">
                {model.formula}
              </div>
            </div>

            {/* Advantages & Disadvantages */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-xs font-bold text-green-400 mb-2">✅ Advantages</p>
                <ul className="space-y-1 text-xs text-slate-300">
                  {model.advantages.map((adv, idx) => (
                    <li key={idx}>{adv}</li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="text-xs font-bold text-red-400 mb-2">❌ Disadvantages</p>
                <ul className="space-y-1 text-xs text-slate-300">
                  {model.disadvantages.map((dis, idx) => (
                    <li key={idx}>{dis}</li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Use Cases & Complexity */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-xs font-bold text-slate-300 mb-2">💡 Use Cases</p>
                <ul className="space-y-1 text-xs text-slate-300">
                  {model.useCases.map((use, idx) => (
                    <li key={idx}>{use}</li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="text-xs font-bold text-slate-300 mb-2">⚙️ Complexity Analysis</p>
                <p className="text-xs text-slate-300 bg-slate-700/30 p-2 rounded font-mono">
                  {model.complexity}
                </p>
                <p className="text-xs text-slate-400 mt-2">
                  📚 <span className="text-cyan-400">{model.references}</span>
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default ModelExplanation;
