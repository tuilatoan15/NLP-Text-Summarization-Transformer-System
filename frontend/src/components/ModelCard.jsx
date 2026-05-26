/**
 * ModelCard.jsx - Detailed model result card
 */

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ChevronDown, Info, Clock, Zap } from 'lucide-react';

const ModelCard = ({ type, modelName, result, metrics, onToggleExplanation }) => {
  const [expanded, setExpanded] = useState(false);

  const isExtractive = type === 'extractive';
  const borderColor = isExtractive ? 'border-emerald-600' : 'border-blue-600';
  const bgColor = isExtractive ? 'bg-emerald-900/20' : 'bg-blue-900/20';
  const accentColor = isExtractive ? 'text-emerald-400' : 'text-blue-400';

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`border ${borderColor} ${bgColor} rounded-xl overflow-hidden transition-all`}
    >
      {/* Header */}
      <div
        onClick={() => setExpanded(!expanded)}
        className="cursor-pointer p-4 border-b border-slate-700 hover:bg-slate-700/30 transition-colors"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div>
              <h3 className={`text-lg font-bold ${accentColor}`}>
                {isExtractive ? '🎯' : '🤖'} {modelName.toUpperCase()}
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                {isExtractive ? 'Graph-based Extraction' : 'Transformer Generation'}
              </p>
            </div>
          </div>
          <ChevronDown
            size={20}
            className={`transition-transform ${expanded ? 'rotate-180' : ''}`}
          />
        </div>
      </div>

      {/* Summary Preview */}
      <div className="p-4 bg-slate-700/20">
        <p className="text-sm text-slate-300 leading-relaxed line-clamp-3">
          {result.summary}
        </p>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-3 gap-2 p-4 bg-slate-800/30 border-t border-slate-700">
        <MetricBadge
          label="ROUGE-1"
          value={metrics.rouge1}
          type="rouge"
        />
        <MetricBadge
          label="Semantic"
          value={metrics.semantic_similarity}
          type="semantic"
        />
        <MetricBadge
          label="Compression"
          value={`${Math.round(result.compression_ratio * 100)}%`}
          type="compression"
        />
      </div>

      {/* Expanded Details */}
      {expanded && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="border-t border-slate-700 p-4 space-y-4"
        >
          {/* Full Summary */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-2">
              📄 Full Summary
            </label>
            <div className="bg-slate-700/50 p-3 rounded-lg text-sm text-slate-300 max-h-48 overflow-y-auto">
              {result.summary}
            </div>
          </div>

          {/* Metrics Details */}
          <div className="grid grid-cols-2 gap-3">
            <MetricDetail label="ROUGE-1" value={metrics.rouge1.toFixed(4)} />
            <MetricDetail label="ROUGE-2" value={metrics.rouge2.toFixed(4)} />
            <MetricDetail label="ROUGE-L" value={metrics.rougeL.toFixed(4)} />
            <MetricDetail label="BERTScore" value={metrics.bertscore_f1.toFixed(4)} />
            <MetricDetail label="Semantic" value={metrics.semantic_similarity.toFixed(4)} />
            <MetricDetail label="Time" value={`${result.inference_time.toFixed(3)}s`} />
          </div>

          {/* Extractive-specific details */}
          {isExtractive && result.ranked_scores && (
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-2">
                🔗 Sentence Importance Scores
              </label>
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {Object.entries(result.ranked_scores).slice(0, 5).map(([sentence, score], idx) => (
                  <div key={idx} className="bg-slate-700/50 p-2 rounded text-xs">
                    <div className="flex justify-between items-start gap-2">
                      <span className="text-slate-300 flex-1">{sentence.substring(0, 60)}...</span>
                      <span className="text-emerald-400 font-bold ml-2">{(score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-1 bg-slate-600 rounded-full mt-1 overflow-hidden">
                      <div
                        className="h-full bg-emerald-500 transition-all"
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Abstractive-specific details */}
          {!isExtractive && (
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-2">
                🧠 Generation Details
              </label>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-700/50 p-2 rounded text-xs">
                  <p className="text-slate-400">Token Count</p>
                  <p className="text-blue-400 font-bold">{result.token_count}</p>
                </div>
                <div className="bg-slate-700/50 p-2 rounded text-xs">
                  <p className="text-slate-400">Compression</p>
                  <p className="text-blue-400 font-bold">{Math.round(result.compression_ratio * 100)}%</p>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      )}
    </motion.div>
  );
};

/**
 * MetricBadge - Compact metric display
 */
const MetricBadge = ({ label, value, type }) => {
  let bgClass = 'bg-slate-700';
  if (type === 'rouge') bgClass = 'bg-purple-900/40 border-purple-600';
  else if (type === 'semantic') bgClass = 'bg-cyan-900/40 border-cyan-600';
  else if (type === 'compression') bgClass = 'bg-orange-900/40 border-orange-600';

  return (
    <div className={`border ${bgClass} rounded-lg p-2 text-center`}>
      <p className="text-xs text-slate-400">{label}</p>
      <p className="text-sm font-bold text-white">
        {typeof value === 'string' ? value : value.toFixed(3)}
      </p>
    </div>
  );
};

/**
 * MetricDetail - Detailed metric row
 */
const MetricDetail = ({ label, value }) => (
  <div className="flex items-center justify-between bg-slate-700/30 p-3 rounded-lg border border-slate-600">
    <span className="text-xs text-slate-400">{label}</span>
    <span className="text-sm font-bold text-slate-200">{value}</span>
  </div>
);

export default ModelCard;
