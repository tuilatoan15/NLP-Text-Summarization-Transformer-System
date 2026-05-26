/**
 * HumanEvaluation.jsx - Human evaluation interface for summaries
 * Cho phép đánh giá summaries theo các tiêu chí
 */

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Star, Download, Plus, Trash2 } from 'lucide-react';

const HumanEvaluation = ({ summaries = [], onSave }) => {
  const [evaluations, setEvaluations] = useState({});
  const [showForm, setShowForm] = useState(null);

  const criteria = [
    { id: 'informativeness', label: 'Informativeness', description: 'Tóm tắt có chứa thông tin quan trọng không?' },
    { id: 'fluency', label: 'Fluency', description: 'Tóm tắt có thành thạo và tự nhiên không?' },
    { id: 'coherence', label: 'Coherence', description: 'Tóm tắt có logic và liên kết tốt không?' },
    { id: 'redundancy', label: 'Redundancy', description: 'Tóm tắt có lặp lại không cần thiết không?' },
    { id: 'readability', label: 'Readability', description: 'Tóm tắt dễ đọc và hiểu không?' },
  ];

  const addEvaluation = (summaryId) => {
    setEvaluations({
      ...evaluations,
      [summaryId]: {
        informativeness: 0,
        fluency: 0,
        coherence: 0,
        redundancy: 0,
        readability: 0,
        comments: '',
      },
    });
    setShowForm(summaryId);
  };

  const updateScore = (summaryId, criterion, score) => {
    setEvaluations({
      ...evaluations,
      [summaryId]: {
        ...evaluations[summaryId],
        [criterion]: score,
      },
    });
  };

  const updateComments = (summaryId, comments) => {
    setEvaluations({
      ...evaluations,
      [summaryId]: {
        ...evaluations[summaryId],
        comments,
      },
    });
  };

  const exportEvaluations = () => {
    const csv = [
      'Summary ID,Model,Informativeness,Fluency,Coherence,Redundancy,Readability,Average,Comments',
      ...Object.entries(evaluations).map(([summaryId, scores]) => {
        const avg = (
          (scores.informativeness +
            scores.fluency +
            scores.coherence +
            scores.redundancy +
            scores.readability) / 5
        ).toFixed(2);
        return `"${summaryId}","",${scores.informativeness},${scores.fluency},${scores.coherence},${scores.redundancy},${scores.readability},${avg},"${scores.comments}"`;
      }),
    ].join('\n');

    const link = document.createElement('a');
    link.href = `data:text/csv;charset=utf-8,${encodeURIComponent(csv)}`;
    link.download = `human_evaluation_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">👥 Human Evaluation</h2>
          <p className="text-sm text-slate-400 mt-1">Đánh giá quality của summaries</p>
        </div>
        <button
          onClick={exportEvaluations}
          disabled={Object.keys(evaluations).length === 0}
          className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 text-sm font-medium"
        >
          <Download size={18} />
          Export CSV
        </button>
      </div>

      {/* Evaluation Forms */}
      <div className="space-y-6">
        {summaries.map((summary) => (
          <motion.div
            key={summary.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-slate-800/50 border border-slate-700 rounded-xl p-6"
          >
            {/* Summary Header */}
            <div className="mb-4">
              <h3 className="text-lg font-bold text-cyan-400">{summary.model}</h3>
              <p className="text-sm text-slate-400 mt-2 leading-relaxed">
                {summary.text.substring(0, 200)}...
              </p>
            </div>

            {/* Evaluation Form */}
            {evaluations[summary.id] ? (
              <EvaluationForm
                criteria={criteria}
                scores={evaluations[summary.id]}
                onScoreChange={(criterion, score) =>
                  updateScore(summary.id, criterion, score)
                }
                onCommentsChange={(comments) =>
                  updateComments(summary.id, comments)
                }
              />
            ) : (
              <button
                onClick={() => addEvaluation(summary.id)}
                className="w-full px-4 py-3 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors flex items-center justify-center gap-2 text-sm font-medium text-slate-300"
              >
                <Plus size={18} />
                Add Evaluation
              </button>
            )}
          </motion.div>
        ))}
      </div>

      {/* Summary Statistics */}
      {Object.keys(evaluations).length > 0 && (
        <div className="bg-blue-900/20 border border-blue-600 rounded-xl p-6">
          <h3 className="text-lg font-bold text-blue-400 mb-4">📊 Evaluation Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {criteria.map((criterion) => {
              const scores = Object.values(evaluations).map((e) => e[criterion.id]);
              const avg = scores.length > 0 ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2) : 0;
              return (
                <div key={criterion.id} className="bg-slate-800/50 p-4 rounded-lg text-center">
                  <p className="text-sm font-bold text-slate-300 mb-2">{criterion.label}</p>
                  <p className="text-2xl font-bold text-cyan-400">{avg}</p>
                  <p className="text-xs text-slate-400 mt-2">
                    ({scores.length} evaluations)
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * EvaluationForm - Form for evaluating a single summary
 */
const EvaluationForm = ({ criteria, scores, onScoreChange, onCommentsChange }) => {
  return (
    <div className="space-y-4">
      {/* Rating Criteria */}
      {criteria.map((criterion) => (
        <div key={criterion.id} className="border border-slate-600 rounded-lg p-4">
          <div className="flex items-between justify-between mb-3">
            <div>
              <label className="text-sm font-bold text-slate-300">{criterion.label}</label>
              <p className="text-xs text-slate-400 mt-1">{criterion.description}</p>
            </div>
            <span className="text-lg font-bold text-cyan-400">
              {scores[criterion.id]}/5
            </span>
          </div>

          {/* Star Rating */}
          <div className="flex gap-2">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                onClick={() => onScoreChange(criterion.id, star)}
                className={`p-2 rounded-lg transition-colors ${
                  star <= scores[criterion.id]
                    ? 'bg-yellow-500 text-yellow-900'
                    : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                }`}
              >
                <Star size={20} fill="currentColor" />
              </button>
            ))}
          </div>
        </div>
      ))}

      {/* Comments */}
      <div className="border border-slate-600 rounded-lg p-4">
        <label className="block text-sm font-bold text-slate-300 mb-3">
          Comments (Tùy chọn)
        </label>
        <textarea
          value={scores.comments}
          onChange={(e) => onCommentsChange(e.target.value)}
          placeholder="Nhập nhận xét chi tiết..."
          className="w-full h-24 bg-slate-700 border border-slate-600 rounded-lg p-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
        />
      </div>

      {/* Summary of Scores */}
      <div className="bg-slate-700/30 rounded-lg p-4">
        <p className="text-sm font-bold text-slate-300 mb-2">Overall Average</p>
        <p className="text-3xl font-bold text-cyan-400">
          {(
            (scores.informativeness +
              scores.fluency +
              scores.coherence +
              scores.redundancy +
              scores.readability) / 5
          ).toFixed(2)}
          /5
        </p>
      </div>
    </div>
  );
};

export default HumanEvaluation;
