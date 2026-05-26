"""Service to compile evaluations, explainability networks, and assets into beautiful Markdown/HTML reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReportGenerator:
    """Generate professional research-grade PDF-ready HTML and Markdown reports."""

    def generate_html(self, payload: dict[str, Any], compare_results: dict[str, Any] | None = None) -> str:
        doc_id = payload.get("document_id", "doc")
        metadata = payload.get("metadata", {})
        quality = payload.get("quality", {}).get("extraction", {})
        assets = payload.get("analysis_assets", {})
        overview = assets.get("overview", {})

        title = metadata.get("title") or "Báo cáo phân tích tài liệu"
        filename = metadata.get("filename") or doc_id
        source_type = metadata.get("source_type") or "txt"
        pages = metadata.get("pages") or 1
        word_count = quality.get("word_count") or len(payload.get("clean_text", "").split())

        # Construct ROUGE/BLEU comparative table
        results_rows = ""
        best_model_name = "N/A"
        best_model_score = 0.0

        if compare_results:
            ranking = compare_results.get("ranking", [])
            if ranking:
                best_model_name = ranking[0].get("algorithm", "N/A")
                best_model_score = ranking[0].get("combined_score", 0.0)

            for idx, r in enumerate(compare_results.get("results", []), start=1):
                m = r.get("metrics", {})
                results_rows += f"""
                <tr class="border-b border-slate-700 hover:bg-slate-800/50 transition">
                    <td class="p-3 text-slate-300 font-medium">{r.get('algorithm')}</td>
                    <td class="p-3 text-emerald-400 font-mono text-center">{m.get('rouge1', 0.0):.4f}</td>
                    <td class="p-3 text-emerald-400 font-mono text-center">{m.get('rouge2', 0.0):.4f}</td>
                    <td class="p-3 text-emerald-400 font-mono text-center">{m.get('rougeL', 0.0):.4f}</td>
                    <td class="p-3 text-emerald-400 font-mono text-center">{m.get('bleu', 0.0):.4f}</td>
                    <td class="p-3 text-sky-400 font-mono text-center">{m.get('bertscore_f1', 0.0):.4f}</td>
                    <td class="p-3 text-sky-400 font-mono text-center">{m.get('semantic_similarity', 0.0):.4f}</td>
                    <td class="p-3 text-amber-400 font-mono text-center">{m.get('processing_time', 0.0):.3f}s</td>
                </tr>
                """
        else:
            results_rows = """
            <tr>
                <td colspan="8" class="p-4 text-center text-slate-400">Chưa chạy phân tích so sánh thuật toán.</td>
            </tr>
            """

        # Quiz rendering
        quiz_html = ""
        quiz_list = assets.get("quiz") or []
        for q in quiz_list:
            opts = ""
            if q.get("options"):
                for idx, opt in enumerate(q["options"]):
                    opts += f"""
                    <div class="flex items-center space-x-2 mt-1">
                        <span class="w-5 h-5 rounded-full bg-slate-800 text-slate-400 text-xs flex items-center justify-center font-bold">{chr(65+idx)}</span>
                        <span class="text-slate-300 text-sm">{opt}</span>
                    </div>
                    """
            quiz_html += f"""
            <div class="p-4 rounded-xl bg-slate-800 border border-slate-700/60 shadow-inner">
                <div class="flex justify-between text-xs text-slate-400 mb-1 font-semibold">
                    <span>Độ khó: {q.get('difficulty', 'Medium').capitalize()}</span>
                    <span>Dạng: {q.get('type', 'Multiple Choice').replace('_', ' ').capitalize()}</span>
                </div>
                <p class="text-slate-200 font-medium text-sm mb-2">{q.get('question')}</p>
                {opts}
                <details class="mt-2 text-xs">
                    <summary class="text-emerald-400 hover:text-emerald-300 cursor-pointer font-bold select-none focus:outline-none">Xem đáp án</summary>
                    <p class="mt-1 text-emerald-300 bg-emerald-950/40 p-2 rounded border border-emerald-900/60 font-mono">Đáp án: {q.get('answer')}</p>
                </details>
            </div>
            """

        # Podcast turns
        podcast_html = ""
        podcast_script = assets.get("podcast") or {}
        for turn in podcast_script.get("turns") or []:
            spk = turn.get("speaker", "Host")
            spk_color = "text-emerald-400" if "A" in spk or "1" in spk else "text-sky-400"
            podcast_html += f"""
            <div class="flex flex-col space-y-1 p-3 rounded-lg bg-slate-800/40 border-l-4 border-slate-600">
                <span class="text-xs font-bold uppercase tracking-wider {spk_color}">{spk}</span>
                <p class="text-slate-300 text-sm">{turn.get('text')}</p>
            </div>
            """

        # Citations checklist (grounding)
        citations_html = ""
        best_result = None
        if compare_results:
            best_model_key = (compare_results.get("best_model") or {}).get("key")
            best_result = next((r for r in compare_results.get("results", []) if r.get("key") == best_model_key), None)
        
        if best_result and best_result.get("citations"):
            for cite in best_result["citations"]:
                status_color = "bg-emerald-500/20 text-emerald-300 border-emerald-800" if cite.get("status") == "grounded" else "bg-amber-500/20 text-amber-300 border-amber-800"
                citations_html += f"""
                <div class="p-3 rounded-xl bg-slate-900/40 border border-slate-700/60 flex flex-col space-y-2">
                    <div class="flex justify-between items-center">
                        <span class="text-xs font-semibold px-2 py-0.5 rounded-full border {status_color}">{cite.get('status', 'needs_review').upper()}</span>
                        <span class="text-xs text-slate-400 font-mono">Độ tương đồng nguồn: {cite.get('best_support_score', 0.0):.2f}</span>
                    </div>
                    <p class="text-slate-200 text-sm italic">"{cite.get('sentence')}"</p>
                    <details class="text-xs">
                        <summary class="text-slate-400 hover:text-slate-300 cursor-pointer select-none">Xem câu gốc đối chiếu</summary>
                        <div class="mt-2 p-2 bg-slate-900 rounded border border-slate-800 text-slate-400 text-xs">
                            {cite.get('evidence', [{}])[0].get('excerpt', 'Không tìm thấy trích dẫn tương ứng')}
                        </div>
                    </details>
                </div>
                """
        else:
            citations_html = "<p class='text-slate-400 text-center italic py-4'>Chưa có dữ liệu trích dẫn grounding.</p>"

        # Complete styled HTML template using TailwindCSS
        return f"""<!DOCTYPE html>
<html lang="vi" class="h-full bg-slate-950 text-slate-100">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo cáo khoa học AI - {title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Outfit', sans-serif;
        }}
        pre, code, .font-mono {{
            font-family: 'JetBrains Mono', monospace;
        }}
        @media print {{
            body {{
                background-color: white !important;
                color: black !important;
            }}
            .no-print {{
                display: none !important;
            }}
            .page-break {{
                page-break-before: always;
            }}
        }}
    </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 pb-20">
    
    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur-xl sticky top-0 z-50 no-print">
        <div class="max-w-6xl mx-auto px-6 py-4 flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <span class="text-2xl font-extrabold bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent">Antigravity AI</span>
                <span class="text-xs px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400 uppercase tracking-widest font-bold">Document Report</span>
            </div>
            <button onclick="window.print()" class="px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold transition shadow-lg shadow-emerald-500/20 text-sm">
                🖨️ Xuất PDF / In báo cáo
            </button>
        </div>
    </header>

    <main class="max-w-5xl mx-auto px-6 mt-12 space-y-12">
        
        <!-- Cover Section -->
        <section class="p-8 rounded-2xl bg-gradient-to-r from-slate-900 to-slate-900/60 border border-slate-800/80 shadow-2xl relative overflow-hidden">
            <div class="absolute -right-24 -top-24 w-80 h-80 rounded-full bg-emerald-500/10 blur-3xl"></div>
            <div class="space-y-6 relative">
                <div class="flex flex-wrap gap-2">
                    <span class="text-xs px-2.5 py-1 rounded-full bg-slate-800 border border-slate-700 text-slate-300 font-semibold uppercase tracking-wider">AI Document Intelligence</span>
                    <span class="text-xs px-2.5 py-1 rounded-full bg-emerald-950/60 border border-emerald-900 text-emerald-400 font-semibold uppercase tracking-wider">Research-Grade</span>
                </div>
                <h1 class="text-4xl font-extrabold text-slate-100 leading-tight">{title}</h1>
                
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-6 pt-6 border-t border-slate-800/60">
                    <div>
                        <span class="block text-xs text-slate-500 uppercase tracking-wider font-semibold">Tên tệp nguồn</span>
                        <span class="text-slate-300 font-medium text-sm font-mono truncate block mt-1">{filename}</span>
                    </div>
                    <div>
                        <span class="block text-xs text-slate-500 uppercase tracking-wider font-semibold">Độ dài tài liệu</span>
                        <span class="text-slate-300 font-medium text-sm block mt-1">{word_count} từ ({pages} trang)</span>
                    </div>
                    <div>
                        <span class="block text-xs text-slate-500 uppercase tracking-wider font-semibold">Thuật toán tối ưu</span>
                        <span class="text-emerald-400 font-bold text-sm block mt-1">{best_model_name}</span>
                    </div>
                    <div>
                        <span class="block text-xs text-slate-500 uppercase tracking-wider font-semibold">Điểm chất lượng</span>
                        <span class="text-sky-400 font-bold text-sm block mt-1">{best_model_score:.4f}</span>
                    </div>
                </div>
            </div>
        </section>

        <!-- Executive Summary & Key Insights -->
        <section class="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div class="md:col-span-2 space-y-4 p-6 rounded-2xl bg-slate-900 border border-slate-800">
                <h2 class="text-xl font-bold text-slate-200 border-b border-slate-800 pb-2 flex items-center space-x-2">
                    <span>📝 Tóm tắt tổng quan</span>
                </h2>
                <p class="text-slate-300 leading-relaxed text-sm">{overview.get('document_overview', 'Không tìm thấy tổng quan.')}</p>
            </div>
            
            <div class="space-y-4 p-6 rounded-2xl bg-slate-900 border border-slate-800">
                <h2 class="text-xl font-bold text-slate-200 border-b border-slate-800 pb-2 flex items-center space-x-2">
                    <span>💡 Nhận định cốt lõi</span>
                </h2>
                <ul class="space-y-3">
                    {"".join(f'<li class="text-slate-300 text-sm flex items-start space-x-2"><span class="text-emerald-400 font-bold mt-0.5">•</span><span>{insight}</span></li>' for insight in (overview.get('key_insights') or []))}
                </ul>
            </div>
        </section>

        <!-- Model Evaluation Matrix -->
        <section class="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-6 page-break">
            <h2 class="text-xl font-bold text-slate-200 border-b border-slate-800 pb-2">
                📊 Ma trận so sánh & Đánh giá nghiên cứu
            </h2>
            <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/40">
                <table class="w-full text-left border-collapse text-sm">
                    <thead>
                        <tr class="bg-slate-800 text-slate-300 border-b border-slate-700">
                            <th class="p-3 font-semibold">Thuật toán</th>
                            <th class="p-3 font-semibold text-center">ROUGE-1</th>
                            <th class="p-3 font-semibold text-center">ROUGE-2</th>
                            <th class="p-3 font-semibold text-center">ROUGE-L</th>
                            <th class="p-3 font-semibold text-center">BLEU</th>
                            <th class="p-3 font-semibold text-center">BERTScore</th>
                            <th class="p-3 font-semibold text-center">Semantic</th>
                            <th class="p-3 font-semibold text-center">Thời gian chạy</th>
                        </tr>
                    </thead>
                    <tbody>
                        {results_rows}
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Citations & Grounding -->
        <section class="grid grid-cols-1 md:grid-cols-2 gap-8 page-break">
            <div class="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
                <h2 class="text-xl font-bold text-slate-200 border-b border-slate-800 pb-2">
                    ⚖️ Đối chiếu nguồn (Citation Grounding)
                </h2>
                <div class="space-y-3">
                    {citations_html}
                </div>
            </div>

            <!-- AI Podcast Script Preview -->
            <div class="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
                <h2 class="text-xl font-bold text-slate-200 border-b border-slate-800 pb-2">
                    🎙️ Kịch bản Audio Podcast (NotebookLM)
                </h2>
                <div class="space-y-4 max-h-[380px] overflow-y-auto pr-2 custom-scrollbar">
                    {podcast_html}
                </div>
            </div>
        </section>

        <!-- Learning Quiz -->
        <section class="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-6 page-break">
            <h2 class="text-xl font-bold text-slate-200 border-b border-slate-800 pb-2">
                🧠 Câu hỏi tương tác (AI Quiz)
            </h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                {quiz_html}
            </div>
        </section>

    </main>

    <!-- Footer -->
    <footer class="max-w-5xl mx-auto px-6 mt-16 pt-8 border-t border-slate-800 text-center text-slate-600 text-xs">
        <p>Báo cáo tự động được tạo bởi Antigravity AI Document Intelligence Platform.</p>
        <p class="mt-1">Dữ liệu đánh giá khoa học dựa trên tiêu chuẩn nghiên cứu NLP quốc tế.</p>
    </footer>
</body>
</html>
"""

    def generate_markdown(self, payload: dict[str, Any], compare_results: dict[str, Any] | None = None) -> str:
        doc_id = payload.get("document_id", "doc")
        metadata = payload.get("metadata", {})
        quality = payload.get("quality", {}).get("extraction", {})
        assets = payload.get("analysis_assets", {})
        overview = assets.get("overview", {})

        title = metadata.get("title") or "Báo cáo phân tích tài liệu"
        word_count = quality.get("word_count") or len(payload.get("clean_text", "").split())
        pages = metadata.get("pages") or 1

        md = f"""# BÁO CÁO PHÂN TÍCH KHOA HỌC — AI DOCUMENT INTELLIGENCE
**Tài liệu:** {title}
**Tên tệp:** {metadata.get('filename') or doc_id}
**Độ dài:** {word_count} từ ({pages} trang)
***

## 1. Tóm tắt tổng quan (Executive Summary)
{overview.get('document_overview', 'Không có tổng quan.')}

### Nhận định cốt lõi (Key Takeaways):
"""
        for insight in (overview.get("key_insights") or []):
            md += f"- {insight}\n"

        md += "\n## 2. Kết quả đánh giá mô hình & so sánh thuật toán\n"
        if compare_results:
            md += "| Thuật toán | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore | Semantic | Thời gian |\n"
            md += "|---|---|---|---|---|---|---|---|\n"
            for r in compare_results.get("results", []):
                m = r.get("metrics", {})
                md += f"| {r.get('algorithm')} | {m.get('rouge1', 0.0):.4f} | {m.get('rouge2', 0.0):.4f} | {m.get('rougeL', 0.0):.4f} | {m.get('bleu', 0.0):.4f} | {m.get('bertscore_f1', 0.0):.4f} | {m.get('semantic_similarity', 0.0):.4f} | {m.get('processing_time', 0.0):.3f}s |\n"
        else:
            md += "*Chưa thực hiện phân tích so sánh.*\n"

        md += "\n## 3. Câu hỏi ôn tập (AI Quiz)\n"
        for q in (assets.get("quiz") or []):
            md += f"### Câu hỏi {q.get('id')}: {q.get('question')}\n"
            if q.get("options"):
                for idx, opt in enumerate(q["options"]):
                    md += f"- **{chr(65+idx)}.** {opt}\n"
            md += f"*Đáp án:* {q.get('answer')}\n\n"

        md += "\n## 4. Kịch bản Audio Podcast\n"
        for turn in (assets.get("podcast", {}).get("turns") or []):
            md += f"**{turn.get('speaker')}:** {turn.get('text')}\n\n"

        md += f"\n***\n*Báo cáo được tạo tự động bởi Antigravity AI Document Intelligence Platform.*"
        return md
