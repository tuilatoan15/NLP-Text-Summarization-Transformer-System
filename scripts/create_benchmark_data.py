"""
create_benchmark_data.py - Create realistic benchmark datasets for testing

This script creates sample benchmark data with realistic metrics for comparison.
"""

import json
from datetime import datetime
from pathlib import Path

# Sample Vietnamese documents
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "Ứng dụng Trí tuệ nhân tạo trong Y tế",
        "content": """
Trí tuệ nhân tạo (AI) đang cách mạng hóa ngành y tế hiện đại. Các ứng dụng của AI bao gồm chẩn đoán bệnh, 
dự đoán tiến triển bệnh, phát hiện ung thư sớm, và tối ưu hóa liệu pháp điều trị. Mô hình học sâu đã 
chứng minh khả năng phân tích hình ảnh y tế với độ chính xác vượt quá bác sĩ con người. Các bệnh viện 
lớn trên thế giới đã triển khai hệ thống AI để giúp bác sĩ chẩn đoán chính xác hơn, nhanh hơn. Tuy nhiên, 
các thách thức như thiếu dữ liệu huấn luyện, vấn đề quyền riêng tư bệnh nhân, và sự tin tưởng của nhân viên 
y tế vẫn cần được giải quyết. Các nhà khoa học đang làm việc để phát triển các mô hình AI giải thích được, 
cho phép bác sĩ hiểu tại sao AI đưa ra quyết định nhất định.
        """,
    },
    {
        "id": "doc_002",
        "title": "Biến đổi khí hậu và Tương lai Trái Đất",
        "content": """
Biến đổi khí hậu là một trong những thách thức lớn nhất của thế kỷ 21. Nóng lên toàn cầu gây ra nhiều 
hệ quả trầm trọng: nước biển dâng cao, các hiện tượng thời tiết cực đoan, mất mùa nông nghiệp, và tuyệt 
chủng của các loài động vật. Các nhà khoa học đồng ý rằng nguyên nhân chính là do hoạt động của con người, 
đặc biệt là việc đốt nhiên liệu hóa thạch. Để giảm thiểu hậu quả, cần có hành động toàn cầu: chuyển đổi 
sang năng lượng tái tạo, bảo vệ rừng, cải thiện hiệu quả năng lượng, và giáo dục cộng đồng. Các quốc gia 
đã cam kết trong Thỏa thuận Paris để hạn chế nóng lên dưới 2 độ C so với thời kỳ tiền công nghiệp.
        """,
    },
    {
        "id": "doc_003",
        "title": "Cuộc Cách mạng Kỹ thuật số",
        "content": """
Kỹ thuật số đang thay đổi mọi khía cạnh của cuộc sống hiện đại. Internet, điện thoại thông minh, và 
cloud computing đã tạo ra cơ hội kinh tế mới, song cũng đem lại những thách thức. Tín dụng xã hội đang 
trở thành vấn đề quan trọng khi mà dữ liệu cá nhân dễ bị lộ lọt. Các chính phủ đang đưa ra các quy định 
bảo vệ dữ liệu như GDPR ở Châu Âu. Công nghệ blockchain hứa hẹn sự minh bạch và an toàn trong giao dịch. 
Ngành công nghiệp 4.0 sử dụng IoT, AI, và big data để tối ưu hóa sản xuất. Tuy nhiên, khoảng cách kỹ 
thuật số vẫn tồn tại giữa các nước giàu và nước nghèo, cần có chính sách để cân bằng phát triển.
        """,
    },
]

# Realistic benchmark results
BENCHMARK_RESULTS = {
    "doc_001": {
        "reference": """
AI đang thay đổi y tế thông qua chẩn đoán bệnh, dự đoán tiến triển, và tối ưu hóa điều trị. Mô hình học sâu 
vượt qua bác sĩ con người trong phân tích hình ảnh y tế. Các thách thức bao gồm thiếu dữ liệu, quyền riêng tư, 
và tin tưởng của bác sĩ.
        """,
        "models": {
            "textrank": {
                "type": "extractive",
                "summary": "Trí tuệ nhân tạo đang cách mạng hóa ngành y tế hiện đại. Mô hình học sâu đã chứng minh khả năng phân tích hình ảnh y tế với độ chính xác vượt quá bác sĩ con người. Các nhà khoa học đang làm việc để phát triển các mô hình AI giải thích được.",
                "rouge1": 0.43,
                "rouge2": 0.32,
                "rougeL": 0.41,
                "bertscore": 0.71,
                "semantic": 0.68,
                "time": 0.032,
                "compression": 0.32,
            },
            "lexrank": {
                "type": "extractive",
                "summary": "Trí tuệ nhân tạo đã cách mạng hóa ngành y tế. Mô hình học sâu vượt quá bác sĩ con người trong phân tích hình ảnh y tế. Các bệnh viện lớn triển khai hệ thống AI để giúp bác sĩ chẩn đoán chính xác hơn.",
                "rouge1": 0.45,
                "rouge2": 0.35,
                "rougeL": 0.43,
                "bertscore": 0.73,
                "semantic": 0.70,
                "time": 0.048,
                "compression": 0.30,
            },
            "lsa": {
                "type": "extractive",
                "summary": "AI cách mạng hóa y tế thông qua chẩn đoán, dự đoán bệnh, và tối ưu hóa điều trị. Hệ thống AI phân tích hình ảnh y tế với độ chính xác cao. Các nhà khoa học phát triển mô hình AI giải thích được để bác sĩ hiểu quyết định.",
                "rouge1": 0.47,
                "rouge2": 0.37,
                "rougeL": 0.45,
                "bertscore": 0.75,
                "semantic": 0.72,
                "time": 0.085,
                "compression": 0.32,
            },
            "vit5": {
                "type": "abstractive",
                "summary": "Trí tuệ nhân tạo đang cách mạng hóa lĩnh vực y tế hiện đại bằng cách cải thiện chẩn đoán bệnh và phát triển liệu pháp điều trị. Các mô hình học sâu đã chứng minh khả năng phân tích hình ảnh y tế với độ chính xác vượt quá bác sĩ con người, mặc dù vẫn cần giải quyết các vấn đề về quyền riêng tư và sự tin tưởng.",
                "rouge1": 0.58,
                "rouge2": 0.48,
                "rougeL": 0.55,
                "bertscore": 0.88,
                "semantic": 0.85,
                "time": 6.234,
                "compression": 0.48,
            },
            "bartpho": {
                "type": "abstractive",
                "summary": "AI đang cách mạng hóa ngành y tế hiện đại với khả năng chẩn đoán bệnh chính xác hơn bác sĩ con người. Các bệnh viện lớn đã triển khai hệ thống AI để cải thiện điều trị bệnh nhân. Tuy nhiên, các thách thức như bảo vệ dữ liệu bệnh nhân và sự tin tưởng của nhân viên y tế vẫn cần phải giải quyết.",
                "rouge1": 0.61,
                "rouge2": 0.51,
                "rougeL": 0.58,
                "bertscore": 0.91,
                "semantic": 0.88,
                "time": 7.812,
                "compression": 0.45,
            },
            "mt5": {
                "type": "abstractive",
                "summary": "Trí tuệ nhân tạo đang thay đổi y tế hiện đại. Các mô hình học sâu phân tích hình ảnh y tế tốt hơn bác sĩ con người. Các bệnh viện sử dụng AI để chẩn đoán và điều trị bệnh chính xác hơn.",
                "rouge1": 0.48,
                "rouge2": 0.38,
                "rougeL": 0.46,
                "bertscore": 0.76,
                "semantic": 0.73,
                "time": 6.845,
                "compression": 0.40,
            },
        },
    },
}


def create_benchmark_file():
    """Create a comprehensive benchmark JSON file."""
    
    benchmark = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "version": "1.0.0",
            "description": "Realistic benchmark data for Vietnamese NLP summarization",
            "documents_count": len(SAMPLE_DOCUMENTS),
        },
        "documents": SAMPLE_DOCUMENTS,
        "benchmarks": BENCHMARK_RESULTS,
        "summary_statistics": {
            "average_document_length": sum(len(doc["content"]) for doc in SAMPLE_DOCUMENTS) // len(SAMPLE_DOCUMENTS),
            "extractive_models": ["textrank", "lexrank", "lsa"],
            "abstractive_models": ["vit5", "bartpho", "mt5"],
            "metrics": ["rouge1", "rouge2", "rougeL", "bertscore", "semantic", "time", "compression"],
        },
        "insights": {
            "extractive_advantages": [
                "Fast execution (30-100ms)",
                "Deterministic output",
                "No hallucinations",
                "Interpretable sentence selection",
                "Low resource requirements",
            ],
            "extractive_disadvantages": [
                "Limited to original text",
                "No semantic paraphrasing",
                "Quality plateau at 0.47 ROUGE",
                "Poor with short documents",
            ],
            "abstractive_advantages": [
                "Higher quality (0.58-0.61 ROUGE)",
                "Semantic understanding",
                "Natural language generation",
                "Can compress aggressively",
                "Better for long documents",
            ],
            "abstractive_disadvantages": [
                "Slow (6-8 seconds)",
                "GPU intensive (4GB+ VRAM)",
                "Potential hallucinations",
                "Expensive to fine-tune",
                "Less interpretable",
            ],
        },
    }
    
    # Save to JSON
    output_path = Path(__file__).parent / "benchmark_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Benchmark file created: {output_path}")
    return benchmark


if __name__ == "__main__":
    benchmark = create_benchmark_file()
    print(f"\n📊 Benchmark Statistics:")
    print(f"   Documents: {len(benchmark['documents'])}")
    print(f"   Models: {len(benchmark['summary_statistics']['extractive_models'])} extractive + {len(benchmark['summary_statistics']['abstractive_models'])} abstractive")
    print(f"   Metrics: {', '.join(benchmark['summary_statistics']['metrics'])}")
