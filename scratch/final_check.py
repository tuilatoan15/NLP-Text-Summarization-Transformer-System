import requests
import json
import time

BASE_URL = 'http://127.0.0.1:8000'

def check():
    print("--- 1. Health Check ---")
    try:
        resp = requests.get(f"{BASE_URL}/health")
        data = resp.json()
        print(f"Status: {data.get('status')}")
        print(f"Preload: {data.get('model_status')}")
        print(f"Device: {data.get('registry', {}).get('device')}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- 2. Summarization Test (ViT5 + TextRank) ---")
    payload = {
        "text": "Trí tuệ nhân tạo (AI) đang là nòng cốt của cuộc cách mạng công nghiệp 4.0. Tại Việt Nam, nhiều doanh nghiệp đã ứng dụng AI vào quy trình sản xuất và chăm sóc khách hàng. Việc tối ưu hóa các mô hình ngôn ngữ lớn giúp hệ thống tóm tắt văn bản hoạt động nhanh và chính xác hơn.",
        "algorithms": ["textrank", "vit5"],
        "extractive_sentences": 1
    }
    try:
        start = time.time()
        resp = requests.post(f"{BASE_URL}/summarize/compare", json=payload)
        elapsed = time.time() - start
        data = resp.json()
        
        print(f"API Wall Time: {elapsed:.2f}s")
        for res in data.get('results', []):
            algo = res.get('algorithm')
            summary = res.get('summary', '')[:100]
            print(f" - [{algo}]: {summary}...")
            
        perf = data.get('performance', {})
        print(f"Total Logic Time: {perf.get('total_wall_time_s')}s")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check()
