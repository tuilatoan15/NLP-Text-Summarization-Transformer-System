import requests
import json
import sys

def test_api():
    print("=== Testing Detailed Comparison Endpoint ===")
    url = 'http://127.0.0.1:8000/research/compare/detailed'
    payload = {
        "text": (
            "Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình "
            "leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay lập "
            "tức và mở hành lang nhân đạo cho người dân vùng chiến sự. Đại diện Mỹ phát biểu "
            "rằng Washington ủng hộ giải pháp hai nhà nước nhưng nhấn mạnh quyền tự vệ hợp "
            "pháp. Nga và Trung Quốc phản đối dự thảo nghị quyết, cho rằng văn kiện còn "
            "thiếu cân bằng. Cuộc khủng hoảng nhân đạo ngày càng nghiêm trọng khi hàng nghìn "
            "thường dân phải di tản. Các tổ chức phi chính phủ kêu gọi cộng đồng quốc tế "
            "hành động khẩn cấp để bảo vệ dân thường."
        ),
        "reference": (
            "Liên Hợp Quốc họp khẩn bàn về tình hình Trung Đông. Các nước kêu gọi ngừng bắn "
            "ngay lập tức để mở hành lang nhân đạo cứu trợ dân thường trong vùng chiến sự."
        ),
        "extractive_sentences": 3,
        "max_abstractive_length": 80,
        "target_length_ratio": 50,
        "use_length_ratio": False,
        "include_visualization": True,
        "save_result": True
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=60)
        print('Status:', resp.status_code)
        if resp.status_code == 200:
            data = resp.json()
            print("Successfully received detailed comparison!")
            print("Extractive models returned:", list(data.get("extractive_results", {}).keys()))
            print("Abstractive models returned:", list(data.get("abstractive_results", {}).keys()))
            
            # Print summaries to confirm they are distinct and not all TextRank fallbacks
            print("\n--- TextRank Summary ---")
            print(data.get("extractive_results", {}).get("textrank", {}).get("summary"))
            
            print("\n--- ViT5 Summary ---")
            print(data.get("abstractive_results", {}).get("vit5", {}).get("summary"))
            
            print("\n--- BARTPho Summary ---")
            print(data.get("abstractive_results", {}).get("bartpho", {}).get("summary"))
        else:
            print("Failed. Details:", resp.text)
    except Exception as e:
        print("API Request failed:", e)

def test_dashboard():
    print("\n=== Testing Analytics Dashboard Endpoint ===")
    url = 'http://127.0.0.1:8000/analytics/dashboard?time_range=30d&limit=10'
    try:
        resp = requests.get(url, timeout=10)
        print('Status:', resp.status_code)
        if resp.status_code == 200:
            print("Dashboard loaded successfully without 500 errors!")
            data = resp.json()
            print("Keys returned:", list(data.keys()))
        else:
            print("Failed. Details:", resp.text)
    except Exception as e:
        print("Dashboard Request failed:", e)

if __name__ == "__main__":
    test_api()
    test_dashboard()
